"""
Quick integration test untuk endpoint /inspect.
Menggunakan TestClient FastAPI + mock google.generativeai
agar tidak memerlukan GEMINI_API_KEY asli.
"""

import io
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path

# pastikan folder app bisa di-import
sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

# import setelah path diset
from main import app, predict_defects, _format_detection_summary
from fastapi.testclient import TestClient


# ── helper: dapatkan satu sampel gambar dari uploads/ ──────────────────────
UPLOADS = Path(__file__).resolve().parent / "uploads"


def sample_image_bytes():
    """Return bytes dari gambar sampel pertama yang ditemukan di uploads/."""
    jpgs = list(UPLOADS.glob("*.jpg")) + list(UPLOADS.glob("*.jpeg"))
    pngs = list(UPLOADS.glob("*.png"))
    candidates = jpgs + pngs
    if not candidates:
        raise FileNotFoundError("Tidak ada gambar di uploads/")
    return candidates[0].read_bytes()


# ── test fungsi format ──────────────────────────────────────────────────────
class TestFormatDetectionSummary(unittest.TestCase):
    def test_empty(self):
        out = _format_detection_summary([])
        self.assertIn("Tidak ada cacat", out)

    def test_with_defects(self):
        from main import DefectDetection

        d = [
            DefectDetection(label="Short Shot", confidence=0.87, bbox=[10, 20, 100, 200]),
            DefectDetection(label="Sink Mark", confidence=0.65, bbox=[300, 150, 380, 210]),
        ]
        out = _format_detection_summary(d)
        self.assertIn("Short Shot", out)
        self.assertIn("0.870", out)
        self.assertIn("[10.0, 20.0, 100.0, 200.0]", out)
        self.assertIn("Total cacat terdeteksi: 2", out)


# ── test endpoint /inspect (dengan mock Gemini) ────────────────────────────
class TestInspectEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # load YOLO sekali saja agar tidak reload berulang
        from main import load_model

        load_model()

    def test_inspect_no_defects(self):
        """Gunakan gambar yang tidak ada cacat (atau hasil YOLO kosong)."""
        img_bytes = sample_image_bytes()
        with patch("google.generativeai.configure"):
            with patch("google.generativeai.GenerativeModel") as mock_model_cls:
                mock_model = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "Laporan test"
                mock_model.generate_content.return_value = mock_response
                mock_model_cls.return_value = mock_model

                # set GEMINI_API_KEY di main supaya tidak return early
                import main as main_module

                main_module.GEMINI_API_KEY = "mock-key"

                with TestClient(app) as client:
                    resp = client.post(
                        "/inspect",
                        files={"file": ("sample.jpg", img_bytes, "image/jpeg")},
                    )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("defects", data)
        self.assertIn("qa_report", data)
        self.assertIn("image_id", data)
        print(f"[TEST] no-defect response keys: {list(data.keys())}")

    def test_inspect_with_defect_and_gemini(self):
        """Test alur lengkap: YOLO detect → Gemini report."""
        img_bytes = sample_image_bytes()

        # mock YOLO supaya return defects (karena COCO umum tidak tahu cacat plastik)
        from main import DefectDetection

        fake_defects = [
            DefectDetection(label="Short Shot", confidence=0.87, bbox=[10, 20, 100, 200]),
            DefectDetection(label="Sink Mark", confidence=0.65, bbox=[300, 150, 380, 210]),
        ]

        with patch("google.generativeai.configure"):
            with patch("google.generativeai.GenerativeModel") as mock_model_cls:
                mock_model = MagicMock()
                mock_response = MagicMock()
                mock_response.text = (
                    "Laporan investigasi: ditemukan Short Shot. "
                    "Rekomendasi: naikkan suhu barrel."
                )
                mock_model.generate_content.return_value = mock_response
                mock_model_cls.return_value = mock_model

                # mock predict_defects agar return defects palsu
                with patch("main.predict_defects", return_value=(fake_defects, 640, 480)):
                    # set GEMINI_API_KEY di main supaya tidak return early
                    import main as main_module

                    main_module.GEMINI_API_KEY = "mock-key"

                    with TestClient(app) as client:
                        resp = client.post(
                            "/inspect",
                            files={"file": ("sample.jpg", img_bytes, "image/jpeg")},
                        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["filename"], "sample.jpg")
        self.assertIn("defects", data)
        self.assertIn("qa_report", data)
        self.assertEqual(data["qa_report"], "Laporan investigasi: ditemukan Short Shot. Rekomendasi: naikkan suhu barrel.")
        print(f"[TEST] full flow OK — defects={len(data['defects'])}, qa_report present={bool(data['qa_report'])}")


# ── test Gemini module presence ─────────────────────────────────────────────
class TestGeminiIntegration(unittest.TestCase):
    def test_import_genai(self):
        try:
            import google.generativeai as genai  # noqa: F401
        except ImportError:
            self.fail("google-generativeai tidak terinstal")

    def test_gemini_model_name(self):
        """Sini hanya dijalankan jika GEMINI_API_KEY tersedia."""
        import os

        if not os.getenv("GEMINI_API_KEY"):
            self.skipTest("GEMINI_API_KEY tidak diatur — skip test list_models")

        import google.generativeai as genai

        models = genai.list_models()
        model_names = [m.name for m in models]
        self.assertIn("models/gemini-1.5-flash", model_names)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ── tambahan: edge-case & schema validation (jalan juga via unittest) ──────
class TestEdgeCases(unittest.TestCase):
    def test_reject_unsupported_format(self):
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            resp = client.post(
                "/inspect",
                files={"file": ("test.txt", b"bukan gambar", "text/plain")},
            )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Format tidak didukung", resp.json()["detail"])

    def test_reject_too_small_file(self):
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            resp = client.post(
                "/inspect",
                files={"file": ("tiny.jpg", b"abc", "image/jpeg")},
            )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("terlalu kecil", resp.json()["detail"])

    def test_response_schema_has_required_fields(self):
        """Cek bahwa response /inspect memuat field wajib sesuai InspectResponse."""
        import io
        from main import DefectDetection

        img_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200  # bytes PNG skeleton fake

        fake_defects = [
            DefectDetection(label="Short Shot", confidence=0.87, bbox=[10, 20, 100, 200]),
        ]

        with patch("google.generativeai.configure"):
            with patch("google.generativeai.GenerativeModel") as mock_model_cls:
                mock_model = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "Laporan uji schema."
                mock_model.generate_content.return_value = mock_response
                mock_model_cls.return_value = mock_model

                with patch("main.predict_defects", return_value=(fake_defects, 640, 480)):
                    import main as main_module

                    main_module.GEMINI_API_KEY = "mock-key"

                    with TestClient(app) as client:
                        resp = client.post(
                            "/inspect",
                            files={"file": ("sample.png", img_bytes, "image/png")},
                        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        required = {
            "image_id",
            "filename",
            "defects",
            "defect_count",
            "has_defects",
            "qa_report",
            "inference_time_ms",
            "gemini_latency_ms",
        }
        missing = required - set(data.keys())
        self.assertEqual(missing, set(), f"Field hilang: {missing}")
        self.assertEqual(data["filename"], "sample.png")
        self.assertEqual(len(data["defects"]), 1)
        self.assertEqual(data["defects"][0]["label"], "Short Shot")
        self.assertAlmostEqual(data["defects"][0]["confidence"], 0.87, places=2)

