"""
Automated QA Reporter — FastAPI + YOLOv8 + Gemini Generative AI
Endpoint:
  POST /inspect   → upload gambar, jalankan inferensi YOLOv8,
                    format hasil deteksi → kirim ke Gemini (gemini-1.5-flash),
                    kembalikan JSON: daftar cacat + laporan investigasi AI.
"""

import os
import io
import uuid
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# ── config ──────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# GEMINI_API_KEY diambil dari environment atau file .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# ── app ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Automated QA Reporter",
    description=(
        "Backend laporan inspeksi kualitas otomatis: "
        "YOLOv8 untuk deteksi cacat + Gemini AI untuk generate laporan investigasi."
    ),
    version="1.0.0",
)


# ── Pydantic schema ──────────────────────────────────────────────────────────
class DefectDetection(BaseModel):
    label: str = Field(..., description="Nama class yang terdeteksi")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score model"
    )
    bbox: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box [x_min, y_min, x_max, y_max] dalam piksel",
    )


class InspectResponse(BaseModel):
    image_id: str
    filename: str
    image_width: int
    image_height: int
    defects: List[DefectDetection]
    defect_count: int
    has_defects: bool
    detection_summary: Optional[str] = None
    qa_report: Optional[str] = None
    inference_time_ms: Optional[float] = None
    gemini_latency_ms: Optional[float] = None


# ── YOLO model (lazy-load) ──────────────────────────────────────────────────
from ultralytics import YOLO

_model_store: dict = {"model": None}


def load_model() -> None:
    """Load YOLOv8 nano dari ultralytics. Dipanggil sekali saat startup."""
    model_path = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
    print(f"[QA-REPORTER] Loading YOLOv8 model: {model_path}")
    _model_store["model"] = YOLO(model_path)
    names = _model_store["model"].names
    print(f"[QA-REPORTER] Model loaded — {len(names)} class terregister")
    print(f"[QA-REPORTER] Sample class: {list(names.items())[:5]}")


def get_model():
    if _model_store["model"] is None:
        load_model()
    return _model_store["model"]


def predict_defects(image_bytes: bytes) -> tuple:
    """
    Jalankan inferensi YOLOv8 pada image bytes.
    Threshold confidence: 0.25. Mengembalikan (defects, width, height).
    """
    import cv2
    import numpy as np

    img_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        return [], 0, 0

    h, w = img.shape[:2]

    model = get_model()
    results = model.predict(img, verbose=False, conf=0.25)
    defects = []
    for result in results[0].boxes:
        x_min, y_min, x_max, y_max = map(float, result.xyxy[0].tolist())
        defects.append(
            DefectDetection(
                label=model.names[int(result.cls[0])],
                confidence=round(float(result.conf[0]), 3),
                bbox=[
                    round(x_min, 1),
                    round(y_min, 1),
                    round(x_max, 1),
                    round(y_max, 1),
                ],
            )
        )
    return defects, w, h


# ── Gemini AI ────────────────────────────────────────────────────────────────
def generate_qa_report(detections: List[DefectDetection]) -> str:
    """
    Kirim data deteksi ke Gemini (gemini-1.5-flash) dan dapatkan
    laporan investigasi QA dalam bahasa Indonesia.
    """
    if not GEMINI_API_KEY:
        return (
            "[GEMINI API KEY TIDAK DIATUR] "
            "Set environment variable GEMINI_API_KEY atau masukkan ke file .env"
        )

    try:
        import google.generativeai as genai
    except ImportError:
        return "[LIBRARY TIDAK TERSEDIA] Install: pip install google-generativeai"

    genai.configure(api_key=GEMINI_API_KEY)

    # ── format data deteksi sebagai teks ──────────────────────────────────
    detection_text = _format_detection_summary(detections)

    # ── system instruction ────────────────────────────────────────────────
    system_instruction = (
        "Kamu adalah Senior QA Engineer di pabrik manufaktur plastik injection molding. "
        "Tugasmu: menyusun laporan investigasi cacat produk plastik berdasarkan data "
        "deteksi objek yang diberikan.\n\n"
        "Syarat laporan:\n"
        "1. Gunakan bahasa Indonesia formal dan profesional.\n"
        "2. Mulailah dengan ringkasan temuan (defect summary).\n"
        "3. Lakukan analisis akar masalah (root cause analysis) untuk setiap cacat "
        "yang terdeteksi, hubungkan dengan parameter mesin injeksi plastik yang umum "
        "(suhu leher tabung/barrel, tekanan injection, kecepatan injection, waktu "
        "dwelling, suhu mold, cooling time, viskositas material, dll).\n"
        "4. Berikan rekomendasi perbaikan parameter mesin yang spesifik dan actionable.\n"
        "5. Jika tidak ada cacat yang terdeteksi, tulis bahwa produk PASS dan tidak "
        "perlu tindakan korektif.\n"
        "6. Gunakan format paragraf yang rapi, bukan bullet point yang berlebihan.\n"
        "7. Panjang laporan: 1-2 paragraf padat yang informatif."
    )

    # ── konfigurasi model ──────────────────────────────────────────────────
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_instruction,
    )

    user_prompt = (
        "Berikut adalah data deteksi cacat dari hasil inferensi YOLOv8:\n\n"
        f"{detection_text}\n\n"
        "Silakan susun laporan investigasi QA berdasarkan data tersebut."
    )

    t0 = time.perf_counter()
    response = model.generate_content(user_prompt)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    report_text = response.text.strip()

    # simpan latency ke response (di-set oleh caller)
    generate_qa_report._last_latency = latency_ms  # type: ignore[attr-defined]
    return report_text


def _format_detection_summary(detections: List[DefectDetection]) -> str:
    """Format list deteksi ke string teks untuk dikirim ke Gemini."""
    if not detections:
        return "Tidak ada cacat atau objek yang terdeteksi."

    lines = ["Hasil deteksi object detection (YOLOv8):", ""]
    for i, d in enumerate(detections, 1):
        bbox = d.bbox
        lines.append(
            f"  {i}. {d.label} (confidence: {d.confidence:.3f})\n"
            f"     Bounding box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}] piksel"
        )
    lines.append("")
    lines.append(
        f"Total cacat terdeteksi: {len(detections)}. "
        f"Confidence threshold: 0.25"
    )
    return "\n".join(lines)


# ── endpoint: /inspect ──────────────────────────────────────────────────────
@app.post("/inspect", response_model=InspectResponse)
async def inspect_image(file: UploadFile = File(...)):
    """
    Upload gambar produk plastic injection → jalankan YOLOv8 inference →
    kirim hasil ke Gemini AI → kembalikan daftar cacat + laporan investigasi.
    """
    # validasi tipe file
    allowed = {".jpg", ".jpeg", ".png", ".webp"}
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Format tidak didukung. Gunakan: {', '.join(allowed)}",
        )

    contents = await file.read()
    if len(contents) < 1024:
        raise HTTPException(
            status_code=400, detail="File terlalu kecil — mungkin bukan gambar"
        )

    image_id = str(uuid.uuid4())

    # simpan untuk audit trail
    save_path = UPLOAD_DIR / f"{image_id}{ext}"
    save_path.write_bytes(contents)

    # ── YOLO inference ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    defects, image_width, image_height = predict_defects(contents)
    inference_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    has_defects = len(defects) > 0
    detection_summary = _format_detection_summary(defects) if has_defects else None

    # ── Gemini report ──────────────────────────────────────────────────────
    qa_report = None
    gemini_latency_ms = None
    if has_defects:
        qa_report = generate_qa_report(defects)
        gemini_latency_ms = getattr(generate_qa_report, "_last_latency", None)  # type: ignore[attr-defined]

    return InspectResponse(
        image_id=image_id,
        filename=file.filename or "unknown",
        image_width=image_width,
        image_height=image_height,
        defects=defects,
        defect_count=len(defects),
        has_defects=has_defects,
        detection_summary=detection_summary,
        qa_report=qa_report,
        inference_time_ms=inference_time_ms,
        gemini_latency_ms=gemini_latency_ms,
    )


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Compatibility endpoint for the existing upload and live-inspection pages."""
    result = await inspect_image(file)
    width = max(result.image_width, 1)
    height = max(result.image_height, 1)
    return {
        "image_id": result.image_id,
        "filename": result.filename,
        "image_width": result.image_width,
        "image_height": result.image_height,
        "status": "Fail" if result.has_defects else "Pass",
        "defects": [
            {
                "label": defect.label,
                "confidence": defect.confidence,
                "bbox": [
                    defect.bbox[0] / width,
                    defect.bbox[1] / height,
                    defect.bbox[2] / width,
                    defect.bbox[3] / height,
                ],
            }
            for defect in result.defects
        ],
        "inference_time_ms": result.inference_time_ms,
    }


# ── health / info ────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model_store["model"] is not None,
        "gemini_api_configured": bool(GEMINI_API_KEY),
        "service": "automated-qa-reporter",
    }


@app.get("/api/info")
async def info():
    return {
        "service": "Automated QA Reporter",
        "endpoints": {
            "POST /inspect": (
                "Upload gambar → YOLOv8 deteksi cacat → Gemini AI laporan "
                "investigasi → JSON { defects[], qa_report }"
            ),
            "GET  /health": "Health check",
        },
        "model": "YOLOv8 nano (yolov8n.pt) + Gemini 1.5 Flash",
        "language": "Laporan QA dalam Bahasa Indonesia",
        "gemini_api_key": "Diatur via GEMINI_API_KEY env var atau file .env",
    }


# ── startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    load_model()
    print("[QA-REPORTER] Server ready — POST /inspect untuk inspeksi + laporan AI")


# ── root (informasi singkat) ────────────────────────────────────────────────
@app.get("/")
async def root():
    page = Path(__file__).resolve().parent.parent / "templates" / "upload.html"
    return FileResponse(page)


@app.get("/live")
async def live_inspection():
    page = Path(__file__).resolve().parent.parent / "templates" / "live.html"
    return FileResponse(page)
