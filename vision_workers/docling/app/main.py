from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI(title="Docling OCR Worker", version="3.0.0")

_CONVERTER = None
_PROFILE = None

DOCLING_THREADS = int(os.getenv("DOCLING_NUM_THREADS", "8"))
DOCLING_DEVICE = os.getenv("DOCLING_DEVICE", "auto")
DOCLING_OCR_SCALE = float(os.getenv("DOCLING_OCR_SCALE", "4.0"))
DOCLING_OCR_CONFIDENCE = float(
    os.getenv("DOCLING_OCR_CONFIDENCE_THRESHOLD", "0.20")
)


def get_converter():
    global _CONVERTER, _PROFILE

    if _CONVERTER is not None:
        return _CONVERTER, _PROFILE

    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        OcrMode,
        PdfPipelineOptions,
    )
    from docling.document_converter import (
        DocumentConverter,
        ImageFormatOption,
    )

    try:
        device = AcceleratorDevice(DOCLING_DEVICE)
    except Exception:
        device = DOCLING_DEVICE

    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = True
    options.accelerator_options = AcceleratorOptions(
        num_threads=DOCLING_THREADS,
        device=device,
    )

    # Para fotografías/imágenes completas, full-page OCR evita depender
    # demasiado de que el layout previo encuentre todos los trazos.
    options.ocr_options = EasyOcrOptions(
        lang=["es"],
        mode=OcrMode.FULL_PAGE,
        scale=DOCLING_OCR_SCALE,
        confidence_threshold=DOCLING_OCR_CONFIDENCE,
    )

    try:
        _CONVERTER = DocumentConverter(
            allowed_formats=[InputFormat.IMAGE],
            format_options={
                InputFormat.IMAGE: ImageFormatOption(
                    pipeline_options=options
                )
            },
        )
        # Inicializa la tubería aquí para detectar pronto dependencias.
        _CONVERTER.initialize_pipeline(InputFormat.IMAGE)
        _PROFILE = "easyocr_es_full_page_quality"
    except Exception:
        # Fallback robusto para instalaciones de Docling que no tengan
        # EasyOCR disponible como backend opcional.
        from docling.datamodel.pipeline_options import OcrAutoOptions

        options.ocr_options = OcrAutoOptions(
            mode=OcrMode.FULL_PAGE,
            scale=DOCLING_OCR_SCALE,
        )
        _CONVERTER = DocumentConverter(
            allowed_formats=[InputFormat.IMAGE],
            format_options={
                InputFormat.IMAGE: ImageFormatOption(
                    pipeline_options=options
                )
            },
        )
        _PROFILE = "auto_full_page_quality"

    return _CONVERTER, _PROFILE


@app.get("/health")
def health() -> dict:
    try:
        import docling  # noqa: F401

        return {
            "status": "ok",
            "service": "docling-worker",
            "version": "3.0.0",
            "device": DOCLING_DEVICE,
            "model": "Docling",
            "profile": "full_page_ocr_quality",
            "ocr_scale": DOCLING_OCR_SCALE,
            "threads": DOCLING_THREADS,
        }
    except Exception as exc:
        return {
            "status": "error",
            "service": "docling-worker",
            "detail": str(exc),
        }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    started = time.perf_counter()
    data = await file.read()
    suffix = Path(file.filename or "image.png").suffix or ".png"

    try:
        with tempfile.TemporaryDirectory(prefix="docling_quality_") as tmp:
            path = Path(tmp) / f"input{suffix}"
            path.write_bytes(data)

            converter, profile = get_converter()
            result = converter.convert(path)
            document = result.document

            try:
                text = document.export_to_text()
            except Exception:
                text = document.export_to_markdown()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Docling no pudo procesar la imagen: {exc}",
        ) from exc

    text = (text or "").strip()

    return {
        "status": "ok",
        "text": text,
        "line_count": len([x for x in text.splitlines() if x.strip()]),
        "average_confidence": None,
        "processing_seconds": round(time.perf_counter() - started, 4),
        "device": DOCLING_DEVICE,
        "model": "Docling",
        "reliable": bool(text),
        "warnings": [] if text else ["EMPTY_TRANSCRIPT"],
        "metadata": {
            "profile": profile,
            "ocr_mode": "full_page",
            "ocr_scale": DOCLING_OCR_SCALE,
            "ocr_confidence_threshold": DOCLING_OCR_CONFIDENCE,
            "threads": DOCLING_THREADS,
        },
    }
