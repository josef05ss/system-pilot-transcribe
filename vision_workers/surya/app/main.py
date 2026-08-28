from __future__ import annotations

import html
import io
import os
import re
import time
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps

app = FastAPI(title="Surya OCR Worker", version="3.0.0")

_MANAGER = None
_LAYOUT = None
_RECOGNITION = None

OCR_MODE = os.getenv("SURYA_OCR_MODE", "layout_blocks").strip().lower()


def get_runtime():
    global _MANAGER, _LAYOUT, _RECOGNITION

    if _RECOGNITION is not None:
        return _MANAGER, _LAYOUT, _RECOGNITION

    from surya.inference import SuryaInferenceManager
    from surya.layout import LayoutPredictor
    from surya.recognition import RecognitionPredictor

    _MANAGER = SuryaInferenceManager()
    _RECOGNITION = RecognitionPredictor(_MANAGER)

    if OCR_MODE == "layout_blocks":
        _LAYOUT = LayoutPredictor(_MANAGER)

    return _MANAGER, _LAYOUT, _RECOGNITION


def _obj(value: Any, key: str, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _html_to_text(value: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", value or "", flags=re.I)
    text = re.sub(r"</(?:p|div|li|h[1-6])>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_page_text(prediction: Any) -> tuple[str, list[float]]:
    blocks = _obj(prediction, "blocks", []) or []
    texts: list[str] = []
    confidences: list[float] = []

    for block in blocks:
        block_html = _obj(block, "html", "") or ""
        block_text = _html_to_text(str(block_html))

        if block_text:
            texts.append(block_text)

        confidence = _obj(block, "confidence", None)
        if confidence is not None:
            try:
                confidences.append(float(confidence))
            except Exception:
                pass

    return "\n".join(texts).strip(), confidences


@app.get("/health")
def health() -> dict:
    try:
        import surya  # noqa: F401

        return {
            "status": "ok",
            "service": "surya-worker",
            "version": "3.0.0",
            "model": "Surya 2",
            "profile": "quality_layout_blocks",
            "ocr_mode": OCR_MODE,
            "detail": (
                "SuryaInferenceManager administrará el backend de inferencia. "
                "Para producción conviene mantenerlo vivo."
            ),
        }
    except Exception as exc:
        return {
            "status": "error",
            "service": "surya-worker",
            "detail": str(exc),
        }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    started = time.perf_counter()
    data = await file.read()

    try:
        image = ImageOps.exif_transpose(
            Image.open(io.BytesIO(data))
        ).convert("RGB")

        _, layout_predictor, recognition_predictor = get_runtime()

        # Para páginas con varios bloques, dibujos y texto distribuido,
        # producción/calidad usa layout -> OCR por bloque.
        if OCR_MODE == "layout_blocks" and layout_predictor is not None:
            layouts = layout_predictor([image])
            predictions = recognition_predictor([image], layouts)
        else:
            predictions = recognition_predictor([image])

        page = predictions[0]
        text, confidences = _extract_page_text(page)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Surya no pudo procesar la imagen: {exc}",
        ) from exc

    avg_conf = (
        sum(confidences) / len(confidences)
        if confidences
        else None
    )

    return {
        "status": "ok",
        "text": text,
        "line_count": len([x for x in text.splitlines() if x.strip()]),
        "average_confidence": (
            round(avg_conf, 4) if avg_conf is not None else None
        ),
        "processing_seconds": round(time.perf_counter() - started, 4),
        "device": "auto",
        "model": "Surya 2",
        "reliable": bool(text),
        "warnings": [] if text else ["EMPTY_TRANSCRIPT"],
        "metadata": {
            "profile": "quality_layout_blocks",
            "ocr_mode": OCR_MODE,
            "keep_alive_recommended": True,
        },
    }
