from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageOps

from app.settings import get_settings

settings = get_settings()
_MODEL = None

DET_MODEL = os.getenv("PADDLE_DETECTION_MODEL", "PP-OCRv5_server_det")
REC_MODEL = os.getenv("PADDLE_RECOGNITION_MODEL", "latin_PP-OCRv5_mobile_rec")

UPSCALE_MIN_SIDE = int(os.getenv("PADDLE_UPSCALE_MIN_SIDE", "1600"))
UPSCALE_MAX_SIDE = int(os.getenv("PADDLE_UPSCALE_MAX_SIDE", "3200"))

DET_LIMIT_SIDE_LEN = int(os.getenv("PADDLE_TEXT_DET_LIMIT_SIDE_LEN", "1280"))
DET_THRESH = float(os.getenv("PADDLE_TEXT_DET_THRESH", "0.20"))
BOX_THRESH = float(os.getenv("PADDLE_TEXT_DET_BOX_THRESH", "0.40"))
UNCLIP_RATIO = float(os.getenv("PADDLE_TEXT_DET_UNCLIP_RATIO", "2.0"))
REC_SCORE_THRESH = float(os.getenv("PADDLE_TEXT_REC_SCORE_THRESH", "0.0"))

MULTIPASS = os.getenv("PADDLE_MULTIPASS", "true").lower() in {"1", "true", "yes", "on"}


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def _result_to_dict(result: Any) -> dict:
    if isinstance(result, dict):
        return _plain(result)

    json_value = getattr(result, "json", None)
    if json_value is not None:
        try:
            if callable(json_value):
                json_value = json_value()
            if isinstance(json_value, str):
                return json.loads(json_value)
            if isinstance(json_value, dict):
                return _plain(json_value)
        except Exception:
            pass

    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        try:
            return _plain(to_dict())
        except Exception:
            pass

    return {"raw": str(result)}


def get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    from paddleocr import PaddleOCR

    _MODEL = PaddleOCR(
        lang=settings.ocr_language,
        device=settings.ocr_device,
        text_detection_model_name=DET_MODEL,
        text_recognition_model_name=REC_MODEL,
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
        text_det_limit_side_len=DET_LIMIT_SIDE_LEN,
        text_det_limit_type="min",
        text_det_thresh=DET_THRESH,
        text_det_box_thresh=BOX_THRESH,
        text_det_unclip_ratio=UNCLIP_RATIO,
        text_rec_score_thresh=REC_SCORE_THRESH,
        return_word_box=True,
    )
    return _MODEL


def _resize_quality(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    min_side = min(w, h)
    max_side = max(w, h)

    if min_side >= UPSCALE_MIN_SIDE:
        return image

    scale = UPSCALE_MIN_SIDE / max(1, min_side)
    if max_side * scale > UPSCALE_MAX_SIDE:
        scale = UPSCALE_MAX_SIDE / max(1, max_side)

    return image.resize(
        (max(1, round(w * scale)), max(1, round(h * scale))),
        Image.Resampling.LANCZOS,
    )


def _enhance(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=0.5)
    gray = ImageEnhance.Contrast(gray).enhance(1.10)
    gray = ImageEnhance.Sharpness(gray).enhance(1.30)
    return gray.convert("RGB")


def _run(path: Path) -> dict:
    started = time.perf_counter()
    raw = list(
        get_model().predict(
            str(path),
            text_det_limit_side_len=DET_LIMIT_SIDE_LEN,
            text_det_limit_type="min",
            text_det_thresh=DET_THRESH,
            text_det_box_thresh=BOX_THRESH,
            text_det_unclip_ratio=UNCLIP_RATIO,
            text_rec_score_thresh=REC_SCORE_THRESH,
        )
    )

    texts, scores = [], []
    for result in raw:
        payload = _result_to_dict(result)
        data = payload.get("res", payload)

        rec_texts = data.get("rec_texts") or []
        rec_scores = data.get("rec_scores") or []

        if isinstance(rec_texts, str):
            rec_texts = [rec_texts]

        texts.extend(str(x).strip() for x in rec_texts if str(x).strip())

        if isinstance(rec_scores, list):
            for score in rec_scores:
                try:
                    scores.append(float(score))
                except Exception:
                    pass

    text = "\n".join(texts).strip()
    avg = sum(scores) / len(scores) if scores else 0.0

    # Favorece confianza y también cobertura; evita elegir una salida corta
    # solo porque tenga confianza alta.
    quality_score = avg * max(1, len(text.replace(" ", "")))

    return {
        "text": text,
        "line_count": len(texts),
        "average_confidence": round(avg, 4) if scores else None,
        "quality_score": quality_score,
        "processing_seconds": round(time.perf_counter() - started, 4),
    }


def transcribe_image_bytes(image_bytes: bytes, extension: str = ".png") -> dict:
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="paddle_quality_") as tmp:
        tmp = Path(tmp)
        src = tmp / f"source{extension or '.png'}"
        src.write_bytes(image_bytes)

        with Image.open(src) as image:
            base = _resize_quality(image)

        color = tmp / "color.png"
        base.save(color, "PNG")
        passes = [("color", _run(color))]

        if MULTIPASS:
            enhanced = tmp / "enhanced.png"
            _enhance(base).save(enhanced, "PNG")
            passes.append(("enhanced", _run(enhanced)))

    selected_name, selected = max(passes, key=lambda x: x[1]["quality_score"])

    return {
        "text": selected["text"],
        "line_count": selected["line_count"],
        "average_confidence": selected["average_confidence"],
        "processing_seconds": round(time.perf_counter() - started, 4),
        "metadata": {
            "profile": "production_quality",
            "selected_pass": selected_name,
            "detection_model": DET_MODEL,
            "recognition_model": REC_MODEL,
            "orientation": True,
            "unwarping": True,
            "textline_orientation": True,
            "return_word_box": True,
            "multipass": MULTIPASS,
            "upscale_min_side": UPSCALE_MIN_SIDE,
            "upscale_max_side": UPSCALE_MAX_SIDE,
            "text_det_limit_side_len": DET_LIMIT_SIDE_LEN,
            "text_det_thresh": DET_THRESH,
            "text_det_box_thresh": BOX_THRESH,
            "text_det_unclip_ratio": UNCLIP_RATIO,
            "text_rec_score_thresh": REC_SCORE_THRESH,
        },
    }
