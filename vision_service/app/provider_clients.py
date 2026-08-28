from __future__ import annotations

import asyncio
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

import httpx

from app.providers.paddle_provider import (
    transcribe_image_bytes as paddle_transcribe,
)
from app.settings import get_settings

settings = get_settings()

REMOTE_PROVIDERS = {
    "docling": settings.docling_worker_url,
    "qwen": settings.qwen_worker_url,
    "surya": settings.surya_worker_url,
}

# Solo modelos reales entran al benchmark.
BENCHMARK_PROVIDERS = ["paddle", "docling", "qwen", "surya"]

# AUTO es el modo recomendado para transcripción normal.
SELECTABLE_PROVIDERS = ["auto", *BENCHMARK_PROVIDERS]
ALL_PROVIDERS = BENCHMARK_PROVIDERS


def _normalize_for_agreement(text: str) -> str:
    value = unicodedata.normalize("NFKD", text or "")
    value = "".join(
        char
        for char in value
        if not unicodedata.combining(char)
    )
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _agreement(a: str, b: str) -> float:
    left = _normalize_for_agreement(a)
    right = _normalize_for_agreement(b)

    if not left or not right:
        return 0.0

    return SequenceMatcher(
        None,
        left,
        right,
        autojunk=False,
    ).ratio()


def _paddle_is_structurally_clean(text: str) -> bool:
    value = (text or "").strip()
    if len(value) < 40:
        return False

    words = re.findall(
        r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9@./:_+-]+",
        value,
    )
    if len(words) < 8:
        return False

    accepted_single = {"a", "y", "o", "e"}
    suspicious_single = sum(
        1
        for word in words
        if len(word) == 1
        and word.lower() not in accepted_single
        and not word.isdigit()
    )

    single_ratio = suspicious_single / max(len(words), 1)

    weird_chars = sum(
        1
        for char in value
        if not (
            char.isalnum()
            or char.isspace()
            or char in ".,;:!?¿¡@#%&()/+-_'\"[]<>="
        )
    )
    weird_ratio = weird_chars / max(len(value), 1)

    return single_ratio <= 0.12 and weird_ratio <= 0.03


async def _remote_health(name: str, url: str) -> dict:
    try:
        async with httpx.AsyncClient(
            timeout=settings.provider_health_timeout_seconds
        ) as client:
            response = await client.get(
                f"{url.rstrip('/')}/health"
            )
            response.raise_for_status()
            payload = response.json()

            return {
                "name": name,
                "available": payload.get("status") == "ok",
                "device": payload.get("device"),
                "model": payload.get("model"),
                "detail": payload.get("detail"),
                "url": url,
            }
    except Exception as exc:
        return {
            "name": name,
            "available": False,
            "device": None,
            "model": None,
            "detail": str(exc),
            "url": url,
        }


async def provider_statuses() -> list[dict]:
    remotes = await asyncio.gather(
        *[
            _remote_health(name, url)
            for name, url in REMOTE_PROVIDERS.items()
        ]
    )

    qwen_status = next(
        (item for item in remotes if item["name"] == "qwen"),
        None,
    )

    auto_detail = (
        "Modo híbrido recomendado: ejecuta PaddleOCR y Qwen en paralelo. "
        "Si ambos coinciden en texto limpio, prioriza Paddle por literalidad; "
        "si divergen, prioriza Qwen para manuscrito/layout complejo."
    )

    if not qwen_status or not qwen_status.get("available"):
        auto_detail += (
            " Qwen no está disponible ahora; AUTO degradará a PaddleOCR."
        )

    return [
        {
            "name": "auto",
            "available": True,
            "device": (
                f"cpu+{qwen_status.get('device')}"
                if qwen_status and qwen_status.get("available")
                else "cpu"
            ),
            "model": "AUTO · PaddleOCR + Qwen-VL",
            "detail": auto_detail,
            "url": None,
        },
        {
            "name": "paddle",
            "available": True,
            "device": settings.ocr_device,
            "model": "PaddleOCR",
            "detail": "OCR literal local del gateway.",
            "url": None,
        },
        *remotes,
    ]


async def _remote_transcribe(
    provider: str,
    image_bytes: bytes,
    extension: str,
    filename: str,
) -> dict:
    url = REMOTE_PROVIDERS[provider].rstrip("/")

    files = {
        "file": (
            filename or f"image{extension}",
            image_bytes,
            "application/octet-stream",
        )
    }

    async with httpx.AsyncClient(
        timeout=settings.provider_timeout_seconds
    ) as client:
        response = await client.post(
            f"{url}/transcribe",
            files=files,
        )

    if response.status_code >= 400:
        try:
            detail: Any = response.json()
        except Exception:
            detail = response.text

        raise RuntimeError(
            f"{provider} respondió {response.status_code}: {detail}"
        )

    payload = response.json()

    return {
        "provider": provider,
        "text": payload.get("text", ""),
        "line_count": payload.get(
            "line_count",
            len((payload.get("text") or "").splitlines()),
        ),
        "average_confidence": payload.get("average_confidence"),
        "processing_seconds": payload.get("processing_seconds"),
        "device": payload.get("device"),
        "model": payload.get("model"),
        "warnings": payload.get("warnings", []),
        "metadata": payload.get("metadata", {}),
    }


async def _paddle_result(
    image_bytes: bytes,
    extension: str,
) -> dict:
    result = await asyncio.to_thread(
        paddle_transcribe,
        image_bytes,
        extension,
    )

    return {
        "provider": "paddle",
        **result,
    }


async def _auto_transcribe(
    image_bytes: bytes,
    extension: str,
    filename: str,
) -> dict:
    # Paddle (CPU) y Qwen (GPU) corren en paralelo para reducir latencia.
    paddle_task = asyncio.create_task(
        _paddle_result(
            image_bytes,
            extension,
        )
    )
    qwen_task = asyncio.create_task(
        _remote_transcribe(
            "qwen",
            image_bytes,
            extension,
            filename,
        )
    )

    paddle_result = None
    qwen_result = None
    paddle_error = None
    qwen_error = None

    try:
        paddle_result = await paddle_task
    except Exception as exc:
        paddle_error = str(exc)

    try:
        qwen_result = await qwen_task
    except Exception as exc:
        qwen_error = str(exc)

    # Fallbacks seguros.
    if qwen_result is None and paddle_result is None:
        raise RuntimeError(
            "AUTO no pudo ejecutar ni Paddle ni Qwen. "
            f"Paddle: {paddle_error}. Qwen: {qwen_error}."
        )

    if qwen_result is None:
        selected = dict(paddle_result)
        selected["provider"] = "auto"
        selected.setdefault("metadata", {})
        selected["metadata"]["auto"] = {
            "selected_provider": "paddle",
            "reason": "qwen_unavailable",
            "qwen_error": qwen_error,
        }
        return selected

    if paddle_result is None:
        selected = dict(qwen_result)
        selected["provider"] = "auto"
        selected.setdefault("metadata", {})
        selected["metadata"]["auto"] = {
            "selected_provider": "qwen",
            "reason": "paddle_unavailable",
            "paddle_error": paddle_error,
        }
        return selected

    paddle_text = paddle_result.get("text", "")
    qwen_text = qwen_result.get("text", "")

    agreement = _agreement(
        paddle_text,
        qwen_text,
    )
    paddle_clean = _paddle_is_structurally_clean(
        paddle_text,
    )

    # Estrategia:
    # - Si ambos motores coinciden fuertemente y Paddle se ve estructuralmente
    #   limpio, usamos Paddle: es OCR más literal y evita "corregir" datos.
    # - Si divergen, usamos Qwen: las pruebas con manuscrito mostraron que
    #   Paddle degrada caracteres mientras Qwen conserva mejor el sentido visual.
    if agreement >= 0.82 and paddle_clean:
        winner = "paddle"
        source = paddle_result
        reason = "high_agreement_prefer_literal_paddle"
    else:
        winner = "qwen"
        source = qwen_result
        reason = "low_or_medium_agreement_prefer_visual_qwen"

    selected = dict(source)
    selected["provider"] = "auto"

    warnings = list(selected.get("warnings") or [])
    if agreement < 0.35:
        warnings.append("LOW_ENGINE_AGREEMENT")
    selected["warnings"] = warnings

    selected.setdefault("metadata", {})
    selected["metadata"]["auto"] = {
        "selected_provider": winner,
        "reason": reason,
        "agreement_score": round(agreement, 4),
        "paddle_structurally_clean": paddle_clean,
        "paddle_model": paddle_result.get("model", "PaddleOCR"),
        "qwen_model": qwen_result.get("model"),
        "paddle_processing_seconds": paddle_result.get(
            "processing_seconds"
        ),
        "qwen_processing_seconds": qwen_result.get(
            "processing_seconds"
        ),
        "low_agreement": agreement < 0.35,
    }

    return selected


async def transcribe_with_provider(
    provider: str,
    image_bytes: bytes,
    extension: str,
    filename: str = "image.png",
) -> dict:
    provider = provider.lower().strip()

    if provider not in SELECTABLE_PROVIDERS:
        raise ValueError(
            f"Proveedor no soportado: {provider}. "
            f"Usa {', '.join(SELECTABLE_PROVIDERS)}."
        )

    if provider == "auto":
        return await _auto_transcribe(
            image_bytes,
            extension,
            filename,
        )

    if provider == "paddle":
        return await _paddle_result(
            image_bytes,
            extension,
        )

    return await _remote_transcribe(
        provider,
        image_bytes,
        extension,
        filename,
    )
