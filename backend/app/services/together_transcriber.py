from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

import httpx

from app.core.config import settings
from app.services.transcriber import ResolvedRuntime, normalize_text


class TogetherTranscriptionError(RuntimeError):
    pass


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error)
        return str(payload.get("message") or payload.get("detail") or payload)
    return str(payload)


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _normalise_word(
    raw_word: dict[str, Any],
    *,
    fallback_speaker: str | None = None,
) -> dict[str, Any] | None:
    token = str(raw_word.get("word") or raw_word.get("text") or "").strip()
    start = _safe_float(raw_word.get("start"))
    end = _safe_float(raw_word.get("end"))

    if not token or start is None or end is None or end <= start:
        return None

    return {
        "word": token,
        "start": start,
        "end": end,
        "speaker_label": (
            raw_word.get("speaker_id")
            or raw_word.get("speaker")
            or fallback_speaker
        ),
    }


def _extract_words(
    payload: dict[str, Any],
    raw_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for raw_word in payload.get("words") or []:
        if isinstance(raw_word, dict):
            word = _normalise_word(raw_word)
            if word:
                candidates.append(word)

    for segment in raw_segments:
        if not isinstance(segment, dict):
            continue
        fallback_speaker = segment.get("speaker_id")
        for raw_word in segment.get("words") or []:
            if not isinstance(raw_word, dict):
                continue
            word = _normalise_word(
                raw_word,
                fallback_speaker=fallback_speaker,
            )
            if word:
                candidates.append(word)

    # Together puede devolver las mismas palabras tanto en `words` como en
    # `speaker_segments[].words`. Se deduplican sin perder el orden temporal.
    unique: dict[tuple[float, float, str, str], dict[str, Any]] = {}
    for word in candidates:
        key = (
            round(float(word["start"]), 3),
            round(float(word["end"]), 3),
            str(word["word"]),
            str(word.get("speaker_label") or ""),
        )
        unique.setdefault(key, word)

    return sorted(
        unique.values(),
        key=lambda item: (float(item["start"]), float(item["end"])),
    )


def transcribe_chunk_together(
    path: str | Path,
    language: str,
    vocabulary: list[str],
    previous_context: str = "",
    *,
    diarize_override: bool | None = None,
) -> dict:
    """
    Proveedor Together AI para chunks del audio ya recortado.

    Solicita timestamps por palabra y por segmento. Las palabras se conservan
    para calcular tiempos muertos precisos incluso dentro de segmentos largos.
    """
    if not settings.together_ready:
        raise TogetherTranscriptionError(
            "Together AI está seleccionado, pero TOGETHER_API_KEY no está "
            "configurada"
        )

    audio_path = Path(path)
    if not audio_path.exists():
        raise TogetherTranscriptionError(
            f"No se encontró el chunk: {audio_path}"
        )

    vocabulary_text = ", ".join(vocabulary[:80])
    prompt_parts: list[str] = []
    if vocabulary_text:
        prompt_parts.append(f"Vocabulario esperado: {vocabulary_text}.")
    if previous_context:
        prompt_parts.append(
            f"Contexto anterior: {previous_context[-500:]}"
        )
    prompt = " ".join(prompt_parts).strip()

    use_diarization = (
        settings.together_diarize
        if diarize_override is None
        else bool(diarize_override)
    )

    fields: list[tuple[str, tuple[None, str]]] = [
        ("model", (None, settings.together_model)),
        ("language", (None, language or "auto")),
        ("response_format", (None, "verbose_json")),
        # Together documenta un arreglo multipart con estas dos claves.
        ("timestamp_granularities[0]", (None, "word")),
        ("timestamp_granularities[1]", (None, "segment")),
        ("temperature", (None, "0")),
    ]

    if prompt:
        fields.append(("prompt", (None, prompt)))

    if use_diarization:
        fields.append(("diarize", (None, "true")))
        if settings.together_min_speakers is not None:
            fields.append(
                (
                    "min_speakers",
                    (None, str(settings.together_min_speakers)),
                )
            )
        if settings.together_max_speakers is not None:
            fields.append(
                (
                    "max_speakers",
                    (None, str(settings.together_max_speakers)),
                )
            )

    mime = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
    url = f"{settings.together_base_url.rstrip('/')}/audio/transcriptions"
    timeout = httpx.Timeout(
        connect=30.0,
        read=float(settings.together_timeout_seconds),
        write=float(settings.together_timeout_seconds),
        pool=30.0,
    )

    try:
        with audio_path.open("rb") as audio_file:
            multipart: list[tuple[str, tuple]] = [*fields]
            multipart.append(
                (
                    "file",
                    (
                        audio_path.name,
                        audio_file,
                        mime,
                    ),
                )
            )
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    url,
                    headers={
                        "Authorization": (
                            f"Bearer {settings.together_api_key}"
                        )
                    },
                    files=multipart,
                )
    except (httpx.HTTPError, OSError) as exc:
        raise TogetherTranscriptionError(
            "No se pudo consultar Together AI. No se reintentó "
            "automáticamente para evitar un posible consumo duplicado: "
            f"{exc}"
        ) from exc

    request_id = (
        response.headers.get("x-request-id")
        or response.headers.get("cf-ray")
        or "sin-id"
    )

    if response.status_code == 503:
        raise TogetherTranscriptionError(
            "Together AI respondió 503 por capacidad temporal del servicio. "
            "No se reintentó automáticamente para evitar consumo duplicado. "
            f"request_id={request_id}"
        )

    if response.status_code == 429:
        retry_after = (
            response.headers.get("x-ratelimit-reset") or "no indicado"
        )
        raise TogetherTranscriptionError(
            "Together AI respondió 429 por límite dinámico. No se reintentó "
            "automáticamente. "
            f"retry_after={retry_after}; request_id={request_id}"
        )

    if response.status_code >= 400:
        raise TogetherTranscriptionError(
            f"Together AI respondió {response.status_code}: "
            f"{_error_detail(response)}; request_id={request_id}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise TogetherTranscriptionError(
            "Together AI devolvió una respuesta no JSON; "
            f"request_id={request_id}"
        ) from exc

    raw_segments = (
        payload.get("speaker_segments")
        if use_diarization and payload.get("speaker_segments")
        else payload.get("segments")
    ) or []

    segments: list[dict[str, Any]] = []
    for segment in raw_segments:
        if not isinstance(segment, dict):
            continue
        segment_text = segment.get("text")
        if not segment_text and segment.get("words"):
            segment_text = " ".join(
                str(word.get("word") or "")
                for word in segment["words"]
                if isinstance(word, dict)
            )
        text = normalize_text(str(segment_text or ""))
        if not text:
            continue

        segments.append(
            {
                "start": float(segment.get("start") or 0.0),
                "end": float(segment.get("end") or 0.0),
                "text": text,
                "avg_logprob": (
                    float(segment["avg_logprob"])
                    if segment.get("avg_logprob") is not None
                    else None
                ),
                "no_speech_prob": (
                    float(segment["no_speech_prob"])
                    if segment.get("no_speech_prob") is not None
                    else None
                ),
                "speaker_label": segment.get("speaker_id"),
            }
        )

    words = _extract_words(payload, raw_segments)
    text = normalize_text(str(payload.get("text") or ""))

    if not segments and text:
        duration = float(payload.get("duration") or 0.0)
        segments = [
            {
                "start": 0.0,
                "end": duration,
                "text": text,
                "avg_logprob": None,
                "no_speech_prob": None,
                "speaker_label": None,
            }
        ]

    return {
        "runtime": ResolvedRuntime(
            device="together-api",
            compute_type="managed",
        ),
        "language": payload.get("language") or language,
        "language_probability": None,
        "text": text,
        "segments": segments,
        "words": words,
        "provider_model": settings.together_model,
        "provider_request_id": request_id,
        "provider_duration_seconds": _safe_float(payload.get("duration")),
        "diarization_requested": use_diarization,
    }