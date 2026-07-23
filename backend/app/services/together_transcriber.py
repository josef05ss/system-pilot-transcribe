from __future__ import annotations

import mimetypes
import time
from pathlib import Path

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


def transcribe_chunk_together(
    path: str | Path,
    language: str,
    vocabulary: list[str],
    previous_context: str = "",
) -> dict:
    """
    Proveedor opcional Together AI.

    IMPORTANTE: esta función solo recibe chunks del audio YA RECORTADO al intervalo
    elegido. Nunca sube las 15 horas de video ni cobra por ese contenido descartado.
    Mientras TRANSCRIPTION_PROVIDER=local, este código no se ejecuta y no consume saldo.
    """
    if not settings.together_ready:
        raise TogetherTranscriptionError(
            "Together AI está seleccionado, pero TOGETHER_API_KEY no está configurada"
        )

    audio_path = Path(path)
    if not audio_path.exists():
        raise TogetherTranscriptionError(f"No se encontró el chunk: {audio_path}")

    vocabulary_text = ", ".join(vocabulary[:80])
    prompt_parts: list[str] = []
    if vocabulary_text:
        prompt_parts.append(f"Vocabulario esperado: {vocabulary_text}.")
    if previous_context:
        prompt_parts.append(f"Contexto anterior: {previous_context[-500:]}")
    prompt = " ".join(prompt_parts).strip()

    fields: list[tuple[str, tuple[None, str]]] = [
        ("model", (None, settings.together_model)),
        ("language", (None, language or "auto")),
        ("response_format", (None, "verbose_json")),
        ("timestamp_granularities", (None, "segment")),
        ("temperature", (None, "0")),
    ]
    if prompt:
        fields.append(("prompt", (None, prompt)))
    if settings.together_diarize:
        fields.append(("diarize", (None, "true")))
        if settings.together_min_speakers is not None:
            fields.append(("min_speakers", (None, str(settings.together_min_speakers))))
        if settings.together_max_speakers is not None:
            fields.append(("max_speakers", (None, str(settings.together_max_speakers))))

    mime = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
    url = f"{settings.together_base_url.rstrip('/')}/audio/transcriptions"
    timeout = httpx.Timeout(
        connect=30.0,
        read=float(settings.together_timeout_seconds),
        write=float(settings.together_timeout_seconds),
        pool=30.0,
    )

    last_error: Exception | None = None
    for attempt in range(settings.together_max_retries + 1):
        try:
            with audio_path.open("rb") as audio_file:
                multipart: list[tuple[str, tuple]] = [*fields]
                # El campo model se envía antes que file, como recomienda Together.
                multipart.append(("file", (audio_path.name, audio_file, mime)))
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(
                        url,
                        headers={"Authorization": f"Bearer {settings.together_api_key}"},
                        files=multipart,
                    )

            if response.status_code == 429 or response.status_code >= 500:
                detail = _error_detail(response)
                if attempt < settings.together_max_retries:
                    time.sleep(min(30, 2**attempt))
                    continue
                raise TogetherTranscriptionError(
                    f"Together AI respondió {response.status_code}: {detail}"
                )

            if response.status_code >= 400:
                raise TogetherTranscriptionError(
                    f"Together AI respondió {response.status_code}: {_error_detail(response)}"
                )

            payload = response.json()
            break
        except (httpx.HTTPError, OSError, ValueError) as exc:
            last_error = exc
            if attempt < settings.together_max_retries:
                time.sleep(min(30, 2**attempt))
                continue
            raise TogetherTranscriptionError(f"No se pudo consultar Together AI: {exc}") from exc
    else:
        raise TogetherTranscriptionError(f"No se pudo consultar Together AI: {last_error}")

    raw_segments = (
        payload.get("speaker_segments")
        if settings.together_diarize and payload.get("speaker_segments")
        else payload.get("segments")
    ) or []
    segments: list[dict] = []
    for segment in raw_segments:
        segment_text = segment.get("text")
        if not segment_text and segment.get("words"):
            segment_text = " ".join(str(word.get("word") or "") for word in segment["words"])
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
        "runtime": ResolvedRuntime(device="together-api", compute_type="managed"),
        "language": payload.get("language") or language,
        "language_probability": None,
        "text": text,
        "segments": segments,
        "provider_model": settings.together_model,
    }
