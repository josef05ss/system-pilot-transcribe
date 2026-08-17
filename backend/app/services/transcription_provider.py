from __future__ import annotations

from pathlib import Path

from app.services.together_transcriber import transcribe_chunk_together
from app.services.transcriber import transcribe_chunk as transcribe_chunk_local


SUPPORTED_PROVIDERS = {"local", "together"}


def transcribe_with_provider(
    provider_name: str,
    path: str | Path,
    model_name: str,
    language: str,
    vocabulary: list[str],
    previous_context: str = "",
    *,
    diarize_override: bool | None = None,
) -> dict:
    if provider_name == "local":
        result = transcribe_chunk_local(
            path,
            model_name,
            language,
            vocabulary,
            previous_context=previous_context,
        )
        result["provider_model"] = model_name
        return result

    if provider_name == "together":
        return transcribe_chunk_together(
            path,
            language,
            vocabulary,
            previous_context=previous_context,
            diarize_override=diarize_override,
        )

    raise ValueError(
        f"Proveedor de transcripción no soportado: {provider_name}"
    )