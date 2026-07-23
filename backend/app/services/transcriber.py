from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import ctranslate2
from faster_whisper import BatchedInferencePipeline, WhisperModel

from app.core.config import settings


@dataclass(frozen=True)
class ResolvedRuntime:
    device: str
    compute_type: str


class ModelCache:
    def __init__(self) -> None:
        self._models: dict[tuple[str, str, str], BatchedInferencePipeline] = {}
        self._lock = Lock()

    def get(self, model_name: str, runtime: ResolvedRuntime) -> BatchedInferencePipeline:
        key = (model_name, runtime.device, runtime.compute_type)
        with self._lock:
            if key not in self._models:
                model = WhisperModel(
                    model_name,
                    device=runtime.device,
                    compute_type=runtime.compute_type,
                    download_root=str(settings.model_cache_dir),
                )
                self._models[key] = BatchedInferencePipeline(model=model)
            return self._models[key]


cache = ModelCache()


def resolve_runtime() -> ResolvedRuntime:
    device = settings.ai_device
    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

    compute_type = settings.ai_compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    return ResolvedRuntime(device=device, compute_type=compute_type)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def deduplicate_overlap(previous: str, current: str, max_words: int = 40) -> str:
    previous_words = previous.split()
    current_words = current.split()
    limit = min(max_words, len(previous_words), len(current_words))

    def normalized(words: list[str]) -> list[str]:
        return [re.sub(r"[^\wáéíóúüñ]", "", word.lower()) for word in words]

    prev_norm = normalized(previous_words)
    curr_norm = normalized(current_words)

    for size in range(limit, 2, -1):
        if prev_norm[-size:] == curr_norm[:size]:
            return " ".join(current_words[size:]).strip()
    return current.strip()


def transcribe_chunk(
    path: str | Path,
    model_name: str,
    language: str,
    vocabulary: list[str],
    previous_context: str = "",
) -> dict:
    runtime = resolve_runtime()
    pipeline = cache.get(model_name, runtime)

    vocabulary_text = ", ".join(vocabulary[:80])
    prompt_parts = []
    if vocabulary_text:
        prompt_parts.append(f"Vocabulario esperado: {vocabulary_text}.")
    if previous_context:
        prompt_parts.append(f"Contexto anterior: {previous_context[-500:]}")
    prompt = " ".join(prompt_parts) or None

    segments_generator, info = pipeline.transcribe(
        str(path),
        language=language or None,
        batch_size=settings.ai_batch_size,
        beam_size=settings.ai_beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt=prompt,
        hotwords=vocabulary_text or None,
    )
    segments = list(segments_generator)
    text = normalize_text(" ".join(segment.text for segment in segments))

    return {
        "runtime": runtime,
        "language": info.language,
        "language_probability": info.language_probability,
        "text": text,
        "segments": [
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": normalize_text(segment.text),
                "avg_logprob": float(segment.avg_logprob),
                "no_speech_prob": float(segment.no_speech_prob),
            }
            for segment in segments
            if normalize_text(segment.text)
        ],
    }
