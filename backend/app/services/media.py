from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChunkPlan:
    index: int
    start_seconds: float
    duration_seconds: float
    path: str

    def as_dict(self) -> dict:
        return asdict(self)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except FileNotFoundError as exc:
        raise MediaError(
            f"No se encontró '{command[0]}'. Instala FFmpeg y agrégalo a PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise MediaError(detail) from exc


def probe_media(path: str | Path) -> dict:
    media_path = Path(path)
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(media_path),
        ]
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    format_info = payload.get("format", {})
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)

    try:
        duration = float(format_info.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0

    if duration <= 0:
        durations = []
        for stream in streams:
            try:
                durations.append(float(stream.get("duration") or 0))
            except (TypeError, ValueError):
                pass
        duration = max(durations, default=0.0)

    return {
        "duration_seconds": duration,
        "container_format": format_info.get("format_name"),
        "video_codec": video.get("codec_name") if video else None,
        "audio_codec": audio.get("codec_name") if audio else None,
        "audio_sample_rate": int(audio.get("sample_rate")) if audio and audio.get("sample_rate") else None,
        "audio_channels": int(audio.get("channels")) if audio and audio.get("channels") else None,
        "has_audio": audio is not None,
        "raw": payload,
    }


def extract_audio_interval(
    input_path: str | Path,
    output_path: str | Path,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    """Extrae primero el intervalo elegido. Ningún proveedor recibe el video completo."""
    if start_seconds < 0 or duration_seconds <= 0:
        raise MediaError("Intervalo de audio inválido")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    codec_args = (
        ["-c:a", "flac", "-compression_level", "5"]
        if output.suffix.lower() == ".flac"
        else ["-c:a", "pcm_s16le"]
    )

    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            # -ss antes de -i permite un seek rápido sobre grabaciones largas.
            "-ss",
            f"{start_seconds:.3f}",
            "-i",
            str(input_path),
            "-t",
            f"{duration_seconds:.3f}",
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            *codec_args,
            str(output),
        ]
    )


def build_chunk_ranges(
    duration_seconds: float,
    chunk_seconds: int,
    overlap_seconds: int,
) -> list[tuple[float, float]]:
    if duration_seconds <= 0:
        return []
    if chunk_seconds <= 0 or overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("Configuración de chunks inválida")

    step = chunk_seconds - overlap_seconds
    total = max(1, math.ceil(max(0.0, duration_seconds - overlap_seconds) / step))
    ranges: list[tuple[float, float]] = []

    for index in range(total):
        start = index * step
        if start >= duration_seconds:
            break
        length = min(chunk_seconds, duration_seconds - start)
        ranges.append((float(start), float(length)))

    return ranges


def create_chunks(
    audio_path: str | Path,
    chunk_dir: str | Path,
    chunk_seconds: int,
    overlap_seconds: int,
    output_format: str = "wav",
) -> list[ChunkPlan]:
    metadata = probe_media(audio_path)
    duration = metadata["duration_seconds"]
    ranges = build_chunk_ranges(duration, chunk_seconds, overlap_seconds)
    output_dir = Path(chunk_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plans: list[ChunkPlan] = []

    normalized_format = output_format.lower().lstrip(".")
    if normalized_format not in {"wav", "flac"}:
        raise ValueError("Formato de chunk no permitido")

    codec_args = (
        ["-c:a", "flac", "-compression_level", "5"]
        if normalized_format == "flac"
        else ["-c:a", "pcm_s16le"]
    )

    for index, (start, length) in enumerate(ranges):
        output = output_dir / f"chunk_{index:05d}.{normalized_format}"
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(audio_path),
                "-t",
                f"{length:.3f}",
                "-ac",
                "1",
                "-ar",
                "16000",
                *codec_args,
                str(output),
            ]
        )
        plans.append(
            ChunkPlan(
                index=index,
                start_seconds=start,
                duration_seconds=length,
                path=str(output.resolve()),
            )
        )

    return plans


def cleanup_directory(path: str | Path) -> None:
    shutil.rmtree(Path(path), ignore_errors=True)
