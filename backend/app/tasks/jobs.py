from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import delete

from app.core.config import settings
from app.db.session import SessionLocal
from app.models import Course, Recording, TranscriptSegment, TranscriptionJob
from app.services.media import (
    cleanup_directory,
    create_chunks,
    extract_audio_interval,
)
from app.services.storage import get_storage_provider
from app.services.transcript_analysis import (
    build_word_gap_analysis,
    long_gap_analyses_agree,
    result_quality_snapshot,
    suppress_unconfirmed_long_gaps,
)
from app.services.transcriber import deduplicate_overlap
from app.services.transcription_provider import transcribe_with_provider
from app.tasks.celery_app import celery_app


def now() -> datetime:
    return datetime.now(timezone.utc)


def update_job(job_id: str, **values) -> None:
    with SessionLocal() as db:
        job = db.get(TranscriptionJob, job_id)
        if not job:
            return

        for key, value in values.items():
            setattr(job, key, value)

        db.commit()


def is_cancelled(job_id: str) -> bool:
    with SessionLocal() as db:
        job = db.get(TranscriptionJob, job_id)
        return bool(job and job.cancel_requested)


def _quality_result_score(snapshot: dict) -> float:
    return (
        float(snapshot.get("word_count") or 0)
        + 0.25 * float(snapshot.get("text_word_count") or 0)
    )


def _select_quality_result(
    first_result: dict,
    second_result: dict,
    *,
    interval_duration_seconds: float,
) -> tuple[dict, dict, bool]:
    first_snapshot = result_quality_snapshot(
        first_result,
        interval_duration_seconds=interval_duration_seconds,
        minimum_gap_seconds=settings.dead_time_min_seconds,
        suspicious_gap_seconds=(
            settings.together_quality_retry_long_gap_seconds
        ),
    )
    second_snapshot = result_quality_snapshot(
        second_result,
        interval_duration_seconds=interval_duration_seconds,
        minimum_gap_seconds=settings.dead_time_min_seconds,
        suspicious_gap_seconds=(
            settings.together_quality_retry_long_gap_seconds
        ),
    )

    analyses_agree = long_gap_analyses_agree(
        first_snapshot["gap_analysis"],
        second_snapshot["gap_analysis"],
        minimum_duration_seconds=(
            settings.together_quality_retry_long_gap_seconds
        ),
        minimum_iou=settings.together_quality_retry_interval_iou,
    )

    first_score = _quality_result_score(first_snapshot)
    second_score = _quality_result_score(second_snapshot)
    improvement_ratio = (
        second_score / first_score
        if first_score > 0
        else (999.0 if second_score > 0 else 1.0)
    )

    use_second = (
        improvement_ratio
        >= settings.together_quality_retry_min_improvement_ratio
        or (
            second_snapshot["suspicious_long_gap_count"]
            < first_snapshot["suspicious_long_gap_count"]
            and second_score >= first_score * 0.9
        )
    )

    selected = second_result if use_second else first_result
    selected_snapshot = second_snapshot if use_second else first_snapshot

    # Si ambos intentos coinciden en un hueco largo, se considera estable.
    # Si el segundo intento mejora claramente la cobertura, también se acepta.
    reliable = analyses_agree or use_second

    details = {
        "first_attempt": {
            key: value
            for key, value in first_snapshot.items()
            if key != "gap_analysis"
        },
        "second_attempt": {
            key: value
            for key, value in second_snapshot.items()
            if key != "gap_analysis"
        },
        "long_gap_analyses_agree": analyses_agree,
        "second_attempt_improvement_ratio": round(improvement_ratio, 3),
        "selected_attempt": 2 if use_second else 1,
        "selected_snapshot": {
            key: value
            for key, value in selected_snapshot.items()
            if key != "gap_analysis"
        },
        "reliable": reliable,
    }
    return selected, details, reliable


@celery_app.task(bind=True, name="app.tasks.jobs.prepare_job")
def prepare_job(self, job_id: str) -> dict:
    """
    Recorta primero el intervalo solicitado.

    Solo el audio resultante pasa al proveedor local o a Together AI.
    """
    work_dir: Path | None = None

    try:
        with SessionLocal() as db:
            job = db.get(TranscriptionJob, job_id)
            if not job:
                raise RuntimeError("Trabajo no encontrado")

            if job.cancel_requested:
                job.status = "CANCELLED"
                job.progress = 0
                job.completed_at = now()
                db.commit()
                return {"cancelled": True}

            job.status = "VALIDATING"
            job.progress = 3
            job.started_at = now()
            job.error_message = None
            db.commit()

        with SessionLocal() as db:
            job = db.get(TranscriptionJob, job_id)
            if not job:
                raise RuntimeError("Trabajo no encontrado")

            recording = db.get(Recording, job.recording_id)
            if not recording:
                raise RuntimeError("Grabación no encontrada")

            source = get_storage_provider().materialize(
                recording.source_uri
            )
            work_dir = settings.storage_root / "work" / job.id
            chunks_dir = work_dir / "chunks"

            # Local usa WAV. Together usa FLAC para reducir el tamaño
            # sin perder calidad.
            audio_extension = (
                "flac"
                if job.provider_name == "together"
                else "wav"
            )
            audio_path = (
                work_dir / f"selected_interval.{audio_extension}"
            )
            work_dir.mkdir(parents=True, exist_ok=True)

            clip_duration = (
                job.offset_end_seconds - job.offset_start_seconds
            )
            job.work_dir = str(work_dir.resolve())
            job.prepared_audio_uri = str(audio_path.resolve())
            job.status = "EXTRACTING"
            job.progress = 8
            db.commit()

            # Edición y corte del intervalo solicitado por el usuario.
            extract_audio_interval(
                source,
                audio_path,
                job.offset_start_seconds,
                clip_duration,
            )

            if is_cancelled(job_id):
                update_job(
                    job_id,
                    status="CANCELLED",
                    progress=0,
                    completed_at=now(),
                )
                if (
                    work_dir
                    and not settings.keep_work_files
                ):
                    cleanup_directory(work_dir)
                return {"cancelled": True}

            # CORRECCIÓN:
            # Esta actualización debe estar fuera del bloque anterior.
            # Antes estaba después del return y nunca se ejecutaba.
            update_job(
                job_id,
                status="CHUNKING",
                progress=15,
            )

            audio_size_mb = (
                audio_path.stat().st_size / (1024 * 1024)
            )

            use_single_together_request = (
                job.provider_name == "together"
                and clip_duration
                <= settings.together_single_request_max_seconds
                and audio_size_mb
                <= settings.together_single_request_max_mb
            )

            if use_single_together_request:
                # Para intervalos normales se envía el FLAC completo
                # en una sola solicitud a Together.
                manifest = [
                    {
                        "index": 0,
                        "start_seconds": 0.0,
                        "duration_seconds": float(
                            clip_duration
                        ),
                        "path": str(audio_path.resolve()),
                    }
                ]
                chunk_strategy = "together_single_request"
            else:
                # Together utiliza chunks largos.
                # El proveedor local conserva sus chunks
                # configurables.
                chunk_seconds = (
                    settings.together_chunk_seconds
                    if job.provider_name == "together"
                    else job.chunk_seconds
                )
                overlap_seconds = (
                    settings.together_chunk_overlap_seconds
                    if job.provider_name == "together"
                    else job.overlap_seconds
                )

                plans = create_chunks(
                    audio_path,
                    chunks_dir,
                    chunk_seconds,
                    overlap_seconds,
                    output_format=audio_extension,
                )
                if not plans:
                    raise RuntimeError(
                        "No se pudieron generar chunks de audio"
                    )

                manifest = [
                    plan.as_dict()
                    for plan in plans
                ]

                chunk_strategy = (
                    "together_adaptive_chunks"
                    if job.provider_name == "together"
                    else "local_chunks"
                )

            job.chunk_manifest = manifest
            job.total_chunks = len(manifest)
            job.completed_chunks = 0
            job.metrics = {
                "chunk_strategy": chunk_strategy,
                "prepared_audio_mb": round(
                    audio_size_mb,
                    3,
                ),
                "selected_interval_seconds": (
                    clip_duration
                ),
            }
            job.status = "QUEUED_TRANSCRIPTION"
            job.progress = 20
            db.commit()

            priority = job.priority

        transcribe_job.apply_async(
            args=[job_id],
            queue="transcription",
            priority=priority,
        )

        return {
            "job_id": job_id,
            "chunks": len(manifest),
            "strategy": chunk_strategy,
        }

    except Exception as exc:
        update_job(
            job_id,
            status="ERROR",
            progress=0,
            error_message=str(exc),
            completed_at=now(),
        )
        if work_dir and not settings.keep_work_files:
            cleanup_directory(work_dir)
        raise


@celery_app.task(
    bind=True,
    name="app.tasks.jobs.transcribe_job",
)
def transcribe_job(self, job_id: str) -> dict:
    started = time.perf_counter()

    try:
        with SessionLocal() as db:
            job = db.get(TranscriptionJob, job_id)
            if not job:
                raise RuntimeError("Trabajo no encontrado")

            recording = db.get(
                Recording,
                job.recording_id,
            )
            course = (
                db.get(Course, job.course_id)
                if job.course_id
                else None
            )
            vocabulary = (
                list(course.vocabulary)
                if course
                else []
            )
            manifest = list(job.chunk_manifest or [])

            if not manifest:
                raise RuntimeError(
                    "El trabajo no tiene chunks preparados"
                )

            db.execute(
                delete(TranscriptSegment).where(
                    TranscriptSegment.job_id == job.id
                )
            )
            job.status = "TRANSCRIBING"
            job.progress = 22
            db.commit()

            merged_parts: list[str] = []
            previous_text = ""
            runtime = None
            language_probability = None
            provider_model = None
            provider_request_ids: list[str] = []
            provider_duration_seconds: list[float] = []
            absolute_words: list[dict] = []
            quality_checks: list[dict] = []
            unconfirmed_long_gap_windows: list[dict] = []
            last_segment_end = -1.0

            for index, chunk in enumerate(manifest):
                db.refresh(job)

                if job.cancel_requested:
                    job.status = "CANCELLED"
                    job.progress = 0
                    job.completed_at = now()
                    db.commit()

                    if (
                        job.work_dir
                        and not settings.keep_work_files
                    ):
                        cleanup_directory(job.work_dir)

                    return {"cancelled": True}

                result = transcribe_with_provider(
                    job.provider_name,
                    chunk["path"],
                    job.model_name,
                    job.language,
                    vocabulary,
                    previous_context=previous_text,
                )

                first_request_id = result.get("provider_request_id")
                if first_request_id:
                    provider_request_ids.append(str(first_request_id))
                first_provider_duration = result.get(
                    "provider_duration_seconds"
                )
                if first_provider_duration is not None:
                    provider_duration_seconds.append(
                        float(first_provider_duration)
                    )

                chunk_duration = float(
                    chunk.get("duration_seconds")
                    or result.get("provider_duration_seconds")
                    or 0.0
                )
                first_snapshot = result_quality_snapshot(
                    result,
                    interval_duration_seconds=chunk_duration,
                    minimum_gap_seconds=settings.dead_time_min_seconds,
                    suspicious_gap_seconds=(
                        settings.together_quality_retry_long_gap_seconds
                    ),
                )
                quality_detail = {
                    "chunk_index": index,
                    "retry_triggered": False,
                    "selected_attempt": 1,
                    "reliable": True,
                    "first_attempt": {
                        key: value
                        for key, value in first_snapshot.items()
                        if key != "gap_analysis"
                    },
                }

                if (
                    job.provider_name == "together"
                    and settings.together_quality_retry_enabled
                    and first_snapshot["suspicious_long_gap_count"] > 0
                ):
                    quality_detail["retry_triggered"] = True
                    try:
                        retry_result = transcribe_with_provider(
                            job.provider_name,
                            chunk["path"],
                            job.model_name,
                            job.language,
                            vocabulary,
                            previous_context=previous_text,
                            diarize_override=False,
                        )
                        retry_request_id = retry_result.get(
                            "provider_request_id"
                        )
                        if retry_request_id:
                            provider_request_ids.append(
                                str(retry_request_id)
                            )
                        retry_provider_duration = retry_result.get(
                            "provider_duration_seconds"
                        )
                        if retry_provider_duration is not None:
                            provider_duration_seconds.append(
                                float(retry_provider_duration)
                            )

                        result, selection_detail, reliable = (
                            _select_quality_result(
                                result,
                                retry_result,
                                interval_duration_seconds=chunk_duration,
                            )
                        )
                        quality_detail.update(selection_detail)
                        quality_detail["reliable"] = reliable

                        if not reliable:
                            selected_snapshot = result_quality_snapshot(
                                result,
                                interval_duration_seconds=chunk_duration,
                                minimum_gap_seconds=(
                                    settings.dead_time_min_seconds
                                ),
                                suspicious_gap_seconds=(
                                    settings
                                    .together_quality_retry_long_gap_seconds
                                ),
                            )
                            for interval in selected_snapshot[
                                "suspicious_long_gaps"
                            ]:
                                unconfirmed_long_gap_windows.append(
                                    {
                                        "start_seconds": (
                                            float(chunk["start_seconds"])
                                            + float(interval["start_seconds"])
                                        ),
                                        "end_seconds": (
                                            float(chunk["start_seconds"])
                                            + float(interval["end_seconds"])
                                        ),
                                    }
                                )
                    except Exception as retry_exc:
                        quality_detail["reliable"] = False
                        quality_detail["retry_error"] = str(retry_exc)
                        for interval in first_snapshot[
                            "suspicious_long_gaps"
                        ]:
                            unconfirmed_long_gap_windows.append(
                                {
                                    "start_seconds": (
                                        float(chunk["start_seconds"])
                                        + float(interval["start_seconds"])
                                    ),
                                    "end_seconds": (
                                        float(chunk["start_seconds"])
                                        + float(interval["end_seconds"])
                                    ),
                                }
                            )

                quality_checks.append(quality_detail)

                runtime = result["runtime"]
                language_probability = result.get(
                    "language_probability"
                )
                provider_model = result.get(
                    "provider_model"
                )
                request_id = result.get("provider_request_id")
                if request_id:
                    provider_request_ids.append(str(request_id))
                provider_duration = result.get(
                    "provider_duration_seconds"
                )
                if provider_duration is not None:
                    provider_duration_seconds.append(
                        float(provider_duration)
                    )
                current_text = result["text"]

                clean_current = (
                    deduplicate_overlap(
                        previous_text,
                        current_text,
                    )
                    if previous_text
                    else current_text
                )

                if clean_current:
                    merged_parts.append(clean_current)

                previous_text = current_text
                chunk_start = float(
                    chunk["start_seconds"]
                )

                for word in result.get("words") or []:
                    absolute_word_start = (
                        chunk_start
                        + float(word["start"])
                    )
                    absolute_word_end = (
                        chunk_start
                        + float(word["end"])
                    )
                    if absolute_word_end <= absolute_word_start:
                        continue

                    word_speaker = word.get("speaker_label")
                    if (
                        job.provider_name == "together"
                        and len(manifest) > 1
                        and word_speaker
                    ):
                        word_speaker = (
                            f"CHUNK_{index + 1:02d}_"
                            f"{word_speaker}"
                        )

                    absolute_words.append(
                        {
                            "word": word["word"],
                            "start": absolute_word_start,
                            "end": absolute_word_end,
                            "speaker_label": word_speaker,
                        }
                    )

                for segment in result["segments"]:
                    absolute_start = (
                        chunk_start
                        + float(segment["start"])
                    )
                    absolute_end = (
                        chunk_start
                        + float(segment["end"])
                    )

                    if (
                        absolute_end
                        <= last_segment_end + 0.05
                    ):
                        continue

                    last_segment_end = max(
                        last_segment_end,
                        absolute_end,
                    )
                    speaker_label = segment.get(
                        "speaker_label"
                    )

                    if (
                        job.provider_name == "together"
                        and len(manifest) > 1
                        and speaker_label
                    ):
                        speaker_label = (
                            f"CHUNK_{index + 1:02d}_"
                            f"{speaker_label}"
                        )

                    db.add(
                        TranscriptSegment(
                            job_id=job.id,
                            chunk_index=index,
                            start_seconds=absolute_start,
                            end_seconds=absolute_end,
                            text=segment["text"],
                            avg_logprob=segment.get(
                                "avg_logprob"
                            ),
                            no_speech_prob=segment.get(
                                "no_speech_prob"
                            ),
                            speaker_label=speaker_label,
                        )
                    )

                job.completed_chunks = index + 1
                job.progress = min(
                    92,
                    22
                    + int(
                        70
                        * (index + 1)
                        / len(manifest)
                    ),
                )
                db.commit()

            job.status = "MERGING"
            job.progress = 95
            db.commit()

            final_text = "\n\n".join(
                part.strip()
                for part in merged_parts
                if part.strip()
            ).strip()

            elapsed = time.perf_counter() - started
            clip_duration = (
                job.offset_end_seconds
                - job.offset_start_seconds
            )
            rtf = (
                elapsed / clip_duration
                if clip_duration > 0
                else None
            )

            job.transcript_text = final_text
            job.processing_seconds = elapsed
            job.real_time_factor = rtf
            job.device_used = (
                runtime.device
                if runtime
                else None
            )
            job.compute_type_used = (
                runtime.compute_type
                if runtime
                else None
            )

            dead_time_analysis = build_word_gap_analysis(
                absolute_words,
                interval_duration_seconds=clip_duration,
                minimum_gap_seconds=(
                    settings.dead_time_min_seconds
                ),
            )

            if unconfirmed_long_gap_windows:
                dead_time_analysis = suppress_unconfirmed_long_gaps(
                    dead_time_analysis,
                    unconfirmed_intervals=(
                        unconfirmed_long_gap_windows
                    ),
                    minimum_duration_seconds=(
                        settings.together_quality_retry_long_gap_seconds
                    ),
                    reason=(
                        "El hueco largo no fue confirmado de forma "
                        "consistente por el segundo intento."
                    ),
                )
            else:
                dead_time_analysis["reliable"] = True
                dead_time_analysis["suppressed_intervals"] = []
                dead_time_analysis["suppressed_interval_count"] = 0

            # Conserva IDs únicos manteniendo el orden de aparición.
            provider_request_ids = list(
                dict.fromkeys(provider_request_ids)
            )
            quality_requires_review = any(
                not bool(item.get("reliable", True))
                for item in quality_checks
            )
            transcription_quality = {
                "requires_review": quality_requires_review,
                "quality_retry_enabled": (
                    settings.together_quality_retry_enabled
                ),
                "suspicious_gap_threshold_seconds": (
                    settings.together_quality_retry_long_gap_seconds
                ),
                "checks": quality_checks,
                "warnings": (
                    [
                        {
                            "code": "UNCONFIRMED_LONG_TIMESTAMP_GAP",
                            "message": (
                                "Se ocultaron huecos largos que no pudieron "
                                "confirmarse en el segundo intento."
                            ),
                        }
                    ]
                    if quality_requires_review
                    else []
                ),
            }

            preparation_metrics = dict(
                job.metrics or {}
            )
            job.metrics = {
                **preparation_metrics,
                "provider": job.provider_name,
                "provider_model": provider_model,
                "language_probability": (
                    language_probability
                ),
                "provider_request_ids": provider_request_ids,
                "provider_duration_seconds": (
                    provider_duration_seconds
                ),
                "word_timestamps_available": bool(absolute_words),
                "word_timestamp_count": len(absolute_words),
                "dead_time_analysis": dead_time_analysis,
                "transcription_quality": transcription_quality,
                "clip_duration_seconds": (
                    clip_duration
                ),
                "recording_duration_seconds": (
                    recording.duration_seconds
                    if recording
                    else None
                ),
                "chunks": len(manifest),
                "batch_size": (
                    settings.ai_batch_size
                    if job.provider_name == "local"
                    else None
                ),
                "beam_size": (
                    settings.ai_beam_size
                    if job.provider_name == "local"
                    else None
                ),
                "important": (
                    "Solo se procesó el intervalo "
                    "seleccionado, no el video completo."
                ),
            }

            job.status = "READY_FOR_REVIEW"
            job.progress = 100
            job.completed_at = now()
            db.commit()

            work_dir = job.work_dir

        if work_dir and not settings.keep_work_files:
            cleanup_directory(work_dir)

        return {
            "job_id": job_id,
            "provider": job.provider_name,
            "processing_seconds": elapsed,
            "rtf": rtf,
        }

    except Exception as exc:
        work_dir = None

        with SessionLocal() as db:
            failed_job = db.get(
                TranscriptionJob,
                job_id,
            )
            work_dir = (
                failed_job.work_dir
                if failed_job
                else None
            )

        update_job(
            job_id,
            status="ERROR",
            progress=0,
            error_message=str(exc),
            completed_at=now(),
        )

        if work_dir and not settings.keep_work_files:
            cleanup_directory(work_dir)

        raise