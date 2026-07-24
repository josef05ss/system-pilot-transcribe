from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import delete

from app.core.config import settings
from app.db.session import SessionLocal
from app.models import Course, Recording, TranscriptSegment, TranscriptionJob
from app.services.media import cleanup_directory, create_chunks, extract_audio_interval
from app.services.storage import get_storage_provider
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


@celery_app.task(bind=True, name="app.tasks.jobs.prepare_job")
def prepare_job(self, job_id: str) -> dict:
    """
    Siempre recorta primero el horario solicitado. Solo el audio resultante pasa al
    proveedor local o, en el futuro, a Together AI.
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

            source = get_storage_provider().materialize(recording.source_uri)
            work_dir = settings.storage_root / "work" / job.id
            chunks_dir = work_dir / "chunks"

            # Local usa WAV. Together usará FLAC para subir menos bytes sin perder calidad.
            audio_extension = "flac" if job.provider_name == "together" else "wav"
            audio_path = work_dir / f"selected_interval.{audio_extension}"
            work_dir.mkdir(parents=True, exist_ok=True)

            clip_duration = job.offset_end_seconds - job.offset_start_seconds
            job.work_dir = str(work_dir.resolve())
            job.prepared_audio_uri = str(audio_path.resolve())
            job.status = "EXTRACTING"
            job.progress = 8
            db.commit()

            # Esta es la edición/corte solicitado por el usuario.
            extract_audio_interval(
                source,
                audio_path,
                job.offset_start_seconds,
                clip_duration,
            )

            if is_cancelled(job_id):
                update_job(job_id, status="CANCELLED", progress=0, completed_at=now())
                if work_dir and not settings.keep_work_files:
                    cleanup_directory(work_dir)
                return {"cancelled": True}

            update_job(job_id, status="CHUNKING", progress=15)
            plans = create_chunks(
                audio_path,
                chunks_dir,
                job.chunk_seconds,
                job.overlap_seconds,
                output_format=audio_extension,
            )
            if not plans:
                raise RuntimeError("No se pudieron generar chunks de audio")

            job.chunk_manifest = [plan.as_dict() for plan in plans]
            job.total_chunks = len(plans)
            job.completed_chunks = 0
            job.status = "QUEUED_TRANSCRIPTION"
            job.progress = 20
            db.commit()
            priority = job.priority

        transcribe_job.apply_async(
            args=[job_id],
            queue="transcription",
            priority=priority,
        )
        return {"job_id": job_id, "chunks": len(plans)}

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


@celery_app.task(bind=True, name="app.tasks.jobs.transcribe_job")
def transcribe_job(self, job_id: str) -> dict:
    started = time.perf_counter()
    try:
        with SessionLocal() as db:
            job = db.get(TranscriptionJob, job_id)
            if not job:
                raise RuntimeError("Trabajo no encontrado")
            recording = db.get(Recording, job.recording_id)
            course = db.get(Course, job.course_id) if job.course_id else None
            vocabulary = list(course.vocabulary) if course else []
            manifest = list(job.chunk_manifest or [])
            if not manifest:
                raise RuntimeError("El trabajo no tiene chunks preparados")

            db.execute(delete(TranscriptSegment).where(TranscriptSegment.job_id == job.id))
            job.status = "TRANSCRIBING"
            job.progress = 22
            db.commit()

            merged_parts: list[str] = []
            previous_text = ""
            runtime = None
            language_probability = None
            provider_model = None
            last_segment_end = -1.0

            for index, chunk in enumerate(manifest):
                db.refresh(job)
                if job.cancel_requested:
                    job.status = "CANCELLED"
                    job.progress = 0
                    job.completed_at = now()
                    db.commit()
                    if job.work_dir and not settings.keep_work_files:
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
                runtime = result["runtime"]
                language_probability = result.get("language_probability")
                provider_model = result.get("provider_model")
                current_text = result["text"]
                clean_current = (
                    deduplicate_overlap(previous_text, current_text)
                    if previous_text
                    else current_text
                )
                if clean_current:
                    merged_parts.append(clean_current)
                previous_text = current_text

                chunk_start = float(chunk["start_seconds"])
                for segment in result["segments"]:
                    absolute_start = chunk_start + float(segment["start"])
                    absolute_end = chunk_start + float(segment["end"])
                    if absolute_end <= last_segment_end + 0.05:
                        continue
                    last_segment_end = max(last_segment_end, absolute_end)
                    db.add(
                        TranscriptSegment(
                            job_id=job.id,
                            chunk_index=index,
                            start_seconds=absolute_start,
                            end_seconds=absolute_end,
                            text=segment["text"],
                            avg_logprob=segment.get("avg_logprob"),
                            no_speech_prob=segment.get("no_speech_prob"),
                            speaker_label=segment.get("speaker_label"),
                        )
                    )

                job.completed_chunks = index + 1
                job.progress = min(92, 22 + int(70 * (index + 1) / len(manifest)))
                db.commit()

            job.status = "MERGING"
            job.progress = 95
            db.commit()

            final_text = "\n\n".join(part.strip() for part in merged_parts if part.strip()).strip()
            elapsed = time.perf_counter() - started
            clip_duration = job.offset_end_seconds - job.offset_start_seconds
            rtf = elapsed / clip_duration if clip_duration > 0 else None

            job.transcript_text = final_text
            job.processing_seconds = elapsed
            job.real_time_factor = rtf
            job.device_used = runtime.device if runtime else None
            job.compute_type_used = runtime.compute_type if runtime else None
            job.metrics = {
                "provider": job.provider_name,
                "provider_model": provider_model,
                "language_probability": language_probability,
                "clip_duration_seconds": clip_duration,
                "recording_duration_seconds": recording.duration_seconds if recording else None,
                "chunks": len(manifest),
                "batch_size": settings.ai_batch_size if job.provider_name == "local" else None,
                "beam_size": settings.ai_beam_size if job.provider_name == "local" else None,
                "important": "Solo se procesó el intervalo seleccionado, no el video completo.",
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
            failed_job = db.get(TranscriptionJob, job_id)
            work_dir = failed_job.work_dir if failed_job else None
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
