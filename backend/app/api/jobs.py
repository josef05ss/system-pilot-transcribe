from __future__ import annotations

import io
import json
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
from app.db.session import get_db
from app.models import Recording, Schedule, TranscriptionJob
from app.schemas import JobCreate, JobOut, ReviewUpdate, TranscriptOut
from app.tasks.jobs import prepare_job

router = APIRouter(prefix="/api/jobs", tags=["Trabajos"])

ACTIVE_STATUSES = {
    "PENDING",
    "VALIDATING",
    "EXTRACTING",
    "CHUNKING",
    "QUEUED_TRANSCRIPTION",
    "TRANSCRIBING",
    "MERGING",
}


def queue_position(db: Session, job: TranscriptionJob) -> int | None:
    if job.status not in {"PENDING", "VALIDATING", "EXTRACTING", "CHUNKING", "QUEUED_TRANSCRIPTION"}:
        return None
    count = db.scalar(
        select(func.count(TranscriptionJob.id)).where(
            TranscriptionJob.created_at < job.created_at,
            TranscriptionJob.status.in_(ACTIVE_STATUSES),
        )
    )
    return int(count or 0) + 1


def serialize_job(db: Session, job: TranscriptionJob) -> JobOut:
    data = JobOut.model_validate(job)
    data.queue_position = queue_position(db, job)
    return data


@router.get("", response_model=list[JobOut])
def list_jobs(
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[JobOut]:
    jobs = db.scalars(
        select(TranscriptionJob).order_by(TranscriptionJob.created_at.desc()).limit(limit)
    ).all()
    return [serialize_job(db, job) for job in jobs]


@router.post("", response_model=JobOut, status_code=status.HTTP_201_CREATED)
def create_job(payload: JobCreate, db: Session = Depends(get_db)) -> JobOut:
    recording = db.get(Recording, payload.recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="Grabación no encontrada")

    recording_end = recording.recording_started_at + timedelta(seconds=recording.duration_seconds)
    if payload.class_started_at < recording.recording_started_at:
        raise HTTPException(status_code=422, detail="La clase inicia antes de la grabación")
    if payload.class_ended_at > recording_end:
        raise HTTPException(status_code=422, detail="La clase termina después de la grabación")

    allowed_models = {"large-v3", "turbo", "medium", "small"}
    if payload.model_name not in allowed_models:
        raise HTTPException(status_code=422, detail="Modelo no permitido")

    provider_name = settings.transcription_provider
    if provider_name == "together" and not settings.together_ready:
        raise HTTPException(
            status_code=503,
            detail="Together AI está seleccionado, pero falta TOGETHER_API_KEY",
        )
    if provider_name == "together" and payload.model_name != "large-v3":
        raise HTTPException(
            status_code=422,
            detail="Together AI está configurado con Whisper Large v3; selecciona large-v3",
        )

    duplicate = db.scalar(
        select(TranscriptionJob.id).where(
            TranscriptionJob.recording_id == recording.id,
            TranscriptionJob.class_started_at == payload.class_started_at,
            TranscriptionJob.class_ended_at == payload.class_ended_at,
            TranscriptionJob.status.in_(ACTIVE_STATUSES),
        )
    )
    if duplicate:
        raise HTTPException(status_code=409, detail="Ese intervalo ya se está procesando")

    offset_start = (payload.class_started_at - recording.recording_started_at).total_seconds()
    offset_end = (payload.class_ended_at - recording.recording_started_at).total_seconds()

    schedule = None
    professor_id = payload.professor_id
    course_id = payload.course_id
    if payload.schedule_id:
        schedule = db.get(Schedule, payload.schedule_id)
        if not schedule or not schedule.active:
            raise HTTPException(status_code=404, detail="Horario no encontrado o inactivo")
        if schedule.classroom_id != recording.classroom_id:
            raise HTTPException(status_code=422, detail="El horario no corresponde al aula de la grabación")
        professor_id = professor_id or schedule.professor_id
        course_id = course_id or schedule.course_id

    job = TranscriptionJob(
        recording_id=recording.id,
        schedule_id=payload.schedule_id,
        professor_id=professor_id,
        course_id=course_id,
        requested_by=payload.requested_by,
        class_started_at=payload.class_started_at,
        class_ended_at=payload.class_ended_at,
        offset_start_seconds=offset_start,
        offset_end_seconds=offset_end,
        status="PENDING",
        progress=0,
        priority=payload.priority,
        provider_name=provider_name,
        model_name=payload.model_name,
        language=payload.language,
        chunk_seconds=payload.chunk_seconds,
        overlap_seconds=payload.overlap_seconds,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    prepare_job.apply_async(args=[job.id], queue="cpu", priority=payload.priority)
    return serialize_job(db, job)


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobOut:
    job = db.get(TranscriptionJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    return serialize_job(db, job)


@router.get("/{job_id}/transcript", response_model=TranscriptOut)
def get_transcript(job_id: str, db: Session = Depends(get_db)) -> TranscriptOut:
    job = db.scalar(
        select(TranscriptionJob)
        .where(TranscriptionJob.id == job_id)
        .options(selectinload(TranscriptionJob.segments))
    )
    if not job:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    final_text = job.reviewed_text or job.transcript_text
    return TranscriptOut(
        job_id=job.id,
        status=job.status,
        automatic_text=job.transcript_text,
        reviewed_text=job.reviewed_text,
        final_text=final_text,
        segments=job.segments,
        metrics=job.metrics,
    )


@router.post("/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: str, db: Session = Depends(get_db)) -> JobOut:
    job = db.get(TranscriptionJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    if job.status not in ACTIVE_STATUSES:
        raise HTTPException(status_code=409, detail="El trabajo ya no puede cancelarse")
    job.cancel_requested = True
    job.status = "CANCEL_REQUESTED"
    db.commit()
    db.refresh(job)
    return serialize_job(db, job)


@router.patch("/{job_id}/review", response_model=TranscriptOut)
def review_job(
    job_id: str,
    payload: ReviewUpdate,
    db: Session = Depends(get_db),
) -> TranscriptOut:
    job = db.scalar(
        select(TranscriptionJob)
        .where(TranscriptionJob.id == job_id)
        .options(selectinload(TranscriptionJob.segments))
    )
    if not job:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    if not job.transcript_text:
        raise HTTPException(status_code=409, detail="Aún no existe una transcripción")
    job.reviewed_text = payload.reviewed_text
    job.status = "APPROVED" if payload.approve else "READY_FOR_REVIEW"
    db.commit()
    return get_transcript(job_id, db)


@router.get("/{job_id}/download")
def download_job(
    job_id: str,
    file_format: str = Query(default="txt", pattern="^(txt|json)$"),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    job = db.scalar(
        select(TranscriptionJob)
        .where(TranscriptionJob.id == job_id)
        .options(selectinload(TranscriptionJob.segments))
    )
    if not job:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    final_text = job.reviewed_text or job.transcript_text or ""

    if file_format == "json":
        payload = {
            "job_id": job.id,
            "status": job.status,
            "text": final_text,
            "metrics": job.metrics,
            "segments": [
                {
                    "start": segment.start_seconds,
                    "end": segment.end_seconds,
                    "speaker": segment.speaker_label,
                    "text": segment.text,
                }
                for segment in job.segments
            ],
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        media_type = "application/json"
        filename = f"transcripcion_{job.id}.json"
    else:
        content = final_text.encode("utf-8")
        media_type = "text/plain; charset=utf-8"
        filename = f"transcripcion_{job.id}.txt"

    return StreamingResponse(
        io.BytesIO(content),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
