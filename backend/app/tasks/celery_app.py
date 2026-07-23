from __future__ import annotations

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "transcriptor_real",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks.jobs"],
)

celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    result_expires=86400,
    broker_connection_retry_on_startup=True,
    task_routes={
        "app.tasks.jobs.prepare_job": {"queue": "cpu"},
        "app.tasks.jobs.transcribe_job": {"queue": "transcription"},
    },
)
