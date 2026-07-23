from __future__ import annotations

import uuid
from datetime import date, datetime, time, timezone

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Time,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Site(Base):
    __tablename__ = "sites"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    code: Mapped[str] = mapped_column(String(40), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    address: Mapped[str | None] = mapped_column(String(240))
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    classrooms: Mapped[list[Classroom]] = relationship(back_populates="site")


class Classroom(Base):
    __tablename__ = "classrooms"
    __table_args__ = (UniqueConstraint("site_id", "code", name="uq_classroom_site_code"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.id"), nullable=False)
    code: Mapped[str] = mapped_column(String(40), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    floor: Mapped[str | None] = mapped_column(String(40))
    capacity: Mapped[int | None] = mapped_column(Integer)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    site: Mapped[Site] = relationship(back_populates="classrooms")
    cameras: Mapped[list[Camera]] = relationship(back_populates="classroom")
    assignments: Mapped[list[CameraAssignment]] = relationship(back_populates="classroom")
    schedules: Mapped[list[Schedule]] = relationship(back_populates="classroom")


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    # Aula actual. El historial se conserva en camera_assignments.
    classroom_id: Mapped[str | None] = mapped_column(ForeignKey("classrooms.id"))
    code: Mapped[str] = mapped_column(String(60), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    brand: Mapped[str | None] = mapped_column(String(80))
    model: Mapped[str | None] = mapped_column(String(100))
    serial_number: Mapped[str | None] = mapped_column(String(120), unique=True)
    source_type: Mapped[str] = mapped_column(String(30), default="pending", nullable=False)
    source_uri: Mapped[str | None] = mapped_column(String(500))
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    classroom: Mapped[Classroom | None] = relationship(back_populates="cameras")
    assignments: Mapped[list[CameraAssignment]] = relationship(
        back_populates="camera", cascade="all, delete-orphan"
    )


class CameraAssignment(Base):
    __tablename__ = "camera_assignments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), nullable=False)
    classroom_id: Mapped[str] = mapped_column(ForeignKey("classrooms.id"), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    notes: Mapped[str | None] = mapped_column(String(300))

    camera: Mapped[Camera] = relationship(back_populates="assignments")
    classroom: Mapped[Classroom] = relationship(back_populates="assignments")


class Professor(Base):
    __tablename__ = "professors"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    code: Mapped[str] = mapped_column(String(40), unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(160), nullable=False)
    email: Mapped[str | None] = mapped_column(String(180), unique=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    schedules: Mapped[list[Schedule]] = relationship(back_populates="professor")


class Course(Base):
    __tablename__ = "courses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    code: Mapped[str] = mapped_column(String(40), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    vocabulary: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    schedules: Mapped[list[Schedule]] = relationship(back_populates="course")


class Schedule(Base):
    __tablename__ = "schedules"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    professor_id: Mapped[str] = mapped_column(ForeignKey("professors.id"), nullable=False)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"), nullable=False)
    classroom_id: Mapped[str] = mapped_column(ForeignKey("classrooms.id"), nullable=False)
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)  # 0=lunes
    start_time: Mapped[time] = mapped_column(Time, nullable=False)
    end_time: Mapped[time] = mapped_column(Time, nullable=False)
    valid_from: Mapped[date | None] = mapped_column(Date)
    valid_until: Mapped[date | None] = mapped_column(Date)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    professor: Mapped[Professor] = relationship(back_populates="schedules")
    course: Mapped[Course] = relationship(back_populates="schedules")
    classroom: Mapped[Classroom] = relationship(back_populates="schedules")


class Recording(Base):
    __tablename__ = "recordings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.id"), nullable=False)
    classroom_id: Mapped[str] = mapped_column(ForeignKey("classrooms.id"), nullable=False)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), nullable=False)

    source_type: Mapped[str] = mapped_column(String(30), default="local", nullable=False)
    source_uri: Mapped[str] = mapped_column(String(1000), nullable=False)
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    recording_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    container_format: Mapped[str | None] = mapped_column(String(120))
    video_codec: Mapped[str | None] = mapped_column(String(80))
    audio_codec: Mapped[str | None] = mapped_column(String(80))
    audio_sample_rate: Mapped[int | None] = mapped_column(Integer)
    audio_channels: Mapped[int | None] = mapped_column(Integer)
    has_audio: Mapped[bool] = mapped_column(Boolean, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class TranscriptionJob(Base):
    __tablename__ = "transcription_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    recording_id: Mapped[str] = mapped_column(ForeignKey("recordings.id"), nullable=False)
    schedule_id: Mapped[str | None] = mapped_column(ForeignKey("schedules.id"))
    professor_id: Mapped[str | None] = mapped_column(ForeignKey("professors.id"))
    course_id: Mapped[str | None] = mapped_column(ForeignKey("courses.id"))

    requested_by: Mapped[str] = mapped_column(String(120), default="Administrador")
    class_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    class_ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    offset_start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    offset_end_seconds: Mapped[float] = mapped_column(Float, nullable=False)

    status: Mapped[str] = mapped_column(String(40), default="PENDING", index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    provider_name: Mapped[str] = mapped_column(String(40), default="local", nullable=False)
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    language: Mapped[str] = mapped_column(String(20), default="es", nullable=False)
    chunk_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    overlap_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    device_used: Mapped[str | None] = mapped_column(String(40))
    compute_type_used: Mapped[str | None] = mapped_column(String(40))

    work_dir: Mapped[str | None] = mapped_column(String(1000))
    prepared_audio_uri: Mapped[str | None] = mapped_column(String(1000))
    chunk_manifest: Mapped[list[dict]] = mapped_column(JSON, default=list, nullable=False)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    transcript_text: Mapped[str | None] = mapped_column(Text)
    reviewed_text: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    processing_seconds: Mapped[float | None] = mapped_column(Float)
    real_time_factor: Mapped[float | None] = mapped_column(Float)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    recording: Mapped[Recording] = relationship()
    schedule: Mapped[Schedule | None] = relationship()
    professor: Mapped[Professor | None] = relationship()
    course: Mapped[Course | None] = relationship()
    segments: Mapped[list[TranscriptSegment]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="TranscriptSegment.start_seconds",
    )


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("transcription_jobs.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    avg_logprob: Mapped[float | None] = mapped_column(Float)
    no_speech_prob: Mapped[float | None] = mapped_column(Float)
    speaker_label: Mapped[str | None] = mapped_column(String(80))

    job: Mapped[TranscriptionJob] = relationship(back_populates="segments")
