from __future__ import annotations

from datetime import date, datetime, time

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ActiveUpdate(BaseModel):
    active: bool


class SiteCreate(BaseModel):
    code: str = Field(min_length=2, max_length=40)
    name: str = Field(min_length=2, max_length=120)
    address: str | None = Field(default=None, max_length=240)


class SiteUpdate(BaseModel):
    code: str | None = Field(default=None, min_length=2, max_length=40)
    name: str | None = Field(default=None, min_length=2, max_length=120)
    address: str | None = Field(default=None, max_length=240)
    active: bool | None = None


class SiteOut(ORMModel):
    id: str
    code: str
    name: str
    address: str | None
    active: bool
    created_at: datetime


class ClassroomCreate(BaseModel):
    site_id: str
    code: str = Field(min_length=1, max_length=40)
    name: str = Field(min_length=2, max_length=120)
    floor: str | None = Field(default=None, max_length=40)
    capacity: int | None = Field(default=None, ge=1, le=10000)


class ClassroomUpdate(BaseModel):
    site_id: str | None = None
    code: str | None = Field(default=None, min_length=1, max_length=40)
    name: str | None = Field(default=None, min_length=2, max_length=120)
    floor: str | None = Field(default=None, max_length=40)
    capacity: int | None = Field(default=None, ge=1, le=10000)
    active: bool | None = None


class ClassroomOut(ORMModel):
    id: str
    site_id: str
    code: str
    name: str
    floor: str | None
    capacity: int | None
    active: bool
    created_at: datetime


class CameraCreate(BaseModel):
    code: str = Field(min_length=2, max_length=60)
    name: str = Field(min_length=2, max_length=120)
    classroom_id: str | None = None
    brand: str | None = Field(default=None, max_length=80)
    model: str | None = Field(default=None, max_length=100)
    serial_number: str | None = Field(default=None, max_length=120)
    source_type: str = Field(default="pending", pattern="^(pending|local|nvr|nas|cloud|external_api)$")
    source_uri: str | None = Field(default=None, max_length=500)


class CameraUpdate(BaseModel):
    code: str | None = Field(default=None, min_length=2, max_length=60)
    name: str | None = Field(default=None, min_length=2, max_length=120)
    classroom_id: str | None = None
    brand: str | None = Field(default=None, max_length=80)
    model: str | None = Field(default=None, max_length=100)
    serial_number: str | None = Field(default=None, max_length=120)
    source_type: str | None = Field(default=None, pattern="^(pending|local|nvr|nas|cloud|external_api)$")
    source_uri: str | None = Field(default=None, max_length=500)
    active: bool | None = None


class CameraOut(ORMModel):
    id: str
    classroom_id: str | None
    code: str
    name: str
    brand: str | None
    model: str | None
    serial_number: str | None
    source_type: str
    source_uri: str | None
    active: bool
    created_at: datetime


class CameraAssignmentCreate(BaseModel):
    camera_id: str
    classroom_id: str
    started_at: datetime | None = None
    notes: str | None = Field(default=None, max_length=300)


class CameraAssignmentOut(ORMModel):
    id: str
    camera_id: str
    classroom_id: str
    started_at: datetime
    ended_at: datetime | None
    active: bool
    notes: str | None


class ProfessorCreate(BaseModel):
    code: str = Field(min_length=2, max_length=40)
    full_name: str = Field(min_length=3, max_length=160)
    email: EmailStr | None = None


class ProfessorUpdate(BaseModel):
    code: str | None = Field(default=None, min_length=2, max_length=40)
    full_name: str | None = Field(default=None, min_length=3, max_length=160)
    email: EmailStr | None = None
    active: bool | None = None


class ProfessorOut(ORMModel):
    id: str
    code: str
    full_name: str
    email: str | None
    active: bool
    created_at: datetime


class CourseCreate(BaseModel):
    code: str = Field(min_length=2, max_length=40)
    name: str = Field(min_length=2, max_length=160)
    description: str | None = None
    vocabulary: list[str] = Field(default_factory=list)


class CourseUpdate(BaseModel):
    code: str | None = Field(default=None, min_length=2, max_length=40)
    name: str | None = Field(default=None, min_length=2, max_length=160)
    description: str | None = None
    vocabulary: list[str] | None = None
    active: bool | None = None


class CourseOut(ORMModel):
    id: str
    code: str
    name: str
    description: str | None
    vocabulary: list[str]
    active: bool
    created_at: datetime


class ScheduleCreate(BaseModel):
    professor_id: str
    course_id: str
    classroom_id: str
    day_of_week: int = Field(ge=0, le=6)
    start_time: time
    end_time: time
    valid_from: date | None = None
    valid_until: date | None = None

    @model_validator(mode="after")
    def validate_times(self) -> ScheduleCreate:
        if self.end_time <= self.start_time:
            raise ValueError("La hora final debe ser posterior a la hora inicial")
        if self.valid_from and self.valid_until and self.valid_until < self.valid_from:
            raise ValueError("La fecha final no puede ser anterior a la inicial")
        return self


class ScheduleUpdate(BaseModel):
    professor_id: str | None = None
    course_id: str | None = None
    classroom_id: str | None = None
    day_of_week: int | None = Field(default=None, ge=0, le=6)
    start_time: time | None = None
    end_time: time | None = None
    valid_from: date | None = None
    valid_until: date | None = None
    active: bool | None = None


class ScheduleOut(ORMModel):
    id: str
    professor_id: str
    course_id: str
    classroom_id: str
    day_of_week: int
    start_time: time
    end_time: time
    valid_from: date | None
    valid_until: date | None
    active: bool
    created_at: datetime


class CatalogsOut(BaseModel):
    sites: list[SiteOut]
    classrooms: list[ClassroomOut]
    cameras: list[CameraOut]
    assignments: list[CameraAssignmentOut]
    professors: list[ProfessorOut]
    courses: list[CourseOut]
    schedules: list[ScheduleOut]


class RecordingOut(ORMModel):
    id: str
    site_id: str
    classroom_id: str
    camera_id: str
    source_type: str
    original_name: str
    recording_started_at: datetime
    duration_seconds: float
    container_format: str | None
    video_codec: str | None
    audio_codec: str | None
    audio_sample_rate: int | None
    audio_channels: int | None
    has_audio: bool
    file_size_bytes: int
    created_at: datetime


class JobCreate(BaseModel):
    recording_id: str
    schedule_id: str | None = None
    professor_id: str | None = None
    course_id: str | None = None
    requested_by: str = Field(default="Administrador", min_length=2, max_length=120)
    class_started_at: datetime
    class_ended_at: datetime
    model_name: str = "large-v3"
    language: str = "es"
    chunk_seconds: int = Field(default=300, ge=60, le=1800)
    overlap_seconds: int = Field(default=3, ge=0, le=30)
    priority: int = Field(default=5, ge=0, le=9)

    @model_validator(mode="after")
    def validate_range(self) -> JobCreate:
        if self.class_ended_at <= self.class_started_at:
            raise ValueError("La hora final debe ser posterior a la hora inicial")
        if self.overlap_seconds >= self.chunk_seconds:
            raise ValueError("El solapamiento debe ser menor que el chunk")
        return self


class SegmentOut(ORMModel):
    id: str
    chunk_index: int
    start_seconds: float
    end_seconds: float
    text: str
    avg_logprob: float | None
    no_speech_prob: float | None
    speaker_label: str | None


class JobOut(ORMModel):
    id: str
    recording_id: str
    schedule_id: str | None
    professor_id: str | None
    course_id: str | None
    requested_by: str
    class_started_at: datetime
    class_ended_at: datetime
    offset_start_seconds: float
    offset_end_seconds: float
    status: str
    progress: int
    priority: int
    provider_name: str
    model_name: str
    language: str
    chunk_seconds: int
    overlap_seconds: int
    device_used: str | None
    compute_type_used: str | None
    total_chunks: int
    completed_chunks: int
    processing_seconds: float | None
    real_time_factor: float | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    queue_position: int | None = None


class TranscriptOut(BaseModel):
    job_id: str
    status: str
    automatic_text: str | None
    reviewed_text: str | None
    final_text: str | None
    segments: list[SegmentOut]
    metrics: dict


class ReviewUpdate(BaseModel):
    reviewed_text: str = Field(min_length=1)
    approve: bool = False
