from __future__ import annotations

from datetime import datetime, time, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Camera, CameraAssignment, Classroom, Course, Professor, Schedule, Site


def seed_catalogs(db: Session) -> None:
    if db.scalar(select(Site.id).limit(1)):
        return

    site = Site(code="SC", name="Sede Central", address="Dirección de prueba")
    db.add(site)
    db.flush()

    classroom = Classroom(
        site_id=site.id,
        code="A-101",
        name="Aula 101",
        floor="1",
        capacity=35,
    )
    db.add(classroom)
    db.flush()

    camera = Camera(
        classroom_id=classroom.id,
        code="CAM-001",
        name="Cámara Aula 101",
        brand="Pendiente",
        source_type="pending",
    )
    professor = Professor(
        code="DOC-001",
        full_name="Profesor de prueba",
        email="profesor@example.com",
    )
    course = Course(
        code="CUR-001",
        name="Curso de prueba",
        description="Registro inicial para validar el flujo.",
        vocabulary=["Whisper", "CUDA", "transcripción"],
    )
    db.add_all([camera, professor, course])
    db.flush()

    db.add(
        CameraAssignment(
            camera_id=camera.id,
            classroom_id=classroom.id,
            started_at=datetime.now(timezone.utc),
            active=True,
            notes="Asignación inicial de demostración",
        )
    )
    db.add(
        Schedule(
            professor_id=professor.id,
            course_id=course.id,
            classroom_id=classroom.id,
            day_of_week=0,
            start_time=time(15, 0),
            end_time=time(18, 0),
        )
    )
    db.commit()
