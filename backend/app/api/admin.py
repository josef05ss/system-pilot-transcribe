from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TypeVar

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Camera, CameraAssignment, Classroom, Course, Professor, Schedule, Site
from app.schemas import (
    CameraAssignmentCreate,
    CameraAssignmentOut,
    CameraCreate,
    CameraOut,
    CameraUpdate,
    ClassroomCreate,
    ClassroomOut,
    ClassroomUpdate,
    CourseCreate,
    CourseOut,
    CourseUpdate,
    ProfessorCreate,
    ProfessorOut,
    ProfessorUpdate,
    ScheduleCreate,
    ScheduleOut,
    ScheduleUpdate,
    SiteCreate,
    SiteOut,
    SiteUpdate,
)

router = APIRouter(prefix="/api/admin", tags=["Administración"])
T = TypeVar("T")


def require(db: Session, model: type[T], object_id: str, label: str) -> T:
    item = db.get(model, object_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"{label} no encontrado")
    return item


def save(db: Session, item: Any, *, duplicate_message: str) -> Any:
    try:
        db.add(item)
        db.commit()
        db.refresh(item)
        return item
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=duplicate_message) from exc


def patch_model(item: Any, payload: Any) -> Any:
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(item, field, value)
    return item


# --------------------------- SEDES ---------------------------
@router.get("/sites", response_model=list[SiteOut])
def list_sites(db: Session = Depends(get_db)) -> list[Site]:
    return db.scalars(select(Site).order_by(Site.name)).all()


@router.post("/sites", response_model=SiteOut, status_code=status.HTTP_201_CREATED)
def create_site(payload: SiteCreate, db: Session = Depends(get_db)) -> Site:
    return save(db, Site(**payload.model_dump()), duplicate_message="El código o nombre de sede ya existe")


@router.patch("/sites/{site_id}", response_model=SiteOut)
def update_site(site_id: str, payload: SiteUpdate, db: Session = Depends(get_db)) -> Site:
    site = require(db, Site, site_id, "Sede")
    return save(db, patch_model(site, payload), duplicate_message="El código o nombre de sede ya existe")


@router.delete("/sites/{site_id}", response_model=SiteOut)
def deactivate_site(site_id: str, db: Session = Depends(get_db)) -> Site:
    site = require(db, Site, site_id, "Sede")
    site.active = False
    return save(db, site, duplicate_message="No se pudo desactivar la sede")


# --------------------------- AULAS ---------------------------
@router.get("/classrooms", response_model=list[ClassroomOut])
def list_classrooms(db: Session = Depends(get_db)) -> list[Classroom]:
    return db.scalars(select(Classroom).order_by(Classroom.code)).all()


@router.post("/classrooms", response_model=ClassroomOut, status_code=status.HTTP_201_CREATED)
def create_classroom(payload: ClassroomCreate, db: Session = Depends(get_db)) -> Classroom:
    require(db, Site, payload.site_id, "Sede")
    return save(db, Classroom(**payload.model_dump()), duplicate_message="El código de aula ya existe en esa sede")


@router.patch("/classrooms/{classroom_id}", response_model=ClassroomOut)
def update_classroom(
    classroom_id: str, payload: ClassroomUpdate, db: Session = Depends(get_db)
) -> Classroom:
    classroom = require(db, Classroom, classroom_id, "Aula")
    if payload.site_id:
        require(db, Site, payload.site_id, "Sede")
    return save(db, patch_model(classroom, payload), duplicate_message="El código de aula ya existe en esa sede")


@router.delete("/classrooms/{classroom_id}", response_model=ClassroomOut)
def deactivate_classroom(classroom_id: str, db: Session = Depends(get_db)) -> Classroom:
    classroom = require(db, Classroom, classroom_id, "Aula")
    classroom.active = False
    return save(db, classroom, duplicate_message="No se pudo desactivar el aula")


# --------------------------- CÁMARAS ---------------------------
@router.get("/cameras", response_model=list[CameraOut])
def list_cameras(db: Session = Depends(get_db)) -> list[Camera]:
    return db.scalars(select(Camera).order_by(Camera.code)).all()


@router.post("/cameras", response_model=CameraOut, status_code=status.HTTP_201_CREATED)
def create_camera(payload: CameraCreate, db: Session = Depends(get_db)) -> Camera:
    if payload.classroom_id:
        require(db, Classroom, payload.classroom_id, "Aula")
    camera = save(db, Camera(**payload.model_dump()), duplicate_message="El código o número de serie ya existe")
    if payload.classroom_id:
        assignment = CameraAssignment(
            camera_id=camera.id,
            classroom_id=payload.classroom_id,
            started_at=datetime.now(timezone.utc),
            active=True,
            notes="Asignación creada junto con la cámara",
        )
        save(db, assignment, duplicate_message="No se pudo crear la asignación inicial")
    return camera


@router.patch("/cameras/{camera_id}", response_model=CameraOut)
def update_camera(camera_id: str, payload: CameraUpdate, db: Session = Depends(get_db)) -> Camera:
    camera = require(db, Camera, camera_id, "Cámara")
    if "classroom_id" in payload.model_fields_set and payload.classroom_id:
        require(db, Classroom, payload.classroom_id, "Aula")
    return save(db, patch_model(camera, payload), duplicate_message="El código o número de serie ya existe")


@router.delete("/cameras/{camera_id}", response_model=CameraOut)
def deactivate_camera(camera_id: str, db: Session = Depends(get_db)) -> Camera:
    camera = require(db, Camera, camera_id, "Cámara")
    camera.active = False
    return save(db, camera, duplicate_message="No se pudo desactivar la cámara")


# ---------------------- ASIGNACIONES DE CÁMARA ----------------------
@router.get("/camera-assignments", response_model=list[CameraAssignmentOut])
def list_assignments(db: Session = Depends(get_db)) -> list[CameraAssignment]:
    return db.scalars(
        select(CameraAssignment).order_by(CameraAssignment.started_at.desc())
    ).all()


@router.post(
    "/camera-assignments",
    response_model=CameraAssignmentOut,
    status_code=status.HTTP_201_CREATED,
)
def create_assignment(
    payload: CameraAssignmentCreate, db: Session = Depends(get_db)
) -> CameraAssignment:
    camera = require(db, Camera, payload.camera_id, "Cámara")
    require(db, Classroom, payload.classroom_id, "Aula")

    now = payload.started_at or datetime.now(timezone.utc)
    active_assignments = db.scalars(
        select(CameraAssignment).where(
            CameraAssignment.camera_id == camera.id,
            CameraAssignment.active.is_(True),
        )
    ).all()
    for old in active_assignments:
        old.active = False
        old.ended_at = now

    assignment = CameraAssignment(
        camera_id=camera.id,
        classroom_id=payload.classroom_id,
        started_at=now,
        active=True,
        notes=payload.notes,
    )
    camera.classroom_id = payload.classroom_id
    try:
        db.add(assignment)
        db.commit()
        db.refresh(assignment)
        return assignment
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="No se pudo asignar la cámara") from exc


@router.delete("/camera-assignments/{assignment_id}", response_model=CameraAssignmentOut)
def close_assignment(assignment_id: str, db: Session = Depends(get_db)) -> CameraAssignment:
    assignment = require(db, CameraAssignment, assignment_id, "Asignación")
    assignment.active = False
    assignment.ended_at = datetime.now(timezone.utc)
    camera = require(db, Camera, assignment.camera_id, "Cámara")
    if camera.classroom_id == assignment.classroom_id:
        camera.classroom_id = None
    return save(db, assignment, duplicate_message="No se pudo cerrar la asignación")


# --------------------------- PROFESORES ---------------------------
@router.get("/professors", response_model=list[ProfessorOut])
def list_professors(db: Session = Depends(get_db)) -> list[Professor]:
    return db.scalars(select(Professor).order_by(Professor.full_name)).all()


@router.post("/professors", response_model=ProfessorOut, status_code=status.HTTP_201_CREATED)
def create_professor(payload: ProfessorCreate, db: Session = Depends(get_db)) -> Professor:
    return save(db, Professor(**payload.model_dump()), duplicate_message="El código o correo del profesor ya existe")


@router.patch("/professors/{professor_id}", response_model=ProfessorOut)
def update_professor(
    professor_id: str, payload: ProfessorUpdate, db: Session = Depends(get_db)
) -> Professor:
    professor = require(db, Professor, professor_id, "Profesor")
    return save(db, patch_model(professor, payload), duplicate_message="El código o correo del profesor ya existe")


@router.delete("/professors/{professor_id}", response_model=ProfessorOut)
def deactivate_professor(professor_id: str, db: Session = Depends(get_db)) -> Professor:
    professor = require(db, Professor, professor_id, "Profesor")
    professor.active = False
    return save(db, professor, duplicate_message="No se pudo desactivar el profesor")


# --------------------------- CURSOS ---------------------------
@router.get("/courses", response_model=list[CourseOut])
def list_courses(db: Session = Depends(get_db)) -> list[Course]:
    return db.scalars(select(Course).order_by(Course.name)).all()


@router.post("/courses", response_model=CourseOut, status_code=status.HTTP_201_CREATED)
def create_course(payload: CourseCreate, db: Session = Depends(get_db)) -> Course:
    return save(db, Course(**payload.model_dump()), duplicate_message="El código del curso ya existe")


@router.patch("/courses/{course_id}", response_model=CourseOut)
def update_course(course_id: str, payload: CourseUpdate, db: Session = Depends(get_db)) -> Course:
    course = require(db, Course, course_id, "Curso")
    return save(db, patch_model(course, payload), duplicate_message="El código del curso ya existe")


@router.delete("/courses/{course_id}", response_model=CourseOut)
def deactivate_course(course_id: str, db: Session = Depends(get_db)) -> Course:
    course = require(db, Course, course_id, "Curso")
    course.active = False
    return save(db, course, duplicate_message="No se pudo desactivar el curso")


# --------------------------- HORARIOS ---------------------------
@router.get("/schedules", response_model=list[ScheduleOut])
def list_schedules(db: Session = Depends(get_db)) -> list[Schedule]:
    return db.scalars(select(Schedule).order_by(Schedule.day_of_week, Schedule.start_time)).all()


@router.post("/schedules", response_model=ScheduleOut, status_code=status.HTTP_201_CREATED)
def create_schedule(payload: ScheduleCreate, db: Session = Depends(get_db)) -> Schedule:
    require(db, Professor, payload.professor_id, "Profesor")
    require(db, Course, payload.course_id, "Curso")
    require(db, Classroom, payload.classroom_id, "Aula")
    return save(db, Schedule(**payload.model_dump()), duplicate_message="No se pudo crear el horario")


@router.patch("/schedules/{schedule_id}", response_model=ScheduleOut)
def update_schedule(
    schedule_id: str, payload: ScheduleUpdate, db: Session = Depends(get_db)
) -> Schedule:
    schedule = require(db, Schedule, schedule_id, "Horario")
    values = payload.model_dump(exclude_unset=True)
    if values.get("professor_id"):
        require(db, Professor, values["professor_id"], "Profesor")
    if values.get("course_id"):
        require(db, Course, values["course_id"], "Curso")
    if values.get("classroom_id"):
        require(db, Classroom, values["classroom_id"], "Aula")
    start = values.get("start_time", schedule.start_time)
    end = values.get("end_time", schedule.end_time)
    if end <= start:
        raise HTTPException(status_code=422, detail="La hora final debe ser posterior a la inicial")
    return save(db, patch_model(schedule, payload), duplicate_message="No se pudo actualizar el horario")


@router.delete("/schedules/{schedule_id}", response_model=ScheduleOut)
def deactivate_schedule(schedule_id: str, db: Session = Depends(get_db)) -> Schedule:
    schedule = require(db, Schedule, schedule_id, "Horario")
    schedule.active = False
    return save(db, schedule, duplicate_message="No se pudo desactivar el horario")
