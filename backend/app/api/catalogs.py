from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Camera, CameraAssignment, Classroom, Course, Professor, Schedule, Site
from app.schemas import CatalogsOut

router = APIRouter(prefix="/api/catalogs", tags=["Catálogos"])


@router.get("", response_model=CatalogsOut)
def get_catalogs(db: Session = Depends(get_db)) -> CatalogsOut:
    return CatalogsOut(
        sites=db.scalars(select(Site).order_by(Site.name)).all(),
        classrooms=db.scalars(select(Classroom).order_by(Classroom.code)).all(),
        cameras=db.scalars(select(Camera).order_by(Camera.code)).all(),
        assignments=db.scalars(
            select(CameraAssignment).order_by(CameraAssignment.started_at.desc())
        ).all(),
        professors=db.scalars(select(Professor).order_by(Professor.full_name)).all(),
        courses=db.scalars(select(Course).order_by(Course.name)).all(),
        schedules=db.scalars(
            select(Schedule).order_by(Schedule.day_of_week, Schedule.start_time)
        ).all(),
    )
