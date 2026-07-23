from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

from app.api.admin import router as admin_router
from app.api.catalogs import router as catalogs_router
from app.api.jobs import router as jobs_router
from app.api.recordings import router as recordings_router
from app.api.system import router as system_router
from app.core.config import settings
from app.db.session import Base, SessionLocal, engine
from app.seed import seed_catalogs


def apply_lightweight_migrations() -> None:
    """Permite abrir esta versión sobre una BD creada por el prototipo anterior."""
    inspector = inspect(engine)
    if "transcription_jobs" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("transcription_jobs")}
    if "provider_name" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE transcription_jobs "
                    "ADD COLUMN provider_name VARCHAR(40) NOT NULL DEFAULT 'local'"
                )
            )


def ensure_schema_v4() -> None:
    inspector = inspect(engine)
    if "sites" not in inspector.get_table_names():
        return
    site_columns = {column["name"] for column in inspector.get_columns("sites")}
    required = {"code", "name", "address", "active"}
    if not required.issubset(site_columns):
        raise RuntimeError(
            "La base pertenece a una versión anterior. Para la primera instalación de v4, "
            "ejecuta scripts/reset_local_database.ps1 y vuelve a iniciar el sistema."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    apply_lightweight_migrations()
    ensure_schema_v4()
    with SessionLocal() as db:
        seed_catalogs(db)
    yield


app = FastAPI(
    title=settings.app_name,
    version="2.0.0",
    description=(
        "API asíncrona con Faster-Whisper local y proveedor Together AI opcional. "
        "El intervalo siempre se recorta antes de la transcripción."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["Sistema"])
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "provider": settings.transcription_provider,
    }


app.include_router(system_router)
app.include_router(admin_router)
app.include_router(catalogs_router)
app.include_router(recordings_router)
app.include_router(jobs_router)
