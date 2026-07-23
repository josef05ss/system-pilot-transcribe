from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Camera, Classroom, Recording
from app.schemas import RecordingOut
from app.services.media import MediaError, probe_media
from app.services.storage import get_storage_provider

router = APIRouter(prefix="/api/recordings", tags=["Grabaciones"])


@router.get("", response_model=list[RecordingOut])
def list_recordings(db: Session = Depends(get_db)) -> list[Recording]:
    return db.scalars(select(Recording).order_by(Recording.created_at.desc()).limit(100)).all()


@router.get("/{recording_id}/stream", response_class=FileResponse)
def stream_recording(recording_id: str, db: Session = Depends(get_db)) -> FileResponse:
    recording = db.get(Recording, recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="Grabación no encontrada")
    if recording.source_type != "local":
        raise HTTPException(status_code=409, detail="La vista previa directa solo está disponible para archivos locales")
    path = Path(recording.source_uri)
    if not path.exists():
        raise HTTPException(status_code=404, detail="El archivo de la grabación no existe")
    return FileResponse(path)


def validate_location(db: Session, site_id: str, classroom_id: str, camera_id: str) -> None:
    classroom = db.get(Classroom, classroom_id)
    camera = db.get(Camera, camera_id)
    if not classroom or not classroom.active or classroom.site_id != site_id:
        raise HTTPException(status_code=422, detail="El aula activa no pertenece a la sede")
    if not camera or not camera.active or camera.classroom_id != classroom_id:
        raise HTTPException(status_code=422, detail="La cámara activa no está asignada al aula")


def create_recording(
    db: Session,
    *,
    site_id: str,
    classroom_id: str,
    camera_id: str,
    recording_started_at: datetime,
    original_name: str,
    uri: str,
    size: int,
) -> Recording:
    try:
        metadata = probe_media(uri)
    except MediaError:
        Path(uri).unlink(missing_ok=True)
        raise

    if metadata["duration_seconds"] <= 0:
        Path(uri).unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="No se pudo determinar la duración")
    if not metadata["has_audio"]:
        Path(uri).unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="La grabación no contiene una pista de audio transcribible",
        )

    recording = Recording(
        site_id=site_id,
        classroom_id=classroom_id,
        camera_id=camera_id,
        source_type="local",
        source_uri=uri,
        original_name=original_name,
        recording_started_at=recording_started_at,
        duration_seconds=metadata["duration_seconds"],
        container_format=metadata["container_format"],
        video_codec=metadata["video_codec"],
        audio_codec=metadata["audio_codec"],
        audio_sample_rate=metadata["audio_sample_rate"],
        audio_channels=metadata["audio_channels"],
        has_audio=True,
        file_size_bytes=size,
    )
    db.add(recording)
    db.commit()
    db.refresh(recording)
    return recording


@router.post("/upload-fast", response_model=RecordingOut, status_code=status.HTTP_201_CREATED)
async def upload_recording_fast(
    request: Request,
    site_id: str = Query(...),
    classroom_id: str = Query(...),
    camera_id: str = Query(...),
    recording_started_at: datetime = Query(...),
    x_file_name: str = Header(..., alias="X-File-Name"),
    x_file_size: int | None = Header(default=None, alias="X-File-Size"),
    db: Session = Depends(get_db),
) -> Recording:
    """
    Subida directa optimizada para videos grandes.

    El navegador envía el File como cuerpo binario y el backend escribe el stream
    directamente al almacenamiento final. Así se evita el archivo temporal multipart
    y una copia adicional de varios GB. El frontend también puede mostrar progreso.
    """
    validate_location(db, site_id, classroom_id, camera_id)
    original_name = unquote(x_file_name).strip() or "grabacion.bin"
    storage = get_storage_provider()
    uri: str | None = None

    try:
        uri, size = await storage.save_stream(
            request.stream(),
            original_name=original_name,
            expected_size=x_file_size,
        )
        return create_recording(
            db,
            site_id=site_id,
            classroom_id=classroom_id,
            camera_id=camera_id,
            recording_started_at=recording_started_at,
            original_name=original_name,
            uri=uri,
            size=size,
        )
    except HTTPException:
        raise
    except (ValueError, MediaError) as exc:
        if uri:
            Path(uri).unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/upload", response_model=RecordingOut, status_code=status.HTTP_201_CREATED)
async def upload_recording_multipart_fallback(
    file: UploadFile = File(...),
    site_id: str = Form(...),
    classroom_id: str = Form(...),
    camera_id: str = Form(...),
    recording_started_at: datetime = Form(...),
    db: Session = Depends(get_db),
) -> Recording:
    """Ruta compatible de respaldo. El dashboard usa /upload-fast por defecto."""
    validate_location(db, site_id, classroom_id, camera_id)
    storage = get_storage_provider()
    original_name = file.filename or "grabacion"
    uri: str | None = None
    try:
        uri, size = await storage.save_upload(file)
        return create_recording(
            db,
            site_id=site_id,
            classroom_id=classroom_id,
            camera_id=camera_id,
            recording_started_at=recording_started_at,
            original_name=original_name,
            uri=uri,
            size=size,
        )
    except HTTPException:
        raise
    except (ValueError, MediaError) as exc:
        if uri:
            Path(uri).unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
