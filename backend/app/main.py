import os
import uuid
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func

from .config import settings
from .database import Base, engine, get_db
from . import models, schemas
from .ocr import extractor, preprocessing, router as ocr_router
from .ocr.color_detection import es_imagen_color

app = FastAPI(
    title="OCR Pipeline API",
    description="Backend de transcripción de imágenes (digital y manuscrita)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    # Crea las tablas si no existen (para un setup simple sin Alembic)
    Base.metadata.create_all(bind=engine)


@app.get("/api/health")
def health():
    return {"status": "ok"}


def _guardar_archivo_temporal(upload: UploadFile) -> str:
    extension = Path(upload.filename).suffix.lower()
    nombre_temporal = f"{uuid.uuid4()}{extension}"
    ruta = os.path.join(settings.UPLOAD_DIR, nombre_temporal)
    with open(ruta, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return ruta


@app.post("/api/ocr/upload", response_model=schemas.DocumentoSchema)
def subir_archivo(archivo: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Punto de entrada equivalente al nodo "USUARIO" del diagrama.
    Acepta:
      - Imagen individual: png/jpg/jpeg/bmp/tiff
      - Documento: pdf/docx/pptx/xlsx (se extraen solo las imágenes)
    Ejecuta preprocesamiento -> OCR router -> Tesseract/TrOCR -> PostgreSQL.
    """
    extension = Path(archivo.filename).suffix.lower()

    if extension in settings.IMAGE_EXTENSIONS:
        tipo_entrada = "imagen"
    elif extension in settings.DOCUMENT_EXTENSIONS:
        tipo_entrada = "documento"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Extensión no soportada: {extension}. "
            f"Soportadas: {sorted(settings.IMAGE_EXTENSIONS | settings.DOCUMENT_EXTENSIONS)}",
        )

    ruta_temporal = _guardar_archivo_temporal(archivo)

    try:
        from PIL import Image

        if tipo_entrada == "imagen":
            imagenes = [Image.open(ruta_temporal).convert("RGB")]
        else:
            imagenes = extractor.extract_images(ruta_temporal, extension)
            if not imagenes:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "No se encontraron imágenes de contenido válidas en el "
                        "documento (se filtran íconos/logos muy pequeños y "
                        "duplicados)."
                    ),
                )

        documento = models.Documento(
            nombre_original=archivo.filename,
            tipo_entrada=tipo_entrada,
            extension=extension,
        )
        db.add(documento)
        db.flush()  # obtiene documento.id sin commitear aún

        for indice, imagen in enumerate(imagenes):
            es_color = es_imagen_color(imagen)
            imagen_preprocesada = preprocessing.enhance_image(imagen)

            clasificacion, motor, texto, confianza = ocr_router.route_and_run(
                imagen_preprocesada, imagen
            )

            resultado = models.ResultadoOCR(
                documento_id=documento.id,
                indice_imagen=indice,
                clasificacion=clasificacion,
                motor_ocr=motor,
                texto=texto,
                confianza=confianza,
                es_color=es_color,
            )
            db.add(resultado)

        db.commit()
        db.refresh(documento)
        return documento

    finally:
        if os.path.exists(ruta_temporal):
            os.remove(ruta_temporal)


@app.get("/api/ocr/results", response_model=List[schemas.DocumentoListItemSchema])
def listar_documentos(db: Session = Depends(get_db)):
    """Lista todos los documentos procesados con su cantidad de resultados."""
    documentos = (
        db.query(
            models.Documento,
            func.count(models.ResultadoOCR.id).label("total_resultados"),
        )
        .outerjoin(models.ResultadoOCR)
        .group_by(models.Documento.id)
        .order_by(models.Documento.creado_en.desc())
        .all()
    )

    respuesta = []
    for documento, total in documentos:
        item = schemas.DocumentoListItemSchema.model_validate(documento)
        item.total_resultados = total
        respuesta.append(item)
    return respuesta


@app.get("/api/ocr/results/{documento_id}", response_model=schemas.DocumentoSchema)
def obtener_documento(documento_id: str, db: Session = Depends(get_db)):
    documento = db.query(models.Documento).filter(
        models.Documento.id == documento_id
    ).first()
    if not documento:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    return documento


@app.delete("/api/ocr/results/{documento_id}")
def eliminar_documento(documento_id: str, db: Session = Depends(get_db)):
    documento = db.query(models.Documento).filter(
        models.Documento.id == documento_id
    ).first()
    if not documento:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    db.delete(documento)
    db.commit()
    return {"status": "eliminado", "id": documento_id}
