import uuid
from datetime import datetime

from sqlalchemy import Column, String, Text, Float, DateTime, Integer, Boolean, ForeignKey
from sqlalchemy.orm import relationship

from .database import Base


class Documento(Base):
    """Representa el archivo subido por el usuario (imagen individual o
    documento PDF/DOCX/PPTX/XLSX)."""

    __tablename__ = "documentos"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    nombre_original = Column(String, nullable=False)
    tipo_entrada = Column(String, nullable=False)  # "imagen" | "documento"
    extension = Column(String, nullable=False)
    creado_en = Column(DateTime, default=datetime.utcnow)

    resultados = relationship(
        "ResultadoOCR", back_populates="documento", cascade="all, delete-orphan"
    )


class ResultadoOCR(Base):
    """Resultado de OCR para cada imagen extraída/procesada.
    Un Documento puede generar N resultados (una imagen -> 1,
    un PDF/DOCX/PPTX/XLSX -> N imágenes extraídas -> N resultados)."""

    __tablename__ = "resultados_ocr"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    documento_id = Column(String, ForeignKey("documentos.id"), nullable=False)
    indice_imagen = Column(Integer, default=0)  # posición dentro del documento

    clasificacion = Column(String, nullable=False)  # "digital" | "manuscrita" | "ia_vision" | "sin_texto" | "baja_confianza"
    motor_ocr = Column(String, nullable=False)  # "tesseract" | "trocr" | "claude_vision" | "ninguno"
    texto = Column(Text, default="")
    confianza = Column(Float, nullable=True)
    es_color = Column(Boolean, nullable=True)  # True=color, False=B&N, None=no evaluado

    imagen_preprocesada_path = Column(String, nullable=True)
    creado_en = Column(DateTime, default=datetime.utcnow)

    documento = relationship("Documento", back_populates="resultados")
