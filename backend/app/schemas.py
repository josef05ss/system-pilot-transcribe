from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, ConfigDict


class ResultadoOCRSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    documento_id: str
    indice_imagen: int
    clasificacion: str
    motor_ocr: str
    texto: str
    confianza: Optional[float] = None
    es_color: Optional[bool] = None
    creado_en: datetime


class DocumentoSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    nombre_original: str
    tipo_entrada: str
    extension: str
    creado_en: datetime
    resultados: List[ResultadoOCRSchema] = []


class DocumentoListItemSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    nombre_original: str
    tipo_entrada: str
    extension: str
    creado_en: datetime
    total_resultados: int = 0
