"""
OCR ROUTER
-----------
Corresponde al nodo "OCR ROUTER" del diagrama, que decide qué motor usa
para transcribir la imagen preprocesada.

Orden de decisión:

1. Si la imagen no tiene contenido visual relevante (ver
   `preprocessing.has_text_content`) -> se descarta de inmediato,
   sin llamar a ningún motor. Clasificación: "sin_texto".

2. Se corre Tesseract (motor local, rápido, gratis). Si su confianza
   >= OCR_MIN_CONFIDENCE -> se acepta como texto DIGITAL.

3. Si Tesseract no alcanza el umbral, y hay una API key de Anthropic
   configurada, se usa el motor de visión en la nube (Claude) como
   respaldo -> clasificación "ia_vision". Este paso cubre tanto
   escritura a mano compleja como texto impreso de baja calidad
   (infografías, bajo contraste, tipografías decorativas) donde los
   modelos locales especializados no rinden bien.

4. Si la API de visión no está configurada, o falla, se intenta con
   TrOCR (motor local especializado en escritura a mano) como último
   recurso -> clasificación "manuscrita".

5. Si NINGÚN motor disponible alcanza el umbral de confianza, el
   resultado se descarta (no se muestra texto poco fiable) y se marca
   como "baja_confianza".

Esta heurística de enrutamiento (paso 2) es intencionalmente simple y
reemplazable: se puede sustituir por un clasificador de imágenes
entrenado (CNN binaria digital/manuscrita) sin tocar el resto del
pipeline.
"""

from typing import Optional, Tuple

from PIL import Image

from ..config import settings
from . import tesseract_engine, trocr_engine, vision_api_engine
from .preprocessing import has_text_content

MENSAJE_SIN_TEXTO = "No se encontró texto."
MENSAJE_BAJA_CONFIANZA = (
    "Se detectó posible texto, pero ningún motor alcanzó el umbral de "
    f"confianza mínimo ({settings.OCR_MIN_CONFIDENCE:.0f}%), por lo que "
    "no se muestra para evitar errores."
)


def route_and_run(
    imagen_preprocesada: Image.Image,
    imagen_original: Optional[Image.Image] = None,
) -> Tuple[str, str, str, float]:
    """Ejecuta el enrutamiento + el/los motor(es) de OCR correspondientes.

    `imagen_preprocesada`: imagen ya pasada por enhance_image (escala de
    grises, escalada, contraste realzado) — usada por Tesseract y TrOCR.
    `imagen_original`: imagen tal cual se subió (a color, sin modificar)
    — se usa para el motor de visión en la nube, que rinde mejor con la
    imagen original que con la versión preprocesada para OCR clásico.

    Devuelve: (clasificacion, motor_usado, texto, confianza)
    clasificacion ∈ {"digital", "manuscrita", "ia_vision", "sin_texto", "baja_confianza"}
    """
    # 0) Filtro previo: ¿la imagen tiene contenido visual relevante?
    if not has_text_content(imagen_preprocesada):
        return "sin_texto", "ninguno", MENSAJE_SIN_TEXTO, 0.0

    # 1) Intento con Tesseract (texto digital/impreso, motor local)
    texto_tesseract, confianza_tesseract = tesseract_engine.run_tesseract(
        imagen_preprocesada
    )
    if texto_tesseract.strip() and confianza_tesseract >= settings.OCR_MIN_CONFIDENCE:
        return "digital", "tesseract", texto_tesseract, confianza_tesseract

    # 2) Respaldo: API de visión en la nube (si está configurada)
    if vision_api_engine.is_configured():
        try:
            imagen_para_vision = (
                imagen_original if imagen_original is not None else imagen_preprocesada
            )
            texto_vision, confianza_vision = vision_api_engine.transcribe(
                imagen_para_vision
            )
            if texto_vision.strip():
                return "ia_vision", "claude_vision", texto_vision, confianza_vision
        except Exception as exc:  # nunca debe tumbar el pipeline
            print(f"[vision_api_engine] Error al llamar la API, usando fallback local: {exc}")

    # 3) Último recurso local: TrOCR (especializado en escritura a mano)
    texto_trocr, confianza_trocr = trocr_engine.run_trocr(imagen_preprocesada)
    if texto_trocr.strip() and confianza_trocr >= settings.OCR_MIN_CONFIDENCE:
        return "manuscrita", "trocr", texto_trocr, confianza_trocr

    # 4) Ningún motor disponible alcanzó el umbral de confianza
    mejor_confianza = max(confianza_tesseract, confianza_trocr)
    if not texto_tesseract.strip() and not texto_trocr.strip():
        return "sin_texto", "ninguno", MENSAJE_SIN_TEXTO, mejor_confianza

    return "baja_confianza", "ninguno", MENSAJE_BAJA_CONFIANZA, mejor_confianza
