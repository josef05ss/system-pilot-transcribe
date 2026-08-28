"""
MOTOR TESSERACT
----------------
Usado para el camino "DIGITAL" del OCR Router: texto impreso/tipeado.
"""

from typing import Tuple

import pytesseract
from PIL import Image

from ..config import settings


def run_tesseract(image: Image.Image) -> Tuple[str, float]:
    """Ejecuta Tesseract sobre la imagen y devuelve (texto, confianza_promedio).

    --oem 3: motor LSTM (el más preciso, default en Tesseract >= 4).
    --psm 3: segmentación automática de página completa, capaz de
             manejar layouts con varias columnas como infografías
             (a diferencia de --psm 6, que asume un único bloque
             uniforme de texto y falla en layouts multi-columna).
    """
    config = "--oem 3 --psm 3"
    data = pytesseract.image_to_data(
        image,
        lang=settings.TESSERACT_LANG,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    palabras = []
    confidencias = []
    for i, word in enumerate(data.get("text", [])):
        word = word.strip()
        conf = data.get("conf", ["-1"])[i]
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = -1.0
        if word:
            palabras.append(word)
        if conf >= 0:
            confidencias.append(conf)

    texto = " ".join(palabras).strip()
    confianza_promedio = sum(confidencias) / len(confidencias) if confidencias else 0.0
    return texto, confianza_promedio


def quick_confidence(image: Image.Image) -> float:
    """Corrida rápida solo para que el OCR Router estime la confianza
    (usada para decidir digital vs manuscrita), sin devolver el texto final."""
    _, confianza = run_tesseract(image)
    return confianza
