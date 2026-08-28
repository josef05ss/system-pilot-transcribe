"""
DETECCIÓN DE COLOR
--------------------
Determina si una imagen es a color o en blanco y negro / escala de
grises. Se ejecuta sobre la imagen ORIGINAL (antes del preprocesamiento
que la convierte a escala de grises para el OCR), y el resultado se
guarda junto con el resto del resultado del OCR.
"""

import cv2
import numpy as np
from PIL import Image

from ..config import settings


def es_imagen_color(image: Image.Image) -> bool:
    """Devuelve True si la imagen es a color, False si es blanco y
    negro / escala de grises.

    Heurística: se convierte a HSV y se mide la saturación promedio del
    canal S. Una imagen en blanco y negro (o casi) tiene saturación
    cercana a 0 en casi todos los píxeles, mientras que una imagen a
    color tiene saturación notablemente mayor en las zonas con color.
    """
    rgb = image.convert("RGB")
    hsv = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2HSV)
    saturacion_promedio = float(np.mean(hsv[:, :, 1]))
    return saturacion_promedio >= settings.COLOR_SATURATION_THRESHOLD
