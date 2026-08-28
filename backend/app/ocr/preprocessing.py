"""
PREPROCESAMIENTO
-----------------
Corresponde al nodo "PREPROCESAMIENTO" del diagrama, entre
"IMAGEN EN MEMORIA" y "OCR ROUTER".

Aplica limpieza de visión por computador para mejorar la exactitud de
ambos motores de OCR (Tesseract y TrOCR):
- Escalado (upscale) si la imagen es pequeña/de baja resolución
- Conversión a escala de grises
- Corrección de inclinación (deskew)
- Reducción de ruido
- Mejora de contraste local (CLAHE)

IMPORTANTE: a propósito NO se aplica una binarización fuerte (blanco/negro
puro) como paso general. Eso funciona bien para escaneos simples de una
sola columna, pero en imágenes complejas -infografías con fotos a color,
múltiples columnas, iconos- una binarización agresiva destruye el
detalle y hace que Tesseract "lea" ruido en vez de texto real (falsos
positivos con confianza alta). Tesseract y TrOCR ya hacen su propia
binarización interna, que es más robusta que un umbral fijo genérico.
"""

import cv2
import numpy as np
from PIL import Image

from ..config import settings

# Ancho mínimo objetivo antes de correr OCR. Imágenes descargadas de la
# web / infografías suelen venir en baja resolución (ej. 638px), lo que
# perjudica mucho la exactitud del OCR.
MIN_TARGET_WIDTH = 1400


def _pil_to_cv(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv_to_pil(mat: np.ndarray) -> Image.Image:
    if len(mat.shape) == 2:
        return Image.fromarray(mat)
    return Image.fromarray(cv2.cvtColor(mat, cv2.COLOR_BGR2RGB))


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Corrige pequeñas inclinaciones de la imagen escaneada/fotografiada."""
    coords = np.column_stack(np.where(gray < 250))
    if coords.shape[0] < 20:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5 or abs(angle) > 20:
        # Ángulos grandes suelen ser falsos positivos en imágenes con
        # fotos/iconos (no son documentos escaneados torcidos de verdad)
        return gray
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _upscale_if_small(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    if max(h, w) >= MIN_TARGET_WIDTH:
        return gray
    scale = MIN_TARGET_WIDTH / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC)


def enhance_image(image: Image.Image) -> Image.Image:
    """Recibe una imagen PIL cruda y devuelve una versión mejorada
    (escala de grises, escalada, con contraste realzado) lista para
    pasar por el OCR Router y los motores de OCR."""
    mat = _pil_to_cv(image)
    gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)

    upscaled = _upscale_if_small(gray)

    # Reducción de ruido conservando bordes de texto
    denoised = cv2.fastNlMeansDenoising(upscaled, h=8)

    deskewed = _deskew(denoised)

    # CLAHE: mejora el contraste local sin perder gradientes (a
    # diferencia de una binarización dura), ideal para fotos + texto
    # mezclados como en infografías
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(deskewed)

    return _cv_to_pil(enhanced)


def has_text_content(enhanced_image: Image.Image) -> bool:
    """Determina si una imagen YA MEJORADA (enhance_image) tiene
    suficiente contenido visual (texto o incluso una foto/gráfico) como
    para valer la pena correr OCR, o si está prácticamente en blanco.

    Heurística: se binariza con Otsu solo para esta medición (no se usa
    para el OCR en sí) y se calcula qué proporción de la imagen es la
    clase minoritaria (contenido) frente a la mayoritaria (fondo),
    sin asumir de antemano si el fondo es claro u oscuro.
    """
    arr = np.array(enhanced_image.convert("L"))
    if arr.size == 0:
        return False
    _, binarized = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ratio_low = float(np.mean(binarized < 128))
    content_ratio = min(ratio_low, 1.0 - ratio_low)
    return settings.MIN_INK_RATIO <= content_ratio <= settings.MAX_INK_RATIO
