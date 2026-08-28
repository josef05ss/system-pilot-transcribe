"""
MOTOR DE VISIÓN (API de Anthropic / Claude)
---------------------------------------------
Motor de respaldo OPCIONAL. Se usa cuando Tesseract y TrOCR no alcanzan
el umbral de confianza mínimo (OCR_MIN_CONFIDENCE) — típicamente:
- Escritura a mano compleja/cursiva (TrOCR es un modelo especializado
  entrenado para reconocer una línea a la vez, mayormente en inglés).
- Texto impreso con bajo contraste, tipografías decorativas, o layouts
  complejos (infografías, fotos con texto superpuesto).

A diferencia de Tesseract/TrOCR (modelos especializados de reconocimiento
de patrones), aquí se usa un modelo de lenguaje multimodal grande, que
interpreta la imagen con contexto semántico y razonamiento — de forma
similar a como lo haría una persona leyendo la imagen.

Requiere una API key de Anthropic (variable de entorno ANTHROPIC_API_KEY).
Si no está configurada, `is_configured()` devuelve False y el resto del
pipeline sigue funcionando 100% local, sin llamadas externas.
"""

import base64
import io
from typing import Tuple

import httpx
from PIL import Image

from ..config import settings

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
NO_TEXT_SENTINEL = "NO_TEXT_FOUND"

SYSTEM_PROMPT = (
    "Eres un sistema de transcripción de texto en imágenes (impreso o "
    "manuscrito). Tu única tarea es transcribir EXACTAMENTE el texto "
    "legible que aparece en la imagen, palabra por palabra, preservando "
    "los saltos de línea originales tal como aparecen. No corrijas "
    "ortografía, no agregues comentarios, explicaciones, ni texto "
    "adicional de ningún tipo: responde solo con la transcripción. "
    f"Si la imagen no contiene ningún texto legible, responde "
    f"únicamente con la palabra: {NO_TEXT_SENTINEL}"
)


def is_configured() -> bool:
    """True si hay una API key configurada para este motor."""
    return bool(settings.ANTHROPIC_API_KEY)


def _image_to_base64_jpeg(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def transcribe(image: Image.Image) -> Tuple[str, float]:
    """Transcribe el texto de una imagen usando la API de Anthropic.

    Devuelve (texto, confianza). Si no se detecta texto, devuelve
    ("", 0.0). Lanza una excepción si la llamada a la API falla (el
    router debe capturarla y hacer fallback al motor local TrOCR).
    """
    if not is_configured():
        raise RuntimeError(
            "ANTHROPIC_API_KEY no está configurada; este motor no está disponible."
        )

    imagen_b64 = _image_to_base64_jpeg(image)

    payload = {
        "model": settings.ANTHROPIC_MODEL,
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": imagen_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Transcribe todo el texto legible de esta imagen.",
                    },
                ],
            }
        ],
    }

    headers = {
        "x-api-key": settings.ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_API_VERSION,
        "content-type": "application/json",
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.post(ANTHROPIC_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    texto = "".join(
        bloque.get("text", "")
        for bloque in data.get("content", [])
        if bloque.get("type") == "text"
    ).strip()

    if not texto or texto == NO_TEXT_SENTINEL:
        return "", 0.0

    return texto, settings.VISION_API_CONFIDENCE
