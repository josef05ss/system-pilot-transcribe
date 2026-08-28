"""
MOTOR TrOCR
------------
Usado para el camino "MANUSCRITA" del OCR Router: texto escrito a mano.

Usa el modelo de HuggingFace `microsoft/trocr-base-handwritten` a través
de `transformers`. El modelo se descarga una única vez (se cachea en
/app/model_cache dentro del contenedor, ver docker-compose volume) la
primera vez que se necesita, por lo que la primera transcripción
manuscrita puede tardar más mientras se descarga (~1.3 GB, requiere
acceso a internet desde el contenedor backend).
"""

from functools import lru_cache
from typing import Tuple

from PIL import Image

from ..config import settings


@lru_cache(maxsize=1)
def _load_model():
    """Carga perezosa (lazy) del procesador y el modelo TrOCR.
    Se ejecuta una sola vez gracias a lru_cache."""
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(settings.TROCR_MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(settings.TROCR_MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device


def run_trocr(image: Image.Image) -> Tuple[str, float]:
    """Ejecuta TrOCR sobre la imagen y devuelve (texto, confianza_aprox)."""
    import torch

    processor, model, device = _load_model()

    rgb_image = image.convert("RGB")
    pixel_values = processor(images=rgb_image, return_tensors="pt").pixel_values.to(
        device
    )

    with torch.no_grad():
        generated = model.generate(
            pixel_values,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=256,
        )

    texto = processor.batch_decode(
        generated.sequences, skip_special_tokens=True
    )[0].strip()

    # Confianza aproximada a partir de los scores de generación
    try:
        scores = torch.stack(generated.scores, dim=1)
        probs = torch.softmax(scores, dim=-1)
        top_probs = probs.max(dim=-1).values
        confianza = float(top_probs.mean().item()) * 100
    except Exception:
        confianza = 0.0

    return texto, confianza
