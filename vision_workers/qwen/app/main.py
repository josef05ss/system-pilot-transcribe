from __future__ import annotations

import asyncio
import io
import os
import re
import time
from collections import Counter

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps

app = FastAPI(
    title="Qwen3-VL OCR Worker",
    version="5.0.0",
    description="OCR visual fiel orientado a producción. Una sola pasada visual.",
)

_MODEL = None
_PROCESSOR = None
_MODEL_LOCK = asyncio.Lock()

MODEL_NAME = os.getenv(
    "QWEN_MODEL_NAME",
    "Qwen/Qwen3-VL-4B-Instruct",
)

# Qwen3-VL usa control de presupuesto visual por píxeles.
# 512K evita perder demasiado detalle en imágenes pequeñas.
# 4M permite conservar páginas fotográficas/documentales con buen detalle
# sin disparar excesivamente VRAM en una GPU de consumo.
MIN_PIXELS = int(os.getenv("QWEN_MIN_PIXELS", str(512 * 1024)))
MAX_PIXELS = int(os.getenv("QWEN_MAX_PIXELS", str(4 * 1024 * 1024)))

MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "4096"))
CUDA_DEVICE = int(os.getenv("QWEN_CUDA_DEVICE", "0"))

# IMPORTANTE:
# No usamos repetition_penalty ni no_repeat_ngram_size.
# En OCR real puede haber repeticiones legítimas y esas penalizaciones
# pueden modificar el texto.
OCR_SYSTEM_PROMPT = """Eres un motor de transcripción visual OCR, no un asistente de redacción.

Tu única función es leer texto visible y devolver una transcripción fiel.
La evidencia visual tiene prioridad sobre lo que "sonaría mejor" en español.
No debes corregir, resumir, completar ni reescribir el contenido.
"""

OCR_USER_PROMPT = """Transcribe TODO el texto visible de esta imagen.

OBJETIVO:
Producir la lectura más fiel posible de lo que realmente está escrito.

REGLAS:
- Devuelve únicamente la transcripción, sin introducciones ni explicaciones.
- No agregues etiquetas como INICIO, FIN, TRANSCRIPCIÓN, OCR o similares.
- No uses bloques Markdown.
- No resumas.
- No traduzcas.
- No normalices ortografía.
- No mejores gramática ni estilo.
- No cambies tú/usted, tiempos verbales ni palabras por otras que suenen mejor.
- Conserva nombres propios, números, fechas, teléfonos, correos, direcciones,
  IP, comandos, fórmulas, códigos y signos exactamente como los veas.
- Usa el contexto lingüístico únicamente para distinguir letras o palabras
  visualmente ambiguas; nunca para inventar una frase que no esté sustentada
  por los trazos visibles.
- Si una palabra tiene una lectura probable, escribe esa lectura.
- Si realmente no puedes leer una palabra, escribe [ilegible].
- Conserva títulos y saltos de línea cuando ayuden a mantener el orden visual.
- Lee de arriba hacia abajo y de izquierda a derecha, respetando bloques.
"""


def _runtime_info() -> dict:
    try:
        import torch

        cuda = torch.cuda.is_available()
        info = {
            "cuda_available": cuda,
            "torch_cuda": torch.version.cuda,
        }
        if cuda:
            info["gpu"] = torch.cuda.get_device_name(CUDA_DEVICE)
            free_bytes, total_bytes = torch.cuda.mem_get_info(CUDA_DEVICE)
            info["gpu_free_gb"] = round(free_bytes / (1024**3), 2)
            info["gpu_total_gb"] = round(total_bytes / (1024**3), 2)
        return info
    except Exception as exc:
        return {"cuda_available": False, "detail": str(exc)}


def get_runtime():
    global _MODEL, _PROCESSOR

    if _MODEL is not None and _PROCESSOR is not None:
        return _MODEL, _PROCESSOR

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cuda = torch.cuda.is_available()

    if cuda:
        torch.cuda.set_device(CUDA_DEVICE)
        # Compatible con RTX 30xx/40xx en Windows sin depender de flash-attn.
        dtype = torch.float16
        model_kwargs = {
            "dtype": dtype,
            "device_map": "auto",
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        }
    else:
        model_kwargs = {
            "dtype": "auto",
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }

    _MODEL = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        **model_kwargs,
    )
    _MODEL.eval()

    _PROCESSOR = AutoProcessor.from_pretrained(MODEL_NAME)

    return _MODEL, _PROCESSOR


def _clean_output(text: str) -> str:
    value = (text or "").strip()

    # Elimina wrappers frecuentes sin tocar el contenido interno.
    value = re.sub(r"^```(?:text|txt|markdown)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value)

    wrapper_patterns = [
        r"^\s*-{2,}\s*INICIO\s*-{2,}\s*",
        r"^\s*INICIO\s*:?\s*",
        r"^\s*TRANSCRIPCI[ÓO]N\s*:?\s*",
        r"\s*-{2,}\s*FIN\s*-{2,}\s*$",
        r"\s*FIN\s*$",
    ]
    for pattern in wrapper_patterns:
        value = re.sub(pattern, "", value, flags=re.I)

    # Limpieza conservadora: no colapsar todos los saltos de línea.
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{4,}", "\n\n\n", value)

    return value.strip()


def _repetition_report(text: str) -> dict:
    lines = [
        re.sub(r"\s+", " ", line.strip().lower())
        for line in (text or "").splitlines()
        if line.strip()
    ]

    counts = Counter(lines)
    repeated = [
        {"line": line, "count": count}
        for line, count in counts.items()
        if count >= 3 and len(line) >= 20
    ]

    return {
        "excessive_repetition": bool(repeated),
        "repeated_lines": repeated[:10],
    }


def _first_model_device(model):
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def _transcribe_sync(image: Image.Image) -> tuple[str, str, dict]:
    import torch

    model, processor = get_runtime()

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": OCR_SYSTEM_PROMPT},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {"type": "text", "text": OCR_USER_PROMPT},
            ],
        },
    ]

    # Qwen3-VL puede procesar directamente el contenido visual desde
    # apply_chat_template en Transformers >= 4.57.
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    target = _first_model_device(model)
    inputs = inputs.to(target)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    text = _clean_output(text)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metadata = {
        "profile": "production_faithful_single_pass",
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": None,
        "no_repeat_ngram_size": None,
        "repetition_analysis": _repetition_report(text),
    }

    return text, device, metadata


@app.get("/health")
def health() -> dict:
    runtime = _runtime_info()
    return {
        "status": "ok",
        "service": "qwen-worker",
        "version": "5.0.0",
        "model": MODEL_NAME,
        "device": "cuda" if runtime.get("cuda_available") else "cpu",
        "model_loaded": _MODEL is not None,
        "profile": "production_faithful_single_pass",
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        **runtime,
    }


@app.post("/warmup")
async def warmup() -> dict:
    async with _MODEL_LOCK:
        started = time.perf_counter()
        try:
            get_runtime()
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"No se pudo cargar Qwen: {exc}",
            ) from exc

        return {
            "status": "ok",
            "model": MODEL_NAME,
            "processing_seconds": round(time.perf_counter() - started, 4),
        }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    started = time.perf_counter()
    data = await file.read()

    if not data:
        raise HTTPException(status_code=400, detail="La imagen está vacía.")

    try:
        image = ImageOps.exif_transpose(
            Image.open(io.BytesIO(data))
        ).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Imagen inválida: {exc}",
        ) from exc

    # Una sola inferencia GPU a la vez evita OOM y resultados inestables
    # en GPUs de consumo.
    async with _MODEL_LOCK:
        try:
            text, device, metadata = await asyncio.to_thread(
                _transcribe_sync,
                image,
            )
        except Exception as exc:
            detail = str(exc)
            if "out of memory" in detail.lower():
                detail = (
                    "CUDA sin memoria. Reduce QWEN_MAX_PIXELS o libera VRAM. "
                    f"Detalle original: {detail}"
                )
            raise HTTPException(
                status_code=500,
                detail=f"Qwen no pudo transcribir la imagen: {detail}",
            ) from exc

    repetition = metadata["repetition_analysis"]
    warnings = []
    if repetition.get("excessive_repetition"):
        warnings.append("EXCESSIVE_REPETITION")

    return {
        "status": "ok",
        "text": text,
        "line_count": len([line for line in text.splitlines() if line.strip()]),
        "average_confidence": None,
        "processing_seconds": round(time.perf_counter() - started, 4),
        "device": device,
        "model": MODEL_NAME,
        "reliable": not repetition.get("excessive_repetition", False),
        "warnings": warnings,
        "metadata": metadata,
    }
