from __future__ import annotations

import asyncio
import io
import math
import os
import re
import time
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, ImageFilter, ImageOps

app = FastAPI(
    title="Qwen3-VL OCR Worker",
    version="6.0.0",
    description=(
        "OCR visual fiel orientado a manuscrito académico y fórmulas. "
        "Presupuesto visual real + bandas de alta resolución + confianza por token."
    ),
)

_MODEL = None
_PROCESSOR = None
_MODEL_LOCK = asyncio.Lock()

MODEL_NAME = os.getenv(
    "QWEN_MODEL_NAME",
    "Qwen/Qwen3-VL-4B-Instruct",
)

CUDA_DEVICE = int(os.getenv("QWEN_CUDA_DEVICE", "0"))
MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "4096"))

# bfloat16 en Ampere+ evita los overflow/NaN que float16 produce en algunos
# bloques de visión. Se puede forzar con QWEN_DTYPE=float16.
DTYPE_NAME = os.getenv("QWEN_DTYPE", "auto").strip().lower()

# Cuantización opcional. Necesaria para cargar modelos 8B en 12 GB de VRAM.
LOAD_IN_4BIT = os.getenv("QWEN_LOAD_IN_4BIT", "false").strip().lower() == "true"

# ---------------------------------------------------------------------------
# Presupuesto visual.
#
# ATENCIÓN: pasar min_pixels/max_pixels dentro del mensaje NO tiene efecto en
# Transformers 5.x; el procesador los ignora. El control real está en
# processor.image_processor.size y en el redimensionado propio de la imagen.
# ---------------------------------------------------------------------------
FULL_MIN_PIXELS = int(os.getenv("QWEN_FULL_MIN_PIXELS", str(1_000_000)))
# 2 MP medido: por encima de ~2.4 MP los pesos + activaciones desbordan los
# 12 GB y el driver de Windows derrama a RAM del sistema (4x más lento).
FULL_MAX_PIXELS = int(os.getenv("QWEN_FULL_MAX_PIXELS", str(2_000_000)))
TILE_MAX_PIXELS = int(os.getenv("QWEN_TILE_MAX_PIXELS", str(1_900_000)))

# Bandas horizontales de alta resolución (nunca verticales: cortarían renglones).
TILING_MODE = os.getenv("QWEN_TILING", "auto").strip().lower()
MAX_TILES = int(os.getenv("QWEN_MAX_TILES", "4"))
TILE_OVERLAP = float(os.getenv("QWEN_TILE_OVERLAP", "0.14"))

# Disparador real: píxeles por renglón, no megapíxeles de la página.
# Qwen funde 32x32 px en un token visual, así que un renglón con paso de 40 px
# ocupa ~1.2 filas de tokens: no hay con qué distinguir una R de una K.
# 100 px de paso ≈ 3 filas de tokens por renglón.
TARGET_LINE_PITCH = float(os.getenv("QWEN_TARGET_LINE_PITCH", "100"))
# Reserva para cuando no se puede medir el texto (página vacía, foto de objeto).
TILE_TRIGGER_RATIO = float(os.getenv("QWEN_TILE_TRIGGER_RATIO", "1.35"))
TILE_MIN_LONG_SIDE = int(os.getenv("QWEN_TILE_MIN_LONG_SIDE", "1000"))

# Enderezado previo. Sin él, la escritura inclinada no tiene ninguna fila
# vacía y todo corte horizontal parte renglones.
SKEW_MAX_ANGLE = float(os.getenv("QWEN_SKEW_MAX_ANGLE", "8"))

# Recorte del cromo del teléfono en capturas de pantalla de apps de escaneo.
CROP_CONTENT = os.getenv("QWEN_CROP_CONTENT", "true").strip().lower() == "true"

# ---------------------------------------------------------------------------
# Modo de reconocimiento.
#
#   page   una sola pasada a la página completa
#   bands  página + franjas horizontales, se elige por confianza
#   lines  segmentación por renglón (lo que usan los HTR de producción)
#
# El modo `lines` es el que da más píxeles por renglón: un renglón aislado
# ocupa todo el presupuesto visual en vez de competir con el resto de la hoja.
# ---------------------------------------------------------------------------
# `lines` por defecto: medido más preciso Y más rápido que `bands`.
# En una carta impresa densa, `bands` (página completa + 4 franjas) supera los
# 10 minutos y agota el timeout de 300 s del gateway, devolviendo texto vacío.
# `lines` resuelve la misma página en menos de un minuto.
RECOGNITION_MODE = os.getenv("QWEN_MODE", "lines").strip().lower()

# Disposición: se le pregunta al propio VLM. Las proyecciones de tinta no
# distinguen una columna de dos (medido en las páginas de prueba reales).
LAYOUT_MODE = os.getenv("QWEN_LAYOUT", "auto").strip().lower()
LAYOUT_MIN_PIXELS = int(os.getenv("QWEN_LAYOUT_MIN_PIXELS", str(1_000_000)))
LAYOUT_MAX_PIXELS = int(os.getenv("QWEN_LAYOUT_MAX_PIXELS", str(1_800_000)))
MAX_COLUMNS = int(os.getenv("QWEN_MAX_COLUMNS", "4"))
LINE_MAX_PIXELS = int(os.getenv("QWEN_LINE_MAX_PIXELS", str(700_000)))
# Renglones por recorte. 2 conserva algo de contexto de la frase sin diluir
# la ampliación; 1 maximiza resolución pero el modelo pierde el hilo.
LINE_GROUP = int(os.getenv("QWEN_LINE_GROUP", "2"))
LINE_PAD_RATIO = float(os.getenv("QWEN_LINE_PAD_RATIO", "0.45"))
# Por debajo del timeout del gateway (300 s): se devuelve lo transcrito hasta
# ese momento, con aviso, en vez de no responder nada.
TIME_BUDGET_SECONDS = float(os.getenv("QWEN_TIME_BUDGET_SECONDS", "240"))

# Segunda lectura de los renglones con números, a otra escala, para detectar
# dígitos inferidos en lugar de leídos. Cuesta una pasada extra por renglón
# numérico; en trabajo que se va a calificar, ese coste está justificado.
RECHECK_NUMBERS = (
    os.getenv("QWEN_RECHECK_NUMBERS", "true").strip().lower() == "true"
)
RECHECK_SCALE = float(os.getenv("QWEN_RECHECK_SCALE", "0.62"))
# Solo se releen las líneas de las que el modelo ya duda. Releerlo todo
# duplica el tiempo y agota el presupuesto antes de terminar la página.
RECHECK_CONFIDENCE_BELOW = float(
    os.getenv("QWEN_RECHECK_CONFIDENCE_BELOW", "0.99")
)

# Aplanado de iluminación: quita sombras de la foto de cuaderno.
FLATTEN_BACKGROUND = (
    os.getenv("QWEN_FLATTEN_BACKGROUND", "true").strip().lower() == "true"
)
FLATTEN_MIN_SPREAD = float(os.getenv("QWEN_FLATTEN_MIN_SPREAD", "28"))

PREPROCESS_MODE = os.getenv("QWEN_PREPROCESS", "auto").strip().lower()
SHARPEN_ON_UPSCALE = (
    os.getenv("QWEN_SHARPEN_ON_UPSCALE", "true").strip().lower() == "true"
)

# Umbral por debajo del cual una palabra se marca para revisión humana.
LOW_CONFIDENCE_WORD = float(os.getenv("QWEN_LOW_CONFIDENCE_WORD", "0.60"))
LOW_CONFIDENCE_PAGE = float(os.getenv("QWEN_LOW_CONFIDENCE_PAGE", "0.75"))
MAX_REPORTED_WORDS = int(os.getenv("QWEN_MAX_REPORTED_WORDS", "60"))

# El patch de Qwen3-VL es 16 con merge 2: los lados múltiplos de 32 evitan
# un reescalado extra dentro del procesador.
PIXEL_ALIGN = 32

# IMPORTANTE:
# No usamos repetition_penalty ni no_repeat_ngram_size.
# En OCR real hay repeticiones legítimas y esas penalizaciones alteran el texto.
OCR_SYSTEM_PROMPT = """Eres un motor de transcripción visual (OCR/HTR), no un asistente de redacción y no un profesor.

Tu única función es leer lo que está escrito y devolverlo tal cual.
La evidencia visual tiene prioridad absoluta sobre lo que "sonaría mejor" o sobre lo que sería matemáticamente correcto.
No corriges, no resuelves, no completas, no calificas.
"""

_FIDELITY_RULES = """IDIOMA:
- El documento está en ESPAÑOL. Transcribe en español.
- NUNCA escribas en portugués, italiano ni catalán. Si una palabra te parece
  portuguesa, es una palabra española que estás leyendo mal.
  · Escribe "abrazo", nunca "abraço".
  · Escribe "corazón", nunca "coração".
  · No uses ç, ã, õ ni terminaciones en -ão.
- Respeta las tildes españolas tal como estén (o falten) en la hoja.

REGLAS DE FIDELIDAD:
- Devuelve únicamente la transcripción, sin introducciones ni comentarios.
- No agregues etiquetas como INICIO, FIN, TRANSCRIPCIÓN u OCR.
- No resumas, no traduzcas, no normalices ortografía, no mejores gramática ni estilo.
- No cambies tú/usted, tiempos verbales ni palabras por otras que suenen mejor.
- PROHIBIDO corregir la ortografía. Copia letra por letra lo que ves, aunque la palabra esté mal escrita y aunque conozcas la forma correcta.
  · Si ves "ejercisio", escribe "ejercisio". NO escribas "ejercicio".
  · Si ves "haber" donde correspondía "a ver", escribe "haber".
  · Si ves "aser", escribe "aser". NO escribas "hacer".
  · Si falta una tilde, no la agregues. Si sobra una tilde, no la quites.
- Las faltas de ortografía del estudiante son parte de lo que el docente va a evaluar: borrarlas destruye la evidencia.
- Conserva nombres, números, fechas, teléfonos, correos, unidades y códigos exactamente como se ven.
- Usa el contexto lingüístico solo para distinguir trazos ambiguos, nunca para inventar texto.
- Si una palabra tiene una lectura probable, escríbela. Si es imposible de leer, escribe [ilegible].

REGLAS DE MATEMÁTICA (crítico: es trabajo de estudiantes que será evaluado):
- Transcribe la matemática tal como está escrita, aunque el procedimiento o el resultado estén MAL.
- Nunca resuelvas, simplifiques, completes ni corrijas una operación.
- Si el estudiante escribió 2+2=5, transcribe 2+2=5.
- Notación LaTeX SOLO cuando la expresión es bidimensional y no se puede escribir en una línea tal cual:
  · fracción con barra horizontal real -> $\\frac{a}{b}$
  · raíz con símbolo radical -> $\\sqrt{x}$
  · exponente o subíndice elevado/bajado gráficamente -> $x^{2}$, $a_{1}$
  · integrales, sumatorias, límites, matrices y sistemas de ecuaciones
- Si la expresión ya está escrita de forma lineal en la hoja (por ejemplo "x^2 + 3x - 4 = 0"),
  cópiala tal cual, SIN convertirla a LaTeX y SIN signos de dólar.
- Respeta el orden y la disposición de los pasos: una línea del desarrollo por línea de texto.
- En operaciones verticales (sumas, divisiones, despejes) conserva un renglón por renglón.

MARCAS PERMITIDAS (las únicas que puedes añadir):
- [ilegible] cuando el trazo no permite ninguna lectura razonable.
- [tachado: texto] cuando el autor tachó algo que aún se puede leer.
- [dibujo] o [diagrama] para figuras, gráficas o esquemas que no son texto.

ESTRUCTURA:
- Respeta el orden visual: de arriba hacia abajo y de izquierda a derecha, por bloques.
- Conserva la numeración de ejercicios tal como aparece (1., a), I., etc.).
- Conserva los saltos de línea que mantienen el orden visual."""

OCR_USER_PROMPT = f"""Transcribe TODO el texto manuscrito e impreso visible en esta imagen.

OBJETIVO:
La lectura más fiel posible de lo que realmente escribió la persona.

{_FIDELITY_RULES}
"""

LAYOUT_PROMPT = """Analiza esta página de tarea escolar. No transcribas nada.

Indica dos cosas:
1. En cuántas columnas de texto está organizada, y dónde empieza y termina
   cada una horizontalmente.
2. Si el texto es IMPRESO (tipografía de ordenador o imprenta) o MANUSCRITO
   (escrito a mano). Si hay de los dos, responde "mixta".

Responde ÚNICAMENTE con JSON, sin explicación:
{"columnas": N, "cajas": [[x1,y1,x2,y2], ...], "escritura": "impresa"}

El campo "escritura" solo admite: "impresa", "manuscrita" o "mixta".
Si la página tiene una sola columna, devuelve una sola caja.
"""

OCR_LINE_PROMPT = f"""Esta imagen contiene UNO O DOS RENGLONES recortados de una página manuscrita.

Transcribe EXACTAMENTE lo que ves en estos renglones.

CRÍTICO:
- Transcribe TODOS los renglones que veas, incluido el último aunque sea muy
  corto (por ejemplo dos palabras sueltas al final). No te detengas antes.
- No continúes la frase. No completes lo que parece faltar.
- No añadas renglones que no estén en la imagen.
- Si el recorte muestra media palabra al principio o al final, transcríbela igual.
- Devuelve un renglón de texto por cada renglón visible, nada más.

{_FIDELITY_RULES}
"""

OCR_TILE_PROMPT = f"""Esta imagen es una FRANJA HORIZONTAL recortada de una página más grande.

Transcribe únicamente el texto que ves dentro de esta franja.
No completes lo que quedó cortado arriba o abajo: transcribe solo el fragmento visible.
No añadas encabezados, ni indiques que es un fragmento.

{_FIDELITY_RULES}
"""


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
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


def _resolve_dtype():
    import torch

    if DTYPE_NAME in {"float16", "fp16", "half"}:
        return torch.float16
    if DTYPE_NAME in {"bfloat16", "bf16"}:
        return torch.bfloat16

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_runtime():
    global _MODEL, _PROCESSOR

    if _MODEL is not None and _PROCESSOR is not None:
        return _MODEL, _PROCESSOR

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if torch.cuda.is_available():
        torch.cuda.set_device(CUDA_DEVICE)
        model_kwargs = {
            "dtype": _resolve_dtype(),
            "device_map": "auto",
            # sdpa es compatible con RTX 30xx en Windows sin flash-attn.
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        }
    else:
        model_kwargs = {
            "dtype": "auto",
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }

    if LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_resolve_dtype(),
        )
        model_kwargs.pop("dtype", None)

    _MODEL = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        **model_kwargs,
    )
    _MODEL.eval()

    _PROCESSOR = AutoProcessor.from_pretrained(MODEL_NAME)

    # Único punto donde el presupuesto visual se aplica de verdad.
    _PROCESSOR.image_processor.size = {
        "shortest_edge": FULL_MIN_PIXELS,
        "longest_edge": max(FULL_MAX_PIXELS, TILE_MAX_PIXELS),
    }

    return _MODEL, _PROCESSOR


def _first_model_device(model):
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


# ---------------------------------------------------------------------------
# Preparación de imagen
# ---------------------------------------------------------------------------
def _percentile_spread(image: Image.Image) -> int:
    """
    Distancia entre el nivel de tinta y el nivel de papel.

    Se usan percentiles extremos (0.2 % / 99.8 %) a propósito: en una hoja
    escrita la tinta ocupa muy pocos píxeles, así que un percentil 2 / 98
    mide papel contra papel y da un rango falsamente bajo en documentos
    perfectamente expuestos. Los extremos siguen siendo robustos frente a
    motas sueltas porque 0.2 % de una página de 3 MP son ~6000 píxeles.
    """
    histogram = image.convert("L").histogram()
    total = sum(histogram) or 1

    low_target = total * 0.002
    high_target = total * 0.998

    low = 0
    high = 255
    running = 0

    for value, count in enumerate(histogram):
        running += count
        if running >= low_target:
            low = value
            break

    running = 0
    for value, count in enumerate(histogram):
        running += count
        if running >= high_target:
            high = value
            break

    return max(high - low, 0)


def _flatten_background(image: Image.Image) -> tuple[Image.Image, dict]:
    """
    Aplana la iluminación dividiendo por el fondo.

    Una foto de cuaderno casi nunca está iluminada de forma uniforme: hay
    sombra de la mano, del lomo del cuaderno o de la lámpara. El autocontraste
    global no lo arregla, porque el problema es local: la misma tinta es
    oscura en una esquina y clarísima en otra.

    Estimando el fondo con un desenfoque fuerte y dividiendo por él, el papel
    queda blanco parejo y el trazo conserva su contraste en toda la hoja. Es
    el paso clásico de los escáneres de documentos. La corrección se calcula
    sobre la luminancia y se aplica a los tres canales, así que el bolígrafo
    azul y las marcas rojas del docente conservan su color.
    """
    if not FLATTEN_BACKGROUND:
        return image, {"applied": False, "reason": "disabled"}

    try:
        import numpy as np
    except Exception:
        return image, {"applied": False, "reason": "numpy_unavailable"}

    radius = max(8, int(max(image.width, image.height) / 28))
    luminance = image.convert("L")
    background = luminance.filter(ImageFilter.GaussianBlur(radius=radius))

    bg = np.asarray(background, dtype=np.float32)
    if bg.size == 0:
        return image, {"applied": False, "reason": "empty"}

    # Desviación de la iluminación: si el fondo ya es parejo, no se toca.
    spread = float(np.percentile(bg, 95) - np.percentile(bg, 5))
    if spread < FLATTEN_MIN_SPREAD:
        return image, {
            "applied": False,
            "reason": "illumination_already_even",
            "background_spread": round(spread, 1),
        }

    target = float(np.percentile(bg, 95))
    ratio = np.clip(target / np.maximum(bg, 1.0), 0.2, 5.0)[:, :, None]

    source = np.asarray(image, dtype=np.float32)
    corrected = np.clip(source * ratio, 0, 255).astype("uint8")

    return Image.fromarray(corrected), {
        "applied": True,
        "background_spread": round(spread, 1),
        "blur_radius": radius,
    }


def _auto_enhance(image: Image.Image) -> tuple[Image.Image, dict]:
    """
    Realce conservador: solo corrige exposición cuando la foto está
    visiblemente plana. No binariza ni filtra: eso destruye trazos de lápiz.
    """
    if PREPROCESS_MODE == "off":
        return image, {"applied": False, "reason": "disabled"}

    spread = _percentile_spread(image)

    if PREPROCESS_MODE != "always" and spread >= 150:
        return image, {
            "applied": False,
            "reason": "contrast_ok",
            "dynamic_range": spread,
        }

    enhanced = ImageOps.autocontrast(image, cutoff=(1, 1))
    return enhanced, {
        "applied": True,
        "reason": "low_contrast",
        "dynamic_range": spread,
    }


def _align(value: int) -> int:
    return max(PIXEL_ALIGN, int(round(value / PIXEL_ALIGN)) * PIXEL_ALIGN)


def _fit_pixels(
    image: Image.Image,
    min_pixels: int,
    max_pixels: int,
) -> tuple[Image.Image, dict]:
    """Lleva la imagen dentro del presupuesto de píxeles con LANCZOS."""
    pixels = image.width * image.height

    if pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
        action = "downscaled"
    elif pixels < min_pixels:
        scale = math.sqrt(min_pixels / pixels)
        action = "upscaled"
    else:
        return image, {
            "action": "unchanged",
            "width": image.width,
            "height": image.height,
            "pixels": pixels,
        }

    width = _align(image.width * scale)
    height = _align(image.height * scale)
    resized = image.resize((width, height), Image.LANCZOS)

    if action == "upscaled" and SHARPEN_ON_UPSCALE:
        # Reconstruye el filo de trazos finos de lápiz tras la interpolación.
        resized = resized.filter(
            ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=3)
        )

    return resized, {
        "action": action,
        "scale": round(scale, 4),
        "width": width,
        "height": height,
        "pixels": width * height,
        "sharpened": action == "upscaled" and SHARPEN_ON_UPSCALE,
    }


def _ink_mask(image: Image.Image, work_width: int = 1000):
    """Máscara booleana de tinta sobre una versión reducida de la página."""
    import numpy as np

    gray = image.convert("L")
    if gray.width > work_width:
        ratio = work_width / gray.width
        gray = gray.resize(
            (work_width, max(1, int(gray.height * ratio))),
            Image.BILINEAR,
        )

    array = np.asarray(gray, dtype=np.float32)
    if array.size == 0:
        return None, 1.0

    paper = float(np.percentile(array, 90))
    return array < (paper - 40), image.height / array.shape[0]


def _content_region(image: Image.Image) -> tuple[Image.Image, dict]:
    """
    Recorta el cromo que rodea al documento.

    Los alumnos entregan capturas de pantalla de apps de escaneo: barra de
    estado, botones de navegación y franjas negras pueden ocupar más de la
    mitad del archivo. Todo ese presupuesto visual se gasta en píxeles que no
    son la tarea, y además el modelo transcribe la hora y la batería.

    Se busca el bloque contiguo de filas y columnas con papel, no se recorta
    desde los bordes: así una miniatura suelta en la franja negra no arrastra
    el recorte hasta el fondo de la imagen.
    """
    if not CROP_CONTENT:
        return image, {"applied": False, "reason": "disabled"}

    try:
        import numpy as np
    except Exception:
        return image, {"applied": False, "reason": "numpy_unavailable"}

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.size == 0:
        return image, {"applied": False, "reason": "empty"}

    def _largest_span(profile, threshold: float) -> tuple[int, int] | None:
        flags = profile > threshold
        best = None
        start = None
        for index, value in enumerate(flags):
            if value and start is None:
                start = index
            elif not value and start is not None:
                if best is None or index - start > best[1] - best[0]:
                    best = (start, index)
                start = None
        if start is not None:
            if best is None or len(flags) - start > best[1] - best[0]:
                best = (start, len(flags))
        return best

    bright = gray > 120
    rows = _largest_span(bright.mean(axis=1), 0.15)
    if rows is None:
        return image, {"applied": False, "reason": "no_paper_found"}

    band = bright[rows[0]: rows[1]]
    cols = _largest_span(band.mean(axis=0), 0.15)
    if cols is None:
        cols = (0, image.width)

    left, right = cols
    top, bottom = rows

    area = (right - left) * (bottom - top)
    ratio = area / max(image.width * image.height, 1)

    # Si apenas se recorta, no vale la pena; si se recorta casi todo,
    # probablemente la detección se equivocó.
    if ratio > 0.92 or ratio < 0.05:
        return image, {
            "applied": False,
            "reason": "nothing_to_crop" if ratio > 0.92 else "suspicious_crop",
            "kept_ratio": round(ratio, 3),
        }

    return image.crop((left, top, right, bottom)), {
        "applied": True,
        "box": {"left": left, "top": top, "right": right, "bottom": bottom},
        "kept_ratio": round(ratio, 3),
        "discarded_percent": round((1 - ratio) * 100, 1),
    }


def _estimate_skew(image: Image.Image) -> float:
    """
    Ángulo de inclinación por perfil de proyección.

    Cuando los renglones están horizontales, la tinta se concentra en pocas
    filas y la varianza del perfil es máxima. Importa para el bandeado: con
    la escritura inclinada no existe ninguna fila realmente vacía, así que
    cualquier corte horizontal parte renglones por la derecha.
    """
    if SKEW_MAX_ANGLE <= 0:
        return 0.0

    try:
        import numpy as np
    except Exception:
        return 0.0

    mask, _ = _ink_mask(image, work_width=700)
    if mask is None or not mask.any():
        return 0.0

    source = Image.fromarray((mask * 255).astype("uint8"))

    best_angle = 0.0
    best_score = -1.0

    step = 0.5
    angle = -SKEW_MAX_ANGLE
    while angle <= SKEW_MAX_ANGLE + 1e-9:
        rotated = source.rotate(
            angle,
            resample=Image.BILINEAR,
            fillcolor=0,
        )
        profile = np.asarray(rotated, dtype=np.float32).sum(axis=1)
        # Varianza alta = tinta agrupada en renglones nítidos.
        score = float(np.var(profile))

        if score > best_score:
            best_score = score
            best_angle = angle

        angle += step

    return best_angle if abs(best_angle) >= step else 0.0


def _text_metrics(image: Image.Image) -> dict | None:
    """
    Estima cuántos píxeles ocupa cada renglón, por proyección horizontal
    de tinta. Es lo que decide si hace falta ampliar: una página de 2 MP con
    letra diminuta necesita más aumento que una de 8 MP con letra grande.
    """
    try:
        import numpy as np
    except Exception:
        return None

    gray = image.convert("L")

    work_width = 1000
    if gray.width > work_width:
        ratio = work_width / gray.width
        gray = gray.resize(
            (work_width, max(1, int(gray.height * ratio))),
            Image.BILINEAR,
        )

    array = np.asarray(gray, dtype=np.float32)
    if array.size == 0:
        return None

    paper = float(np.percentile(array, 90))
    ink = array < (paper - 40)
    row_ink = ink.mean(axis=1)

    peak = float(row_ink.max())
    if peak <= 0:
        return None

    is_text = row_ink > max(0.004, peak * 0.15)

    runs: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(is_text):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(is_text)))

    runs = [(a, b) for a, b in runs if b - a >= 2]
    if len(runs) < 3:
        return None

    # Fusionar trozos del MISMO renglón. En cursiva, las astas altas (l, t, d)
    # y las bajas (g, j, p) generan huecos de tinta que parten un renglón en
    # dos o tres tramos; sin fusionar, 15 renglones se detectan como 24.
    starts = sorted(runs[i + 1][0] - runs[i][0] for i in range(len(runs) - 1))
    rough_pitch = starts[len(starts) // 2] if starts else 0

    if rough_pitch > 0:
        merged_runs = [list(runs[0])]
        for top, bottom in runs[1:]:
            if top - merged_runs[-1][1] < rough_pitch * 0.35:
                merged_runs[-1][1] = bottom
            else:
                merged_runs.append([top, bottom])
        runs = [(a, b) for a, b in merged_runs]

    if len(runs) < 2:
        return None

    back = image.height / array.shape[0]

    # Filas SIN NADA de tinta en todo el ancho. Solo ahí se puede cortar sin
    # partir un renglón: promediar por fila engaña con escritura inclinada,
    # donde una fila "casi vacía" a la izquierda corta texto a la derecha.
    empty_rows = (~ink.any(axis=1)).nonzero()[0]
    empty_original = [int(row * back) for row in empty_rows.tolist()]
    gaps = sorted(
        runs[i + 1][0] - runs[i][0] for i in range(len(runs) - 1)
    )
    heights = sorted(b - a for a, b in runs)

    return {
        "lines": len(runs),
        "line_height": round(heights[len(heights) // 2] * back, 1),
        "line_pitch": round(gaps[len(gaps) // 2] * back, 1),
        # Coordenadas en la imagen original: sirven para cortar las franjas
        # justo en el hueco entre renglones y no partir ninguno.
        "runs": [(int(a * back), int(b * back)) for a, b in runs],
        "empty_rows": empty_original,
    }


def _plan_bands(image: Image.Image, text: dict | None) -> int:
    """
    Cuántas franjas horizontales conviene usar.

    El criterio es el aumento que le falta al renglón para llegar a
    TARGET_LINE_PITCH. Antes se usaban megapíxeles, y eso fallaba justo en el
    caso que importa: un escaneo de 1.1 MP con letra muy pequeña no disparaba
    nada, aunque fuera el que más aumento necesitaba.
    """
    if TILING_MODE == "off":
        return 0

    pixels = image.width * image.height

    if TILING_MODE == "always":
        return max(2, min(MAX_TILES, math.ceil(pixels / TILE_MAX_PIXELS)))

    if max(image.width, image.height) < TILE_MIN_LONG_SIDE:
        return 0

    if text and text.get("line_pitch"):
        pitch = text["line_pitch"]
        if pitch >= TARGET_LINE_PITCH:
            return 0

        # Área extra necesaria para que el renglón alcance el paso objetivo.
        zoom = TARGET_LINE_PITCH / pitch
        needed = pixels * zoom * zoom
        return max(2, min(MAX_TILES, math.ceil(needed / TILE_MAX_PIXELS)))

    # Sin medición fiable: se vuelve al criterio antiguo por megapíxeles.
    if pixels < FULL_MAX_PIXELS * TILE_TRIGGER_RATIO:
        return 0

    return max(2, min(MAX_TILES, math.ceil(pixels / TILE_MAX_PIXELS)))


def _horizontal_bands(
    image: Image.Image,
    bands: int,
    overlap: float,
    empty_rows: list | None = None,
) -> tuple[list[Image.Image], bool]:
    """
    Franjas horizontales. Siempre horizontales, incluso en páginas apaisadas:
    cortar en vertical partiría los renglones por la mitad.

    Si se conocen los renglones, se corta en el hueco entre dos de ellos y sin
    solape. Eso evita de raíz el problema real: con cortes proporcionales cada
    franja leía la zona compartida de forma distinta ("AQUI EN CARACAS" contra
    "AQUÍ EN CACACAS") y la costura no podía reconciliarlas, dejando líneas
    duplicadas y contradictorias.

    Devuelve (franjas, cortes_limpios).
    """
    height = image.height

    if empty_rows and len(empty_rows) >= bands - 1:
        # Solo filas totalmente limpias, y solo si están cerca del corte ideal.
        # Si el corte más cercano queda lejos, la página no tiene huecos
        # utilizables (típico de escritura inclinada) y no se fuerza.
        tolerance = height / (bands * 3)

        cuts: list[int] = []
        for index in range(1, bands):
            target = height * index / bands
            candidate = min(
                (row for row in empty_rows if row not in cuts),
                key=lambda row: abs(row - target),
                default=None,
            )
            if candidate is None or abs(candidate - target) > tolerance:
                cuts = []
                break
            cuts.append(candidate)

        cuts = sorted(set(cuts))

        if len(cuts) == bands - 1:
            edges = [0, *cuts, height]
            crops = [
                image.crop((0, edges[i], image.width, edges[i + 1]))
                for i in range(len(edges) - 1)
            ]
            if all(crop.height > 0 for crop in crops):
                return crops, True

    # Reserva: corte proporcional con solape y costura difusa.
    step = height / bands
    margin = step * overlap

    crops = []
    for index in range(bands):
        top = max(0, int(index * step - margin))
        bottom = min(height, int((index + 1) * step + margin))
        crops.append(image.crop((0, top, image.width, bottom)))

    return crops, False


# ---------------------------------------------------------------------------
# Confianza por token
# ---------------------------------------------------------------------------
def _detect_columns(image: Image.Image) -> tuple[list[tuple[int, int]], dict]:
    """
    Columnas de la página, según el propio modelo de visión.

    Se probaron tres detectores por proyección de tinta y ninguno distingue una
    página de una columna de una de dos: en las muestras reales, la profundidad
    del valle central de una carta a una columna (0.32) cae entre las de dos
    tareas a dos columnas (0.29 y 0.35). El modelo, en cambio, acierta el
    recuento porque ve la estructura en lugar de medir estadísticas.

    Se usa solo el rango horizontal: recortar también en vertical arriesga
    perder renglones si la caja se queda corta.
    """
    if LAYOUT_MODE == "off":
        return [(0, image.width)], {"applied": False, "reason": "disabled"}

    import json

    try:
        fitted, _ = _fit_pixels(image, LAYOUT_MIN_PIXELS, LAYOUT_MAX_PIXELS)
        result = _generate(fitted, LAYOUT_PROMPT)
    except Exception as exc:
        return [(0, image.width)], {
            "applied": False,
            "reason": "layout_call_failed",
            "detail": str(exc)[:200],
        }

    raw = (result.get("raw_text") or "").strip()
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return [(0, image.width)], {
            "applied": False,
            "reason": "unparseable",
            "raw": raw[:200],
        }

    try:
        payload = json.loads(match.group(0))
        boxes = payload.get("cajas") or []
    except Exception:
        return [(0, image.width)], {
            "applied": False,
            "reason": "invalid_json",
            "raw": raw[:200],
        }

    script = str(payload.get("escritura") or "").strip().lower()
    if script not in ("impresa", "manuscrita", "mixta"):
        script = "desconocida"

    spans: list[tuple[int, int]] = []
    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) < 3:
            continue
        try:
            x1, x2 = float(box[0]), float(box[2])
        except (TypeError, ValueError):
            continue

        # El modelo responde en un espacio normalizado 0-1000, no en píxeles.
        scale = image.width / 1000.0
        left = int(max(0, min(x1, x2)) * scale)
        right = int(min(1000, max(x1, x2)) * scale)

        if right - left >= image.width * 0.15:
            spans.append((left, right))

    spans.sort()
    if len(spans) < 2 or len(spans) > MAX_COLUMNS:
        return [(0, image.width)], {
            "applied": False,
            "reason": "single_column" if len(spans) < 2 else "too_many_columns",
            "reported": len(spans),
            "script": script,
            "seconds": result.get("seconds"),
        }

    # Ensanchar hasta los bordes y hasta tocarse: mejor solapar un poco que
    # dejar fuera una palabra que sobresale de la caja.
    merged: list[tuple[int, int]] = []
    for index, (left, right) in enumerate(spans):
        start = 0 if index == 0 else (spans[index - 1][1] + left) // 2
        end = image.width if index == len(spans) - 1 else right
        merged.append((max(0, start), min(image.width, end)))

    return merged, {
        "applied": True,
        "columns": len(merged),
        "spans": merged,
        "script": script,
        "seconds": result.get("seconds"),
    }


_NUMBER_RE = re.compile(r"\d+")


def _numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text or "")


def _numeric_disagreement(first: str, second: str) -> list[dict]:
    """
    Compara los números de dos lecturas del MISMO recorte.

    Un modelo de visión no sabe cuándo se equivoca: en la tarea de conjuntos
    dio 24 con 0.94 de confianza donde la hoja pone 17. Pero leído a otra
    escala dijo otra cosa. Ese desacuerdo consigo mismo es la única señal
    disponible de que el número no está siendo leído, sino inferido.

    No se elige ninguna lectura ni se corrige nada: se marca para que lo mire
    una persona. Elegir sería sustituir una invención por otra.
    """
    left = _numbers(first)
    right = _numbers(second)

    if left == right:
        return []

    # Alineación antes de comparar. Comparando por posición cruda, que una
    # lectura tuviera un número de más desplazaba todo lo siguiente y marcaba
    # la línea entera: doce "desacuerdos" donde solo había uno. Con doce
    # marcas el docente los ignora todos, incluido el bueno.
    matcher = SequenceMatcher(None, left, right, autojunk=False)

    disputed = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            # Un número que solo aparece en una lectura no es un desacuerdo
            # sobre su valor; se refleja aparte, en `count_mismatch`.
            continue
        for i, j in zip(range(i1, i2), range(j1, j2)):
            disputed.append(
                {
                    "position": i,
                    "lectura_1": left[i],
                    "lectura_2": right[j],
                }
            )

    return disputed[:12]


def _is_blank(image: Image.Image, min_ink: float = 0.010) -> bool:
    """
    ¿El recorte está prácticamente vacío?

    El final de una columna corta suele ser papel en blanco. Pedirle al modelo
    que transcriba papel vacío produce basura: en la tarea de conjuntos devolvió
    "c)", "?", "×" y "[ilegible]" sobre una zona donde no había nada escrito.

    El umbral de tinta es 60, no 45, porque en papel cuadriculado la retícula
    impresa cuenta como tinta y una zona vacía llegaba a 0.0098 — por encima
    del filtro. Medido sobre las tareas reales, con delta 60 hay 25x de
    separación: escrito 0.047-0.060 contra vacío 0.0019-0.0020.
    """
    try:
        import numpy as np
    except Exception:
        return False

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.size == 0:
        return True

    paper = float(np.percentile(gray, 90))
    ink = gray < (paper - 60)

    if not ink.any():
        return True

    # Criterio principal: qué parte del recorte abarca la tinta.
    # La fracción global sola descarta renglones cortos: "a mano?" en una
    # página ancha da 0.0113, pegado al umbral, y se perdía. Su caja de tinta,
    # en cambio, ocupa el 31 % del recorte, frente al 1.3 % de una zona vacía.
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    box_area = (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)
    box_ratio = box_area / max(gray.size, 1)

    return box_ratio < 0.05 or float(ink.mean()) < 0.003


def _line_crops(
    image: Image.Image,
    runs: list,
    pitch: float,
    group: int,
) -> list[tuple[Image.Image, tuple[int, int]]]:
    """
    Recorta la página en grupos de renglones.

    Cada recorte se lleva todo el ancho (nunca se parte una palabra) y un
    margen vertical generoso, para no cortar tildes arriba ni rabos abajo.
    """
    if not runs:
        return []

    padding = max(4, int(pitch * LINE_PAD_RATIO))
    group = max(1, group)

    crops = []
    for index in range(0, len(runs), group):
        chunk = runs[index: index + group]
        top = max(0, chunk[0][0] - padding)
        bottom = min(image.height, chunk[-1][1] + padding)

        if bottom - top < 8:
            continue

        crops.append(
            (image.crop((0, top, image.width, bottom)), (top, bottom))
        )

    return crops


def _make_loop_stopper():
    """
    Corta la generación cuando el modelo entra en bucle.

    Con una imagen ilegible, el modelo repite la misma expresión hasta agotar
    el presupuesto de tokens: en la prueba con la captura de ecuaciones gastó
    447 segundos escribiendo 230 veces la misma integral. Detectarlo pronto
    ahorra el tiempo y deja una señal limpia de que la página no se pudo leer.
    """
    from transformers import StoppingCriteria

    class _RepeatLoopStopper(StoppingCriteria):
        def __init__(self, window: int = 24, check_every: int = 16) -> None:
            self.window = window
            self.check_every = check_every
            self.start = None
            self.steps = 0
            self.tripped = False

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            if self.start is None:
                # Primera llamada: todo lo anterior es el prompt.
                self.start = input_ids.shape[1] - 1

            self.steps += 1
            if self.steps % self.check_every:
                return False

            generated = input_ids[0, self.start:].tolist()
            if len(generated) < self.window * 3:
                return False

            tail = generated[-self.window:]
            earlier = generated[:-self.window]

            # ¿Esta cola ya apareció antes tal cual?
            for offset in range(len(earlier) - self.window + 1):
                if earlier[offset: offset + self.window] == tail:
                    self.tripped = True
                    return True

            return False

    return _RepeatLoopStopper()


def _make_recorder():
    """
    Registra la probabilidad del token elegido en cada paso.

    Con decodificación greedy el token elegido es el argmax de los logits ya
    procesados, así que basta con guardar el máximo del softmax. Guardamos
    tensores 0-dim para no sincronizar la GPU en cada paso.
    """
    import torch
    from transformers import LogitsProcessor

    class _GreedyProbRecorder(LogitsProcessor):
        def __init__(self) -> None:
            self.probs: list = []

        def __call__(self, input_ids, scores):
            self.probs.append(
                torch.softmax(scores[0].float(), dim=-1).max().detach()
            )
            return scores

    return _GreedyProbRecorder()


def _word_confidences(
    token_ids: list[int],
    probs: list[float],
    tokenizer,
) -> list[dict]:
    """
    Agrupa tokens en palabras y asigna a cada palabra su token más dudoso.

    Se usa convert_ids_to_tokens para detectar el inicio de palabra ('Ġ'/'Ċ')
    porque decodificar token a token rompe los caracteres acentuados
    multibyte del español.
    """
    if not token_ids:
        return []

    try:
        pieces = tokenizer.convert_ids_to_tokens(token_ids)
    except Exception:
        return []

    groups: list[list[int]] = []
    for index, piece in enumerate(pieces):
        starts_word = (
            index == 0
            or not piece
            or piece[0] in ("Ġ", "Ċ")  # 'Ġ' espacio, 'Ċ' salto
        )
        if starts_word or not groups:
            groups.append([index])
        else:
            groups[-1].append(index)

    words = []
    for group in groups:
        confidence = min(probs[i] for i in group)
        text = tokenizer.decode(
            [token_ids[i] for i in group],
            skip_special_tokens=True,
        ).strip()

        if not text:
            continue

        words.append(
            {
                "word": text,
                "confidence": round(confidence, 4),
            }
        )

    return words


# ---------------------------------------------------------------------------
# Limpieza y diagnóstico
# ---------------------------------------------------------------------------
_WRAPPER_LINE = re.compile(
    r"^\s*(?:-{2,}\s*)?"
    r"(?:INICIO|FIN|TRANSCRIPCI[ÓO]N(?:\s+FIEL)?|OCR|TEXTO)"
    r"\s*(?:-{2,})?\s*:?\s*$",
    re.I,
)


def _clean_output(text: str) -> tuple[str, list[str]]:
    """
    Limpieza conservadora.

    Solo se elimina un envoltorio si ocupa una LÍNEA COMPLETA en el borde del
    texto y queda contenido suficiente. Un OCR fiel no puede borrar una línea
    que el estudiante realmente escribió, así que lo eliminado se reporta.
    """
    value = (text or "").strip()

    value = re.sub(r"^```(?:text|txt|markdown|latex)?\s*\n?", "", value, flags=re.I)
    value = re.sub(r"\n?\s*```$", "", value)

    value = value.replace("\r\n", "\n").replace("\r", "\n")

    lines = value.split("\n")
    removed: list[str] = []

    while len(lines) > 2 and _WRAPPER_LINE.match(lines[0]):
        removed.append(lines.pop(0).strip())
    while len(lines) > 2 and _WRAPPER_LINE.match(lines[-1]):
        removed.append(lines.pop().strip())

    value = "\n".join(lines)
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{4,}", "\n\n\n", value)

    return value.strip(), removed


def _foreign_language_report(text: str) -> dict:
    """
    Marca rastros de portugués o italiano, sin corregir nada.

    El modelo es multilingüe y a veces resbala: escribió "Um abraço" donde la
    hoja decía "Un abrazo". Reescribirlo automáticamente sería sustituir una
    invención por otra, así que solo se señala para que lo mire una persona.
    """
    value = text or ""

    hits: list[str] = []
    for token in re.findall(r"\b[\wÀ-ÿ]+\b", value):
        lowered = token.lower()
        if any(char in lowered for char in ("ç", "ã", "õ")):
            hits.append(token)
        elif lowered.endswith(("ção", "ões", "inho", "inha")):
            hits.append(token)

    unique = list(dict.fromkeys(hits))
    return {
        "foreign_tokens_found": bool(unique),
        "tokens": unique[:20],
    }


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

    # Proporción de líneas que son copia de otra: distingue una repetición
    # legítima (un ejercicio parecido al anterior) de un bucle degenerado.
    duplicate_ratio = 0.0
    if lines:
        duplicate_ratio = round(1 - len(set(lines)) / len(lines), 3)

    return {
        "excessive_repetition": bool(repeated) or duplicate_ratio > 0.5,
        "duplicate_line_ratio": duplicate_ratio,
        "repeated_lines": repeated[:10],
    }


# ---------------------------------------------------------------------------
# Costura de franjas (determinista, sin reescritura por el modelo)
# ---------------------------------------------------------------------------
def _norm_line(line: str) -> str:
    """Normalización agresiva, solo para puntuar parecido global."""
    value = unicodedata.normalize("NFKD", line or "")
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _signature(line: str) -> str:
    """
    Firma para comparar renglones entre franjas.

    Conserva símbolos: una línea de fórmula como "$x_2 = 1$" se reduce a nada
    con la normalización alfanumérica y dejaría de detectarse como solape.
    """
    value = unicodedata.normalize("NFKD", line or "")
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", "", value).lower()


def _same_line(left: str, right: str, threshold: float = 0.75) -> bool:
    """
    Renglones largos se comparan de forma difusa (tolera ruido de OCR).
    Renglones cortos exigen coincidencia exacta: con pocas letras el parecido
    difuso confunde "$x_1=-4$" con "$x_2=1$" y borraría un paso del alumno.
    """
    if not left or not right:
        return False

    if len(left) >= 12 and len(right) >= 12:
        return (
            SequenceMatcher(None, left, right, autojunk=False).ratio()
            >= threshold
        )

    return len(left) >= 3 and left == right


def _overlap_offset(
    previous: list[str],
    current: list[str],
    window: int = 12,
) -> int:
    """Índice de `current` desde el cual el contenido deja de ser repetido."""
    prev_idx = [i for i, line in enumerate(previous) if line.strip()][-window:]
    curr_idx = [
        i for i, line in enumerate(current[:window]) if line.strip()
    ]

    if not prev_idx or not curr_idx:
        return 0

    prev_sig = [_signature(previous[i]) for i in prev_idx]
    curr_sig = [_signature(current[i]) for i in curr_idx]

    for size in range(min(len(prev_sig), len(curr_sig)), 0, -1):
        left = prev_sig[-size:]
        right = curr_sig[:size]

        # Un ancla demasiado corta coincide por azar.
        if sum(len(item) for item in right) < 6:
            continue

        if all(_same_line(a, b) for a, b in zip(left, right)):
            return curr_idx[size - 1] + 1

    # Segundo intento, sin exigir alineación. Cuando dos recortes contiguos
    # comparten margen, el segundo puede volver a leer una línea que NO es la
    # última del primero — un título, por ejemplo. En la tarea de conjuntos
    # "Sesión 4" salía dos veces porque encabezaba ambos recortes.
    dropped = 0
    for position, index in enumerate(curr_idx[:3]):
        candidate = curr_sig[position]
        if len(candidate) < 4:
            break
        if any(_same_line(candidate, sig) for sig in prev_sig):
            dropped = index + 1
        else:
            break

    return dropped


def _resolve_seam(
    merged: list[str],
    incoming: list[str],
) -> tuple[list[str], list[str]]:
    """Elimina el renglón partido por el corte, conservando la versión completa."""
    left_idx = next(
        (i for i in range(len(merged) - 1, -1, -1) if merged[i].strip()),
        None,
    )
    right_idx = next(
        (i for i, line in enumerate(incoming) if line.strip()),
        None,
    )

    if left_idx is None or right_idx is None:
        return merged, incoming

    left = _signature(merged[left_idx])
    right = _signature(incoming[right_idx])

    if len(left) >= 6 and len(right) >= 6:
        if left in right:
            merged = merged[:left_idx] + merged[left_idx + 1:]
        elif right in left:
            incoming = incoming[:right_idx] + incoming[right_idx + 1:]

    return merged, incoming


def _stitch_bands(
    texts: list[str],
    clean_cuts: bool = False,
) -> tuple[str, dict]:
    """
    Une las franjas.

    Con cortes limpios no hay solape que reconciliar: basta concatenar, más
    una comprobación barata por si el modelo repitió el renglón del borde.
    Con cortes proporcionales hay que alinear de forma difusa.
    """
    merged: list[str] = []
    dropped = 0

    for text in texts:
        lines = (text or "").split("\n")

        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        if not lines:
            continue

        if not merged:
            merged = lines
            continue

        if clean_cuts:
            # Solo se descarta un primer renglón que repita literalmente el
            # último ya aceptado; nada más se toca.
            first = next(
                (i for i, line in enumerate(lines) if line.strip()),
                None,
            )
            last = next(
                (
                    i
                    for i in range(len(merged) - 1, -1, -1)
                    if merged[i].strip()
                ),
                None,
            )
            if first is not None and last is not None:
                if _same_line(
                    _signature(merged[last]),
                    _signature(lines[first]),
                    threshold=0.90,
                ):
                    lines = lines[:first] + lines[first + 1:]
                    dropped += 1

            merged.extend(lines)
            continue

        offset = _overlap_offset(merged, lines)
        dropped += offset
        incoming = lines[offset:]

        merged, incoming = _resolve_seam(merged, incoming)
        merged.extend(incoming)

    return "\n".join(merged), {
        "dropped_overlap_lines": dropped,
        "clean_cuts": clean_cuts,
    }


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------
def _generate(image: Image.Image, prompt: str) -> dict:
    import torch
    from transformers import LogitsProcessorList, StoppingCriteriaList

    model, processor = get_runtime()
    started = time.perf_counter()

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": OCR_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(_first_model_device(model))

    recorder = _make_recorder()
    stopper = _make_loop_stopper()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            logits_processor=LogitsProcessorList([recorder]),
            stopping_criteria=StoppingCriteriaList([stopper]),
        )

    prompt_length = inputs.input_ids.shape[1]
    new_ids = generated_ids[0][prompt_length:].tolist()

    probs: list[float] = []
    if recorder.probs:
        probs = torch.stack(recorder.probs).float().cpu().tolist()

    # generate registra un paso por token nuevo; recortamos por seguridad.
    size = min(len(new_ids), len(probs))
    new_ids = new_ids[:size]
    probs = probs[:size]

    text = processor.batch_decode(
        [new_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return {
        "raw_text": text,
        "token_ids": new_ids,
        "token_probs": probs,
        "image_size": [image.width, image.height],
        "truncated": len(new_ids) >= MAX_NEW_TOKENS,
        "looped": stopper.tripped,
        "seconds": round(time.perf_counter() - started, 2),
    }


def _generate_with_oom_ladder(
    image: Image.Image,
    prompt: str,
    max_pixels: int,
    min_pixels: int | None = None,
) -> tuple[dict, dict]:
    """Reintenta con menos presupuesto visual antes de rendirse por OOM."""
    import torch

    attempts = []
    budget = max_pixels

    for attempt in range(3):
        floor = FULL_MIN_PIXELS if min_pixels is None else min_pixels
        fitted, fit_info = _fit_pixels(
            image,
            min(floor, budget),
            budget,
        )

        try:
            result = _generate(fitted, prompt)
            fit_info["oom_retries"] = attempt
            fit_info["pixel_budget"] = budget
            return result, fit_info
        except torch.cuda.OutOfMemoryError:
            attempts.append(budget)
            torch.cuda.empty_cache()
            budget = int(budget * 0.6)
        except Exception as exc:
            if "out of memory" not in str(exc).lower():
                raise
            attempts.append(budget)
            torch.cuda.empty_cache()
            budget = int(budget * 0.6)

    raise RuntimeError(
        "CUDA sin memoria incluso con presupuesto reducido "
        f"(intentos: {attempts}). Baja QWEN_FULL_MAX_PIXELS o libera VRAM."
    )


def _transcribe_by_lines(
    prepared: Image.Image,
    text_metrics: dict | None,
    original: dict,
    crop_info: dict,
    enhance_info: dict,
    skew: float,
    started: float,
) -> dict:
    """
    Reconocimiento renglón a renglón, como los HTR de producción.

    Ventajas sobre la página completa:
      - cada renglón usa todo el presupuesto visual (10x más píxeles)
      - el orden de lectura es el de la segmentación, no lo decide el modelo
      - el modelo no puede "seguir escribiendo" más allá de lo que ve
      - cada línea queda ligada a su recorte, para revisión humana
    """
    import torch

    # Una tarea a dos columnas leída como renglones de ancho completo mezcla
    # el ejercicio 1 con el 9. La disposición se resuelve antes de segmentar.
    columns, layout_info = _detect_columns(prepared)

    crops: list[tuple[Image.Image, tuple[int, int]]] = []
    column_of: list[int] = []
    full_panel: list[bool] = []

    segmented_columns = 0

    # El troceado por renglón existe porque la cursiva necesita píxeles. El
    # texto impreso no: una sola pasada por columna lo lee igual de bien y en
    # una fracción del tiempo. Sin este atajo, una carta impresa tardaba 249 s
    # y agotaba el presupuesto; por columna entera baja a unos 25 s.
    printed = layout_info.get("script") == "impresa"

    for column_index, (left, right) in enumerate(columns):
        panel = prepared.crop((left, 0, right, prepared.height))
        panel_metrics = (
            _text_metrics(panel) if len(columns) > 1 else text_metrics
        )

        panel_crops = []
        if printed:
            panel_crops = [(panel, (0, prepared.height))]
        elif panel_metrics:
            panel_crops = _line_crops(
                panel,
                panel_metrics.get("runs") or [],
                panel_metrics.get("line_pitch") or 40.0,
                LINE_GROUP,
            )

        whole_panel = printed
        if not panel_crops:
            # Sin renglones separables, la columna entera va como un recorte.
            panel_crops = [(panel, (0, prepared.height))]
            whole_panel = True
        elif not printed:
            segmented_columns += 1

        for crop, box in panel_crops:
            crops.append((crop, box))
            column_of.append(column_index)
            full_panel.append(whole_panel)

    runs = (text_metrics or {}).get("runs") or []

    lines: list[dict] = []
    all_probs: list[float] = []
    all_words: list[dict] = []
    passes: list[dict] = []
    processor = get_runtime()[1]

    skipped_blank = 0
    budget_exhausted = False

    for index, (crop, (top, bottom)) in enumerate(crops):
        # El gateway corta a los 300 s. Sin este freno, una página densa se
        # queda sin responder y el docente ve "(No se reconoció texto)" en vez
        # de una transcripción parcial y un aviso. Mejor devolver lo leído.
        if time.perf_counter() - started > TIME_BUDGET_SECONDS:
            budget_exhausted = True
            break

        if _is_blank(crop):
            skipped_blank += 1
            continue

        # Una columna entera necesita el prompt y el presupuesto de página;
        # un recorte de dos renglones, los suyos.
        is_panel = full_panel[index]
        budget = FULL_MAX_PIXELS if is_panel else LINE_MAX_PIXELS

        result, fit = _generate_with_oom_ladder(
            crop,
            OCR_USER_PROMPT if is_panel else OCR_LINE_PROMPT,
            budget,
            min_pixels=budget,
        )
        text, _ = _clean_output(result["raw_text"])

        probs = result["token_probs"]
        confidence = round(sum(probs) / len(probs), 4) if probs else None

        all_probs.extend(probs)
        all_words.extend(
            _word_confidences(
                result["token_ids"],
                probs,
                processor.tokenizer,
            )
        )

        column_index = column_of[index]
        left, right = columns[column_index]

        lines.append(
            {
                "index": index,
                "crop_index": index,
                "column": column_index,
                "text": text,
                "confidence": confidence,
                "is_panel": is_panel,
                "budget": budget,
                "disputed_numbers": [],
                # Coordenadas en la página enderezada: permiten mostrar al
                # docente el recorte exacto junto a la transcripción.
                "box": {
                    "top": top,
                    "bottom": bottom,
                    "left": left,
                    "right": right,
                },
            }
        )
        passes.append(
            {
                "kind": f"line_{index + 1}",
                "image_size": result["image_size"],
                "crop_scale": fit.get("scale"),
                "tokens": len(result["token_ids"]),
                "seconds": result["seconds"],
                "truncated": result["truncated"],
                "looped": result["looped"],
            }
        )

    # --- Verificación de números, DESPUÉS de transcribirlo todo ------------
    # Antes iba dentro del bucle y se comía el presupuesto: las dos tareas de
    # matemática salían con la transcripción a medias. La transcripción es el
    # producto; la verificación es una mejora. Con el presupuesto que quede se
    # revisan primero las líneas de las que el modelo está menos seguro.
    rechecked = 0
    candidates = sorted(
        (
            item for item in lines
            if _numbers(item["text"])
            and (item["confidence"] or 1.0) < RECHECK_CONFIDENCE_BELOW
        ),
        key=lambda item: item["confidence"] or 1.0,
    )

    if RECHECK_NUMBERS:
        for item in candidates:
            if time.perf_counter() - started > TIME_BUDGET_SECONDS:
                break

            crop = crops[item["crop_index"]][0]
            scale = int(item["budget"] * RECHECK_SCALE)

            try:
                second, _ = _generate_with_oom_ladder(
                    crop,
                    OCR_USER_PROMPT if item["is_panel"] else OCR_LINE_PROMPT,
                    scale,
                    min_pixels=scale,
                )
                second_text, _ = _clean_output(second["raw_text"])
                item["disputed_numbers"] = _numeric_disagreement(
                    item["text"], second_text
                )
                rechecked += 1
            except Exception:
                continue

    for item in lines:
        item.pop("is_panel", None)
        item.pop("budget", None)

    # Los recortes llevan margen vertical para no cortar tildes ni rabos, y ese
    # margen hace que dos recortes contiguos compartan renglones. Se cosen con
    # el mismo alineador difuso de las franjas, columna por columna, para que
    # el solape no se convierta en texto duplicado.
    stitched: list[str] = []
    for column_index in range(len(columns)):
        column_texts = [
            item["text"]
            for item in lines
            if item["column"] == column_index and item["text"].strip()
        ]
        if not column_texts:
            continue
        column_text, _ = _stitch_bands(column_texts, clean_cuts=False)
        stitched.append(column_text)

    final_text = "\n".join(stitched)

    average_confidence = (
        round(sum(all_probs) / len(all_probs), 4) if all_probs else None
    )
    low_words = sorted(
        (w for w in all_words if w["confidence"] < LOW_CONFIDENCE_WORD),
        key=lambda item: item["confidence"],
    )

    return {
        "text": final_text,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "average_confidence": average_confidence,
        "truncated": any(item["truncated"] for item in passes),
        "metadata": {
            "profile": "faithful_handwriting_v6_lines",
            "model": MODEL_NAME,
            "mode": "lines",
            "original_size": original,
            "content_crop": crop_info,
            "preprocess": enhance_info,
            "deskew_degrees": round(skew, 2),
            "text_metrics": (
                {
                    k: v
                    for k, v in text_metrics.items()
                    if k not in ("runs", "empty_rows")
                }
                if text_metrics
                else None
            ),
            "layout": layout_info,
            "numeric_verification": {
                "enabled": RECHECK_NUMBERS,
                "recheck_scale": RECHECK_SCALE,
                "lines_rechecked": rechecked,
                "lines_pending_recheck": max(0, len(candidates) - rechecked),
                "lines_with_disputed_numbers": sum(
                    1 for item in lines if item["disputed_numbers"]
                ),
                "disputed": [
                    {
                        "line": item["index"],
                        "column": item["column"],
                        "text": item["text"][:160],
                        "numbers": item["disputed_numbers"],
                    }
                    for item in lines
                    if item["disputed_numbers"]
                ][:20],
            },
            "line_segmentation": {
                "detected_lines": len(runs),
                "columns": len(columns),
                "columns_with_line_split": segmented_columns,
                "crops": len(crops),
                "crops_transcribed": len(lines),
                "blank_crops_skipped": skipped_blank,
                "time_budget_exhausted": budget_exhausted,
                "lines_per_crop": LINE_GROUP,
                "line_max_pixels": LINE_MAX_PIXELS,
                "padding_ratio": LINE_PAD_RATIO,
            },
            "lines": lines,
            "passes": passes,
            "selection": {
                "source": "lines",
                "reason": "line_level_recognition",
                "full_page_vs_bands_agreement": None,
            },
            "confidence": {
                "average": average_confidence,
                "scored_tokens": len(all_probs),
                "low_confidence_threshold": LOW_CONFIDENCE_WORD,
                "low_confidence_word_count": len(low_words),
                "low_confidence_words": low_words[:MAX_REPORTED_WORDS],
            },
            "repetition_analysis": _repetition_report(final_text),
            "stripped_wrapper_lines": [],
            "inference_seconds": round(time.perf_counter() - started, 4),
        },
    }


def _transcribe_sync(
    image: Image.Image,
    tiling_override: str | None,
) -> dict:
    import torch

    started = time.perf_counter()

    original = {"width": image.width, "height": image.height}

    # Fuera el cromo del teléfono antes de gastar presupuesto visual en él.
    image, crop_info = _content_region(image)

    # Primero la iluminación (problema local), después la exposición (global).
    image, flatten_info = _flatten_background(image)
    prepared, enhance_info = _auto_enhance(image)
    enhance_info = {**enhance_info, "flatten_background": flatten_info}

    # Enderezar antes de medir: los renglones inclinados no producen filas
    # limpias y el bandeado no tendría dónde cortar.
    skew = _estimate_skew(prepared)
    if skew:
        prepared = prepared.rotate(
            skew,
            resample=Image.BICUBIC,
            expand=True,
            fillcolor=(255, 255, 255),
        )

    text_metrics = _text_metrics(prepared)

    # --- Modo renglón: cada línea recibe todo el presupuesto visual ---------
    # No se exige que la segmentación funcione: en papel rayado con escritura
    # densa la proyección da un solo bloque, y aun así separar por columnas
    # es una mejora grande por sí sola.
    if (tiling_override or RECOGNITION_MODE) == "lines":
        return _transcribe_by_lines(
            prepared,
            text_metrics,
            original,
            crop_info,
            enhance_info,
            skew,
            started,
        )

    mode = (tiling_override or TILING_MODE).strip().lower()
    if mode == "off":
        bands = 0
    elif mode == "always":
        bands = max(2, min(MAX_TILES, math.ceil(
            (prepared.width * prepared.height) / TILE_MAX_PIXELS
        )))
    else:
        bands = _plan_bands(prepared, text_metrics)

    # 1) Página completa: da el contexto global y el orden de lectura.
    page_result, page_fit = _generate_with_oom_ladder(
        prepared,
        OCR_USER_PROMPT,
        FULL_MAX_PIXELS,
    )
    page_text, page_removed = _clean_output(page_result["raw_text"])

    passes = [
        {
            "kind": "full_page",
            "image_size": page_result["image_size"],
            "tokens": len(page_result["token_ids"]),
            "seconds": page_result["seconds"],
            "truncated": page_result["truncated"],
            "looped": page_result["looped"],
        }
    ]

    band_text = None
    band_stitch = None
    band_probs: list[float] = []
    band_words: list[dict] = []

    # 2) Franjas horizontales: cada renglón recibe más píxeles reales.
    if bands >= 2:
        crops, clean_cuts = _horizontal_bands(
            prepared,
            bands,
            TILE_OVERLAP,
            (text_metrics or {}).get("empty_rows"),
        )
        band_texts = []

        for index, crop in enumerate(crops):
            # La franja se renderiza SIEMPRE al presupuesto completo, aunque
            # haya que ampliarla: ese aumento es justamente el objetivo.
            crop_result, crop_fit = _generate_with_oom_ladder(
                crop,
                OCR_TILE_PROMPT,
                TILE_MAX_PIXELS,
                min_pixels=TILE_MAX_PIXELS,
            )
            cleaned, _ = _clean_output(crop_result["raw_text"])
            band_texts.append(cleaned)
            band_probs.extend(crop_result["token_probs"])
            band_words.extend(
                _word_confidences(
                    crop_result["token_ids"],
                    crop_result["token_probs"],
                    get_runtime()[1].tokenizer,
                )
            )

            passes.append(
                {
                    "kind": f"band_{index + 1}",
                    "image_size": crop_result["image_size"],
                    "crop_scale": crop_fit.get("scale"),
                    "tokens": len(crop_result["token_ids"]),
                    "seconds": crop_result["seconds"],
                    "truncated": crop_result["truncated"],
                    "looped": crop_result["looped"],
                }
            )

        band_text, band_stitch = _stitch_bands(band_texts, clean_cuts)

    processor = get_runtime()[1]
    page_words = _word_confidences(
        page_result["token_ids"],
        page_result["token_probs"],
        processor.tokenizer,
    )

    # 3) Selección final. La banda gana por resolución, salvo que se degrade.
    page_repetition = _repetition_report(page_text)
    band_repetition = (
        _repetition_report(band_text) if band_text is not None else None
    )

    agreement = None
    if band_text is not None:
        agreement = round(
            SequenceMatcher(
                None,
                _norm_line(" ".join(page_text.split())),
                _norm_line(" ".join(band_text.split())),
                autojunk=False,
            ).ratio(),
            4,
        )

    def _mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    page_confidence = _mean(page_result["token_probs"])
    bands_confidence = _mean(band_probs)

    use_page = True
    if band_text is None:
        reason = "tiling_not_applied"
    elif band_repetition["excessive_repetition"] and not page_repetition[
        "excessive_repetition"
    ]:
        reason = "bands_degenerated_into_repetition"
    elif len(band_text) < len(page_text) * 0.6:
        reason = "bands_lost_too_much_text"
    elif len(band_text) > len(page_text) * 1.15:
        # La página completa se quedó corta: bandeando apareció más contenido.
        use_page = False
        reason = "bands_recovered_more_text"
    elif (
        agreement is not None
        and agreement >= 0.90
        and page_confidence is not None
        and bands_confidence is not None
    ):
        # Dicen casi lo mismo: gana la lectura de la que el modelo está más
        # seguro. En página impresa suele ganar la página completa; en
        # manuscrito pequeño suelen ganar las franjas.
        use_page = page_confidence >= bands_confidence
        reason = (
            "near_identical_prefer_higher_confidence_"
            + ("full_page" if use_page else "bands")
        )
    else:
        use_page = False
        reason = "bands_have_higher_effective_resolution"

    if use_page:
        final_text = page_text
        source = "full_page"
        probs = page_result["token_probs"]
        words = page_words
        repetition = page_repetition
    else:
        final_text = band_text
        source = "stitched_bands"
        probs = band_probs
        words = band_words
        repetition = band_repetition

    average_confidence = (
        round(sum(probs) / len(probs), 4) if probs else None
    )
    low_words = sorted(
        (word for word in words if word["confidence"] < LOW_CONFIDENCE_WORD),
        key=lambda item: item["confidence"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    metadata = {
        "profile": "faithful_handwriting_v6",
        "model": MODEL_NAME,
        "dtype": str(_resolve_dtype()).replace("torch.", ""),
        "load_in_4bit": LOAD_IN_4BIT,
        "original_size": original,
        "content_crop": crop_info,
        "preprocess": enhance_info,
        # `runs` son cientos de coordenadas: útiles internamente, ruido en la API.
        "text_metrics": (
            {
                k: v
                for k, v in text_metrics.items()
                if k not in ("runs", "empty_rows")
            }
            if text_metrics
            else None
        ),
        "deskew_degrees": round(skew, 2),
        "target_line_pitch": TARGET_LINE_PITCH,
        "full_page_fit": page_fit,
        "pixel_budget": {
            "full_min": FULL_MIN_PIXELS,
            "full_max": FULL_MAX_PIXELS,
            "tile_max": TILE_MAX_PIXELS,
        },
        "tiling": {
            "mode": mode,
            "bands": bands,
            "overlap": TILE_OVERLAP,
            "orientation": "horizontal",
            "stitch": band_stitch,
        },
        "passes": passes,
        "selection": {
            "source": source,
            "reason": reason,
            "full_page_vs_bands_agreement": agreement,
            "full_page_confidence": (
                round(page_confidence, 4)
                if page_confidence is not None
                else None
            ),
            "bands_confidence": (
                round(bands_confidence, 4)
                if bands_confidence is not None
                else None
            ),
        },
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": MAX_NEW_TOKENS,
            "repetition_penalty": None,
            "no_repeat_ngram_size": None,
        },
        "confidence": {
            "average": average_confidence,
            "scored_tokens": len(probs),
            "low_confidence_threshold": LOW_CONFIDENCE_WORD,
            "low_confidence_word_count": len(low_words),
            "low_confidence_words": low_words[:MAX_REPORTED_WORDS],
        },
        "repetition_analysis": repetition,
        "stripped_wrapper_lines": page_removed,
        "alternatives": {
            "full_page_text": page_text if source != "full_page" else None,
            "stitched_bands_text": (
                band_text if source != "stitched_bands" else None
            ),
        },
        "inference_seconds": round(time.perf_counter() - started, 4),
    }

    return {
        "text": final_text,
        "device": device,
        "average_confidence": average_confidence,
        "metadata": metadata,
        "truncated": any(item["truncated"] for item in passes),
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    runtime = _runtime_info()
    return {
        "status": "ok",
        "service": "qwen-worker",
        "version": "6.0.0",
        "model": MODEL_NAME,
        "device": "cuda" if runtime.get("cuda_available") else "cpu",
        "model_loaded": _MODEL is not None,
        "profile": "faithful_handwriting_v6",
        "load_in_4bit": LOAD_IN_4BIT,
        "full_min_pixels": FULL_MIN_PIXELS,
        "full_max_pixels": FULL_MAX_PIXELS,
        "tile_max_pixels": TILE_MAX_PIXELS,
        "tiling": TILING_MODE,
        "max_tiles": MAX_TILES,
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
async def transcribe(
    file: UploadFile = File(...),
    tiling: str | None = Query(
        default=None,
        description=(
            "Sobrescribe el modo para esta petición: "
            "lines | auto | always | off."
        ),
    ),
) -> dict:
    started = time.perf_counter()
    data = await file.read()

    if not data:
        raise HTTPException(status_code=400, detail="La imagen está vacía.")

    if tiling is not None and tiling.lower() not in {
        "lines",
        "auto",
        "always",
        "off",
    }:
        raise HTTPException(
            status_code=400,
            detail="tiling debe ser lines, auto, always u off.",
        )

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
            result = await asyncio.to_thread(
                _transcribe_sync,
                image,
                tiling,
            )
        except Exception as exc:
            detail = str(exc)
            if "out of memory" in detail.lower():
                detail = (
                    "CUDA sin memoria. Reduce QWEN_FULL_MAX_PIXELS o libera "
                    f"VRAM. Detalle original: {detail}"
                )
            raise HTTPException(
                status_code=500,
                detail=f"Qwen no pudo transcribir la imagen: {detail}",
            ) from exc

    text = result["text"]
    metadata = result["metadata"]
    confidence = metadata["confidence"]
    repetition = metadata["repetition_analysis"]
    looped = any(item.get("looped") for item in metadata["passes"])

    degenerated = (
        repetition.get("excessive_repetition", False)
        or looped
        or result["truncated"]
    )

    # CRÍTICO: la probabilidad por token mide fluidez, no acierto. En un bucle
    # el modelo se copia a sí mismo y la confianza sube a ~0.998 justo cuando
    # la salida no vale nada. Si hubo degeneración, no se publica el número:
    # un consumidor que solo mire `average_confidence` daría por buena la
    # peor transcripción posible.
    if degenerated:
        confidence["average"] = None
        confidence["invalidated_by"] = (
            "loop_detected" if looped
            else "excessive_repetition" if repetition.get("excessive_repetition")
            else "output_truncated"
        )
        result["average_confidence"] = None

    foreign = _foreign_language_report(text)
    metadata["foreign_language_analysis"] = foreign

    warnings = []
    if metadata.get("line_segmentation", {}).get("time_budget_exhausted"):
        warnings.append("PARTIAL_TIME_BUDGET")
    verification = metadata.get("numeric_verification", {})
    if verification.get("lines_with_disputed_numbers"):
        warnings.append("NUMERIC_DISAGREEMENT")

    # CRÍTICO: "ningún número disputado" y "no dio tiempo a comprobarlos" se
    # ven idénticos desde fuera. Para quien califica, esa diferencia lo es
    # todo: sin este aviso daría por verificada una página que nadie revisó.
    if verification.get("lines_pending_recheck"):
        warnings.append("VERIFICATION_INCOMPLETE")
    if foreign.get("foreign_tokens_found"):
        warnings.append("FOREIGN_LANGUAGE_TOKENS")
    if repetition.get("excessive_repetition"):
        warnings.append("EXCESSIVE_REPETITION")
    if looped:
        warnings.append("GENERATION_LOOP")
    if result["truncated"]:
        warnings.append("OUTPUT_TRUNCATED")
    if confidence["low_confidence_word_count"] > 0:
        warnings.append("LOW_CONFIDENCE_WORDS")
    if (
        confidence["average"] is not None
        and confidence["average"] < LOW_CONFIDENCE_PAGE
    ):
        warnings.append("LOW_CONFIDENCE_PAGE")

    agreement = metadata["selection"]["full_page_vs_bands_agreement"]
    if agreement is not None and agreement < 0.55:
        warnings.append("PAGE_VS_BANDS_DISAGREEMENT")

    reliable = not degenerated and (
        confidence["average"] is not None
        and confidence["average"] >= LOW_CONFIDENCE_PAGE
    )

    return {
        "status": "ok",
        "text": text,
        "line_count": len([line for line in text.splitlines() if line.strip()]),
        "average_confidence": result["average_confidence"],
        "processing_seconds": round(time.perf_counter() - started, 4),
        "device": result["device"],
        "model": MODEL_NAME,
        "reliable": reliable,
        "warnings": warnings,
        "metadata": metadata,
    }
