from __future__ import annotations

import io
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.benchmark_metrics import evaluate_text
from app.document_images import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    extract_document_images,
    extract_pdf_native_text,
    render_pdf_pages,
)
from app.provider_clients import (
    BENCHMARK_PROVIDERS,
    provider_statuses,
    transcribe_with_provider,
)
from app.settings import get_settings

settings = get_settings()

app = FastAPI(
    title="Vision OCR Service",
    version="3.0.0",
    description=(
        "Gateway OCR de producción. AUTO combina PaddleOCR + Qwen3-VL "
        "y conserva benchmarking independiente por modelo."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "vision-ocr-gateway",
        "version": "3.0.0",
        "default_provider": settings.ocr_provider,
        "device": settings.ocr_device,
        "production_mode": "single_pass_auto_hybrid",
    }


@app.get("/api/v1/vision/providers")
async def providers() -> dict:
    return {
        "default": settings.ocr_provider,
        "providers": await provider_statuses(),
    }


def _validate_image(
    filename: str,
    data: bytes,
) -> tuple[str, int, int]:
    extension = Path(filename).suffix.lower()

    if extension not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Usa PNG, JPG, JPEG, WEBP o BMP.",
        )

    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.width, image.height
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="El archivo no es una imagen válida.",
        ) from exc

    return extension, width, height


def _format_image_result(result: dict) -> str:
    location = []

    if result.get("page"):
        location.append(f"Página {result['page']}")

    if result.get("sheet"):
        location.append(f"Hoja {result['sheet']}")

    if result.get("cell"):
        location.append(f"Celda {result['cell']}")

    if result.get("media_name"):
        location.append(str(result["media_name"]))

    title = f"Imagen {result['image_number']}"
    if location:
        title += f" ({' · '.join(location)})"

    body = result.get("text") or "(No se reconoció texto)"

    if result.get("error"):
        body += f"\n\n[Error: {result['error']}]"

    return f"{title}:\n{body}"


async def _single_image_response(
    provider: str,
    filename: str,
    data: bytes,
) -> dict:
    started = time.perf_counter()
    extension, width, height = _validate_image(
        filename,
        data,
    )

    try:
        ocr = await transcribe_with_provider(
            provider,
            data,
            extension,
            filename,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"No se pudo ejecutar {provider}: {exc}",
        ) from exc

    image_result = {
        "image_number": 1,
        "source": "uploaded_image",
        "page": None,
        "sheet": None,
        "cell": None,
        "media_name": filename,
        "width": width,
        "height": height,
        **ocr,
    }

    return {
        "status": "ok",
        "provider": provider,
        "input_type": "image",
        "filename": filename,
        "images_found": 1,
        "images_processed": 1,
        "processing_seconds": round(
            time.perf_counter() - started,
            4,
        ),
        "images": [image_result],
        "full_text": _format_image_result(image_result),
        "warnings": image_result.get("warnings", []),
    }


@app.post("/api/v1/vision/image/transcribe")
async def transcribe_image(
    file: UploadFile = File(...),
    provider: str = Query(default="auto"),
) -> dict:
    data = await file.read()

    return await _single_image_response(
        provider.lower().strip(),
        file.filename or "image.png",
        data,
    )


@app.post("/api/v1/vision/image/benchmark")
async def benchmark_image(
    file: UploadFile = File(...),
    ground_truth: str = Form(default=""),
) -> dict:
    """
    Benchmark justo:
    AUTO no participa. Cada modelo real ejecuta una sola inferencia.
    """
    started = time.perf_counter()

    filename = file.filename or "image.png"
    data = await file.read()

    extension, width, height = _validate_image(
        filename,
        data,
    )

    statuses = await provider_statuses()
    availability = {
        item["name"]: bool(item.get("available"))
        for item in statuses
    }

    results = []

    for provider in BENCHMARK_PROVIDERS:
        if not availability.get(provider):
            detail = next(
                (
                    item.get("detail")
                    for item in statuses
                    if item["name"] == provider
                ),
                "Worker no disponible.",
            )

            results.append(
                {
                    "provider": provider,
                    "status": "unavailable",
                    "text": "",
                    "processing_seconds": None,
                    "cer": None,
                    "wer": None,
                    "detail": detail,
                }
            )
            continue

        one_started = time.perf_counter()

        try:
            ocr = await transcribe_with_provider(
                provider,
                data,
                extension,
                filename,
            )

            quality = evaluate_text(
                ground_truth,
                ocr.get("text", ""),
            )

            results.append(
                {
                    "provider": provider,
                    "status": "ok",
                    **ocr,
                    **quality,
                    "wall_seconds": round(
                        time.perf_counter() - one_started,
                        4,
                    ),
                }
            )

        except Exception as exc:
            results.append(
                {
                    "provider": provider,
                    "status": "error",
                    "text": "",
                    "processing_seconds": None,
                    "cer": None,
                    "wer": None,
                    "detail": str(exc),
                }
            )

    successful = [
        item
        for item in results
        if item.get("status") == "ok"
    ]

    best_cer = None
    with_cer = [
        item
        for item in successful
        if item.get("cer") is not None
    ]

    if with_cer:
        best = min(
            with_cer,
            key=lambda item: item["cer"],
        )
        best_cer = {
            "provider": best["provider"],
            "cer": best["cer"],
        }

    fastest = None
    with_time = [
        item
        for item in successful
        if isinstance(item.get("wall_seconds"), (int, float))
    ]

    if with_time:
        best = min(
            with_time,
            key=lambda item: item["wall_seconds"],
        )
        fastest = {
            "provider": best["provider"],
            "wall_seconds": best["wall_seconds"],
        }

    return {
        "status": "ok",
        "mode": "benchmark",
        "filename": filename,
        "width": width,
        "height": height,
        "ground_truth_provided": bool(ground_truth.strip()),
        "providers_requested": BENCHMARK_PROVIDERS,
        "providers_successful": len(successful),
        "processing_seconds": round(
            time.perf_counter() - started,
            4,
        ),
        "best_cer": best_cer,
        "fastest": fastest,
        "results": results,
    }


async def _transcribe_extracted_images(
    provider: str,
    extracted: list[dict],
) -> tuple[list[dict], str]:
    results = []
    consolidated = []

    # Secuencial por imagen:
    # el worker Qwen serializa GPU internamente para evitar OOM.
    for original in extracted:
        item = dict(original)
        raw = item.pop("bytes")

        image_filename = (
            item.get("media_name")
            or f"image_{item['image_number']}.png"
        )

        extension = (
            item.get("extension")
            or Path(image_filename).suffix
            or ".png"
        )

        try:
            ocr = await transcribe_with_provider(
                provider,
                raw,
                extension,
                image_filename,
            )

            result = {
                **item,
                **ocr,
            }
        except Exception as exc:
            result = {
                **item,
                "provider": provider,
                "text": "",
                "line_count": 0,
                "average_confidence": None,
                "processing_seconds": None,
                "warnings": [],
                "error": str(exc),
            }

        results.append(result)
        consolidated.append(
            _format_image_result(result)
        )

    return results, "\n\n".join(consolidated)


@app.post("/api/v1/vision/document/transcribe-images")
async def transcribe_document_images(
    file: UploadFile = File(...),
    provider: str = Query(default="auto"),
) -> dict:
    started = time.perf_counter()

    filename = file.filename or "document"
    extension = Path(filename).suffix.lower()
    provider = provider.lower().strip()

    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Usa PDF, DOCX o XLSX.",
        )

    data = await file.read()

    try:
        extracted = extract_document_images(
            filename,
            data,
            settings.document_min_image_width,
            settings.document_min_image_height,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudieron extraer las imágenes: {exc}",
        ) from exc

    # El texto que el PDF ya trae es exacto: pasarlo por OCR solo añade
    # errores. Se transcribe con visión únicamente lo que es imagen.
    native_pages: list[dict] = []
    rendered_fallback = False

    if extension == ".pdf":
        try:
            native_pages = extract_pdf_native_text(data)
        except Exception:
            native_pages = []

        # Ni texto ni imágenes: la página en sí es el documento.
        if not native_pages and not extracted:
            try:
                extracted = render_pdf_pages(data)
                rendered_fallback = True
            except Exception:
                extracted = []

    results, consolidated = await _transcribe_extracted_images(
        provider,
        extracted,
    )

    if native_pages:
        native_block = "\n\n".join(
            f"Página {item['page']} (texto del documento):\n{item['text']}"
            for item in native_pages
        )
        consolidated = (
            f"{native_block}\n\n{consolidated}"
            if consolidated
            else native_block
        )

    warnings = []

    if native_pages:
        warnings.append(
            f"{len(native_pages)} página(s) traían texto propio "
            f"({sum(item['characters'] for item in native_pages)} caracteres). "
            "Ese texto se tomó tal cual, sin OCR, porque ya es exacto."
        )

    if rendered_fallback:
        warnings.append(
            "El PDF no traía texto ni imágenes incrustadas: se renderizaron "
            "las páginas completas para poder transcribirlas."
        )

    if extension == ".docx":
        warnings.append(
            "En DOCX se identifica la imagen incrustada; "
            "la página visual exacta depende del renderizado de Word."
        )

    if not extracted and not native_pages:
        warnings.append(
            "No se encontraron imágenes incrustadas que cumplan "
            "el tamaño mínimo configurado."
        )

    # Resume alertas internas de AUTO/Qwen sin ensuciar el texto final.
    if any(
        "LOW_ENGINE_AGREEMENT" in (item.get("warnings") or [])
        for item in results
    ):
        warnings.append(
            "Una o más imágenes tuvieron baja concordancia entre motores. "
            "La salida se conservó, pero conviene revisar visualmente esas páginas."
        )

    return {
        "status": "ok",
        "provider": provider,
        "input_type": extension.lstrip("."),
        "filename": filename,
        "images_found": len(extracted),
        "images_processed": len(results),
        "native_text_pages": len(native_pages),
        "native_text_characters": sum(
            item["characters"] for item in native_pages
        ),
        "rendered_pages_fallback": rendered_fallback,
        "processing_seconds": round(
            time.perf_counter() - started,
            4,
        ),
        "images": results,
        "full_text": consolidated,
        "warnings": warnings,
    }


@app.post("/api/v1/vision/document/benchmark-images")
async def benchmark_document_images(
    file: UploadFile = File(...),
) -> dict:
    """
    Benchmark de las imágenes extraídas del documento.
    AUTO no participa.
    """
    started = time.perf_counter()

    filename = file.filename or "document"
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Usa PDF, DOCX o XLSX.",
        )

    data = await file.read()

    try:
        extracted = extract_document_images(
            filename,
            data,
            settings.document_min_image_width,
            settings.document_min_image_height,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudieron extraer las imágenes: {exc}",
        ) from exc

    limited = extracted[
        : settings.benchmark_max_document_images
    ]

    statuses = await provider_statuses()
    availability = {
        item["name"]: bool(item.get("available"))
        for item in statuses
    }

    provider_results = []

    for provider in BENCHMARK_PROVIDERS:
        if not availability.get(provider):
            provider_results.append(
                {
                    "provider": provider,
                    "status": "unavailable",
                    "images": [],
                }
            )
            continue

        one_started = time.perf_counter()

        results, consolidated = await _transcribe_extracted_images(
            provider,
            limited,
        )

        provider_results.append(
            {
                "provider": provider,
                "status": "ok",
                "processing_seconds": round(
                    time.perf_counter() - one_started,
                    4,
                ),
                "images": results,
                "full_text": consolidated,
            }
        )

    return {
        "status": "ok",
        "mode": "document_benchmark",
        "filename": filename,
        "images_found": len(extracted),
        "images_benchmarked": len(limited),
        "max_images": settings.benchmark_max_document_images,
        "processing_seconds": round(
            time.perf_counter() - started,
            4,
        ),
        "results": provider_results,
    }
