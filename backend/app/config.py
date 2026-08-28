import os


class Settings:
    """Configuración centralizada, leída desde variables de entorno
    (definidas en docker-compose.yml)."""

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://ocr_user:ocr_password@db:5432/ocr_db",
    )

    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/app/uploads")

    # Idiomas para Tesseract (español + inglés)
    TESSERACT_LANG: str = os.getenv("TESSERACT_LANG", "spa+eng")

    # Modelo de HuggingFace para texto manuscrito
    TROCR_MODEL_NAME: str = os.getenv(
        "TROCR_MODEL_NAME", "microsoft/trocr-base-handwritten"
    )

    # --- Detección de "imagen sin texto" ---
    # Antes de correr cualquier motor de OCR, medimos qué proporción de
    # píxeles de "contenido" (texto u otro elemento) tiene la imagen. Si
    # está fuera de este rango, se asume que no hay nada relevante y NO
    # se llama a Tesseract ni a TrOCR (evita que TrOCR "alucine"
    # caracteres en imágenes en blanco o sin contenido).
    MIN_INK_RATIO: float = float(os.getenv("MIN_INK_RATIO", "0.001"))
    MAX_INK_RATIO: float = float(os.getenv("MAX_INK_RATIO", "0.45"))

    # --- Umbral único de confianza aceptada (0-100) ---
    # Se aplica tanto a Tesseract (digital) como a TrOCR (manuscrita).
    # Una transcripción solo se acepta y se guarda como texto real si
    # su confianza es >= este umbral; si ningún motor lo alcanza, el
    # resultado se marca como "baja_confianza" y no se muestra texto,
    # para evitar mostrar transcripciones poco fiables.
    OCR_MIN_CONFIDENCE: float = float(os.getenv("OCR_MIN_CONFIDENCE", "80"))

    # --- Detección de color ---
    # Saturación promedio (canal S en HSV, escala 0-255) mínima para
    # considerar que una imagen es a color. Por debajo de esto se
    # considera blanco y negro / escala de grises.
    COLOR_SATURATION_THRESHOLD: float = float(
        os.getenv("COLOR_SATURATION_THRESHOLD", "12")
    )

    # --- Filtrado de imágenes extraídas de documentos ---
    # Al extraer imágenes de PDF/DOCX/PPTX/XLSX, se descartan las que
    # sean más pequeñas que esto en cualquier dimensión (íconos,
    # viñetas, logos diminutos que no son contenido real a transcribir).
    MIN_EXTRACTED_IMAGE_SIZE: int = int(os.getenv("MIN_EXTRACTED_IMAGE_SIZE", "60"))

    # --- Motor de respaldo: API de visión de Anthropic (Claude) ---
    # Opcional. Si se configura ANTHROPIC_API_KEY, se usa como motor de
    # respaldo cuando Tesseract y TrOCR no alcanzan el umbral de
    # confianza mínimo (típicamente escritura a mano compleja o texto
    # impreso con bajo contraste/tipografías decorativas). Si no se
    # configura, el pipeline sigue funcionando 100% local sin llamadas
    # externas, igual que antes.
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    # Claude no devuelve un score de confianza numérico; al responder
    # con una transcripción se le asigna esta confianza fija (por
    # defecto claramente por encima del umbral de aceptación).
    VISION_API_CONFIDENCE: float = float(os.getenv("VISION_API_CONFIDENCE", "95"))

    # Extensiones soportadas
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx"}

    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))

    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")


settings = Settings()
