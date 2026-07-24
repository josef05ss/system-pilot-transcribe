from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Sistema de Transcripción"
    environment: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000"

    database_url: str = (
        "postgresql+psycopg://transcriptor:transcriptor123@localhost:5432/"
        "transcriptor_db"
    )
    redis_url: str = "redis://localhost:6379/0"

    storage_provider: Literal["local"] = "local"
    storage_root: Path = PROJECT_ROOT / "storage"
    keep_work_files: bool = False
    max_upload_gb: int = Field(default=80, ge=1, le=500)

    # Proveedor activo. En localhost se mantiene LOCAL para usar la RTX 3060.
    # Cambiar a "together" únicamente cuando exista API key y saldo autorizado.
    transcription_provider: Literal["local", "together"] = "local"

    # Perfil local: Whisper large-v3 ejecutado con Faster-Whisper/CTranslate2.
    whisper_model: str = "large-v3"
    ai_device: Literal["auto", "cuda", "cpu"] = "cuda"
    ai_compute_type: str = "int8_float16"
    ai_batch_size: int = Field(default=2, ge=1, le=32)
    ai_beam_size: int = Field(default=3, ge=1, le=10)
    ai_language: str = "es"
    model_cache_dir: Path = PROJECT_ROOT / "storage" / "model-cache"

    # Together AI queda implementado, pero desactivado por defecto.
    # No se realiza ninguna llamada ni consumo mientras TRANSCRIPTION_PROVIDER=local.
    together_api_key: str | None = None
    together_base_url: str = "https://api.together.ai/v1"
    together_model: str = "openai/whisper-large-v3"
    together_timeout_seconds: int = Field(default=1800, ge=60, le=7200)
    together_max_retries: int = Field(default=3, ge=0, le=8)
    together_diarize: bool = False
    together_min_speakers: int | None = Field(default=None, ge=1, le=50)
    together_max_speakers: int | None = Field(default=None, ge=1, le=50)

    default_chunk_seconds: int = Field(default=300, ge=60, le=1800)
    default_overlap_seconds: int = Field(default=3, ge=0, le=30)

    diarization_enabled: bool = False
    huggingface_token: str | None = None

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def together_ready(self) -> bool:
        return bool(self.together_api_key and self.together_api_key.strip())

    def ensure_directories(self) -> None:
        for path in (
            self.storage_root,
            self.storage_root / "uploads",
            self.storage_root / "work",
            self.storage_root / "results",
            self.model_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
if not settings.storage_root.is_absolute():
    settings.storage_root = (PROJECT_ROOT / settings.storage_root).resolve()
if not settings.model_cache_dir.is_absolute():
    settings.model_cache_dir = (PROJECT_ROOT / settings.model_cache_dir).resolve()
settings.ensure_directories()
