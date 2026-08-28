from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vision_host: str = "0.0.0.0"
    vision_port: int = 8100
    vision_cors_origins: str = "http://localhost:3000"

    # Producción: AUTO = Paddle literal + Qwen visual.
    ocr_provider: str = "auto"
    ocr_device: str = "cpu"
    ocr_language: str = "es"

    docling_worker_url: str = "http://localhost:8112"
    qwen_worker_url: str = "http://localhost:8113"
    surya_worker_url: str = "http://localhost:8114"

    provider_timeout_seconds: float = 300.0
    provider_health_timeout_seconds: float = 2.0

    document_min_image_width: int = 40
    document_min_image_height: int = 40
    benchmark_max_document_images: int = 10

    @property
    def cors_origins(self) -> list[str]:
        return [
            item.strip()
            for item in self.vision_cors_origins.split(",")
            if item.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
