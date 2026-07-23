from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable
from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import UploadFile

from app.core.config import settings


class StorageProvider(ABC):
    @abstractmethod
    async def save_upload(self, upload: UploadFile) -> tuple[str, int]:
        raise NotImplementedError

    @abstractmethod
    async def save_stream(
        self,
        chunks: AsyncIterable[bytes],
        original_name: str,
        expected_size: int | None = None,
    ) -> tuple[str, int]:
        """Guarda el cuerpo HTTP directamente, sin el spool multipart de UploadFile."""
        raise NotImplementedError

    @abstractmethod
    def materialize(self, uri: str) -> Path:
        raise NotImplementedError


class LocalStorageProvider(StorageProvider):
    def __init__(self, root: Path) -> None:
        self.upload_dir = root / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_name(name: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
        return clean or "recording.bin"

    def _destination(self, original_name: str) -> Path:
        safe_name = self._safe_name(original_name)
        return self.upload_dir / f"{uuid4()}_{safe_name}"

    async def save_upload(self, upload: UploadFile) -> tuple[str, int]:
        """Ruta multipart compatible. Para archivos grandes se prefiere save_stream."""
        destination = self._destination(upload.filename or "recording.bin")
        total = 0
        max_bytes = settings.max_upload_gb * 1024**3

        try:
            async with aiofiles.open(destination, "wb", buffering=16 * 1024 * 1024) as output:
                while chunk := await upload.read(16 * 1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(
                            f"El archivo supera el máximo de {settings.max_upload_gb} GB"
                        )
                    await output.write(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

        return str(destination.resolve()), total

    async def save_stream(
        self,
        chunks: AsyncIterable[bytes],
        original_name: str,
        expected_size: int | None = None,
    ) -> tuple[str, int]:
        """
        Escritura directa desde Request.stream(). Evita que Starlette guarde primero
        todo el multipart en un archivo temporal y después lo vuelva a copiar.
        """
        destination = self._destination(original_name)
        total = 0
        max_bytes = settings.max_upload_gb * 1024**3

        if expected_size is not None and expected_size > max_bytes:
            raise ValueError(f"El archivo supera el máximo de {settings.max_upload_gb} GB")

        try:
            async with aiofiles.open(destination, "wb", buffering=16 * 1024 * 1024) as output:
                async for chunk in chunks:
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(
                            f"El archivo supera el máximo de {settings.max_upload_gb} GB"
                        )
                    await output.write(chunk)

            if expected_size is not None and total != expected_size:
                raise ValueError(
                    f"La subida quedó incompleta: se esperaban {expected_size} bytes y llegaron {total}"
                )
        except Exception:
            destination.unlink(missing_ok=True)
            raise

        return str(destination.resolve()), total

    def materialize(self, uri: str) -> Path:
        path = Path(uri)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"No se encontró el archivo: {uri}")
        return path


_storage = LocalStorageProvider(settings.storage_root)


def get_storage_provider() -> StorageProvider:
    return _storage
