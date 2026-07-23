from __future__ import annotations

import platform
import shutil
import subprocess

import ctranslate2
from fastapi import APIRouter

from app.core.config import settings

router = APIRouter(prefix="/api/system", tags=["Sistema"])


def nvidia_summary() -> list[dict]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,memory.total,driver_version,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        output = []
        for line in result.stdout.splitlines():
            index, name, memory, driver, utilization = [item.strip() for item in line.split(",", 4)]
            output.append(
                {
                    "index": int(index),
                    "name": name,
                    "memory_mb": int(memory),
                    "driver": driver,
                    "utilization_percent": int(utilization),
                }
            )
        return output
    except Exception:
        return []


@router.get("")
def system_info() -> dict:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ctranslate2": ctranslate2.__version__,
        "cuda_devices": ctranslate2.get_cuda_device_count(),
        "configured_device": settings.ai_device,
        "configured_model": settings.whisper_model,
        "transcription_provider": settings.transcription_provider,
        "together_ready": settings.together_ready,
        "together_model": settings.together_model,
        "gpus": nvidia_summary(),
    }
