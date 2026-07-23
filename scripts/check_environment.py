from __future__ import annotations

import platform
import shutil
import subprocess


def check_command(name: str) -> None:
    path = shutil.which(name)
    print(f"{name}: {'OK - ' + path if path else 'NO ENCONTRADO'}")


print("=== Sistema ===")
print(platform.platform())
print("Python:", platform.python_version())

print("\n=== Comandos ===")
for command in ("ffmpeg", "ffprobe", "nvidia-smi"):
    check_command(command)

print("\n=== CTranslate2 ===")
try:
    import ctranslate2

    print("Versión:", ctranslate2.__version__)
    print("Dispositivos CUDA:", ctranslate2.get_cuda_device_count())
except Exception as exc:
    print("ERROR:", exc)

if shutil.which("nvidia-smi"):
    print("\n=== GPU ===")
    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=False,
    )
