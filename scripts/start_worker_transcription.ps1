param([int]$GpuIndex = 0)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location "$Root\backend"
$env:PYTHONPATH = "$Root\backend"

# En modo local, este worker utiliza la GPU indicada.
# En modo Together, la misma cola funciona como worker HTTP/CPU y CUDA no es necesaria.
$env:CUDA_VISIBLE_DEVICES = "$GpuIndex"
Write-Host "Worker de transcripción iniciado (GPU local $GpuIndex o Together según .env)" -ForegroundColor Cyan
& "$Root\.venv\Scripts\python.exe" -m celery -A app.tasks.celery_app:celery_app worker --pool=solo -Q transcription --hostname="transcription$GpuIndex@%h" --concurrency=1 --loglevel=INFO
