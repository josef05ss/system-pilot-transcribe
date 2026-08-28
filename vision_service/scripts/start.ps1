$ErrorActionPreference = "Stop"

$ServiceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ServiceRoot ".venv\Scripts\python.exe"

if (!(Test-Path $Python)) {
    throw "Primero ejecuta .\vision_service\scripts\setup_cpu.ps1"
}

Set-Location $ServiceRoot
& $Python -m uvicorn app.main:app --host 0.0.0.0 --port 8100 --reload
