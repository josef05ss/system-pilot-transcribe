$ErrorActionPreference = "Stop"

$ServiceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $ServiceRoot "..")).Path
$DriveRoot = [System.IO.Path]::GetPathRoot($RepoRoot)

# Use a short path for the virtual environment to avoid Windows path-length
# problems with PaddlePaddle / PaddleOCR dependencies.
$Venv = Join-Path $DriveRoot "vision_service_venv"

Write-Host ""
Write-Host "Vision Service - instalacion CPU"
Write-Host "Proyecto: $RepoRoot"
Write-Host "Entorno virtual corto: $Venv"
Write-Host ""

if (Test-Path $Venv) {
    Write-Host "Eliminando entorno virtual anterior..."
    Remove-Item -Recurse -Force $Venv
}

Write-Host "Creando entorno virtual..."
& py -3.11 -m venv $Venv
if ($LASTEXITCODE -ne 0) {
    throw "No se pudo crear el entorno virtual."
}

$Python = Join-Path $Venv "Scripts\python.exe"

Write-Host "Actualizando pip..."
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Fallo al actualizar pip."
}

Write-Host "Instalando PaddlePaddle CPU..."
& $Python -m pip install "paddlepaddle==3.2.0" --index-url "https://www.paddlepaddle.org.cn/packages/stable/cpu/"
if ($LASTEXITCODE -ne 0) {
    throw "Fallo al instalar PaddlePaddle CPU."
}

Write-Host "Instalando dependencias de Vision Service..."
& $Python -m pip install -r "$ServiceRoot\requirements.txt"
if ($LASTEXITCODE -ne 0) {
    throw "Fallo al instalar las dependencias de Vision Service."
}

Write-Host "Verificando instalacion..."
& $Python -c "import paddle; import paddleocr; import fastapi; import fitz; import openpyxl; print('OK: dependencias principales importadas')"
if ($LASTEXITCODE -ne 0) {
    throw "La verificacion de dependencias fallo."
}

$EnvFile = Join-Path $ServiceRoot ".env"
$EnvExample = Join-Path $ServiceRoot ".env.example"
if (!(Test-Path $EnvFile) -and (Test-Path $EnvExample)) {
    Copy-Item $EnvExample $EnvFile
}

Write-Host ""
Write-Host "========================================"
Write-Host "VISION SERVICE INSTALADO CORRECTAMENTE"
Write-Host "========================================"
Write-Host "Venv: $Venv"
Write-Host ""
Write-Host "Siguiente comando:"
Write-Host "powershell -ExecutionPolicy Bypass -File .\vision_service\scripts\start_shortpath.ps1"
