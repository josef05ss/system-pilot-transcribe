$ErrorActionPreference = "Stop"

$ServiceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Venv = Join-Path $ServiceRoot ".venv"

Write-Host "Creando entorno virtual de Vision Service..."
py -3.11 -m venv $Venv

$Python = Join-Path $Venv "Scripts\python.exe"

& $Python -m pip install --upgrade pip

Write-Host "Instalando PaddlePaddle CPU..."
& $Python -m pip install "paddlepaddle==3.2.0" -i "https://www.paddlepaddle.org.cn/packages/stable/cpu/"

Write-Host "Instalando Vision Service..."
& $Python -m pip install -r "$ServiceRoot\requirements.txt"

if (!(Test-Path "$ServiceRoot\.env")) {
    Copy-Item "$ServiceRoot\.env.example" "$ServiceRoot\.env"
}

Write-Host ""
Write-Host "Vision Service instalado."
Write-Host "La primera ejecución de OCR descargará los modelos de PaddleOCR."
