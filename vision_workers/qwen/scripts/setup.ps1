$ErrorActionPreference = "Stop"

$WorkerRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DriveRoot = [System.IO.Path]::GetPathRoot($WorkerRoot)
$Venv = Join-Path $DriveRoot "vision_qwen_venv"

if (Test-Path $Venv) {
    Remove-Item -Recurse -Force $Venv
}

py -3.11 -m venv $Venv
if ($LASTEXITCODE -ne 0) { throw "No se pudo crear el venv." }

$Python = Join-Path $Venv "Scripts\python.exe"
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Fallo pip." }


Write-Host "Instalando PyTorch CPU para Qwen..."
& $Python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "Fallo instalando PyTorch CPU." }


& $Python -m pip install -r "$WorkerRoot\requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "Fallo instalando dependencias de qwen." }

Write-Host ""
Write-Host "QWEN WORKER INSTALADO"
Write-Host "Venv: $Venv"
