$ErrorActionPreference = "Stop"

$WorkerRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DriveRoot = [System.IO.Path]::GetPathRoot($WorkerRoot)
$Venv = Join-Path $DriveRoot "vision_docling_venv"

if (Test-Path $Venv) {
    Remove-Item -Recurse -Force $Venv
}

py -3.11 -m venv $Venv
if ($LASTEXITCODE -ne 0) { throw "No se pudo crear el venv." }

$Python = Join-Path $Venv "Scripts\python.exe"
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Fallo pip." }



& $Python -m pip install -r "$WorkerRoot\requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "Fallo instalando dependencias de docling." }

Write-Host ""
Write-Host "DOCLING WORKER INSTALADO"
Write-Host "Venv: $Venv"
