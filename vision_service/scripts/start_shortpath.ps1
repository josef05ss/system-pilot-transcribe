$ErrorActionPreference = "Stop"

$ServiceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $ServiceRoot "..")).Path
$DriveRoot = [System.IO.Path]::GetPathRoot($RepoRoot)

$Venv = Join-Path $DriveRoot "vision_service_venv"
$Python = Join-Path $Venv "Scripts\python.exe"

if (!(Test-Path $Python)) {
    throw "No existe $Python. Ejecuta primero setup_cpu_shortpath_v2.ps1."
}

Set-Location $ServiceRoot

Write-Host "Iniciando Vision Service..."
Write-Host "API: http://localhost:8100"
Write-Host "Swagger: http://localhost:8100/docs"
Write-Host ""

& $Python -m uvicorn app.main:app --host 0.0.0.0 --port 8100 --reload
if ($LASTEXITCODE -ne 0) {
    throw "Vision Service termino con codigo $LASTEXITCODE."
}
