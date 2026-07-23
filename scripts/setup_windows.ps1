$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "=== Instalación del sistema real de transcripción ===" -ForegroundColor Cyan

function Require-Command($Name, $Message) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Message (comando faltante: $Name)"
    }
}

Require-Command "py" "Instala Python 3.11 x64"
Require-Command "node" "Instala Node.js 20.9 o superior"
Require-Command "npm" "npm debe instalarse junto con Node.js"
Require-Command "docker" "Instala e inicia Docker Desktop"
Require-Command "ffmpeg" "Instala FFmpeg y agrégalo a PATH"
Require-Command "ffprobe" "ffprobe debe estar junto a FFmpeg"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Se creó .env" -ForegroundColor Green
}
if (-not (Test-Path "frontend\.env.local")) {
    Copy-Item "frontend\.env.local.example" "frontend\.env.local"
    Write-Host "Se creó frontend/.env.local" -ForegroundColor Green
}

if (-not (Test-Path ".venv")) {
    py -3.11 -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
& ".\.venv\Scripts\python.exe" -m pip install -r "backend\requirements.txt"

Push-Location frontend
npm install
Pop-Location

New-Item -ItemType Directory -Force -Path "storage\uploads", "storage\work", "storage\results", "storage\model-cache" | Out-Null

Write-Host ""
Write-Host "Instalación terminada." -ForegroundColor Green
Write-Host "Siguiente paso: .\scripts\start_all_windows.ps1"
