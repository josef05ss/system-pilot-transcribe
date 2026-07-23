$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
docker compose -f docker-compose.infra.yml up -d --wait
Write-Host "PostgreSQL y Redis iniciados." -ForegroundColor Green
