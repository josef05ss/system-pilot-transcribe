$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "ADVERTENCIA: se eliminarán la base PostgreSQL y la cola Redis locales de este proyecto." -ForegroundColor Yellow
$answer = Read-Host "Escribe REINICIAR para continuar"
if ($answer -ne "REINICIAR") {
    Write-Host "Operación cancelada."
    exit 0
}

docker compose -f docker-compose.infra.yml down -v
Write-Host "Infraestructura local reiniciada. En el próximo arranque se creará el modelo v4." -ForegroundColor Green
