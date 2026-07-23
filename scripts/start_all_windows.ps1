$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

& "$Root\scripts\start_infra.ps1"
Start-Sleep -Seconds 3

$commands = @(
    "$Root\scripts\start_backend.ps1",
    "$Root\scripts\start_worker_cpu.ps1",
    "$Root\scripts\start_worker_transcription.ps1",
    "$Root\scripts\start_frontend.ps1"
)

foreach ($script in $commands) {
    Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $script
}

Write-Host "Servicios iniciados en nuevas ventanas." -ForegroundColor Green
Write-Host "Dashboard: http://localhost:3000"
Write-Host "Swagger:   http://localhost:8000/docs"
