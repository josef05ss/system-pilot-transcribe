$ErrorActionPreference = "Stop"

$WorkerRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DriveRoot = [System.IO.Path]::GetPathRoot($WorkerRoot)
$Venv = Join-Path $DriveRoot "vision_docling_venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$EnvFile = Join-Path $WorkerRoot ".env"

if (!(Test-Path $Python)) {
    throw "No existe $Python. Ejecuta primero setup.ps1 del worker docling."
}

# Cargar .env del worker al proceso actual
if (Test-Path $EnvFile) {
    Write-Host "Cargando variables desde: $EnvFile"

    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()

        if ($line -eq "" -or $line.StartsWith("#") -or !$line.Contains("=")) {
            return
        }

        $parts = $line.Split("=", 2)
        $nameVar = $parts[0].Trim()
        $value = $parts[1].Trim()

        if (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        if ($nameVar) {
            [Environment]::SetEnvironmentVariable(
                $nameVar,
                $value,
                "Process"
            )
            Write-Host "  OK $nameVar"
        }
    }
}
else {
    Write-Host "No existe .env en $WorkerRoot; se usarán valores por defecto."
}

# Host/puerto configurables desde .env, con fallback seguro
$HostAddress = [Environment]::GetEnvironmentVariable("DOCLING_HOST", "Process")
$Port = [Environment]::GetEnvironmentVariable("DOCLING_PORT", "Process")

if ([string]::IsNullOrWhiteSpace($HostAddress)) {
    $HostAddress = "0.0.0.0"
}

if ([string]::IsNullOrWhiteSpace($Port)) {
    $Port = "8112"
}

Set-Location $WorkerRoot

Write-Host ""
Write-Host "Iniciando docling worker..."
Write-Host "Host: $HostAddress"
Write-Host "Puerto: $Port"
Write-Host "Health: http://localhost:$Port/health"
Write-Host ""

& $Python -m uvicorn app.main:app --host $HostAddress --port $Port

if ($LASTEXITCODE -ne 0) {
    throw "docling worker termino con codigo $LASTEXITCODE."
}
