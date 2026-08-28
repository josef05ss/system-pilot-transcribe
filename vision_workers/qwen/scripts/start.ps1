$ErrorActionPreference = "Stop"

$WorkerRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = "D:\vision_qwen_venv\Scripts\python.exe"
$EnvFile = Join-Path $WorkerRoot ".env"

if (!(Test-Path $Python)) {
    throw "No existe $Python."
}

if (Test-Path $EnvFile) {
    Write-Host "Cargando variables desde: $EnvFile"

    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()

        if (
            $line -eq "" -or
            $line.StartsWith("#") -or
            !$line.Contains("=")
        ) {
            return
        }

        $parts = $line.Split("=", 2)
        $name = $parts[0].Trim()
        $value = $parts[1].Trim()

        if (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        if ($name) {
            [Environment]::SetEnvironmentVariable(
                $name,
                $value,
                "Process"
            )
            Write-Host "  OK $name"
        }
    }
}

$HostAddress = $env:QWEN_HOST
$Port = $env:QWEN_PORT

if ([string]::IsNullOrWhiteSpace($HostAddress)) {
    $HostAddress = "0.0.0.0"
}

if ([string]::IsNullOrWhiteSpace($Port)) {
    $Port = "8113"
}

Set-Location $WorkerRoot

Write-Host ""
Write-Host "Iniciando Qwen3-VL OCR..."
Write-Host "Health: http://localhost:$Port/health"
Write-Host ""

& $Python -m uvicorn app.main:app `
    --host $HostAddress `
    --port $Port

if ($LASTEXITCODE -ne 0) {
    throw "Qwen worker termino con codigo $LASTEXITCODE."
}
