param([int]$GpuIndex = 0)
# Alias conservado para compatibilidad. El nombre correcto ahora es worker de transcripción.
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
& "$Root\scripts\start_worker_transcription.ps1" -GpuIndex $GpuIndex
