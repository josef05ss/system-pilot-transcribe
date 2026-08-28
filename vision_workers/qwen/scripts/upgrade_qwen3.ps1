$ErrorActionPreference = "Stop"

$Python = "D:\vision_qwen_venv\Scripts\python.exe"

if (!(Test-Path $Python)) {
    throw "No existe D:\vision_qwen_venv. Instala primero el worker Qwen."
}

Write-Host "Verificando PyTorch CUDA actual..."
& $Python -c "import torch; print('Torch:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available()); print('CUDA PyTorch:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'SIN CUDA')"
if ($LASTEXITCODE -ne 0) {
    throw "No se pudo verificar PyTorch."
}

Write-Host ""
Write-Host "IMPORTANTE: este script NO reinstala torch."
Write-Host "Se conserva tu PyTorch CUDA actual."
Write-Host ""

& $Python -m pip install --upgrade `
    "transformers>=4.57,<6" `
    "qwen-vl-utils==0.0.14" `
    "accelerate>=1.0,<2" `
    "safetensors>=0.4" `
    "pillow>=10,<13"

if ($LASTEXITCODE -ne 0) {
    throw "Fallo actualizando dependencias de Qwen3-VL."
}

Write-Host ""
Write-Host "Verificando soporte Qwen3-VL..."
& $Python -c "from transformers import AutoModelForImageTextToText, AutoProcessor; import transformers; print('Transformers:', transformers.__version__); print('Qwen3-VL API: OK')"

if ($LASTEXITCODE -ne 0) {
    throw "Transformers no tiene soporte compatible con Qwen3-VL."
}

Write-Host ""
Write-Host "Actualización Qwen3-VL terminada."
Write-Host "La primera transcripción descargará Qwen3-VL-4B-Instruct."
