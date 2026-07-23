# Primera instalación — resumen

No ejecutes los pasos hasta verificar primero los programas instalados.

Requisitos:

- Windows 10/11 x64.
- NVIDIA RTX 3060 y driver actualizado.
- Python 3.11 x64.
- Node.js 20 o superior.
- Docker Desktop abierto.
- FFmpeg y ffprobe disponibles en PATH.

Comandos principales:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\scripts\start_all_windows.ps1
```

Direcciones:

- Dashboard: `http://localhost:3000`
- Swagger: `http://localhost:8000/docs`

La primera carga de `large-v3` descarga los pesos y puede tardar según la conexión.
