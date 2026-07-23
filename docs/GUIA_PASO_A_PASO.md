# Guía paso a paso

## 1. Verificar herramientas

Ejecutar en PowerShell:

```powershell
nvidia-smi
py -3.11 --version
node --version
npm --version
docker --version
ffmpeg -version
ffprobe -version
```

## 2. Preparar el proyecto

Abrir PowerShell en la carpeta raíz y ejecutar:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

El script crea `.env`, el entorno `.venv`, instala Python, instala Next.js y prepara las carpetas de almacenamiento.

## 3. Iniciar

```powershell
.\scripts\start_all_windows.ps1
```

Se abrirán ventanas separadas para:

- FastAPI.
- Worker CPU.
- Worker de transcripción.
- Next.js.

Además, Docker ejecutará PostgreSQL y Redis.

## 4. Configurar datos maestros

En `http://localhost:3000` entrar a **Configuración CRUD** y completar:

1. Sede.
2. Aula.
3. Cámara.
4. Asignación de cámara al aula.
5. Profesor.
6. Curso.
7. Horario.

El proyecto incluye registros de prueba para validar el flujo.

## 5. Primera transcripción

1. Abrir **Nueva transcripción**.
2. Seleccionar sede, aula y cámara.
3. Escribir el inicio real de la grabación.
4. Seleccionar un video corto para la primera prueba.
5. Subir e inspeccionar.
6. Seleccionar la grabación creada.
7. Elegir un horario o mover los marcadores.
8. Confirmar el intervalo.
9. Presionar **Recortar y enviar a transcripción**.
10. Revisar la cola y el resultado.

## 6. Verificar CUDA

Durante `TRANSCRIBING`, abrir otra terminal:

```powershell
nvidia-smi -l 1
```

Debe aparecer un proceso Python utilizando memoria de la RTX 3060.

## 7. Base antigua

Solo si ya ejecutaste otra versión y aparece un error de esquema:

```powershell
.\scripts\reset_local_database.ps1
```

Después vuelve a ejecutar `start_all_windows.ps1`.
