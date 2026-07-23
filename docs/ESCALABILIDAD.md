# Escalabilidad

## Etapa local

- Un servidor o PC.
- Una RTX 3060.
- Un worker CPU.
- Un worker de transcripción con concurrencia 1.
- Varios trabajos permanecen en Redis.

## Más usuarios con una GPU

Los usuarios pueden crear solicitudes simultáneamente. FastAPI responde rápido y las transcripciones esperan en cola. Solo un trabajo usa la GPU a la vez.

## Más GPU

Ejecutar un worker por GPU y fijar `CUDA_VISIBLE_DEVICES`:

```powershell
.\scripts\start_worker_transcription.ps1 -GpuIndex 0
.\scripts\start_worker_transcription.ps1 -GpuIndex 1
```

Cada worker conserva una copia de Faster-Whisper en su GPU.

## Together AI

Si se activa Together, la cola sigue siendo necesaria para:

- controlar concurrencia;
- reintentar HTTP 429/5xx;
- mantener orden y prioridad;
- evitar saturar límites del proveedor;
- guardar resultados de forma idempotente.

## Varios servidores

Todos deben compartir Redis, PostgreSQL y almacenamiento común como NAS o S3.

## Por qué no Spark

La ruta operativa distribuye trabajos de inferencia, no DataFrames masivos. Celery + Redis encaja mejor. Spark puede utilizarse después para analítica histórica.
