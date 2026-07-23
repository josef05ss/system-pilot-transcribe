# Arquitectura

```text
Usuarios
  ↓
Next.js
  ↓
FastAPI
  ├── CRUD y modelado → PostgreSQL
  ├── subida binaria → almacenamiento local
  └── trabajos → Redis/Celery
                  ├── worker CPU: inspección, recorte, normalización y chunks
                  └── worker transcripción:
                        ├── local: Faster-Whisper large-v3 + CUDA
                        └── futuro: Together AI Whisper API
```

## Escalamiento

Con una GPU existe un worker de transcripción y los trabajos restantes esperan en cola. Con varias GPU se inicia un worker por GPU conectado a la misma cola. Con Together AI, el worker realiza solicitudes HTTP y no necesita CUDA para la inferencia.

## Decisión crítica

El recorte ocurre en el worker CPU antes de la inferencia. Ningún proveedor recibe la grabación completa cuando solo se necesita una sección.
