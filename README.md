# Sistema de Transcripción Híbrido v4

Base funcional para administrar sedes, aulas, cámaras, profesores, cursos y horarios; cargar una grabación larga; seleccionar visualmente el intervalo de una clase; recortarlo antes de la inferencia; y transcribirlo mediante Whisper `large-v3`.

## Configuración activa para pruebas

- Frontend: Next.js.
- API: FastAPI.
- Base de datos: PostgreSQL.
- Cola: Redis + Celery.
- Motor local: Faster-Whisper `large-v3`.
- GPU local: NVIDIA RTX 3060 mediante CUDA.
- Perfil de memoria: `int8_float16`, un worker GPU y concurrencia 1.
- Proveedor Together AI: implementado, documentado y desactivado hasta disponer de API Key y saldo.

Mientras `TRANSCRIPTION_PROVIDER=local`, el sistema no realiza llamadas ni genera consumo en Together AI.

## Flujo implementado

1. Crear una sede.
2. Crear aulas asociadas a la sede.
3. Registrar cámaras.
4. Asignar una cámara a un aula conservando el historial.
5. Crear profesores y cursos.
6. Crear horarios recurrentes.
7. Subir una grabación mediante streaming binario.
8. Elegir una grabación y un horario o ajustar manualmente los marcadores.
9. Recortar únicamente el intervalo seleccionado con FFmpeg.
10. Extraer audio mono a 16 kHz.
11. Crear chunks con solapamiento.
12. Transcribir con Faster-Whisper local o Together AI cuando se active.
13. Revisar, aprobar y descargar TXT o JSON.

## CRUD incluidos

- Sedes.
- Aulas.
- Cámaras.
- Historial de asignaciones cámara–aula.
- Profesores.
- Cursos y vocabulario técnico.
- Horarios.

Las eliminaciones operativas son lógicas: los registros se desactivan para conservar el historial.

## Estructura

```text
backend/
  app/api/admin.py          CRUD y relaciones
  app/api/recordings.py     subida rápida, inspección y vista previa
  app/api/jobs.py           trabajos asíncronos
  app/services/             FFmpeg, almacenamiento y proveedores ASR
  app/tasks/                workers Celery CPU/transcripción
frontend/
  app/page.tsx              dashboard, CRUD, editor y resultados
storage/
  uploads/ work/ results/ model-cache/
scripts/
  setup_windows.ps1
  start_all_windows.ps1
  reset_local_database.ps1
docs/
```

## Editor de intervalo

El navegador muestra la duración completa, dos marcadores y las horas absolutas. El backend calcula los offsets y ejecuta FFmpeg antes de seleccionar el proveedor:

```text
Video completo
  → intervalo seleccionado
  → audio recortado
  → chunks
  → Faster-Whisper local o Together AI
```

Por eso Together AI, cuando se habilite, solo consumirá los minutos realmente seleccionados.

## Together AI preparado, pero inactivo

Para el futuro existe una abstracción de proveedor:

```text
TranscriptionProvider
├── FasterWhisperLocalProvider
└── TogetherWhisperProvider
```

La activación futura se hará en `.env`:

```env
TRANSCRIPTION_PROVIDER=together
TOGETHER_API_KEY=...
TOGETHER_MODEL=openai/whisper-large-v3
```

No actives estas variables todavía.

## Nota sobre la base de datos

Esta versión amplía el modelo de las versiones anteriores. Para la primera ejecución de v4 usa una base limpia. Si ya habías iniciado una versión previa, ejecuta:

```powershell
.\scripts\reset_local_database.ps1
```

Esto elimina únicamente los volúmenes locales de PostgreSQL y Redis de este proyecto.

## Estado de la entrega

- Sintaxis Python verificada.
- Modelo SQLAlchemy creado y probado con SQLite temporal.
- Operaciones principales de los CRUD probadas directamente.
- Página TSX validada con el compilador TypeScript mediante declaraciones temporales.
- La inferencia completa con CUDA debe probarse en la RTX 3060 del equipo local.
- Antes de producción faltan autenticación corporativa, permisos, auditoría, HTTPS, backups, observabilidad y pruebas de carga.
