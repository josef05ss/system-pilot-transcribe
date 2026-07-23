# Together AI preparado y desactivado

## Estado actual

El sistema utiliza:

```env
TRANSCRIPTION_PROVIDER=local
```

Por tanto, todos los chunks se procesan con Faster-Whisper en la RTX 3060. No se realiza ninguna petición a Together AI y no existe consumo de saldo.

## Dónde está implementado

```text
backend/app/services/together_transcriber.py
backend/app/services/transcription_provider.py
```

El proveedor Together incluye:

- autenticación Bearer;
- `openai/whisper-large-v3`;
- `verbose_json`;
- timestamps por segmento;
- prompt con vocabulario del curso;
- reintentos ante HTTP 429 y errores 5xx;
- timeout configurable;
- diarización opcional;
- conversión de la respuesta al mismo formato usado por el proveedor local.

## Orden del procesamiento

```text
Video completo
→ selección del horario
→ extracción con FFmpeg
→ audio del intervalo
→ chunks FLAC
→ Together AI
```

Se usa FLAC cuando el proveedor es Together para disminuir los bytes enviados sin perder calidad de audio. El cobro del proveedor se basa en el audio procesado según sus condiciones vigentes; el sistema no envía las horas descartadas.

## Activación futura

1. Obtener saldo y API Key.
2. Copiar a `.env`:

```env
TRANSCRIPTION_PROVIDER=together
TOGETHER_API_KEY=clave_real
TOGETHER_MODEL=openai/whisper-large-v3
TOGETHER_BASE_URL=https://api.together.ai/v1
TOGETHER_TIMEOUT_SECONDS=1800
TOGETHER_MAX_RETRIES=3
```

3. Reiniciar FastAPI y workers.
4. Probar primero con 1–5 minutos.
5. Revisar factura, límites, precisión y latencia.
6. Autorizar el envío de audio a un tercero antes del uso real.

## Límites externos

Los límites y precios pueden cambiar. Revisar antes de activar:

- https://docs.together.ai/docs/inference/transcription/overview
- https://docs.together.ai/docs/inference/transcription/features
- https://docs.together.ai/reference/audio-transcriptions
- https://www.together.ai/pricing

A julio de 2026, la documentación oficial indica un máximo de cuatro horas por solicitud y formatos como WAV, MP3, FLAC, OGG, Opus y AAC. Nuestro sistema usa chunks de cinco minutos, muy por debajo de ese límite.
