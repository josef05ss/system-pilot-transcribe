# Plan de pruebas

## Benchmark mínimo

Usar exactamente el mismo video e intervalo:

1. `small + CPU + int8` como línea base.
2. `small + CUDA + float16`.
3. `large-v3 + CUDA + int8_float16`.
4. `large-v3 + CUDA + float16`.

Registrar:

- duración seleccionada;
- tiempo de extracción;
- tiempo de transcripción;
- tiempo total;
- RTF;
- VRAM;
- uso de CPU/GPU;
- errores de palabras frente a una transcripción humana.

## Casos de formato

- MP4 H.264 + AAC.
- MP4 HEVC/H.265 + AAC.
- video sin audio: debe rechazarse.
- archivo dañado: debe devolverse error controlado.
- intervalo fuera de la grabación: debe rechazarse.

## Carga concurrente

Crear tres trabajos seguidos con una GPU. Deben permanecer en cola y completarse
sin bloquear FastAPI. Luego repetir con dos workers de transcripción cuando exista una segunda GPU.


## Subida

- archivo corto: porcentaje y MB/s;
- archivo grande: memoria estable y escritura directa;
- interrupción: archivo parcial eliminado;
- tamaño incorrecto: rechazo controlado.

## Together futuro

- confirmar que TRANSCRIPTION_PROVIDER=local no realiza solicitudes externas;
- activar con un audio de 1–5 minutos;
- validar que solo se envió el intervalo recortado;
- simular HTTP 429 y verificar reintentos.
