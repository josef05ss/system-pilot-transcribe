# Configuración local para RTX 3060

El archivo `.env.example` ya viene preparado con:

```env
WHISPER_MODEL=large-v3
AI_DEVICE=cuda
AI_COMPUTE_TYPE=int8_float16
AI_BATCH_SIZE=2
AI_BEAM_SIZE=3
AI_LANGUAGE=es
```

## Motivo

- `large-v3`: modelo principal de máxima calidad dentro de la familia Whisper.
- `int8_float16`: menor consumo de VRAM para la RTX 3060.
- `batch 2`: punto de partida conservador y estable.
- `beam 3`: equilibrio entre calidad y tiempo de respuesta.
- `concurrency 1`: una sola instancia del modelo en la GPU.

## Si aparece falta de memoria

Primero cambia únicamente:

```env
AI_BATCH_SIZE=1
```

No abras dos workers GPU sobre la misma RTX 3060. Reinicia el worker después de
modificar `.env`.

## Verificación

Mientras un trabajo esté en estado `TRANSCRIBING`, ejecuta:

```powershell
nvidia-smi -l 1
```

Debe aparecer el proceso de Python usando memoria de la GPU.
