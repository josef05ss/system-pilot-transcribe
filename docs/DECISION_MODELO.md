# Decisión del modelo

## Perfil activo para la RTX 3060

```text
Proveedor: local
Modelo: Whisper large-v3
Motor: Faster-Whisper / CTranslate2
Dispositivo: CUDA
Precisión: int8_float16
Batch: 2
Beam size: 3
Worker de transcripción: 1
```

Esta configuración mantiene `large-v3` y reduce el uso de VRAM. No existe pago por minuto porque la inferencia se ejecuta en la GPU local.

## Alternativas del dashboard

| Modelo | Uso |
|---|---|
| `large-v3` | principal y evaluación de calidad |
| `turbo` | comparación de velocidad |
| `medium` / `small` | diagnóstico en hardware menor |

## Together AI

El proveedor Together está implementado como alternativa futura. No se activa hasta configurar:

```env
TRANSCRIPTION_PROVIDER=together
TOGETHER_API_KEY=...
```

Together usaría `openai/whisper-large-v3`, pero la inferencia ocurriría en infraestructura externa y tendría cobro por uso según la tarifa vigente.
