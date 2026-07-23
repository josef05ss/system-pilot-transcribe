# Subida rápida de grabaciones

## Implementación actual

El dashboard utiliza `XMLHttpRequest` para mostrar progreso de subida y envía el archivo sin `FormData`:

```text
POST /api/recordings/upload-fast
Content-Type: application/octet-stream
X-File-Name: nombre_codificado.mp4
X-File-Size: tamaño_en_bytes
```

FastAPI consume `Request.stream()` y escribe directamente al destino con `aiofiles`.

## Beneficios

- evita el spool multipart para archivos muy grandes;
- evita una copia adicional del temporal al almacenamiento final;
- no carga el video completo en RAM;
- muestra porcentaje y MB/s;
- detecta subidas incompletas;
- mantiene un endpoint multipart de respaldo.

## Qué no puede acelerar el software

La velocidad máxima está limitada por:

- lectura del disco del usuario;
- red LAN o Internet;
- escritura del disco del servidor;
- antivirus;
- proxy o reverse proxy;
- límites del navegador.

## Producción recomendada

Cuando se confirme dónde están las grabaciones, se debe preferir:

```text
NVR/NAS/S3 → backend obtiene referencia o intervalo
```

antes que:

```text
NVR → PC del usuario → navegador → backend
```

Si los videos están en S3 u otra nube, la evolución adecuada es subida multipart directa con URL prefirmada o selección del objeto existente. Si están en NVR/NAS, el backend debe recuperar el tramo solicitado directamente.
