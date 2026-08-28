# OCR Pipeline — Transcripción de imágenes (digital + manuscrita)

Implementación completa de la arquitectura del diagrama:

```
USUARIO → imagen individual | documento (pdf/docx/pptx/xlsx)
        → EXTRACTOR DE IMÁGENES (solo si es documento, extrae solo imágenes)
        → IMAGEN EN MEMORIA
        → PREPROCESAMIENTO (grises, deskew, denoise, binarización)
        → OCR ROUTER (decide digital vs manuscrita)
            ├── DIGITAL     → TESSERACT
            └── MANUSCRITA  → TrOCR (HuggingFace transformers)
        → OCR RESULTADO → PostgreSQL → API (FastAPI) → FRONTEND
```

## Estructura del proyecto

```
ocr-app/
├── docker-compose.yml
├── backend/                  # FastAPI + OCR pipeline
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py           # API REST
│       ├── config.py
│       ├── database.py       # conexión PostgreSQL (SQLAlchemy)
│       ├── models.py         # tablas: documentos, resultados_ocr
│       ├── schemas.py
│       └── ocr/
│           ├── extractor.py      # extrae imágenes de PDF/DOCX/PPTX/XLSX
│           ├── preprocessing.py  # OpenCV: grises/deskew/denoise/binarización
│           ├── router.py         # OCR Router: digital vs manuscrita
│           ├── tesseract_engine.py
│           └── trocr_engine.py
└── frontend/                 # HTML/CSS/JS estático servido por Nginx
    ├── Dockerfile
    ├── nginx.conf             # proxy /api -> backend:8000
    └── src/ (index.html, styles.css, app.js)
```

## Cómo correrlo

Requisitos: Docker y Docker Compose instalados, y **conexión a internet**
la primera vez que se ejecuta (para descargar el modelo TrOCR de
HuggingFace, ~1.3 GB, y las imágenes base de Docker).

```bash
cd ocr-app
docker compose up --build
```

Cuando termine de levantar:

- **Frontend** → http://localhost:3000
- **API (FastAPI, con docs interactivas Swagger)** → http://localhost:8000/docs
- **PostgreSQL** → localhost:5432 (usuario `ocr_user`, password `ocr_password`, db `ocr_db`)

La primera transcripción de un texto manuscrito tardará más porque el
contenedor `backend` descarga el modelo `microsoft/trocr-base-handwritten`
la primera vez (queda cacheado en el volumen `ocr_model_cache` para las
siguientes ejecuciones).

## Motor de respaldo en la nube (opcional): API de Anthropic (Claude)

Cuando Tesseract y TrOCR no alcanzan el umbral de confianza (típicamente
en escritura a mano compleja, o texto impreso con bajo contraste/
tipografías decorativas), el pipeline puede usar un modelo de visión de
Anthropic como respaldo, mucho más preciso porque interpreta la imagen
con contexto semántico en vez de solo reconocer patrones de trazos.

**Es 100% opcional.** Sin configurarlo, la app funciona exactamente
igual que antes (Tesseract + TrOCR, sin llamadas externas ni costo).

### Cómo activarlo

1. Consigue una API key en https://console.anthropic.com/settings/keys
2. Copia `.env.example` a `.env` (misma carpeta que `docker-compose.yml`)
   y pega tu key:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```
3. Vuelve a levantar el proyecto: `docker compose up --build`

El archivo `.env` **no se sube a git** (está en `.gitignore`) — nunca
compartas tu API key públicamente.

### Costo

Cada vez que se usa este motor se hace una llamada real a la API de
Anthropic, con costo por imagen procesada. Revisa el precio actual del
modelo que uses en https://www.anthropic.com/pricing. Para controlar el
gasto, puedes usar un modelo más económico cambiando `ANTHROPIC_MODEL`
en `.env` (por ejemplo `claude-haiku-4-5-20251001`).

### Cómo se identifica en los resultados

Los resultados transcritos con este motor aparecen con clasificación
`ia_vision` / motor `claude_vision`, badge 🤖 en el frontend — así
siempre sabes qué imágenes usaron el motor local (gratis) y cuáles la
API (con costo).

## Endpoints principales
| Método | Ruta | Descripción |
|---|---|---|
| POST | `/api/ocr/upload` | Sube una imagen o documento y ejecuta todo el pipeline |
| GET | `/api/ocr/results` | Lista todos los documentos procesados |
| GET | `/api/ocr/results/{id}` | Detalle de un documento con todos sus resultados de OCR |
| DELETE | `/api/ocr/results/{id}` | Elimina un documento y sus resultados |
| GET | `/api/health` | Healthcheck |

## Notas técnicas importantes

1. **OCR Router (heurística).** Decide "digital vs manuscrita" corriendo
   primero Tesseract; si su confianza no alcanza el umbral, intenta con
   TrOCR. Si NINGUNO de los dos motores alcanza el umbral de confianza,
   el resultado se descarta como `baja_confianza` (no se muestra texto
   poco fiable). Está aislado en `app/ocr/router.py` para que puedas
   reemplazarlo fácilmente por un clasificador entrenado sin tocar el
   resto del pipeline.

2. **Umbral de confianza único (80% por defecto).** Variable
   `OCR_MIN_CONFIDENCE` en `docker-compose.yml`. Una transcripción solo
   se acepta y se guarda como texto real si su confianza es igual o
   mayor a este umbral. Bájalo si sientes que se está descartando texto
   válido; súbelo si sigues viendo transcripciones poco confiables.

3. **Detección de "sin texto".** Antes de correr cualquier motor, se
   mide si la imagen tiene contenido visual relevante
   (`app/ocr/preprocessing.has_text_content`). Si no lo tiene, el
   resultado es inmediatamente `sin_texto` con el mensaje
   "No se encontró texto." — sin gastar cómputo en Tesseract/TrOCR.

4. **Detección de color.** Cada resultado incluye `es_color`
   (true/false), calculado sobre la imagen original según su
   saturación promedio (`app/ocr/color_detection.py`). Ajustable con
   `COLOR_SATURATION_THRESHOLD`.

5. **Extracción de documentos.** Al extraer imágenes de PDF/DOCX/PPTX/
   XLSX, se descartan automáticamente imágenes muy pequeñas (íconos,
   viñetas, logos — controlado por `MIN_EXTRACTED_IMAGE_SIZE`, default
   60px) y duplicados exactos (mismo logo repetido en varias páginas/
   diapositivas), para que solo lleguen a OCR las imágenes de contenido
   real.

6. **Idiomas de Tesseract**: viene configurado con `spa+eng` (español +
   inglés). Se cambia con la variable de entorno `TESSERACT_LANG`.

7. **Recursos**: TrOCR usa `transformers` + `torch` (CPU por defecto). Si
   tu servidor tiene GPU NVIDIA con drivers/CUDA configurados en Docker,
   el código ya detecta `cuda` automáticamente (`trocr_engine.py`).

8. **Persistencia**: los datos de PostgreSQL y el caché del modelo TrOCR
   se guardan en volúmenes de Docker (`ocr_pg_data`, `ocr_model_cache`),
   así que sobreviven a un `docker compose down` (no a un `-v`).

> ⚠️ **Importante si ya tenías el proyecto corriendo antes:** esta
> versión agrega una columna nueva (`es_color`) a la tabla de
> resultados. La app crea tablas automáticamente solo si no existen
> (`Base.metadata.create_all`), pero **no** modifica tablas ya creadas.
> Si tu base de datos ya existía de una ejecución anterior, corre
> `docker compose down -v` (borra los datos existentes) antes de volver
> a levantar con `docker compose up --build`, o la app fallará al
> intentar guardar el nuevo campo.



## Comandos útiles

```bash
# Ver logs en vivo
docker compose logs -f backend

# Apagar todo
docker compose down

# Apagar y borrar también los datos (Postgres + modelo TrOCR)
docker compose down -v

# Reconstruir solo el backend tras un cambio de código
docker compose up --build backend
```
