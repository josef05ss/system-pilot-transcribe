# Arquitectura Detallada del Sistema de Transcripción Híbrido

## 1. Visión General (C4 Model)

```mermaid
C4Container
    title Diagrama de Contenedores - Sistema de Transcripción Híbrido v4

    Person(admin, "Administrador", "Gestiona sedes, profesores, cursos y realiza transcripciones.")
  
    System_Ext(together, "Together AI API", "Proveedor de IA opcional (Fallback) para transcripción (Whisper Large v3).")
    System_Ext(gpu, "Local GPU (NVIDIA RTX 3060)", "Hardware local para inferencia con Faster-Whisper.")

    Container_Boundary(c1, "Sistema Híbrido v4") {
        Container(frontend, "Frontend", "Next.js, TypeScript, React", "Interfaz de usuario para CRUDs, subida de grabaciones, y editor de transcripción.")
        Container(celery, "Celery Workers", "Python, Celery", "Workers que ejecutan FFmpeg y la transcripción pesada en segundo plano.")
        Container(storage, "File Storage Local", "Volúmenes", "Almacena videos subidos (uploads), trabajos en curso (work) y resultados (results).")
        Container(redis, "Cola de Mensajes", "Redis", "Gestiona las colas de trabajos asíncronos.")
        Container(backend, "Backend API", "FastAPI, Python", "Maneja la lógica de negocio, autenticación (futura) y gestión de trabajos.")
        ContainerDb(db, "Base de Datos Relacional", "PostgreSQL", "Almacena datos operativos, configuraciones de aula y metadatos de transcripción.")
    }

    Rel(admin, frontend, "Usa", "Navegador Web")
    Rel(frontend, backend, "Solicita CRUD y trabajos", "REST API (JSON)")
    Rel(frontend, storage, "Sube/Descarga archivos", "HTTP (Multipart)")
    Rel(backend, db, "Lee y Escribe datos", "SQLAlchemy ORM")
    Rel(backend, redis, "Encola tareas de transcripción", "Celery Protocol")
    Rel(celery, redis, "Consume tareas", "Celery Protocol")
    Rel(celery, db, "Actualiza estados y guarda segmentos", "SQLAlchemy ORM")
    Rel(celery, storage, "Lee video, extrae audio y guarda chunks", "File System")
  
    Rel(celery, gpu, "Inferencia local", "CUDA")
    Rel(celery, together, "Inferencia remota (si activo)", "HTTPS API")
```

## 2. Modelo de Datos (ERD)

```mermaid
erDiagram
    Site ||--o{ Classroom : "has"
    Site ||--o{ Recording : "has"
    Classroom ||--o{ Camera : "has"
    Classroom ||--o{ CameraAssignment : "history of"
    Classroom ||--o{ Schedule : "has"
    Classroom ||--o{ Recording : "has"
    Camera ||--o{ CameraAssignment : "history of"
    Camera ||--o{ Recording : "records"
    Professor ||--o{ Schedule : "teaches"
    Course ||--o{ Schedule : "taught in"
    Recording ||--o{ TranscriptionJob : "processed by"
    Schedule ||--o{ TranscriptionJob : "associated with"
    TranscriptionJob ||--o{ TranscriptSegment : "contains"

    Site {
        string id PK
        string code
        string name
        boolean active
    }
  
    Classroom {
        string id PK
        string site_id FK
        string code
        string name
        int capacity
    }
  
    Camera {
        string id PK
        string classroom_id FK
        string code
        string serial_number
        string source_type
    }
  
    CameraAssignment {
        string id PK
        string camera_id FK
        string classroom_id FK
        datetime started_at
        datetime ended_at
    }
  
    Professor {
        string id PK
        string code
        string full_name
        string email
    }
  
    Course {
        string id PK
        string code
        string name
        json vocabulary
    }
  
    Schedule {
        string id PK
        string professor_id FK
        string course_id FK
        string classroom_id FK
        int day_of_week
        time start_time
        time end_time
    }
  
    Recording {
        string id PK
        string source_uri
        string original_name
        float duration_seconds
        int file_size_bytes
    }
  
    TranscriptionJob {
        string id PK
        string recording_id FK
        string status
        float offset_start_seconds
        float offset_end_seconds
        string provider_name
        int progress
    }
  
    TranscriptSegment {
        string id PK
        string job_id FK
        float start_seconds
        float end_seconds
        string text
        string speaker_label
    }
```

## 3. Flujo de Trabajo (Diagrama de Secuencia)

El siguiente diagrama detalla cómo se procesa una grabación, desde el recorte hasta la transcripción final.

```mermaid
sequenceDiagram
    actor User as Administrador
    participant UI as Frontend (Next.js)
    participant API as FastAPI Backend
    participant DB as PostgreSQL
    participant Worker as Celery Worker
    participant Storage as File Storage
    participant GPU as Faster-Whisper (CUDA)

    User->>UI: Sube video y selecciona Aula/Horario
    UI->>API: POST /api/recordings/upload (Multipart)
    API->>Storage: Guarda archivo de video original
    API->>DB: Crea registro en `recordings`
    API-->>UI: Devuelve metadatos del video

    User->>UI: Selecciona intervalo de tiempo (Start/End) y da click en "Transcribir"
    UI->>API: POST /api/jobs (Petición de Trabajo)
    API->>DB: Crea `TranscriptionJob` (estado: PENDING)
    API->>Worker: Encola tarea en Redis
    API-->>UI: Devuelve JobID

    Worker->>DB: Actualiza `TranscriptionJob` (estado: PROCESSING)
    Worker->>Storage: Lee video original
    Worker->>Worker: Ejecuta FFmpeg para extraer y recortar audio (16kHz)
    Worker->>Worker: Divide el audio en chunks superpuestos
  
    loop Para cada chunk
        Worker->>GPU: Transcribe chunk (Faster-Whisper)
        GPU-->>Worker: Devuelve texto y timestamps
        Worker->>DB: Guarda `TranscriptSegment`
        Worker->>DB: Actualiza `progress` del Job
    end

    Worker->>DB: Actualiza `TranscriptionJob` (estado: COMPLETED)
  
    UI->>API: GET /api/jobs/{id} (Polling)
    API-->>UI: Retorna progreso o completado
    User->>UI: Revisa transcripción y descarga TXT/JSON
```

## 4. Estrategia de Despliegue Local

Este proyecto está optimizado para funcionar localmente en hardware accesible, específicamente aprovechando una **NVIDIA RTX 3060**.

* **Motor de Inferencia:** Se utiliza `Faster-Whisper` con el modelo `large-v3`.
* **Perfil de Memoria:** Debido a que el modelo `large-v3` es pesado, se utiliza cuantización `int8_float16` que permite correr el modelo dentro de los 12GB de VRAM que tiene la RTX 3060.
* **Concurrencia:** Los workers de Celery se configuran con `concurrency=1` para evitar conflictos en la VRAM y "Out of Memory Errors".
* **Proveedor de Respaldo:** El sistema soporta usar la API de *Together AI* como fallback (cambiando `TRANSCRIPTION_PROVIDER=together` en el `.env`), lo que delega la transcripción a la nube si la GPU local está saturada o no disponible, cobrando solo por el tiempo de los chunks recortados y no por el video completo.
