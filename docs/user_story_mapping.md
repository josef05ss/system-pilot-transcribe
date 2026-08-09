# User Story Mapping (USM) - Sistema de Transcripción Híbrido

Basado en la arquitectura del proyecto, el comportamiento del sistema se puede dividir lógicamente en **dos grandes sistemas** de negocio complementarios. A continuación se presenta el User Story Mapping para ambos, estructurado por Actividades (Epics), Pasos (Steps) y Historias de Usuario (Backlog).

## 1. Sistema de Gestión (CRUD e Infraestructura)

Este sistema engloba la administración de todos los recursos (sedes, aulas, hardware) y la estructura académica (profesores, cursos, horarios). Es un pre-requisito funcional para el motor de IA.

| Persona | Actividad (Epic)                    | Paso (Step)            | Historia de Usuario                                                                                                              | Prioridad   |
| ------- | ----------------------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| Admin   | **Gestión de Sedes y Aulas** | Crear Sede             | Como administrador, quiero registrar sedes (campus) para organizar físicamente las aulas.                                       | Media (MVP) |
| Admin   |                                     | Administrar Aulas      | Como administrador, quiero agregar aulas y definir su capacidad y ubicación.                                                    | Alta (MVP)  |
| Admin   |                                     | Asignar Hardware       | Como administrador, quiero registrar cámaras (fuentes de video) y asociarlas a aulas específicas.                              | Media       |
| Admin   | **Gestión Académica**       | Administrar Cursos     | Como administrador, quiero crear cursos e incluir un diccionario de vocabulario para mejorar la precisión de la transcripción. | Media(MVP)  |
| Admin   |                                     | Gestión de Profesores | Como administrador, quiero registrar perfiles de profesores y su información de contacto.                                       | Media(MVP)  |
| Admin   |                                     | Programar Horarios     | Como administrador, quiero vincular un profesor, un curso y un aula en días y horas específicas (`Schedule`).                | Media(MVP)  |

---

## 2. Sistema de Transcripción (Carga y Procesamiento)

Este sistema maneja el ciclo de vida del video: subida, extracción de audio, encolamiento de tareas (Celery/Redis), inferencia de IA (Together AI) y edición post-procesado.

| Persona | Actividad (Epic)                  | Paso (Step)              | Historia de Usuario                                                                                                           | Prioridad   |
| ------- | --------------------------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | ----------- |
| Admin   | **Gestión de Grabaciones** | Subir Archivo Original   | Como administrador, quiero subir un video o audio pesado al almacenamiento del servidor.                                      | Alta (MVP)  |
| Admin   |                                   | Vincular al Horario      | Como administrador, quiero asociar la grabación subida a la clase de un profesor específico.                                | Alta (MVP)  |
| Admin   | **Procesamiento de IA**     | Solicitar Transcripción | Como administrador, quiero seleccionar un rango de tiempo del video y encolar el trabajo de transcripción.                   | Alta (MVP)  |
| Sistema |                                   | Usar Motor               | Como sistema, quiero usar Together AI para crear las transcripciones.                                                         | Alta (MVP) |
| Admin   |                                   | Monitorear Progreso      | Como administrador, quiero visualizar el progreso de mi tarea en tiempo real a medida que los workers procesan los*chunks*. | Alta (MVP)  |
| Admin   | **Revisión y Edición**    | Visualizar Segmentos     | Como administrador, quiero ver la transcripción dividida en segmentos con*timestamps* y etiqueta de hablante.              | Alta (MVP)  |
| Admin   |                                   | Corrección Manual       | Como administrador, quiero editar los textos de los segmentos en caso de que la IA haya cometido errores.                     | Alta (MVP)  |
| Admin   |                                   | Exportar Resultados      | Como administrador, quiero descargar la transcripción final en formatos útiles (JSON, TXT, SRT) para su distribución.      | Alta (MVP)  |
