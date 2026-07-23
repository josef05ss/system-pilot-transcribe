# Modelo y orden de configuración

## Relaciones

```text
Sede 1 ── N Aula
Aula 1 ── N Horario
Profesor 1 ── N Horario
Curso 1 ── N Horario
Cámara N ── N Aula mediante historial de asignaciones
Cámara 1 ── N Grabación
Grabación 1 ── N Trabajo de transcripción
Trabajo 1 ── N Segmento transcrito
```

## Orden operativo recomendado

1. Sedes.
2. Aulas.
3. Cámaras.
4. Asignaciones cámara–aula.
5. Profesores.
6. Cursos.
7. Horarios.
8. Grabaciones.
9. Transcripciones.

## Cámara y fuente de video

La fuente real todavía puede ser local, NVR, NAS, nube o API externa. Por eso cada cámara almacena `source_type` y `source_uri`, pero la prueba local utiliza carga manual.

## Horario frente a intervalo

El horario propone el inicio y fin de la clase. El operador puede ajustar visualmente los marcadores antes de crear el trabajo. El intervalo final guardado en el trabajo es el que realmente se recorta y transcribe.
