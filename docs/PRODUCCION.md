# Lista mínima antes de producción

El paquete es una base funcional para validar el procesamiento. Antes de exponerlo
a usuarios reales deben completarse estas tareas:

## Seguridad

- autenticación corporativa o JWT;
- roles y permisos;
- HTTPS y proxy inverso;
- auditoría de cargas, descargas, correcciones y eliminaciones;
- análisis de archivos y límites de tamaño;
- secretos fuera del repositorio;
- política de retención de videos y transcripciones.

## Confiabilidad

- almacenamiento compartido NAS/S3 para varios servidores;
- backups de PostgreSQL;
- Redis/RabbitMQ con persistencia y monitoreo;
- reintentos idempotentes;
- alertas por trabajos fallidos;
- limpieza programada de audios y chunks temporales.

## Escalabilidad

- un worker de transcripción por GPU cuando el proveedor sea local;
- pruebas de carga con múltiples usuarios;
- límites por usuario y prioridades;
- métricas de tiempo de espera, RTF, GPU, VRAM y almacenamiento;
- proveedor de grabaciones para NVR, nube, NAS o API externa.

## Calidad

- conjunto de audios reales representativos;
- transcripción humana de referencia;
- medición de WER;
- vocabulario por curso o dominio;
- revisión humana antes de marcar una transcripción como oficial.


## Proveedores externos

- aprobación legal y de privacidad antes de enviar audio;
- presupuesto y alertas de consumo;
- API Key en un gestor de secretos;
- límites de concurrencia y circuit breaker;
- proveedor local como respaldo cuando sea viable.
