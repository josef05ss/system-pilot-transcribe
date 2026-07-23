# API principal

## Administración

- `GET/POST /api/admin/sites`
- `PATCH/DELETE /api/admin/sites/{id}`
- `GET/POST /api/admin/classrooms`
- `PATCH/DELETE /api/admin/classrooms/{id}`
- `GET/POST /api/admin/cameras`
- `PATCH/DELETE /api/admin/cameras/{id}`
- `GET/POST /api/admin/camera-assignments`
- `DELETE /api/admin/camera-assignments/{id}`
- `GET/POST /api/admin/professors`
- `PATCH/DELETE /api/admin/professors/{id}`
- `GET/POST /api/admin/courses`
- `PATCH/DELETE /api/admin/courses/{id}`
- `GET/POST /api/admin/schedules`
- `PATCH/DELETE /api/admin/schedules/{id}`

## Grabaciones

- `POST /api/recordings/upload-fast`
- `POST /api/recordings/upload`
- `GET /api/recordings`
- `GET /api/recordings/{id}/stream`

## Trabajos

- `POST /api/jobs`
- `GET /api/jobs`
- `GET /api/jobs/{id}`
- `GET /api/jobs/{id}/transcript`
- `PATCH /api/jobs/{id}/review`
- `POST /api/jobs/{id}/cancel`
- `GET /api/jobs/{id}/download`

Swagger expone todos los contratos en `http://localhost:8000/docs`.
