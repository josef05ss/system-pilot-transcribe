"use client";

import { FormEvent, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Site = { id: string; code: string; name: string; address?: string; active: boolean };
type Classroom = { id: string; site_id: string; code: string; name: string; floor?: string; capacity?: number; active: boolean };
type Camera = { id: string; classroom_id?: string; code: string; name: string; brand?: string; model?: string; serial_number?: string; source_type: string; source_uri?: string; active: boolean };
type Assignment = { id: string; camera_id: string; classroom_id: string; started_at: string; ended_at?: string; active: boolean; notes?: string };
type Professor = { id: string; code: string; full_name: string; email?: string; active: boolean };
type Course = { id: string; code: string; name: string; description?: string; vocabulary: string[]; active: boolean };
type Schedule = { id: string; professor_id: string; course_id: string; classroom_id: string; day_of_week: number; start_time: string; end_time: string; valid_from?: string; valid_until?: string; active: boolean };
type Catalogs = { sites: Site[]; classrooms: Classroom[]; cameras: Camera[]; assignments: Assignment[]; professors: Professor[]; courses: Course[]; schedules: Schedule[] };
type Recording = { id: string; site_id: string; classroom_id: string; camera_id: string; original_name: string; recording_started_at: string; duration_seconds: number; container_format?: string; video_codec?: string; audio_codec?: string; audio_sample_rate?: number; audio_channels?: number; file_size_bytes: number };
type Job = { id: string; recording_id: string; schedule_id?: string; professor_id?: string; course_id?: string; requested_by: string; class_started_at: string; class_ended_at: string; status: string; progress: number; provider_name: string; model_name: string; total_chunks: number; completed_chunks: number; processing_seconds?: number; real_time_factor?: number; device_used?: string; compute_type_used?: string; error_message?: string; queue_position?: number; created_at: string };
type Transcript = { job_id: string; status: string; automatic_text?: string; reviewed_text?: string; final_text?: string; metrics: Record<string, unknown>; segments: Array<{ id: string; start_seconds: number; end_seconds: number; text: string; speaker_label?: string }> };
type SystemInfo = { configured_device: string; configured_model: string; transcription_provider: "local" | "together"; together_ready: boolean; cuda_devices: number; gpus: Array<{ index: number; name: string; memory_mb: number; utilization_percent: number }> };

type View = "transcribe" | "settings" | "jobs";
type ConfigTab = "sites" | "classrooms" | "cameras" | "assignments" | "professors" | "courses" | "schedules";

const emptyCatalogs: Catalogs = { sites: [], classrooms: [], cameras: [], assignments: [], professors: [], courses: [], schedules: [] };
const days = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"];
const statusLabel: Record<string, string> = { PENDING: "En cola", VALIDATING: "Validando", EXTRACTING: "Recortando intervalo", CHUNKING: "Preparando audio", QUEUED_TRANSCRIPTION: "Esperando motor", TRANSCRIBING: "Transcribiendo", MERGING: "Uniendo resultados", READY_FOR_REVIEW: "Pendiente de revisión", APPROVED: "Aprobada", ERROR: "Error", CANCEL_REQUESTED: "Cancelando", CANCELLED: "Cancelada" };

function formatDuration(value?: number): string {
  if (value === undefined || Number.isNaN(value)) return "—";
  const total = Math.max(0, Math.round(value));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return `${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

function formatBytes(bytes: number): string {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index > 1 ? 2 : 0)} ${units[index]}`;
}

function localDateTime(date: Date): string {
  const offset = date.getTimezoneOffset();
  return new Date(date.getTime() - offset * 60_000).toISOString().slice(0, 16);
}

function absoluteAt(recording: Recording, offsetSeconds: number): Date {
  return new Date(new Date(recording.recording_started_at).getTime() + offsetSeconds * 1000);
}

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || "La solicitud no pudo completarse");
  return payload as T;
}

function uploadRawFile(file: File, metadata: { siteId: string; classroomId: string; cameraId: string; recordingStartedAt: string }, onProgress: (percent: number, mbps: number) => void): Promise<Recording> {
  return new Promise((resolve, reject) => {
    const params = new URLSearchParams({ site_id: metadata.siteId, classroom_id: metadata.classroomId, camera_id: metadata.cameraId, recording_started_at: metadata.recordingStartedAt });
    const xhr = new XMLHttpRequest();
    const started = performance.now();
    xhr.open("POST", `${API}/api/recordings/upload-fast?${params.toString()}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.setRequestHeader("X-File-Name", encodeURIComponent(file.name));
    xhr.setRequestHeader("X-File-Size", String(file.size));
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      const seconds = Math.max(0.001, (performance.now() - started) / 1000);
      onProgress(Math.round((event.loaded / event.total) * 100), event.loaded / 1024 / 1024 / seconds);
    };
    xhr.onerror = () => reject(new Error("La subida se interrumpió"));
    xhr.onload = () => {
      const payload = JSON.parse(xhr.responseText || "{}");
      if (xhr.status >= 200 && xhr.status < 300) resolve(payload as Recording);
      else reject(new Error(payload.detail || "No se pudo subir el video"));
    };
    xhr.send(file);
  });
}

export default function Dashboard() {
  const [view, setView] = useState<View>("transcribe");
  const [configTab, setConfigTab] = useState<ConfigTab>("sites");
  const [catalogs, setCatalogs] = useState<Catalogs>(emptyCatalogs);
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [message, setMessage] = useState("Conectando con FastAPI...");
  const [busy, setBusy] = useState(false);

  const [siteId, setSiteId] = useState("");
  const [classroomId, setClassroomId] = useState("");
  const [cameraId, setCameraId] = useState("");
  const [recordingStart, setRecordingStart] = useState(localDateTime(new Date()));
  const [file, setFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadSpeed, setUploadSpeed] = useState(0);

  const [recordingId, setRecordingId] = useState("");
  const [scheduleId, setScheduleId] = useState("");
  const [professorId, setProfessorId] = useState("");
  const [courseId, setCourseId] = useState("");
  const [requestedBy, setRequestedBy] = useState("Operador de transcripción");
  const [offsetStart, setOffsetStart] = useState(0);
  const [offsetEnd, setOffsetEnd] = useState(1800);
  const [modelName, setModelName] = useState("large-v3");
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [transcript, setTranscript] = useState<Transcript | null>(null);
  const [reviewText, setReviewText] = useState("");
  const videoRef = useRef<HTMLVideoElement>(null);

  const [siteForm, setSiteForm] = useState({ id: "", code: "", name: "", address: "" });
  const [classroomForm, setClassroomForm] = useState({ id: "", site_id: "", code: "", name: "", floor: "", capacity: "" });
  const [cameraForm, setCameraForm] = useState({ id: "", code: "", name: "", classroom_id: "", brand: "", model: "", serial_number: "", source_type: "pending", source_uri: "" });
  const [assignmentForm, setAssignmentForm] = useState({ camera_id: "", classroom_id: "", notes: "" });
  const [professorForm, setProfessorForm] = useState({ id: "", code: "", full_name: "", email: "" });
  const [courseForm, setCourseForm] = useState({ id: "", code: "", name: "", description: "", vocabulary: "" });
  const [scheduleForm, setScheduleForm] = useState({ id: "", professor_id: "", course_id: "", classroom_id: "", day_of_week: "0", start_time: "15:00", end_time: "18:00", valid_from: "", valid_until: "" });

  const activeSites = useMemo(() => catalogs.sites.filter((x) => x.active), [catalogs.sites]);
  const activeClassrooms = useMemo(() => catalogs.classrooms.filter((x) => x.active), [catalogs.classrooms]);
  const uploadClassrooms = useMemo(() => activeClassrooms.filter((x) => x.site_id === siteId), [activeClassrooms, siteId]);
  const uploadCameras = useMemo(() => catalogs.cameras.filter((x) => x.active && x.classroom_id === classroomId), [catalogs.cameras, classroomId]);
  const selectedRecording = useMemo(() => recordings.find((x) => x.id === recordingId), [recordings, recordingId]);
  const recordingSchedules = useMemo(() => catalogs.schedules.filter((x) => x.active && (!selectedRecording || x.classroom_id === selectedRecording.classroom_id)), [catalogs.schedules, selectedRecording]);

  const loadAll = useCallback(async () => {
    try {
      const [catalogData, recordingData, jobData, systemData] = await Promise.all([
        api<Catalogs>("/api/catalogs"), api<Recording[]>("/api/recordings"), api<Job[]>("/api/jobs"), api<SystemInfo>("/api/system"),
      ]);
      setCatalogs(catalogData); setRecordings(recordingData); setJobs(jobData); setSystem(systemData);
      setSiteId((v) => v || catalogData.sites.find((x) => x.active)?.id || "");
      setClassroomForm((v) => ({ ...v, site_id: v.site_id || catalogData.sites.find((x) => x.active)?.id || "" }));
      setMessage("Sistema conectado y listo.");
    } catch (error) { setMessage(error instanceof Error ? error.message : "No se pudo conectar"); }
  }, []);

  useEffect(() => { loadAll(); }, [loadAll]);
  useEffect(() => {
    const timer = window.setInterval(async () => {
      try {
        const latest = await api<Job[]>("/api/jobs");
        setJobs(latest);
        if (selectedJob) setSelectedJob(latest.find((x) => x.id === selectedJob.id) || selectedJob);
      } catch { /* conserva último estado */ }
    }, 3000);
    return () => window.clearInterval(timer);
  }, [selectedJob]);

  useEffect(() => {
    if (!uploadClassrooms.some((x) => x.id === classroomId)) setClassroomId(uploadClassrooms[0]?.id || "");
  }, [uploadClassrooms, classroomId]);
  useEffect(() => {
    if (!uploadCameras.some((x) => x.id === cameraId)) setCameraId(uploadCameras[0]?.id || "");
  }, [uploadCameras, cameraId]);
  useEffect(() => {
    if (!selectedRecording) return;
    setOffsetStart(0);
    setOffsetEnd(Math.min(selectedRecording.duration_seconds, 3 * 3600));
  }, [selectedRecording]);
  useEffect(() => {
    if (!selectedJob || !["READY_FOR_REVIEW", "APPROVED"].includes(selectedJob.status)) return;
    api<Transcript>(`/api/jobs/${selectedJob.id}/transcript`).then((data) => { setTranscript(data); setReviewText(data.reviewed_text || data.automatic_text || ""); }).catch(() => undefined);
  }, [selectedJob]);

  async function submitJson(path: string, method: "POST" | "PATCH", body: unknown) {
    setBusy(true);
    try {
      await api(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      await loadAll(); setMessage("Cambios guardados correctamente.");
    } catch (error) { setMessage(error instanceof Error ? error.message : "No se pudo guardar"); throw error; }
    finally { setBusy(false); }
  }

  async function deactivate(path: string) {
    if (!window.confirm("Se desactivará el registro, pero se conservará su historial. ¿Continuar?")) return;
    setBusy(true);
    try { await api(path, { method: "DELETE" }); await loadAll(); setMessage("Registro desactivado."); }
    catch (error) { setMessage(error instanceof Error ? error.message : "No se pudo desactivar"); }
    finally { setBusy(false); }
  }

  async function uploadRecording(event: FormEvent) {
    event.preventDefault();
    if (!file || !siteId || !classroomId || !cameraId) return setMessage("Completa sede, aula, cámara y archivo.");
    setBusy(true); setUploadProgress(0);
    try {
      const recording = await uploadRawFile(file, { siteId, classroomId, cameraId, recordingStartedAt: new Date(recordingStart).toISOString() }, (p, speed) => { setUploadProgress(p); setUploadSpeed(speed); });
      await loadAll(); setRecordingId(recording.id); setFile(null); setMessage("Video subido e inspeccionado. Ahora selecciona el intervalo.");
    } catch (error) { setMessage(error instanceof Error ? error.message : "No se pudo subir"); }
    finally { setBusy(false); }
  }

  function applySchedule(id: string) {
    setScheduleId(id);
    const schedule = catalogs.schedules.find((x) => x.id === id);
    if (!schedule || !selectedRecording) return;
    setProfessorId(schedule.professor_id); setCourseId(schedule.course_id);
    const base = new Date(selectedRecording.recording_started_at);
    const [sh, sm] = schedule.start_time.split(":").map(Number);
    const [eh, em] = schedule.end_time.split(":").map(Number);
    const start = new Date(base); start.setHours(sh, sm, 0, 0);
    const end = new Date(base); end.setHours(eh, em, 0, 0);
    const startOffset = Math.max(0, (start.getTime() - base.getTime()) / 1000);
    const endOffset = Math.min(selectedRecording.duration_seconds, (end.getTime() - base.getTime()) / 1000);
    if (endOffset > startOffset) { setOffsetStart(startOffset); setOffsetEnd(endOffset); }
  }

  function previewAt(seconds: number) {
    if (!videoRef.current) return;
    videoRef.current.currentTime = Math.max(0, seconds);
    videoRef.current.play().catch(() => undefined);
  }

  async function createJob(event: FormEvent) {
    event.preventDefault();
    if (!selectedRecording) return setMessage("Selecciona una grabación.");
    if (offsetEnd <= offsetStart) return setMessage("El fin debe ser posterior al inicio.");
    setBusy(true);
    try {
      const job = await api<Job>("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
        recording_id: selectedRecording.id, schedule_id: scheduleId || null, professor_id: professorId || null, course_id: courseId || null,
        requested_by: requestedBy, class_started_at: absoluteAt(selectedRecording, offsetStart).toISOString(), class_ended_at: absoluteAt(selectedRecording, offsetEnd).toISOString(),
        model_name: modelName, language: "es", chunk_seconds: 300, overlap_seconds: 3, priority: 5,
      }) });
      setSelectedJob(job); setTranscript(null); setView("jobs"); await loadAll(); setMessage("Trabajo creado. El recorte ocurre antes de la transcripción.");
    } catch (error) { setMessage(error instanceof Error ? error.message : "No se pudo crear el trabajo"); }
    finally { setBusy(false); }
  }

  async function saveReview(approve: boolean) {
    if (!selectedJob || !reviewText.trim()) return;
    setBusy(true);
    try {
      const data = await api<Transcript>(`/api/jobs/${selectedJob.id}/review`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reviewed_text: reviewText, approve }) });
      setTranscript(data); await loadAll(); setMessage(approve ? "Transcripción aprobada." : "Revisión guardada.");
    } catch (error) { setMessage(error instanceof Error ? error.message : "No se pudo guardar"); }
    finally { setBusy(false); }
  }

  const siteName = (id: string) => catalogs.sites.find((x) => x.id === id)?.name || "—";
  const classroomName = (id?: string) => catalogs.classrooms.find((x) => x.id === id)?.name || "Sin asignar";
  const professorName = (id: string) => catalogs.professors.find((x) => x.id === id)?.full_name || "—";
  const courseName = (id: string) => catalogs.courses.find((x) => x.id === id)?.name || "—";
  const cameraName = (id: string) => catalogs.cameras.find((x) => x.id === id)?.name || "—";

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark">T</div><div><strong>Transcriptor</strong><span>Motor híbrido empresarial</span></div></div>
        <nav>
          <button className={view === "transcribe" ? "active" : ""} onClick={() => setView("transcribe")}>Nueva transcripción</button>
          <button className={view === "jobs" ? "active" : ""} onClick={() => setView("jobs")}>Cola y resultados</button>
          <button className={view === "settings" ? "active" : ""} onClick={() => setView("settings")}>Configuración CRUD</button>
        </nav>
        <div className="provider-card">
          <span>Proveedor activo</span>
          <strong>{system?.transcription_provider === "together" ? "Together AI" : "Faster-Whisper local"}</strong>
          <small>{system?.transcription_provider === "local" ? `${system?.configured_model || "large-v3"} · CUDA RTX 3060` : "Whisper Large v3 administrado"}</small>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div><p className="eyebrow">SISTEMA REAL DE TRANSCRIPCIÓN</p><h1>{view === "transcribe" ? "Preparar y transcribir una clase" : view === "jobs" ? "Cola de procesamiento" : "Modelado y configuración"}</h1></div>
          <button className="outline" onClick={loadAll}>Actualizar datos</button>
        </header>

        {view === "transcribe" && (
          <div className="content-grid">
            <form className="card" onSubmit={uploadRecording}>
              <div className="card-title"><span>1</span><div><h2>Cargar grabación</h2><p>Subida binaria por streaming: no carga el archivo completo en RAM.</p></div></div>
              <div className="field-grid three">
                <label>Sede<select value={siteId} onChange={(e) => setSiteId(e.target.value)}>{activeSites.map((x) => <option key={x.id} value={x.id}>{x.code} · {x.name}</option>)}</select></label>
                <label>Aula<select value={classroomId} onChange={(e) => setClassroomId(e.target.value)}>{uploadClassrooms.map((x) => <option key={x.id} value={x.id}>{x.code} · {x.name}</option>)}</select></label>
                <label>Cámara<select value={cameraId} onChange={(e) => setCameraId(e.target.value)}>{uploadCameras.map((x) => <option key={x.id} value={x.id}>{x.code} · {x.name}</option>)}</select></label>
              </div>
              {uploadCameras.length === 0 && <div className="notice warning">Primero asigna una cámara activa al aula desde Configuración CRUD.</div>}
              <label>Inicio real de la grabación<input type="datetime-local" value={recordingStart} onChange={(e) => setRecordingStart(e.target.value)} /></label>
              <label className="dropzone"><input type="file" accept="video/*,audio/*,.mp4,.mkv,.avi,.mov,.ts" onChange={(e) => setFile(e.target.files?.[0] || null)} /><strong>{file?.name || "Seleccionar video o audio"}</strong><span>{file ? formatBytes(file.size) : "La inspección detectará duración, códecs y pista de audio"}</span></label>
              {uploadProgress > 0 && <div><div className="progress"><i style={{ width: `${uploadProgress}%` }} /></div><div className="progress-meta"><span>{uploadProgress}%</span><span>{uploadSpeed.toFixed(1)} MB/s</span></div></div>}
              <button className="primary" disabled={busy || !file || !cameraId}>Subir e inspeccionar</button>
            </form>

            <form className="card wide" onSubmit={createJob}>
              <div className="card-title"><span>2</span><div><h2>Editor de intervalo</h2><p>El sistema recorta este fragmento antes de usar CUDA o consumir Together AI.</p></div></div>
              <label>Grabación<select value={recordingId} onChange={(e) => setRecordingId(e.target.value)}><option value="">Seleccionar grabación</option>{recordings.map((x) => <option key={x.id} value={x.id}>{x.original_name} · {formatDuration(x.duration_seconds)}</option>)}</select></label>

              {selectedRecording && <>
                <div className="media-facts"><span>{selectedRecording.container_format || "contenedor"}</span><span>Video {selectedRecording.video_codec || "—"}</span><span>Audio {selectedRecording.audio_codec || "—"}</span><span>{formatBytes(selectedRecording.file_size_bytes)}</span></div>
                <video ref={videoRef} className="preview" controls preload="metadata" src={`${API}/api/recordings/${selectedRecording.id}/stream`} />
                <small className="muted">Si el navegador no reproduce HEVC/H.265, los controles horarios siguen funcionando y FFmpeg sí puede recortar el audio.</small>

                <div className="timeline-card">
                  <div className="timeline-head"><strong>Fragmento seleccionado</strong><span>{formatDuration(offsetEnd - offsetStart)}</span></div>
                  <label>Inicio · {formatDuration(offsetStart)} · {absoluteAt(selectedRecording, offsetStart).toLocaleString()}<input type="range" min={0} max={Math.max(1, selectedRecording.duration_seconds)} step={1} value={offsetStart} onChange={(e) => setOffsetStart(Math.min(Number(e.target.value), offsetEnd - 1))} /></label>
                  <label>Fin · {formatDuration(offsetEnd)} · {absoluteAt(selectedRecording, offsetEnd).toLocaleString()}<input type="range" min={0} max={Math.max(1, selectedRecording.duration_seconds)} step={1} value={offsetEnd} onChange={(e) => setOffsetEnd(Math.max(Number(e.target.value), offsetStart + 1))} /></label>
                  <div className="inline-actions"><button type="button" className="outline" onClick={() => previewAt(offsetStart)}>Previsualizar inicio</button><button type="button" className="outline" onClick={() => previewAt(Math.max(offsetStart, offsetEnd - 10))}>Previsualizar final</button></div>
                </div>

                <div className="field-grid two">
                  <label>Horario registrado<select value={scheduleId} onChange={(e) => applySchedule(e.target.value)}><option value="">Ajuste manual</option>{recordingSchedules.map((x) => <option key={x.id} value={x.id}>{days[x.day_of_week]} {x.start_time.slice(0,5)}–{x.end_time.slice(0,5)} · {courseName(x.course_id)}</option>)}</select></label>
                  <label>Solicitado por<input value={requestedBy} onChange={(e) => setRequestedBy(e.target.value)} /></label>
                  <label>Profesor<select value={professorId} onChange={(e) => setProfessorId(e.target.value)}><option value="">Sin asignar</option>{catalogs.professors.filter((x) => x.active).map((x) => <option key={x.id} value={x.id}>{x.full_name}</option>)}</select></label>
                  <label>Curso<select value={courseId} onChange={(e) => setCourseId(e.target.value)}><option value="">Sin asignar</option>{catalogs.courses.filter((x) => x.active).map((x) => <option key={x.id} value={x.id}>{x.code} · {x.name}</option>)}</select></label>
                  <label>Modelo<select value={modelName} disabled={system?.transcription_provider === "together"} onChange={(e) => setModelName(e.target.value)}><option value="large-v3">large-v3 · máxima calidad</option><option value="turbo">turbo · comparación rápida</option></select></label>
                  <div className="summary-box"><span>Ruta de procesamiento</span><strong>Recorte → audio FLAC → chunks → {system?.transcription_provider === "together" ? "Together API" : "Faster-Whisper CUDA"}</strong></div>
                </div>
                <button className="primary" disabled={busy}>Recortar y enviar a transcripción</button>
              </>}
            </form>
          </div>
        )}

        {view === "settings" && (
          <div>
            <div className="tabs">{(["sites","classrooms","cameras","assignments","professors","courses","schedules"] as ConfigTab[]).map((tab) => <button key={tab} className={configTab === tab ? "active" : ""} onClick={() => setConfigTab(tab)}>{({sites:"Sedes",classrooms:"Aulas",cameras:"Cámaras",assignments:"Asignaciones",professors:"Profesores",courses:"Cursos",schedules:"Horarios"} as Record<ConfigTab,string>)[tab]}</button>)}</div>

            {configTab === "sites" && <div className="crud-layout"><form className="card" onSubmit={async (e) => { e.preventDefault(); await submitJson(siteForm.id ? `/api/admin/sites/${siteForm.id}` : "/api/admin/sites", siteForm.id ? "PATCH" : "POST", { code: siteForm.code, name: siteForm.name, address: siteForm.address || null }); setSiteForm({id:"",code:"",name:"",address:""}); }}><h2>{siteForm.id ? "Editar sede" : "Nueva sede"}</h2><label>Código<input value={siteForm.code} onChange={(e)=>setSiteForm({...siteForm,code:e.target.value})} required /></label><label>Nombre<input value={siteForm.name} onChange={(e)=>setSiteForm({...siteForm,name:e.target.value})} required /></label><label>Dirección<input value={siteForm.address} onChange={(e)=>setSiteForm({...siteForm,address:e.target.value})} /></label><button className="primary" disabled={busy}>Guardar sede</button></form><CrudTable headers={["Código","Nombre","Dirección","Estado","Acciones"]}>{catalogs.sites.map((x)=><tr key={x.id}><td>{x.code}</td><td>{x.name}</td><td>{x.address||"—"}</td><td><Status active={x.active}/></td><td><button onClick={()=>setSiteForm({id:x.id,code:x.code,name:x.name,address:x.address||""})}>Editar</button>{x.active&&<button onClick={()=>deactivate(`/api/admin/sites/${x.id}`)}>Desactivar</button>}</td></tr>)}</CrudTable></div>}

            {configTab === "classrooms" && <div className="crud-layout"><form className="card" onSubmit={async(e)=>{e.preventDefault();await submitJson(classroomForm.id?`/api/admin/classrooms/${classroomForm.id}`:"/api/admin/classrooms",classroomForm.id?"PATCH":"POST",{site_id:classroomForm.site_id,code:classroomForm.code,name:classroomForm.name,floor:classroomForm.floor||null,capacity:classroomForm.capacity?Number(classroomForm.capacity):null});setClassroomForm({id:"",site_id:activeSites[0]?.id||"",code:"",name:"",floor:"",capacity:""});}}><h2>{classroomForm.id?"Editar aula":"Nueva aula"}</h2><label>Sede<select value={classroomForm.site_id} onChange={(e)=>setClassroomForm({...classroomForm,site_id:e.target.value})}>{activeSites.map((x)=><option key={x.id} value={x.id}>{x.name}</option>)}</select></label><label>Código<input value={classroomForm.code} onChange={(e)=>setClassroomForm({...classroomForm,code:e.target.value})} required/></label><label>Nombre<input value={classroomForm.name} onChange={(e)=>setClassroomForm({...classroomForm,name:e.target.value})} required/></label><div className="field-grid two"><label>Piso<input value={classroomForm.floor} onChange={(e)=>setClassroomForm({...classroomForm,floor:e.target.value})}/></label><label>Capacidad<input type="number" value={classroomForm.capacity} onChange={(e)=>setClassroomForm({...classroomForm,capacity:e.target.value})}/></label></div><button className="primary">Guardar aula</button></form><CrudTable headers={["Sede","Código","Aula","Piso","Capacidad","Estado","Acciones"]}>{catalogs.classrooms.map((x)=><tr key={x.id}><td>{siteName(x.site_id)}</td><td>{x.code}</td><td>{x.name}</td><td>{x.floor||"—"}</td><td>{x.capacity||"—"}</td><td><Status active={x.active}/></td><td><button onClick={()=>setClassroomForm({id:x.id,site_id:x.site_id,code:x.code,name:x.name,floor:x.floor||"",capacity:x.capacity?String(x.capacity):""})}>Editar</button>{x.active&&<button onClick={()=>deactivate(`/api/admin/classrooms/${x.id}`)}>Desactivar</button>}</td></tr>)}</CrudTable></div>}

            {configTab === "cameras" && <div className="crud-layout"><form className="card" onSubmit={async(e)=>{e.preventDefault();await submitJson(cameraForm.id?`/api/admin/cameras/${cameraForm.id}`:"/api/admin/cameras",cameraForm.id?"PATCH":"POST",{...cameraForm,id:undefined,classroom_id:cameraForm.classroom_id||null,brand:cameraForm.brand||null,model:cameraForm.model||null,serial_number:cameraForm.serial_number||null,source_uri:cameraForm.source_uri||null});setCameraForm({id:"",code:"",name:"",classroom_id:"",brand:"",model:"",serial_number:"",source_type:"pending",source_uri:""});}}><h2>{cameraForm.id?"Editar cámara":"Nueva cámara"}</h2><div className="field-grid two"><label>Código<input value={cameraForm.code} onChange={(e)=>setCameraForm({...cameraForm,code:e.target.value})} required/></label><label>Nombre<input value={cameraForm.name} onChange={(e)=>setCameraForm({...cameraForm,name:e.target.value})} required/></label><label>Marca<input value={cameraForm.brand} onChange={(e)=>setCameraForm({...cameraForm,brand:e.target.value})}/></label><label>Modelo<input value={cameraForm.model} onChange={(e)=>setCameraForm({...cameraForm,model:e.target.value})}/></label><label>N.° de serie<input value={cameraForm.serial_number} onChange={(e)=>setCameraForm({...cameraForm,serial_number:e.target.value})}/></label><label>Aula actual<select value={cameraForm.classroom_id} onChange={(e)=>setCameraForm({...cameraForm,classroom_id:e.target.value})}><option value="">Sin asignar</option>{activeClassrooms.map((x)=><option key={x.id} value={x.id}>{siteName(x.site_id)} · {x.name}</option>)}</select></label><label>Fuente<select value={cameraForm.source_type} onChange={(e)=>setCameraForm({...cameraForm,source_type:e.target.value})}><option value="pending">Pendiente</option><option value="local">Local</option><option value="nvr">NVR</option><option value="nas">NAS</option><option value="cloud">Nube</option><option value="external_api">API externa</option></select></label><label>URI/identificador<input value={cameraForm.source_uri} onChange={(e)=>setCameraForm({...cameraForm,source_uri:e.target.value})}/></label></div><button className="primary">Guardar cámara</button></form><CrudTable headers={["Código","Cámara","Aula actual","Fuente","Estado","Acciones"]}>{catalogs.cameras.map((x)=><tr key={x.id}><td>{x.code}</td><td>{x.name}<small>{[x.brand,x.model].filter(Boolean).join(" ")}</small></td><td>{classroomName(x.classroom_id)}</td><td>{x.source_type}</td><td><Status active={x.active}/></td><td><button onClick={()=>setCameraForm({id:x.id,code:x.code,name:x.name,classroom_id:x.classroom_id||"",brand:x.brand||"",model:x.model||"",serial_number:x.serial_number||"",source_type:x.source_type,source_uri:x.source_uri||""})}>Editar</button>{x.active&&<button onClick={()=>deactivate(`/api/admin/cameras/${x.id}`)}>Desactivar</button>}</td></tr>)}</CrudTable></div>}

            {configTab === "assignments" && <div className="crud-layout"><form className="card" onSubmit={async(e)=>{e.preventDefault();await submitJson("/api/admin/camera-assignments","POST",assignmentForm);setAssignmentForm({camera_id:"",classroom_id:"",notes:""});}}><h2>Asignar cámara a aula</h2><p className="muted">Una asignación nueva cierra automáticamente la anterior y conserva el historial.</p><label>Cámara<select value={assignmentForm.camera_id} onChange={(e)=>setAssignmentForm({...assignmentForm,camera_id:e.target.value})}><option value="">Seleccionar</option>{catalogs.cameras.filter((x)=>x.active).map((x)=><option key={x.id} value={x.id}>{x.code} · {x.name}</option>)}</select></label><label>Aula<select value={assignmentForm.classroom_id} onChange={(e)=>setAssignmentForm({...assignmentForm,classroom_id:e.target.value})}><option value="">Seleccionar</option>{activeClassrooms.map((x)=><option key={x.id} value={x.id}>{siteName(x.site_id)} · {x.name}</option>)}</select></label><label>Observación<input value={assignmentForm.notes} onChange={(e)=>setAssignmentForm({...assignmentForm,notes:e.target.value})}/></label><button className="primary">Crear asignación</button></form><CrudTable headers={["Cámara","Aula","Inicio","Fin","Estado","Acciones"]}>{catalogs.assignments.map((x)=><tr key={x.id}><td>{cameraName(x.camera_id)}</td><td>{classroomName(x.classroom_id)}</td><td>{new Date(x.started_at).toLocaleString()}</td><td>{x.ended_at?new Date(x.ended_at).toLocaleString():"—"}</td><td><Status active={x.active}/></td><td>{x.active&&<button onClick={()=>deactivate(`/api/admin/camera-assignments/${x.id}`)}>Cerrar</button>}</td></tr>)}</CrudTable></div>}

            {configTab === "professors" && <div className="crud-layout"><form className="card" onSubmit={async(e)=>{e.preventDefault();await submitJson(professorForm.id?`/api/admin/professors/${professorForm.id}`:"/api/admin/professors",professorForm.id?"PATCH":"POST",{code:professorForm.code,full_name:professorForm.full_name,email:professorForm.email||null});setProfessorForm({id:"",code:"",full_name:"",email:""});}}><h2>{professorForm.id?"Editar profesor":"Nuevo profesor"}</h2><label>Código<input value={professorForm.code} onChange={(e)=>setProfessorForm({...professorForm,code:e.target.value})} required/></label><label>Nombre completo<input value={professorForm.full_name} onChange={(e)=>setProfessorForm({...professorForm,full_name:e.target.value})} required/></label><label>Correo<input type="email" value={professorForm.email} onChange={(e)=>setProfessorForm({...professorForm,email:e.target.value})}/></label><button className="primary">Guardar profesor</button></form><CrudTable headers={["Código","Profesor","Correo","Estado","Acciones"]}>{catalogs.professors.map((x)=><tr key={x.id}><td>{x.code}</td><td>{x.full_name}</td><td>{x.email||"—"}</td><td><Status active={x.active}/></td><td><button onClick={()=>setProfessorForm({id:x.id,code:x.code,full_name:x.full_name,email:x.email||""})}>Editar</button>{x.active&&<button onClick={()=>deactivate(`/api/admin/professors/${x.id}`)}>Desactivar</button>}</td></tr>)}</CrudTable></div>}

            {configTab === "courses" && <div className="crud-layout"><form className="card" onSubmit={async(e)=>{e.preventDefault();await submitJson(courseForm.id?`/api/admin/courses/${courseForm.id}`:"/api/admin/courses",courseForm.id?"PATCH":"POST",{code:courseForm.code,name:courseForm.name,description:courseForm.description||null,vocabulary:courseForm.vocabulary.split(",").map((x)=>x.trim()).filter(Boolean)});setCourseForm({id:"",code:"",name:"",description:"",vocabulary:""});}}><h2>{courseForm.id?"Editar curso":"Nuevo curso"}</h2><label>Código<input value={courseForm.code} onChange={(e)=>setCourseForm({...courseForm,code:e.target.value})} required/></label><label>Nombre<input value={courseForm.name} onChange={(e)=>setCourseForm({...courseForm,name:e.target.value})} required/></label><label>Descripción<textarea value={courseForm.description} onChange={(e)=>setCourseForm({...courseForm,description:e.target.value})}/></label><label>Vocabulario técnico, separado por comas<input value={courseForm.vocabulary} onChange={(e)=>setCourseForm({...courseForm,vocabulary:e.target.value})}/></label><button className="primary">Guardar curso</button></form><CrudTable headers={["Código","Curso","Vocabulario","Estado","Acciones"]}>{catalogs.courses.map((x)=><tr key={x.id}><td>{x.code}</td><td>{x.name}</td><td>{x.vocabulary.join(", ")||"—"}</td><td><Status active={x.active}/></td><td><button onClick={()=>setCourseForm({id:x.id,code:x.code,name:x.name,description:x.description||"",vocabulary:x.vocabulary.join(", ")})}>Editar</button>{x.active&&<button onClick={()=>deactivate(`/api/admin/courses/${x.id}`)}>Desactivar</button>}</td></tr>)}</CrudTable></div>}

            {configTab === "schedules" && <div className="crud-layout"><form className="card" onSubmit={async(e)=>{e.preventDefault();await submitJson(scheduleForm.id?`/api/admin/schedules/${scheduleForm.id}`:"/api/admin/schedules",scheduleForm.id?"PATCH":"POST",{professor_id:scheduleForm.professor_id,course_id:scheduleForm.course_id,classroom_id:scheduleForm.classroom_id,day_of_week:Number(scheduleForm.day_of_week),start_time:scheduleForm.start_time,end_time:scheduleForm.end_time,valid_from:scheduleForm.valid_from||null,valid_until:scheduleForm.valid_until||null});setScheduleForm({id:"",professor_id:"",course_id:"",classroom_id:"",day_of_week:"0",start_time:"15:00",end_time:"18:00",valid_from:"",valid_until:""});}}><h2>{scheduleForm.id?"Editar horario":"Nuevo horario"}</h2><label>Profesor<select value={scheduleForm.professor_id} onChange={(e)=>setScheduleForm({...scheduleForm,professor_id:e.target.value})}><option value="">Seleccionar</option>{catalogs.professors.filter((x)=>x.active).map((x)=><option key={x.id} value={x.id}>{x.full_name}</option>)}</select></label><label>Curso<select value={scheduleForm.course_id} onChange={(e)=>setScheduleForm({...scheduleForm,course_id:e.target.value})}><option value="">Seleccionar</option>{catalogs.courses.filter((x)=>x.active).map((x)=><option key={x.id} value={x.id}>{x.name}</option>)}</select></label><label>Aula<select value={scheduleForm.classroom_id} onChange={(e)=>setScheduleForm({...scheduleForm,classroom_id:e.target.value})}><option value="">Seleccionar</option>{activeClassrooms.map((x)=><option key={x.id} value={x.id}>{siteName(x.site_id)} · {x.name}</option>)}</select></label><div className="field-grid three"><label>Día<select value={scheduleForm.day_of_week} onChange={(e)=>setScheduleForm({...scheduleForm,day_of_week:e.target.value})}>{days.map((x,i)=><option key={x} value={i}>{x}</option>)}</select></label><label>Inicio<input type="time" value={scheduleForm.start_time} onChange={(e)=>setScheduleForm({...scheduleForm,start_time:e.target.value})}/></label><label>Fin<input type="time" value={scheduleForm.end_time} onChange={(e)=>setScheduleForm({...scheduleForm,end_time:e.target.value})}/></label></div><button className="primary">Guardar horario</button></form><CrudTable headers={["Día","Horario","Profesor","Curso","Aula","Estado","Acciones"]}>{catalogs.schedules.map((x)=><tr key={x.id}><td>{days[x.day_of_week]}</td><td>{x.start_time.slice(0,5)}–{x.end_time.slice(0,5)}</td><td>{professorName(x.professor_id)}</td><td>{courseName(x.course_id)}</td><td>{classroomName(x.classroom_id)}</td><td><Status active={x.active}/></td><td><button onClick={()=>setScheduleForm({id:x.id,professor_id:x.professor_id,course_id:x.course_id,classroom_id:x.classroom_id,day_of_week:String(x.day_of_week),start_time:x.start_time.slice(0,5),end_time:x.end_time.slice(0,5),valid_from:x.valid_from||"",valid_until:x.valid_until||""})}>Editar</button>{x.active&&<button onClick={()=>deactivate(`/api/admin/schedules/${x.id}`)}>Desactivar</button>}</td></tr>)}</CrudTable></div>}
          </div>
        )}

        {view === "jobs" && <div className="jobs-layout"><div className="card"><div className="card-title simple"><div><h2>Solicitudes recientes</h2><p>Con una GPU se procesan por turno; con más GPU se agregan workers.</p></div></div><div className="job-list">{jobs.map((job)=><button key={job.id} className={`job ${selectedJob?.id===job.id?"selected":""}`} onClick={()=>{setSelectedJob(job);setTranscript(null);}}><div><strong>{job.requested_by}</strong><span>{new Date(job.created_at).toLocaleString()}</span></div><b>{statusLabel[job.status]||job.status}</b><div className="progress"><i style={{width:`${job.progress}%`}}/></div><small>{job.provider_name} · {job.model_name} · {job.completed_chunks}/{job.total_chunks||"—"} chunks {job.queue_position?`· cola #${job.queue_position}`:""}</small>{job.error_message&&<em>{job.error_message}</em>}</button>)}</div></div><div className="card result-card">{selectedJob?<><div className="card-title simple"><div><h2>Trabajo {selectedJob.id.slice(0,8)}</h2><p>{statusLabel[selectedJob.status]||selectedJob.status}</p></div><div className="download"><a href={`${API}/api/jobs/${selectedJob.id}/download?file_format=txt`}>TXT</a><a href={`${API}/api/jobs/${selectedJob.id}/download?file_format=json`}>JSON</a></div></div><div className="metrics"><div><span>Proveedor</span><strong>{selectedJob.provider_name}</strong></div><div><span>Dispositivo</span><strong>{selectedJob.device_used||"Pendiente"}</strong></div><div><span>RTF</span><strong>{selectedJob.real_time_factor?.toFixed(3)||"—"}</strong></div><div><span>Progreso</span><strong>{selectedJob.progress}%</strong></div></div>{["READY_FOR_REVIEW","APPROVED"].includes(selectedJob.status)?<><textarea className="transcript" value={reviewText} onChange={(e)=>setReviewText(e.target.value)} placeholder="Cargando texto..."/><div className="inline-actions"><button className="outline" onClick={()=>saveReview(false)}>Guardar revisión</button><button className="primary" onClick={()=>saveReview(true)}>Aprobar</button></div></>:<div className="processing"><div className="spinner"/><strong>{statusLabel[selectedJob.status]||selectedJob.status}</strong><p>Los workers continúan aunque cierres el navegador.</p></div>}</>:<div className="empty">Selecciona un trabajo para ver su resultado.</div>}</div></div>}

        <footer className="statusbar"><span className="status-dot" />{message}</footer>
      </section>
    </main>
  );
}

function Status({ active }: { active: boolean }) { return <span className={`state ${active ? "on" : "off"}`}>{active ? "Activo" : "Inactivo"}</span>; }
function CrudTable({ headers, children }: { headers: string[]; children: ReactNode }) { return <div className="card table-card"><div className="table-scroll"><table><thead><tr>{headers.map((x)=><th key={x}>{x}</th>)}</tr></thead><tbody>{children}</tbody></table></div></div>; }
