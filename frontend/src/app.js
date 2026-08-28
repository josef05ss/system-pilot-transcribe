const API_BASE = "/api";

const el = (id) => document.getElementById(id);
const dropzone = el("dropzone");
const fileInput = el("fileInput");
const dropzoneHint = el("dropzoneHint");
const scanline = el("scanline");
const resultsList = el("resultsList");
const detailOverlay = el("detailOverlay");
const detailTitle = el("detailTitle");
const detailBody = el("detailBody");

const TRACE_STEPS = ["upload", "extract", "preprocess", "route", "engine", "store"];

function setTraceStep(activeIndex, done = false) {
  TRACE_STEPS.forEach((step, i) => {
    const node = document.querySelector(`.trace__step[data-step="${step}"]`);
    node.classList.remove("active", "done");
    if (i < activeIndex || (i === activeIndex && done)) node.classList.add("done");
    else if (i === activeIndex) node.classList.add("active");
  });
}

function resetTrace() {
  TRACE_STEPS.forEach((step) => {
    document.querySelector(`.trace__step[data-step="${step}"]`).classList.remove("active", "done");
  });
}

async function checkHealth() {
  const dot = el("apiDot");
  const text = el("apiStatusText");
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error();
    dot.classList.add("ok");
    dot.classList.remove("down");
    text.textContent = "backend conectado";
  } catch {
    dot.classList.add("down");
    dot.classList.remove("ok");
    text.textContent = "backend no disponible";
  }
}

function formatFecha(iso) {
  const d = new Date(iso);
  return d.toLocaleString();
}

function badgeClasificacion(clasificacion) {
  if (clasificacion === "digital") return "🖨️ digital · tesseract";
  if (clasificacion === "manuscrita") return "✍️ manuscrita · trocr";
  if (clasificacion === "ia_vision") return "🤖 IA visión · claude";
  if (clasificacion === "baja_confianza") return "⚠️ confianza < 80%";
  return "— sin texto detectado";
}

function badgeColor(esColor) {
  if (esColor === null || esColor === undefined) return null;
  return esColor ? "🎨 color" : "◻️ blanco y negro";
}

async function cargarResultados() {
  resultsList.innerHTML = `<p class="empty-state">cargando…</p>`;
  try {
    const res = await fetch(`${API_BASE}/ocr/results`);
    const documentos = await res.json();

    if (!documentos.length) {
      resultsList.innerHTML = `<p class="empty-state">Todavía no hay documentos procesados.</p>`;
      return;
    }

    resultsList.innerHTML = "";
    documentos.forEach((doc) => {
      const card = document.createElement("div");
      card.className = "doc-card";
      card.innerHTML = `
        <div>
          <p class="doc-card__name">${doc.nombre_original}</p>
          <p class="doc-card__meta">${doc.tipo_entrada} · ${doc.extension} · ${formatFecha(doc.creado_en)}</p>
        </div>
        <span class="badge">${doc.total_resultados} imagen(es)</span>
      `;
      card.addEventListener("click", () => abrirDetalle(doc.id));
      resultsList.appendChild(card);
    });
  } catch (err) {
    resultsList.innerHTML = `<p class="empty-state">Error al cargar resultados. ¿Está el backend corriendo?</p>`;
  }
}

async function abrirDetalle(documentoId) {
  detailOverlay.classList.add("open");
  detailTitle.textContent = "Cargando…";
  detailBody.innerHTML = "";

  const res = await fetch(`${API_BASE}/ocr/results/${documentoId}`);
  if (!res.ok) {
    detailTitle.textContent = "Error";
    detailBody.innerHTML = "<p>No se pudo cargar el documento.</p>";
    return;
  }
  const doc = await res.json();
  detailTitle.textContent = doc.nombre_original;

  if (!doc.resultados.length) {
    detailBody.innerHTML = "<p>Sin resultados de OCR.</p>";
    return;
  }

  detailBody.innerHTML = doc.resultados
    .sort((a, b) => a.indice_imagen - b.indice_imagen)
    .map((r) => {
      const badgeColorHtml = badgeColor(r.es_color)
        ? `<span class="badge">${badgeColor(r.es_color)}</span>`
        : "";
      return `
      <div class="result-block">
        <div class="result-block__meta">
          <span class="badge">imagen #${r.indice_imagen + 1}</span>
          <span class="badge">${badgeClasificacion(r.clasificacion)}</span>
          <span class="badge">confianza: ${r.confianza ? r.confianza.toFixed(1) : "—"}%</span>
          ${badgeColorHtml}
        </div>
        <div class="result-block__text">${r.texto ? escapeHtml(r.texto) : "(no se detectó texto en esta imagen)"}</div>
      </div>
    `;
    })
    .join("");
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

el("closeDetail").addEventListener("click", () => detailOverlay.classList.remove("open"));
detailOverlay.addEventListener("click", (e) => {
  if (e.target === detailOverlay) detailOverlay.classList.remove("open");
});

el("refreshBtn").addEventListener("click", cargarResultados);

async function subirArchivo(file) {
  scanline.classList.add("active");
  dropzoneHint.textContent = `procesando: ${file.name}`;
  setTraceStep(0);

  const formData = new FormData();
  formData.append("archivo", file);

  // Simulación visual del avance por las etapas del pipeline mientras
  // la petición real está en curso en el backend.
  const isDoc = /\.(pdf|docx|pptx|xlsx)$/i.test(file.name);
  const secuenciaVisual = isDoc ? [0, 1, 2, 3, 4] : [0, 2, 3, 4];
  let idx = 0;
  const interval = setInterval(() => {
    if (idx < secuenciaVisual.length) {
      setTraceStep(secuenciaVisual[idx]);
      idx++;
    }
  }, 700);

  try {
    const res = await fetch(`${API_BASE}/ocr/upload`, {
      method: "POST",
      body: formData,
    });

    clearInterval(interval);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Error desconocido" }));
      throw new Error(err.detail || "Error al procesar el archivo");
    }

    setTraceStep(5, true);
    dropzoneHint.textContent = `✔ completado: ${file.name}`;
    await cargarResultados();
  } catch (err) {
    dropzoneHint.textContent = `✗ ${err.message}`;
  } finally {
    scanline.classList.remove("active");
    setTimeout(() => {
      resetTrace();
      dropzoneHint.textContent = "esperando archivo…";
    }, 3500);
  }
}

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) subirArchivo(file);
});

["dragover", "dragenter"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);

["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) subirArchivo(file);
});

checkHealth();
cargarResultados();
setInterval(checkHealth, 15000);
