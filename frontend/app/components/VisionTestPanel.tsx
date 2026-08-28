"use client";

import {
  ChangeEvent,
  useEffect,
  useMemo,
  useState,
} from "react";

const VISION_API =
  process.env.NEXT_PUBLIC_VISION_API_URL || "http://localhost:8100";

type Mode = "image" | "document";

type ProviderStatus = {
  name: string;
  available: boolean;
  device?: string | null;
  model?: string | null;
  detail?: string | null;
};

type ImageResult = {
  image_number: number;
  source: string;
  page?: number | null;
  sheet?: string | null;
  cell?: string | null;
  media_name?: string | null;
  width?: number | null;
  height?: number | null;
  text: string;
  line_count: number;
  average_confidence?: number | null;
  processing_seconds?: number | null;
  provider?: string;
  error?: string;
};

type VisionResponse = {
  status: string;
  provider: string;
  input_type: string;
  filename: string;
  images_found: number;
  images_processed: number;
  processing_seconds: number;
  full_text: string;
  images: ImageResult[];
  warnings?: string[];
};

type BenchmarkItem = {
  provider: string;
  status: string;
  text?: string;
  processing_seconds?: number | null;
  wall_seconds?: number | null;
  cer?: number | null;
  wer?: number | null;
  device?: string | null;
  model?: string | null;
  detail?: string | null;
  images?: ImageResult[];
  full_text?: string;
};

type BenchmarkResponse = {
  status: string;
  mode: string;
  filename: string;
  processing_seconds: number;
  providers_successful?: number;
  ground_truth_provided?: boolean;
  best_cer?: {
    provider: string;
    cer: number;
  } | null;
  fastest?: {
    provider: string;
    wall_seconds: number;
  } | null;
  images_found?: number;
  images_benchmarked?: number;
  max_images?: number;
  results: BenchmarkItem[];
};

function fmtTime(seconds?: number | null) {
  if (seconds === undefined || seconds === null) return "—";
  return `${seconds.toFixed(3)} s`;
}

function fmtRate(value?: number | null) {
  if (value === undefined || value === null) return "—";
  return `${(value * 100).toFixed(2)}%`;
}

function locationLabel(item: ImageResult) {
  const parts: string[] = [];
  if (item.page) parts.push(`Página ${item.page}`);
  if (item.sheet) parts.push(`Hoja ${item.sheet}`);
  if (item.cell) parts.push(`Celda ${item.cell}`);
  if (item.media_name) parts.push(item.media_name);
  return parts.length ? parts.join(" · ") : item.source;
}

export default function VisionTestPanel() {
  const [mode, setMode] = useState<Mode>("image");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<VisionResponse | null>(null);
  const [benchmark, setBenchmark] =
    useState<BenchmarkResponse | null>(null);
  const [providers, setProviders] = useState<ProviderStatus[]>([]);
  const [provider, setProvider] = useState("paddle");
  const [groundTruth, setGroundTruth] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetch(`${VISION_API}/api/v1/vision/providers`)
      .then((response) => response.json())
      .then((payload) => {
        const list = Array.isArray(payload.providers)
          ? payload.providers
          : [];
        setProviders(list);

        const preferred =
          list.find(
            (item: ProviderStatus) =>
              item.name === payload.default && item.available
          ) ||
          list.find((item: ProviderStatus) => item.available);

        if (preferred) setProvider(preferred.name);
      })
      .catch(() => {
        setProviders([
          {
            name: "paddle",
            available: true,
            device: "cpu",
            model: "PaddleOCR",
          },
        ]);
      });
  }, []);

  const previewUrl = useMemo(() => {
    if (mode !== "image" || !file) return null;
    return URL.createObjectURL(file);
  }, [file, mode]);

  function resetOutput() {
    setResult(null);
    setBenchmark(null);
    setMessage("");
  }

  function onFile(event: ChangeEvent<HTMLInputElement>) {
    setFile(event.target.files?.[0] || null);
    resetOutput();
  }

  async function runSingle() {
    if (!file) {
      setMessage("Selecciona un archivo primero.");
      return;
    }

    setBusy(true);
    resetOutput();
    setMessage(`Procesando con ${provider}...`);

    try {
      const form = new FormData();
      form.append("file", file);

      const endpoint =
        mode === "image"
          ? `/api/v1/vision/image/transcribe?provider=${encodeURIComponent(
              provider
            )}`
          : `/api/v1/vision/document/transcribe-images?provider=${encodeURIComponent(
              provider
            )}`;

      const response = await fetch(`${VISION_API}${endpoint}`, {
        method: "POST",
        body: form,
      });

      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          payload.detail ||
            payload.message ||
            "No se pudo procesar el archivo."
        );
      }

      setResult(payload as VisionResponse);
      setMessage(`Finalizado con ${provider}.`);
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "Error durante el OCR."
      );
    } finally {
      setBusy(false);
    }
  }

  async function runBenchmark() {
    if (!file) {
      setMessage("Selecciona un archivo primero.");
      return;
    }

    setBusy(true);
    resetOutput();
    setMessage(
      "Ejecutando benchmark secuencial. Los modelos no compiten entre sí."
    );

    try {
      const form = new FormData();
      form.append("file", file);
      if (mode === "image") {
        form.append("ground_truth", groundTruth);
      }

      const endpoint =
        mode === "image"
          ? "/api/v1/vision/image/benchmark"
          : "/api/v1/vision/document/benchmark-images";

      const response = await fetch(`${VISION_API}${endpoint}`, {
        method: "POST",
        body: form,
      });

      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          payload.detail ||
            payload.message ||
            "No se pudo ejecutar el benchmark."
        );
      }

      setBenchmark(payload as BenchmarkResponse);
      setMessage("Benchmark finalizado.");
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Error durante el benchmark."
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="content-grid">
      <div className="card">
        <div className="card-title">
          <span>1</span>
          <div>
            <h2>Prueba de OCR y benchmarking</h2>
            <p>
              Interfaz temporal para validar los endpoints antes de integrar el
              módulo al B-learning.
            </p>
          </div>
        </div>

        <div className="tabs">
          <button
            type="button"
            className={mode === "image" ? "active" : ""}
            onClick={() => {
              setMode("image");
              setFile(null);
              resetOutput();
            }}
          >
            Imagen de tarea
          </button>
          <button
            type="button"
            className={mode === "document" ? "active" : ""}
            onClick={() => {
              setMode("document");
              setFile(null);
              resetOutput();
            }}
          >
            PDF / Word / Excel
          </button>
        </div>

        <label>
          Modelo / proveedor
          <select
            value={provider}
            onChange={(event) => setProvider(event.target.value)}
          >
            {providers.map((item) => (
              <option
                key={item.name}
                value={item.name}
                disabled={!item.available}
              >
                {item.name}
                {item.available
                  ? ` · ${item.device || "auto"}`
                  : " · no disponible"}
              </option>
            ))}
          </select>
        </label>

        <div className="notice">
          {providers.map((item) => (
            <div key={item.name}>
              <strong>{item.name}</strong>:{" "}
              {item.available ? "disponible" : "no disponible"}
              {item.model ? ` · ${item.model}` : ""}
              {item.detail ? ` · ${item.detail}` : ""}
            </div>
          ))}
        </div>

        {mode === "image" ? (
          <>
            <div className="notice">
              Simula el botón que verá el profesor para convertir a texto la
              imagen de una tarea.
            </div>
            <label className="dropzone">
              <input
                type="file"
                accept=".png,.jpg,.jpeg,.webp,.bmp"
                onChange={onFile}
              />
              <strong>{file?.name || "Seleccionar imagen de tarea"}</strong>
              <span>PNG, JPG, JPEG, WEBP o BMP</span>
            </label>

            <label>
              Texto real para medir CER/WER (opcional)
              <textarea
                rows={5}
                value={groundTruth}
                onChange={(event) => setGroundTruth(event.target.value)}
                placeholder="Pega aquí la transcripción correcta de la imagen para medir precisión."
              />
            </label>
          </>
        ) : (
          <>
            <div className="notice">
              El gateway extrae las imágenes una sola vez y luego puede
              enviarlas al modelo seleccionado o a todos para comparar.
            </div>
            <label className="dropzone">
              <input
                type="file"
                accept=".pdf,.docx,.xlsx"
                onChange={onFile}
              />
              <strong>{file?.name || "Seleccionar documento"}</strong>
              <span>PDF, DOCX o XLSX</span>
            </label>
          </>
        )}

        {previewUrl && (
          <img
            src={previewUrl}
            alt="Vista previa"
            style={{
              maxWidth: "100%",
              maxHeight: 420,
              objectFit: "contain",
              borderRadius: 10,
              marginTop: 16,
            }}
          />
        )}

        <div className="inline-actions">
          <button
            type="button"
            className="primary"
            disabled={busy || !file}
            onClick={runSingle}
          >
            {busy ? "Procesando..." : `Transcribir con ${provider}`}
          </button>

          <button
            type="button"
            className="outline"
            disabled={busy || !file}
            onClick={runBenchmark}
          >
            Benchmark con todos
          </button>
        </div>

        {message && <div className="notice">{message}</div>}
      </div>

      <div className="card wide">
        <div className="card-title">
          <span>2</span>
          <div>
            <h2>Resultado</h2>
            <p>
              PaddleOCR, Docling, Qwen2.5-VL y Surya se comparan detrás del
              mismo endpoint del gateway.
            </p>
          </div>
        </div>

        {!result && !benchmark && (
          <div className="notice">
            Todavía no hay resultado.
          </div>
        )}

        {result && (
          <>
            <div className="metrics">
              <div>
                <span>Proveedor</span>
                <strong>{result.provider}</strong>
              </div>
              <div>
                <span>Imágenes</span>
                <strong>{result.images_processed}</strong>
              </div>
              <div>
                <span>Tiempo total</span>
                <strong>{fmtTime(result.processing_seconds)}</strong>
              </div>
            </div>

            {result.warnings?.map((warning, index) => (
              <div className="notice warning" key={index}>
                {warning}
              </div>
            ))}

            <div className="job-list">
              {result.images.map((item) => (
                <div className="job selected" key={item.image_number}>
                  <div>
                    <strong>Imagen {item.image_number}</strong>
                    <span>{locationLabel(item)}</span>
                  </div>
                  <small>
                    {item.line_count} líneas ·{" "}
                    {fmtTime(item.processing_seconds)}
                  </small>
                  <textarea
                    readOnly
                    value={
                      item.text ||
                      item.error ||
                      "(No se reconoció texto)"
                    }
                    rows={8}
                  />
                </div>
              ))}
            </div>

            <label>
              Texto consolidado
              <textarea readOnly value={result.full_text} rows={16} />
            </label>
          </>
        )}

        {benchmark && (
          <>
            <div className="metrics">
              <div>
                <span>Tiempo benchmark</span>
                <strong>{fmtTime(benchmark.processing_seconds)}</strong>
              </div>
              <div>
                <span>Mejor CER</span>
                <strong>
                  {benchmark.best_cer
                    ? `${benchmark.best_cer.provider} · ${fmtRate(
                        benchmark.best_cer.cer
                      )}`
                    : "Sin ground truth"}
                </strong>
              </div>
              <div>
                <span>Más rápido</span>
                <strong>
                  {benchmark.fastest
                    ? `${benchmark.fastest.provider} · ${fmtTime(
                        benchmark.fastest.wall_seconds
                      )}`
                    : "—"}
                </strong>
              </div>
            </div>

            <div className="job-list">
              {benchmark.results.map((item) => (
                <div className="job selected" key={item.provider}>
                  <div>
                    <strong>{item.provider}</strong>
                    <span>
                      Estado: {item.status}
                      {item.device ? ` · ${item.device}` : ""}
                      {item.model ? ` · ${item.model}` : ""}
                    </span>
                  </div>

                  <small>
                    Tiempo:{" "}
                    {fmtTime(item.wall_seconds ?? item.processing_seconds)}
                    {" · "}CER: {fmtRate(item.cer)}
                    {" · "}WER: {fmtRate(item.wer)}
                  </small>

                  {mode === "image" ? (
                    <textarea
                      readOnly
                      value={
                        item.text ||
                        item.detail ||
                        "(Sin resultado)"
                      }
                      rows={10}
                    />
                  ) : (
                    <>
                      {item.images?.map((image) => (
                        <div key={image.image_number}>
                          <strong>Imagen {image.image_number}</strong>
                          <span>{locationLabel(image)}</span>
                          <textarea
                            readOnly
                            value={
                              image.text ||
                              image.error ||
                              "(Sin resultado)"
                            }
                            rows={6}
                          />
                        </div>
                      ))}
                    </>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
