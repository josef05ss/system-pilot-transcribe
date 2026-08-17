"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type ClippedVideoPlayerProps = {
  src: string;
  clipStart: number;
  clipEnd: number;
  originalDuration: number;
  seekToRelativeSeconds?: number | null;
  onSeekApplied?: () => void;
};

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}

function formatTime(value: number): string {
  const safe = Math.max(0, Number.isFinite(value) ? value : 0);
  const totalSeconds = Math.floor(safe);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  const milliseconds = Math.floor((safe - totalSeconds) * 1000);

  return `${hours.toString().padStart(2, "0")}:${minutes
    .toString()
    .padStart(2, "0")}:${seconds.toString().padStart(2, "0")}.${milliseconds
    .toString()
    .padStart(3, "0")}`;
}

export default function ClippedVideoPlayer({
  src,
  clipStart,
  clipEnd,
  originalDuration,
  seekToRelativeSeconds = null,
  onSeekApplied,
}: ClippedVideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [relativeTime, setRelativeTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);
  const [ready, setReady] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const safeStart = useMemo(
    () => clamp(Number(clipStart) || 0, 0, Math.max(0, originalDuration)),
    [clipStart, originalDuration],
  );
  const safeEnd = useMemo(
    () => clamp(Number(clipEnd) || originalDuration, safeStart, Math.max(safeStart, originalDuration)),
    [clipEnd, originalDuration, safeStart],
  );
  const clipDuration = Math.max(0, safeEnd - safeStart);

  const seekRelative = useCallback(
    (seconds: number) => {
      const video = videoRef.current;
      if (!video) return;
      const relative = clamp(seconds, 0, clipDuration);
      video.currentTime = safeStart + relative;
      setRelativeTime(relative);
    },
    [clipDuration, safeStart],
  );

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.pause();
    setPlaying(false);
    setReady(false);
    setRelativeTime(0);
    setErrorMessage("");
    video.load();
  }, [src, safeStart, safeEnd]);

  useEffect(() => {
    if (
      seekToRelativeSeconds === null ||
      seekToRelativeSeconds === undefined ||
      !ready
    ) {
      return;
    }

    seekRelative(seekToRelativeSeconds);

    const video = videoRef.current;
    if (video) {
      video
        .play()
        .then(() => setPlaying(true))
        .catch(() =>
          setErrorMessage(
            "El navegador ubicó el intervalo, pero bloqueó la reproducción automática. Pulsa Reproducir.",
          ),
        );
    }

    onSeekApplied?.();
  }, [onSeekApplied, ready, seekRelative, seekToRelativeSeconds]);

  function handleLoadedMetadata() {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = safeStart;
    setRelativeTime(0);
    setReady(true);
  }

  function handleTimeUpdate() {
    const video = videoRef.current;
    if (!video) return;

    if (video.currentTime < safeStart - 0.05) {
      video.currentTime = safeStart;
    }

    if (video.currentTime >= safeEnd - 0.02) {
      video.pause();
      video.currentTime = safeEnd;
      setPlaying(false);
      setRelativeTime(clipDuration);
      return;
    }

    setRelativeTime(clamp(video.currentTime - safeStart, 0, clipDuration));
  }

  async function togglePlayback() {
    const video = videoRef.current;
    if (!video || !ready) return;

    if (!video.paused) {
      video.pause();
      setPlaying(false);
      return;
    }

    if (video.currentTime >= safeEnd - 0.05) {
      seekRelative(0);
    } else if (video.currentTime < safeStart) {
      video.currentTime = safeStart;
    }

    try {
      await video.play();
      setPlaying(true);
    } catch {
      setErrorMessage("El navegador no pudo iniciar la reproducción.");
    }
  }

  function toggleMute() {
    const video = videoRef.current;
    if (!video) return;
    video.muted = !video.muted;
    setMuted(video.muted);
  }

  async function openFullscreen() {
    const video = videoRef.current;
    if (!video?.requestFullscreen) return;
    await video.requestFullscreen().catch(() => undefined);
  }

  return (
    <section
      aria-label="Reproductor del fragmento analizado"
      style={{
        display: "grid",
        gap: "12px",
        padding: "14px",
        border: "1px solid rgba(148, 163, 184, 0.22)",
        borderRadius: "14px",
        background: "rgba(15, 23, 42, 0.38)",
      }}
    >
      <video
        ref={videoRef}
        src={src}
        preload="metadata"
        playsInline
        muted={muted}
        onLoadedMetadata={handleLoadedMetadata}
        onTimeUpdate={handleTimeUpdate}
        onPause={() => setPlaying(false)}
        onPlay={() => setPlaying(true)}
        onEnded={() => setPlaying(false)}
        onError={() =>
          setErrorMessage(
            "No se pudo reproducir el formato en este navegador. El reporte y la descarga siguen disponibles.",
          )
        }
        style={{
          width: "100%",
          maxHeight: "430px",
          borderRadius: "10px",
          background: "#020617",
        }}
      />

      <div style={{ display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" }}>
        <button type="button" className="outline" onClick={togglePlayback} disabled={!ready}>
          {playing ? "Pausar" : "Reproducir"}
        </button>
        <button type="button" className="outline" onClick={toggleMute}>
          {muted ? "Activar audio" : "Silenciar"}
        </button>
        <button type="button" className="outline" onClick={openFullscreen}>
          Pantalla completa
        </button>
        <strong style={{ marginLeft: "auto", fontVariantNumeric: "tabular-nums" }}>
          {formatTime(relativeTime)} / {formatTime(clipDuration)}
        </strong>
      </div>

      <input
        aria-label="Posición dentro del fragmento"
        type="range"
        min={0}
        max={Math.max(0.001, clipDuration)}
        step={0.1}
        value={Math.min(relativeTime, Math.max(0.001, clipDuration))}
        onChange={(event) => seekRelative(Number(event.target.value))}
        disabled={!ready || clipDuration <= 0}
        style={{ width: "100%" }}
      />

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: "8px",
          fontSize: "0.86rem",
        }}
      >
        <span>Vista del fragmento: 00:00.000–{formatTime(clipDuration)}</span>
        <span>Posición original: {formatTime(safeStart)}–{formatTime(safeEnd)}</span>
        <span>Duración original: {formatTime(originalDuration)}</span>
      </div>

      {errorMessage && <div className="notice warning">{errorMessage}</div>}
    </section>
  );
}