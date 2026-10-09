import { Dropzone, Text, TextColor, TextVariant } from "@voxel51/voodo";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { mediaKindOf } from "../format";
import type { MediaKind, UploadedQuery } from "../types";

const MAX_UPLOAD_MB = 50;
const ACCEPT = "image/*,video/*,audio/*,.mcap";

// Matches the model's default audio cap, so nothing recorded is cut off
const MAX_RECORD_SECONDS = 30;

type Source = "upload" | "webcam" | "microphone";

const VIDEO_TYPES = ["video/webm;codecs=vp8,opus", "video/webm", "video/mp4"];
const AUDIO_TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];

const ICONS: Record<string, React.ReactNode> = {
  upload: (
    <>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="M17 8l-5-5-5 5M12 3v12" />
    </>
  ),
  webcam: (
    <>
      <path d="M23 7l-7 5 7 5V7z" />
      <rect x="1" y="5" width="15" height="14" rx="2" />
    </>
  ),
  microphone: (
    <>
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M19 10v1a7 7 0 0 1-14 0v-1M12 18v4" />
    </>
  ),
  camera: (
    <>
      <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
      <circle cx="12" cy="13" r="4" />
    </>
  ),
  close: <path d="M18 6L6 18M6 6l12 12" />,
};

function Glyph({ name, size = 16 }: { name: keyof typeof ICONS; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {ICONS[name]}
    </svg>
  );
}

const FRAME: React.CSSProperties = {
  border: "1px solid var(--fo-palette-divider)",
  borderRadius: 8,
  overflow: "hidden",
  background: "var(--fo-palette-background-level1)",
};

const STAGE: React.CSSProperties = {
  position: "relative",
  minHeight: 220,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  background: "#000",
};

const FILL: React.CSSProperties = { display: "block", width: "100%", maxHeight: 280, objectFit: "contain" };

const BAR: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  gap: 4,
  padding: 6,
  borderTop: "1px solid var(--fo-palette-divider)",
};

const CONTROLS: React.CSSProperties = {
  position: "absolute",
  bottom: 12,
  left: 0,
  right: 0,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  gap: 12,
};

const ROUND: React.CSSProperties = {
  width: 44,
  height: 44,
  borderRadius: "50%",
  border: "3px solid #fff",
  background: "rgba(0, 0, 0, 0.35)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  cursor: "pointer",
  color: "#fff",
  padding: 0,
};

const CLOSE: React.CSSProperties = {
  ...ROUND,
  position: "absolute",
  top: 8,
  right: 8,
  width: 28,
  height: 28,
  border: "none",
  background: "rgba(0, 0, 0, 0.6)",
};

const TIMER: React.CSSProperties = {
  position: "absolute",
  top: 10,
  left: 10,
  padding: "2px 8px",
  borderRadius: 10,
  background: "rgba(0, 0, 0, 0.6)",
  color: "#fff",
  fontSize: 12,
  fontVariantNumeric: "tabular-nums",
};

function sourceButton(active: boolean): React.CSSProperties {
  return {
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "4px 10px",
    borderRadius: 6,
    border: "none",
    cursor: "pointer",
    fontSize: 13,
    background: active ? "var(--fo-palette-background-level3)" : "transparent",
    color: active ? "var(--fo-palette-primary-main)" : "var(--fo-palette-text-secondary)",
  };
}

function stamp(): string {
  return new Date().toISOString().slice(11, 19).replace(/:/g, "");
}

function readAsDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

function describeError(e: unknown): string {
  const name = (e as { name?: string })?.name;
  if (name === "NotAllowedError") return "Access was blocked. Allow the camera and microphone in the browser's site settings";
  if (name === "NotFoundError") return "No camera or microphone was found";
  if (name === "NotReadableError") return "The device is in use by another app";
  return String((e as Error)?.message ?? e);
}

async function openStream(source: Source): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("The camera and microphone need the App on https or localhost");
  }
  if (source === "microphone") return navigator.mediaDevices.getUserMedia({ audio: true });
  try {
    return await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  } catch (e) {
    // Webcams without a microphone still record video
    if ((e as { name?: string })?.name !== "NotFoundError") throw e;
    return navigator.mediaDevices.getUserMedia({ video: true });
  }
}

/** Draws the microphone's live waveform. */
function Waveform({ stream }: { stream: MediaStream }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const audio = new AudioContext();
    const analyser = audio.createAnalyser();
    analyser.fftSize = 1024;
    audio.createMediaStreamSource(stream).connect(analyser);
    const samples = new Uint8Array(analyser.fftSize);
    let frame = 0;
    const draw = () => {
      const canvas = canvasRef.current;
      const g = canvas?.getContext("2d");
      if (canvas && g) {
        analyser.getByteTimeDomainData(samples);
        g.clearRect(0, 0, canvas.width, canvas.height);
        g.strokeStyle = "#ff6d04";
        g.lineWidth = 2;
        g.beginPath();
        samples.forEach((v, i) => {
          const x = (i / (samples.length - 1)) * canvas.width;
          const y = (v / 255) * canvas.height;
          if (i === 0) g.moveTo(x, y);
          else g.lineTo(x, y);
        });
        g.stroke();
      }
      frame = requestAnimationFrame(draw);
    };
    draw();
    return () => {
      cancelAnimationFrame(frame);
      audio.close();
    };
  }, [stream]);

  return <canvas ref={canvasRef} width={600} height={160} style={{ width: "100%", height: 160 }} />;
}

type Props = {
  value: UploadedQuery | null;
  onChange: (value: UploadedQuery | null) => void;
};

/** One media box, like Gradio's: upload a file, or record from the webcam
 * or microphone. */
export default function MediaInput({ value, onChange }: Props) {
  const [source, setSource] = useState<Source>("upload");
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [attempt, setAttempt] = useState(0);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const live = !value && source !== "upload";

  useEffect(() => {
    if (!live) return;
    let opened: MediaStream | null = null;
    let cancelled = false;
    setError(null);
    openStream(source)
      .then((s) => {
        if (cancelled) s.getTracks().forEach((t) => t.stop());
        else setStream((opened = s));
      })
      .catch((e) => !cancelled && setError(describeError(e)));
    return () => {
      cancelled = true;
      const recorder = recorderRef.current;
      if (recorder?.state === "recording") {
        recorder.onstop = null;
        recorder.stop();
      }
      setRecording(false);
      opened?.getTracks().forEach((t) => t.stop());
      setStream(null);
    };
  }, [live, source, attempt]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.srcObject = stream;
  }, [stream]);

  useEffect(() => {
    if (!recording) return;
    const started = Date.now();
    const timer = window.setInterval(() => {
      const elapsed = Math.floor((Date.now() - started) / 1000);
      setSeconds(elapsed);
      if (elapsed >= MAX_RECORD_SECONDS) recorderRef.current?.stop();
    }, 250);
    return () => window.clearInterval(timer);
  }, [recording]);

  const onFiles = (files: File[]) => {
    const file = files[0];
    if (!file) return;
    const kind = mediaKindOf(file.name);
    if (!kind) {
      setError(`Can't embed ${file.name}: use an image, video, audio or .mcap file`);
      return;
    }
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setError(`${file.name} is over the ${MAX_UPLOAD_MB} MB limit`);
      return;
    }
    readAsDataUrl(file)
      .then((data) => {
        setError(null);
        onChange({ name: file.name, data, kind });
      })
      .catch(() => setError(`Failed to read ${file.name}`));
  };

  const startRecording = useCallback(() => {
    if (!stream) return;
    const kind: MediaKind = source === "microphone" ? "audio" : "video";
    const mimeType = (kind === "audio" ? AUDIO_TYPES : VIDEO_TYPES).find((t) => MediaRecorder.isTypeSupported(t));
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    const chunks: Blob[] = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size) chunks.push(e.data);
    };
    recorder.onstop = async () => {
      setRecording(false);
      const type = recorder.mimeType || mimeType || "";
      const mp4 = type.includes("mp4");
      const ext = kind === "audio" ? (mp4 ? ".m4a" : ".weba") : mp4 ? ".mp4" : ".webm";
      const blob = new Blob(chunks, { type });
      if (!blob.size) {
        setError("Nothing was recorded");
        return;
      }
      onChange({ name: `recording-${stamp()}${ext}`, data: await readAsDataUrl(blob), kind });
    };
    recorderRef.current = recorder;
    setSeconds(0);
    setRecording(true);
    recorder.start(250);
  }, [stream, source, onChange]);

  const takePhoto = () => {
    const video = videoRef.current;
    if (!video?.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    onChange({ name: `photo-${stamp()}.png`, data: canvas.toDataURL("image/png"), kind: "image" });
  };

  const recordButton = recording ? (
    <button type="button" title="Stop" style={ROUND} onClick={() => recorderRef.current?.stop()}>
      <span style={{ width: 14, height: 14, borderRadius: 3, background: "#e5484d" }} />
    </button>
  ) : (
    <button type="button" title="Record" style={ROUND} onClick={startRecording} disabled={!stream}>
      <span style={{ width: 18, height: 18, borderRadius: "50%", background: "#e5484d" }} />
    </button>
  );

  let stage: React.ReactNode;
  if (value) {
    stage = (
      <>
        {value.kind === "image" ? (
          <img src={value.data} alt={value.name} style={FILL} />
        ) : value.kind === "video" ? (
          <video src={value.data} controls preload="metadata" style={FILL} />
        ) : value.kind === "audio" ? (
          <audio src={value.data} controls preload="metadata" style={{ width: "90%" }} />
        ) : (
          <Text variant={TextVariant.BodySecondary} color={TextColor.Muted}>
            No preview for MCAP files
          </Text>
        )}
        <button type="button" title="Clear" style={CLOSE} onClick={() => onChange(null)}>
          <Glyph name="close" size={14} />
        </button>
      </>
    );
  } else if (source === "upload") {
    stage = (
      <div style={{ width: "100%", padding: 8 }}>
        <Dropzone
          title="Drop an image, video or audio file"
          description={`Up to ${MAX_UPLOAD_MB} MB; the file isn't added to the dataset`}
          accept={ACCEPT}
          onFiles={onFiles}
        />
      </div>
    );
  } else if (error) {
    stage = (
      <div style={{ padding: 16, textAlign: "center", color: "#ccc", fontSize: 13 }}>
        <div style={{ marginBottom: 10 }}>{error}</div>
        <button type="button" style={sourceButton(true)} onClick={() => setAttempt((a) => a + 1)}>
          Try again
        </button>
      </div>
    );
  } else {
    stage = (
      <>
        {source === "webcam" ? (
          <video ref={videoRef} autoPlay muted playsInline style={{ ...FILL, transform: "scaleX(-1)" }} />
        ) : stream ? (
          <Waveform stream={stream} />
        ) : null}
        {!stream && (
          <span style={{ position: "absolute", color: "#aaa", fontSize: 13 }}>
            Waiting for the {source === "webcam" ? "camera" : "microphone"}…
          </span>
        )}
        {recording && (
          <span style={TIMER}>
            <span style={{ color: "#e5484d" }}>●</span> {Math.floor(seconds / 60)}:
            {String(seconds % 60).padStart(2, "0")} / 0:{MAX_RECORD_SECONDS}
          </span>
        )}
        {stream && (
          <div style={CONTROLS}>
            {recordButton}
            {source === "webcam" && !recording && (
              <button type="button" title="Take a photo" style={ROUND} onClick={takePhoto}>
                <Glyph name="camera" size={18} />
              </button>
            )}
          </div>
        )}
      </>
    );
  }

  return (
    <div>
      <input
        ref={fileRef}
        type="file"
        accept={ACCEPT}
        style={{ display: "none" }}
        onChange={(e) => {
          onFiles(Array.from(e.target.files ?? []));
          e.target.value = "";
        }}
      />
      <div style={FRAME}>
        <div style={{ ...STAGE, background: source === "upload" && !value ? "transparent" : STAGE.background }}>
          {stage}
        </div>
        <div style={BAR}>
          {value ? (
            <Text variant={TextVariant.BodySecondary} color={TextColor.Secondary}>
              {value.name}
            </Text>
          ) : (
            (["upload", "webcam", "microphone"] as Source[]).map((s) => (
              <button
                key={s}
                type="button"
                title={s === "upload" ? "Upload a file" : s === "webcam" ? "Record from the webcam" : "Record from the microphone"}
                style={sourceButton(source === s)}
                onClick={() => {
                  setError(null);
                  setSource(s);
                  if (s === "upload") fileRef.current?.click();
                }}
                disabled={recording}
              >
                <Glyph name={s} />
                {s === "upload" ? "Upload" : s === "webcam" ? "Webcam" : "Microphone"}
              </button>
            ))
          )}
        </div>
      </div>
      {error && source === "upload" && !value && (
        <Text variant={TextVariant.BodySecondary} color={TextColor.Failure}>
          {error}
        </Text>
      )}
    </div>
  );
}
