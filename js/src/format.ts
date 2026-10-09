import type { MediaKind, Modality, Run } from "./types";

export const MIDDLE_DOT = "\u00B7";

export const MODALITY_LABELS: Record<Modality, string> = {
  image: "Image",
  video: "Video",
  audio: "Audio",
  "video+audio": "Video + audio",
};

/** What each kind of media can be embedded as; mirrors the Python engine. */
export const QUERY_MODALITIES: Record<Exclude<MediaKind, null>, Modality[]> = {
  image: ["image"],
  video: ["video", "audio", "video+audio"],
  audio: ["audio"],
  mcap: ["video", "audio", "video+audio"],
};

const AUDIO_EXTENSIONS = [
  ".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".oga", ".ogg", ".opus",
  ".wav", ".weba", ".wma",
];
const VIDEO_EXTENSIONS = [".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"];
const IMAGE_EXTENSIONS = [".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"];

export function mediaKindOf(name: string): MediaKind {
  const dot = name.lastIndexOf(".");
  const ext = dot >= 0 ? name.slice(dot).toLowerCase() : "";
  if (AUDIO_EXTENSIONS.includes(ext)) return "audio";
  if (VIDEO_EXTENSIONS.includes(ext)) return "video";
  if (IMAGE_EXTENSIONS.includes(ext)) return "image";
  if (ext === ".mcap") return "mcap";
  return null;
}

/** One line saying what a search was asked with. */
export function formatQuery(run: Run): string {
  if (run.query_type === "text") return `"${run.query ?? ""}"`;
  if (run.query_label) return run.query_label;
  if (Array.isArray(run.query)) {
    return `${run.query.length} selected ${run.query.length === 1 ? "sample" : "samples"}`;
  }
  return run.query ?? "";
}

/** "just now", "5 min ago", "3 h ago", then the date. */
export function formatTime(iso: string, now: number = Date.now()): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const minutes = Math.floor((now - then) / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  return new Date(iso).toLocaleDateString();
}
