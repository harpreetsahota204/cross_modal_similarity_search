import type { MediaKind, Modality } from "./types";

export const MODALITY_LABELS: Record<Modality, string> = {
  image: "Image",
  video: "Video",
  audio: "Audio",
  "video+audio": "Video + audio",
};

/** VOODO theme variables, so accents follow the App's light/dark mode. */
export const MODALITY_COLORS: Record<Modality, string> = {
  image: "var(--color-content-icon-info)",
  video: "var(--color-brand-accent)",
  audio: "var(--color-content-icon-success)",
  "video+audio": "var(--color-content-icon-warning)",
};

export const BORDER = "var(--color-content-border-default)";
export const CARD = "var(--color-content-bg-card)";
export const MUTED_BG = "var(--color-content-bg-muted)";

/** What each kind of media can be embedded as; mirrors the Python engine. */
export const QUERY_MODALITIES: Record<Exclude<MediaKind, null>, Modality[]> = {
  image: ["image"],
  video: ["video", "audio", "video+audio"],
  audio: ["audio"],
  mcap: ["video", "audio", "video+audio"],
};

const AUDIO_EXTENSIONS = [
  ".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".oga", ".ogg", ".opus",
  ".wav", ".wma",
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

export function basename(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

/** Cosine similarity as a 0-1 bar width; EG2 scores rarely leave 0.4-1. */
export function scoreFraction(score: number): number {
  return Math.max(0, Math.min(1, (score - 0.4) / 0.6));
}

/** The modality to query a hit with when pivoting from it. */
export function pivotModality(kind: MediaKind, sectionModality: Modality): Modality | null {
  if (!kind) return null;
  const options = QUERY_MODALITIES[kind];
  return options.includes(sectionModality) ? sectionModality : options[0];
}
