import type { CSSProperties } from "react";
import type { Modality } from "./types";

// The App's palette variables, as FiftyOne's Similarity Search panel uses them

export const PAGE: CSSProperties = {
  padding: 16,
  height: "100%",
  overflow: "auto",
  boxSizing: "border-box",
};

export const INFO_CARD: CSSProperties = {
  borderRadius: 6,
  padding: 12,
  background: "var(--fo-palette-background-level2)",
  border: "1px solid var(--fo-palette-divider)",
};

export const ACTIVE_BOX: CSSProperties = {
  ...INFO_CARD,
  border: "1px solid var(--fo-palette-primary-main)",
};

export const POINTER: CSSProperties = { cursor: "pointer" };

export const HIGHLIGHT: CSSProperties = {
  boxShadow: "0 0 8px 2px rgba(255, 109, 4, 0.4)",
  borderRadius: 6,
};

export const TOOLTIP_TEXT: CSSProperties = {
  color: "var(--color-content-text-primary)",
};

// Picked from FiftyOne's default label color pool, skipping the brand orange
// that marks the applied search and the colors too dark for the dark theme
const MODALITY_COLORS: Record<Modality, string> = {
  image: "#0066ff",
  video: "#cc33cc",
  audio: "#009999",
  "video+audio": "#ee6600",
};

/** A result chip tinted with its modality's color, filled solid when applied. */
export const modalityChip = (modality: Modality, active: boolean): CSSProperties => {
  const color = MODALITY_COLORS[modality];
  return {
    // Hex alpha suffixes: 2e is ~18% (tint) and 80 is 50% (border)
    background: active ? color : `${color}2e`,
    border: `1px solid ${active ? color : `${color}80`}`,
    color: active ? "#fff" : "var(--color-content-text-primary)",
  };
};
