import type { CSSProperties } from "react";

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
