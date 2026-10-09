import React from "react";
import { MODALITY_COLORS, MODALITY_LABELS } from "../format";
import type { Modality } from "../types";

/** A small colored tag naming a modality; the color is the same everywhere. */
export default function ModalityPill({ modality }: { modality: Modality }) {
  const color = MODALITY_COLORS[modality];
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 5,
        padding: "1px 8px",
        borderRadius: 999,
        border: `1px solid ${color}`,
        color,
        fontSize: 11,
        fontWeight: 600,
        whiteSpace: "nowrap",
        lineHeight: "16px",
      }}
    >
      <span style={{ width: 6, height: 6, borderRadius: 3, background: color }} />
      {MODALITY_LABELS[modality]}
    </span>
  );
}
