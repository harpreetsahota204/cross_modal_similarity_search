import themeCss from "@voxel51/voodo/theme.css?inline";

const STYLE_ID = "cross-modal-retrieval-voodo-theme";

export function ensureTheme() {
  if (typeof document === "undefined" || document.getElementById(STYLE_ID)) {
    return;
  }
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = themeCss;
  document.head.appendChild(style);
}
