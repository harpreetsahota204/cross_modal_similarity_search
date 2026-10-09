import { getSampleSrc } from "@fiftyone/state";
import { Icon, IconColor, IconName, Text, TextColor, TextVariant } from "@voxel51/voodo";
import React, { useRef } from "react";
import { MUTED_BG } from "../format";
import type { MediaKind } from "../types";

type Props = {
  filepath: string;
  kind: MediaKind;
  /** Start videos muted; off for results found by their sound */
  muted: boolean;
  width: number;
  height: number;
};

// One clip plays at a time across the whole panel
let playing: HTMLMediaElement | null = null;

function onPlay(ev: React.SyntheticEvent<HTMLMediaElement>) {
  const el = ev.currentTarget;
  if (playing && playing !== el) playing.pause();
  playing = el;
}

/** Previews an image, video or audio file with the browser's own players. */
export default function Media({ filepath, kind, muted, width, height }: Props) {
  const src = getSampleSrc(filepath);
  const box: React.CSSProperties = {
    width,
    height,
    borderRadius: 6,
    overflow: "hidden",
    background: MUTED_BG,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
  };

  if (kind === "image") {
    return (
      <div style={box}>
        <img src={src} loading="lazy" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </div>
    );
  }

  if (kind === "video") {
    return (
      <div style={box}>
        <HoverVideo src={src} muted={muted} />
      </div>
    );
  }

  if (kind === "audio") {
    return (
      <div style={{ ...box, gap: 8 }}>
        <div style={{ width: 36, height: 36 }}>
          <Icon name={IconName.VolumeUp} color={IconColor.Muted} />
        </div>
        <audio src={src} controls preload="none" onPlay={onPlay} style={{ width: "92%", height: 32 }} />
      </div>
    );
  }

  return (
    <div style={box}>
      <Text variant={TextVariant.Caption} color={TextColor.Muted}>
        No preview
      </Text>
    </div>
  );
}

/** Plays silently on hover; the controls unmute or scrub it. */
function HoverVideo({ src, muted }: { src: string; muted: boolean }) {
  const ref = useRef<HTMLVideoElement>(null);
  const hovering = useRef(false);

  return (
    <video
      ref={ref}
      src={src}
      muted={muted}
      controls
      preload="metadata"
      playsInline
      loop
      onPlay={onPlay}
      onMouseEnter={() => {
        const el = ref.current;
        if (el && el.paused) {
          hovering.current = true;
          el.play().catch(() => undefined);
        }
      }}
      onMouseLeave={() => {
        const el = ref.current;
        if (el && hovering.current) el.pause();
        hovering.current = false;
      }}
      // a click on the controls means the user wants it to keep playing
      onClick={() => (hovering.current = false)}
      style={{ width: "100%", height: "100%", objectFit: "cover", background: "#000" }}
    />
  );
}