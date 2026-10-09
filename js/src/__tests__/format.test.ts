import { describe, expect, it } from "vitest";
import { basename, mediaKindOf, pivotModality, scoreFraction } from "../format";

describe("mediaKindOf", () => {
  it("reads the kind from the extension", () => {
    expect(mediaKindOf("a/b/clip.MP4")).toBe("video");
    expect(mediaKindOf("song.wav")).toBe("audio");
    expect(mediaKindOf("photo.jpeg")).toBe("image");
    expect(mediaKindOf("drive.mcap")).toBe("mcap");
    expect(mediaKindOf("notes.txt")).toBeNull();
    expect(mediaKindOf("noextension")).toBeNull();
  });
});

describe("pivotModality", () => {
  it("keeps the section's modality when the media supports it", () => {
    expect(pivotModality("video", "audio")).toBe("audio");
    expect(pivotModality("video", "video+audio")).toBe("video+audio");
  });

  it("falls back to what the media can be embedded as", () => {
    expect(pivotModality("image", "audio")).toBe("image");
    expect(pivotModality("audio", "video")).toBe("audio");
    expect(pivotModality(null, "video")).toBeNull();
  });
});

describe("scoreFraction", () => {
  it("maps the useful score range to 0-1 and clamps", () => {
    expect(scoreFraction(0.4)).toBe(0);
    expect(scoreFraction(1)).toBe(1);
    expect(scoreFraction(0.1)).toBe(0);
    expect(scoreFraction(0.7)).toBeCloseTo(0.5);
  });
});

describe("basename", () => {
  it("handles both separators", () => {
    expect(basename("/a/b/c.wav")).toBe("c.wav");
    expect(basename("C:\\x\\y.png")).toBe("y.png");
  });
});
