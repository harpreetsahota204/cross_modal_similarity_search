import { defineConfig } from "@voxel51/fiftyone-js-plugin-build";
import { dirname } from "path";
import { fileURLToPath } from "url";

const dir = dirname(fileURLToPath(import.meta.url));

// The build plugin requires FIFTYONE_DIR, but only reads it for private
// @fiftyone/* packages, which this plugin never imports
process.env.FIFTYONE_DIR ??= dir;

export default defineConfig(dir, {
  buildConfigOverride: { sourcemap: true },
});
