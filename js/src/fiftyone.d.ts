// Types for the @fiftyone/* globals the App provides at runtime. The plugin
// build externalizes these packages, so they are not installed.

declare module "@fiftyone/plugins" {
  import type { FC } from "react";

  export enum PluginComponentType {
    Visualizer = 1,
    Plot = 2,
    Panel = 3,
    Component = 4,
  }

  export function registerComponent(options: {
    name: string;
    label: string;
    component: FC<any>;
    type: PluginComponentType;
    activator: (ctx?: unknown) => boolean;
  }): void;
}

declare module "@fiftyone/operators" {
  export function useTriggerPanelEvent(): (
    event: string,
    params?: Record<string, unknown>,
    prompt?: boolean,
    callback?: (result: unknown) => void,
  ) => void;
}

declare module "@fiftyone/spaces" {
  export function usePanelId(): string;
}

declare module "@fiftyone/state" {
  export function getSampleSrc(url: string): string;
}
