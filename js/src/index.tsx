import { PluginComponentType, registerComponent } from "@fiftyone/plugins";
import CrossModalRetrievalView from "./CrossModalRetrievalView";
import { ensureTheme } from "./theme";

ensureTheme();

registerComponent({
  name: "CrossModalRetrievalView",
  label: "CrossModalRetrievalView",
  component: CrossModalRetrievalView,
  type: PluginComponentType.Component,
  activator: () => true,
});
