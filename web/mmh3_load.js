import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_CLASS = "MMH3Load";
const ROUTE = "/mmh3_media/files";

app.registerExtension({
    name: "mmh3.media.load.refresh",
    async nodeCreated(node) {
        if (node.comfyClass !== NODE_CLASS) return;

        const button = document.createElement("button");
        button.type = "button";
        button.textContent = "↻  REFRESH .MMH3 FILES";
        button.title = "Rescan ComfyUI input/output folders for .mmh3 files";
        Object.assign(button.style, {
            width: "100%",
            minHeight: "34px",
            marginTop: "6px",
            borderRadius: "7px",
            border: "1px solid var(--border-color, #777)",
            fontWeight: "700",
            letterSpacing: "0.035em",
            cursor: "pointer",
        });

        button.addEventListener("click", async () => {
            const fileWidget = node.widgets?.find((w) => w.name === "file");
            if (!fileWidget) return;

            const oldText = button.textContent;
            button.disabled = true;
            button.textContent = "↻  REFRESHING…";
            try {
                const response = await api.fetchApi(ROUTE);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const values = await response.json();
                if (!Array.isArray(values)) throw new Error("Invalid file-list response");

                fileWidget.options ??= {};
                fileWidget.options.values = values.length ? values : ["(none)"];
                fileWidget.value = fileWidget.options.values[0];
                fileWidget.callback?.(fileWidget.value, app.canvas, node, app.canvas?.graph_mouse, {});
                node.graph?.setDirtyCanvas(true, true);
            } catch (error) {
                console.error("[MMH3 Media] Failed to refresh .mmh3 files", error);
                button.textContent = "⚠  REFRESH FAILED — RETRY";
                window.setTimeout(() => { button.textContent = oldText; }, 1800);
                return;
            } finally {
                button.disabled = false;
            }
            button.textContent = oldText;
        });

        node.addDOMWidget("mmh3_refresh", "MMH3 refresh", button, {
            serialize: false,
            hideOnZoom: false,
        });
    },
});
