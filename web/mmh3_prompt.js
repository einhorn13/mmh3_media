import { app } from "/scripts/app.js";

const NODE_CLASS = "MMH3Create";
const PROMPT_HEIGHT = 180;
const MIN_NODE_WIDTH = 360;

app.registerExtension({
    name: "mmh3.media.create.prompt",
    async nodeCreated(node) {
        if (node.comfyClass !== NODE_CLASS) return;

        const promptWidget = node.widgets?.find((widget) => widget.name === "prompt");
        if (!promptWidget) return;

        const originalComputeSize = promptWidget.computeSize?.bind(promptWidget);
        promptWidget.computeSize = (width) => {
            const measured = originalComputeSize?.(width) ?? [width, 0];
            return [measured[0] ?? width, Math.max(Number(measured[1]) || 0, PROMPT_HEIGHT)];
        };

        const required = node.computeSize?.() ?? node.size;
        node.setSize?.([
            Math.max(Number(node.size?.[0]) || 0, MIN_NODE_WIDTH),
            Math.max(Number(node.size?.[1]) || 0, Number(required?.[1]) || 0),
        ]);
        node.graph?.setDirtyCanvas(true, true);
    },
});
