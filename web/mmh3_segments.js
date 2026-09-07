import { app } from "/scripts/app.js";

const summaries = new Set(["MMH3H3SegmentPrepare", "MMH3SegmentReview", "MMH3H3ContinuationHandover", "MMH3VideoUpscale", "MMH3H3StitchUpscale"]);

function upstream(node, inputName = "packet") {
    const input = node.inputs?.find(item => item.name === inputName);
    const links = node.graph?.links;
    const link = links?.get?.(input?.link) ?? links?.[input?.link];
    return link ? node.graph.getNodeById(link.origin_id) : null;
}

function findSource(node, kind) {
    const seen = new Set();
    while (node && !seen.has(node.id)) {
        if (node.comfyClass === kind) return node;
        seen.add(node.id);
        node = upstream(node);
    }
    return null;
}

function setValue(node, name, value) {
    const widget = node?.widgets?.find(item => item.name === name);
    if (!widget) return false;
    widget.value = value;
    widget.callback?.(value, app.canvas, node, app.canvas?.graph_mouse, {});
    return true;
}

function syncSegmentHelpers(node) {
    const action = node.widgets?.find(item => item.name === "action")?.value;
    // Only manage helper loaders explicitly marked by the bundled F04 workflow.
    // User-created/shared input branches are never muted automatically.
    for (const [input, role, enabled] of [
        ["chain_packet", "reroll_chain", action === "Reroll accepted"],
        ["reanchor_image", "reanchor_image", action === "Reanchor"],
    ]) {
        const helper = upstream(node, input);
        if (helper?.properties?.mmh3_segment_helper !== role) continue;
        if (role === "reanchor_image" && action === "Reroll accepted") continue;
        helper.mode = enabled ? 0 : 2;
    }
    node.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "mmh3.media.segments",
    nodeCreated(node) {
        if (node.comfyClass === "MMH3H3SegmentPrepare") {
            const action = node.widgets?.find(item => item.name === "action");
            if (action) {
                const changed = action.callback;
                action.callback = function(...args) {
                    const result = changed?.apply(this, args);
                    syncSegmentHelpers(node);
                    return result;
                };
            }
        }
        if (summaries.has(node.comfyClass)) {
            const text = document.createElement("div");
            text.setAttribute("role", "status");
            Object.assign(text.style, { padding: "6px 8px", fontSize: "12px", lineHeight: "1.4", whiteSpace: "pre-wrap", overflow: "auto" });
            text.textContent = node.comfyClass === "MMH3H3SegmentPrepare"
                ? "Prompts: one per segment, separated by ---. Blank inherits the saved plan."
                : "";
            node.addDOMWidget("mmh3_segment_status", "status", text, {
                serialize: false, getMinHeight: () => 48, getMaxHeight: () => 64,
            });
            const executed = node.onExecuted;
            node.onExecuted = function(message, ...args) {
                const result = executed?.call(this, message, ...args);
                text.textContent = (message?.text ?? []).join("\n");
                return result;
            };
        }
        if (node.comfyClass !== "MMH3Save") return;
        let accepted = null;
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = "Continue from this result";
        button.disabled = true;
        button.title = "Available after an accepted segment is saved. Updates Source and selects Draft; does not queue.";
        Object.assign(button.style, { width: "100%", padding: "6px", fontSize: "12px" });
        node.addDOMWidget("mmh3_continue", "button", button, {
            serialize: false, getMinHeight: () => accepted ? 32 : 0, getMaxHeight: () => accepted ? 32 : 0,
        });
        button.hidden = true;
        const executed = node.onExecuted;
        node.onExecuted = function(message, ...args) {
            const result = executed?.call(this, message, ...args);
            const saved = message?.mmh3_saved?.[0];
            accepted = saved?.segment_state === "accepted" ? saved : null;
            button.hidden = !accepted;
            button.disabled = !accepted;
            node.graph?.setDirtyCanvas(true, true);
            return result;
        };
        button.addEventListener("click", () => {
            if (!accepted) return;
            const review = findSource(upstream(node), "MMH3SegmentReview");
            const prepare = findSource(review, "MMH3H3SegmentPrepare");
            const source = findSource(upstream(prepare ?? {}), "MMH3Load");
            const loads = new Set([source]);
            for (const item of [prepare, review]) {
                const chain = item && upstream(item, "chain_packet");
                if (chain) loads.add(findSource(chain, "MMH3Load"));
            }
            // Only update a fully understood local path; never pick an unrelated Load node.
            if (!review || !prepare || [...loads].some(load => !load?.widgets?.some(w => w.name === "path_override"))) {
                button.title = "Connect Save through Review and Prepare to an MMH3 Load source.";
                return;
            }
            node.graph?.beforeChange?.();
            for (const load of loads) setValue(load, "path_override", accepted.path);
            setValue(prepare, "action", "Continue");
            setValue(review, "decision", "Draft");
            accepted = null;
            button.disabled = true;
            button.hidden = true;
            node.graph?.afterChange?.();
            node.graph?.setDirtyCanvas(true, true);
        });
    },
});
