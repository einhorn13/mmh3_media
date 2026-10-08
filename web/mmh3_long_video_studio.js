import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { openProjectManager } from "./mmh3_project_manager.js";

app.registerExtension({
    name: "mmh3.media.longVideoStudio",
    nodeCreated(node) {
        if (node.comfyClass !== "MMH3LongVideoStudio") return;
        const panel = document.createElement("div");
        panel.style.cssText = "padding:8px;display:grid;gap:8px;color:var(--input-text);background:var(--comfy-input-bg);font:13px sans-serif;";
        const buttons = document.createElement("div");
        buttons.style.cssText = "display:flex;gap:5px;flex-wrap:wrap";
        const manager = document.createElement("button");manager.type="button";manager.textContent="Open Project Manager";
        manager.onclick=()=>{node.properties ??= {};node.properties.mmh3_project_id="studio::"+node.widgets.find(w=>w.name==="project").value.trim().replaceAll("\\","/");void openProjectManager(node);};
        buttons.append(manager);
        const status = document.createElement("pre");
        status.style.cssText = "white-space:pre-wrap;max-height:180px;overflow:auto;margin:0";
        status.textContent = "Choose an identity image, master audio and Ref2VA models. Generate → review → accept → generate next → assemble.";
        const selector = document.createElement("select");
        selector.setAttribute("aria-label", "Choose a take");
        selector.style.cssText = "width:100%;min-height:30px;display:none";
        const video = document.createElement("video");
        video.controls = true;
        video.preload = "metadata";
        video.style.cssText = "width:100%;max-height:260px;display:none";
        let candidates = [];
        function preview(value) {
            if (!value) { video.removeAttribute("src"); video.style.display = "none"; return; }
            video.src = api.apiURL(`/view?${new URLSearchParams(value)}`);
            video.style.display = "block";
        }
        selector.onchange = () => {
            const widget = node.widgets.find(w => w.name === "candidate_id");
            if (widget) widget.value = selector.value;
            preview(candidates.find(c => c.candidate_id === selector.value)?.preview);
        };
        for (const [action, label] of [["generate_next", "Generate next"], ["accept", "Accept take"], ["reroll", "Reroll"], ["assemble", "Assemble"], ["inspect", "Refresh"], ["recover", "Recover"]]) {
            const button = document.createElement("button");
            button.type = "button";
            button.textContent = label;
            button.onclick = async () => {
                const widget = node.widgets.find(w => w.name === "action");
                if (!widget) return;
                widget.value = action;
                button.disabled = true;
                try { await app.queuePrompt(0, 1); }
                catch (error) { status.textContent = String(error); }
                finally { button.disabled = false; }
            };
            buttons.append(button);
        }
        panel.append(buttons, selector, video, status);
        node.addDOMWidget("studio_review", "studio_review", panel, { serialize: false, hideOnZoom: false });
        const executed = node.onExecuted;
        node.onExecuted = function(message) {
            executed?.apply(this, arguments);
            const data = message.mmh3_studio?.at(-1);
            if (!data) return;
            status.textContent = data.status + (data.path ? `\n${data.path}` : "");
            candidates = data.candidates ?? [];
            selector.style.display = candidates.length ? "block" : "none";
            selector.replaceChildren();
            for (const take of candidates) {
                const option = document.createElement("option");
                option.value = take.candidate_id;
                option.textContent = `${take.job_id} · take ${take.attempt}${take.label ? ` · ${take.label}` : ""}`;
                selector.append(option);
            }
            const current = node.widgets.find(w => w.name === "candidate_id");
            const chosen = candidates.find(c => c.packet_path === data.path) ?? candidates.find(c => c.candidate_id === current?.value) ?? candidates.at(-1);
            if (chosen) { selector.value = chosen.candidate_id; if (current) current.value = chosen.candidate_id; }
            preview(data.preview ?? chosen?.preview);
            node.setSize([Math.max(node.size[0], 470), Math.max(node.size[1], 1100)]);
            node.graph?.setDirtyCanvas(true, true);
        };
    },
});
