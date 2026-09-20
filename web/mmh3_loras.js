import { app } from "/scripts/app.js";

const NODE = "MMH3H3TurboLoRAs";
const legacyName = /^(lora|strength)_[123]$/;

app.registerExtension({
    name: "mmh3.media.load_loras",
    nodeCreated(node) {
        const labels = () => {
            for (const port of [...(node.inputs ?? []), ...(node.outputs ?? [])]) {
                if (port.name === "turbo_loras_json") port.label = "LoRAs";
            }
        };
        labels();
        if (node.comfyClass !== NODE) return;
        const widget = name => node.widgets?.find(w => w.name === name);
        const state = widget("lora_entries_json");
        if (!state) return;
        const mode = widget("mode");
        const originalWidgets = new Map();
        for (const w of node.widgets) {
            if (!legacyName.test(w.name) && w !== state) continue;
            originalWidgets.set(w, { type: w.type, computeSize: w.computeSize, draw: w.draw, hidden: w.hidden });
        }
        const root = document.createElement("div");
        Object.assign(root.style, { display: "grid", gap: "8px", padding: "8px", boxSizing: "border-box",
            color: "var(--input-text, #ddd)", font: "12px sans-serif" });
        const rows = document.createElement("div");
        Object.assign(rows.style, { display: "grid", gap: "6px", maxHeight: "300px", overflowY: "auto" });
        const status = document.createElement("div");
        status.setAttribute("role", "status");
        Object.assign(status.style, { lineHeight: "1.4", opacity: "0.8" });
        const add = document.createElement("button");
        add.type = "button";
        add.textContent = "+ Add LoRA";
        add.setAttribute("aria-label", "Add LoRA");
        Object.assign(add.style, { padding: "7px", cursor: "pointer" });
        root.append(rows, add, status);
        const read = () => {
            if (state.value) {
                const entries = JSON.parse(state.value);
                if (!Array.isArray(entries) || entries.some(e => !e || typeof e.name !== "string" ||
                    typeof e.strength !== "number" || !Number.isFinite(e.strength) || e.strength < -10 || e.strength > 10)) {
                    throw new Error("Invalid saved LoRA list. Restore the workflow before editing.");
                }
                return entries;
            }
            const entries = [1, 2, 3].map(i => ({ name: widget(`lora_${i}`)?.value ?? "None",
                strength: widget(`strength_${i}`)?.value ?? 1 })).filter(e => e.name !== "None");
            return entries.length ? entries : [{ name: "None", strength: 1 }];
        };
        const linkedLegacy = () => node.inputs?.some(p => legacyName.test(p.name) && p.link != null);
        const edit = action => {
            try {
                if (linkedLegacy()) return;
                const entries = read();
                action(entries);
                node.graph?.beforeChange?.();
                try {
                    state.value = JSON.stringify(entries);
                    state.callback?.(state.value);
                } finally {
                    node.graph?.afterChange?.();
                }
            } catch (error) {
                status.textContent = error.message;
            }
        };
        const render = () => {
            labels();
            if (["H3 Turbo LoRAs", "Turbo LoRAs", "MMH3H3TurboLoRAs"].includes(node.title)) node.title = "Load LoRAs";
            const linked = linkedLegacy();
            for (const [w, original] of originalWidgets) {
                const hide = w === state || !linked;
                w.type = hide ? "converted-widget" : original.type;
                w.computeSize = hide ? () => [0, -4] : original.computeSize;
                w.draw = hide ? () => {} : original.draw;
                w.hidden = hide ? true : original.hidden;
                if (w.inputEl) w.inputEl.style.display = hide ? "none" : "";
                if (w.element) w.element.hidden = hide;
            }
            rows.replaceChildren();
            add.disabled = !!linked;
            if (linked) {
                status.textContent = "Using connected legacy slots. Disconnect them to edit the expandable list.";
                return;
            }
            let entries;
            try { entries = read(); }
            catch (error) { status.textContent = error.message; return; }
            // Once migrated, obsolete hidden combo values must not block queue
            // validation (e.g. a removed or renamed legacy LoRA file).
            if (state.value) {
                for (let i = 1; i <= 3; i++) {
                    widget(`lora_${i}`).value = "None";
                    widget(`strength_${i}`).value = 1;
                }
            }
            let options = widget("lora_1")?.options?.values ?? ["None"];
            if (typeof options === "function") options = options();
            entries.forEach((entry, index) => {
                const row = document.createElement("div");
                Object.assign(row.style, { display: "grid", gridTemplateColumns: "minmax(0, 1fr) 70px 28px", gap: "5px" });
                const file = document.createElement("select");
                file.setAttribute("aria-label", `LoRA ${index + 1}`);
                file.style.minWidth = "0";
                file.style.width = "100%";
                for (const name of new Set([...options, entry.name])) {
                    const option = document.createElement("option");
                    option.value = name;
                    option.textContent = name;
                    file.append(option);
                }
                file.value = entry.name;
                file.title = entry.name;
                file.addEventListener("change", () => edit(items => { items[index].name = file.value; }));
                const strength = document.createElement("input");
                strength.type = "number";
                strength.min = "-10";
                strength.max = "10";
                strength.step = "0.05";
                strength.value = String(entry.strength);
                strength.style.width = "100%";
                strength.style.boxSizing = "border-box";
                strength.setAttribute("aria-label", `Strength ${index + 1}`);
                strength.addEventListener("change", () => edit(items => {
                    const value = Number(strength.value);
                    if (!strength.value.trim() || !Number.isFinite(value) || value < -10 || value > 10) {
                        throw new Error("Strength must be a number between -10 and 10.");
                    }
                    items[index].strength = value;
                }));
                const remove = document.createElement("button");
                remove.type = "button";
                remove.textContent = "×";
                remove.title = `Remove LoRA ${index + 1}`;
                remove.setAttribute("aria-label", remove.title);
                remove.addEventListener("click", () => edit(items => { items.splice(index, 1); }));
                row.append(file, strength, remove);
                rows.append(row);
            });
            const hint = mode.value === "auto" ? "Auto uses preset/source LoRAs. Select Custom or Extension to use this list."
                : mode.value === "disabled" ? "Preset LoRAs disabled. This list is retained."
                : mode.value === "extension" ? "Adds this ordered list to preset/source LoRAs."
                : "Replaces preset LoRAs with this ordered list.";
            status.textContent = hint;
            const height = Math.min(300, Math.max(32, entries.length * 34)) + 95;
            dom.options.getMinHeight = () => height;
            dom.options.getMaxHeight = () => height;
            node.setSize?.([Math.max(node.size?.[0] ?? 400, 360), node.computeSize?.()[1] ?? height + 70]);
            node.graph?.setDirtyCanvas(true, true);
        };
        const dom = node.addDOMWidget("mmh3_lora_editor", "LoRA list", root,
            { serialize: false, hideOnZoom: false, getMinHeight: () => 150, getMaxHeight: () => 395 });
        add.addEventListener("click", () => edit(items => { items.push({ name: "None", strength: 1 }); }));
        for (const w of [state, mode, ...originalWidgets.keys()]) {
            if (w._mmh3LoraCallback) continue;
            w._mmh3LoraCallback = true;
            const previous = w.callback;
            w.callback = function(...args) { const result = previous?.apply(this, args); render(); return result; };
        }
        const configure = node.onConfigure;
        node.onConfigure = function(...args) { const result = configure?.apply(this, args); render(); return result; };
        const connections = node.onConnectionsChange;
        node.onConnectionsChange = function(...args) { const result = connections?.apply(this, args); render(); return result; };
        render();
    },
});
