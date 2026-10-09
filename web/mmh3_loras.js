import { app } from "/scripts/app.js";

const NODE = "MMH3H3TurboLoRAs";
let comboId = 0;

function searchableLoRA(names, selected, label, commit) {
    const host = document.createElement("div");
    host.style.cssText = "min-width:0;display:flex;flex-direction:column;gap:4px";
    const input = document.createElement("input");
    input.type = "search";
    input.value = selected;
    input.title = selected;
    input.placeholder = "Filter LoRAs…";
    input.setAttribute("aria-label", label);
    input.setAttribute("role", "combobox");
    input.setAttribute("aria-autocomplete", "list");
    input.setAttribute("aria-expanded", "false");
    input.style.cssText = "width:100%;min-width:0;box-sizing:border-box;min-height:32px;background:#252525;color:#f2f2f2;border:1px solid #777;border-radius:4px;padding:6px 8px";
    const list = document.createElement("div");
    list.id = `mmh3-lora-options-${++comboId}`;
    list.setAttribute("role", "listbox");
    list.setAttribute("aria-label", `${label} matches`);
    // The top-layer picker does not resize the DOM widget or get clipped by
    // the canvas/node. Its width is independent of the narrow filename field.
    list.popover = "auto";
    list.style.cssText = "position:fixed;inset:auto;margin:0;box-sizing:border-box;overflow:auto;padding:6px;border:1px solid #777;border-radius:5px;background:#252525;color:#f2f2f2;font:13px/1.4 sans-serif;box-shadow:0 6px 20px #0008";
    input.setAttribute("aria-controls", list.id);
    let matches = [], active = -1;
    const close = () => {
        list.hidePopover();
        input.setAttribute("aria-expanded", "false");
        input.removeAttribute("aria-activedescendant");
        input.value = selected;
    };
    const highlight = () => {
        [...list.querySelectorAll('[role="option"]')].forEach((option, index) => {
            option.setAttribute("aria-selected", String(index === active));
            option.style.background = index === active ? "#40566f" : "transparent";
            if (index === active) {
                input.setAttribute("aria-activedescendant", option.id);
                option.scrollIntoView({block: "nearest"});
            }
        });
    };
    const filter = (query) => {
        const found = names.filter(name => name.toLocaleLowerCase().includes(query.toLocaleLowerCase()));
        matches = found.slice(0, 100);
        active = -1;
        input.removeAttribute("aria-activedescendant");
        list.replaceChildren();
        for (const [index, name] of matches.entries()) {
            const option = document.createElement("button");
            option.type = "button";
            option.id = `${list.id}-${index}`;
            option.setAttribute("role", "option");
            option.setAttribute("aria-selected", "false");
            option.textContent = name;
            option.title = name;
            option.style.cssText = "display:block;width:100%;text-align:left;overflow-wrap:anywhere;min-width:0;padding:8px;cursor:pointer;background:transparent;color:#f2f2f2;border:0;border-radius:3px;font:inherit";
            option.addEventListener("pointerenter", () => { active = index; highlight(); });
            option.addEventListener("mousedown", event => event.preventDefault());
            option.addEventListener("click", () => commit(name));
            list.append(option);
        }
        if (!found.length || found.length > matches.length) {
            const hint = document.createElement("div");
            hint.textContent = found.length ? `${found.length} matches. Type more to narrow the list.` : "No matching LoRAs";
            hint.setAttribute("role", "status");
            list.append(hint);
        }
        const rect = input.getBoundingClientRect();
        const width = Math.min(Math.max(480, rect.width), window.innerWidth - 24);
        const below = window.innerHeight - rect.bottom - 12;
        const above = rect.top - 12;
        const upward = below < 200 && above > below;
        Object.assign(list.style, {
            width: `${width}px`,
            left: `${Math.max(12, Math.min(rect.left, window.innerWidth - width - 12))}px`,
            top: upward ? "auto" : `${rect.bottom + 4}px`,
            bottom: upward ? `${window.innerHeight - rect.top + 4}px` : "auto",
            maxHeight: `${Math.max(60, Math.min(340, upward ? above : below) - 4)}px`,
        });
        if (!list.matches(":popover-open")) list.showPopover();
        input.setAttribute("aria-expanded", "true");
    };
    input.addEventListener("focus", () => { input.select(); filter(""); });
    input.addEventListener("input", () => filter(input.value));
    input.addEventListener("keydown", event => {
        if (["ArrowDown", "ArrowUp"].includes(event.key)) {
            event.preventDefault();
            if (!list.matches(":popover-open")) filter(input.value === selected ? "" : input.value);
            if (matches.length) active = active < 0 ? (event.key === "ArrowDown" ? 0 : matches.length - 1)
                : (active + (event.key === "ArrowDown" ? 1 : -1) + matches.length) % matches.length;
            highlight();
        } else if (event.key === "Enter") {
            event.preventDefault();
            const choice = matches[active] ?? matches.find(name => name === input.value) ?? (matches.length === 1 ? matches[0] : null);
            if (choice) commit(choice);
        } else if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            close();
        }
    });
    host.addEventListener("focusout", event => { if (!host.contains(event.relatedTarget)) close(); });
    list.addEventListener("toggle", event => {
        if (event.newState === "closed") {
            input.setAttribute("aria-expanded", "false");
            input.removeAttribute("aria-activedescendant");
            input.value = selected;
        }
    });
    host.append(input, list);
    return host;
}

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
        const catalog = widget("lora_catalog");
        const root = document.createElement("div");
        Object.assign(root.style, { display: "flex", flexDirection: "column", gap: "8px", padding: "8px", boxSizing: "border-box",
            color: "var(--input-text, #ddd)", font: "12px sans-serif" });
        const rows = document.createElement("div");
        Object.assign(rows.style, { display: "grid", gap: "6px", maxHeight: "300px", overflowY: "auto", flexShrink: "0" });
        const status = document.createElement("div");
        status.setAttribute("role", "status");
        Object.assign(status.style, { lineHeight: "1.4", opacity: "0.8", flexShrink: "0" });
        const add = document.createElement("button");
        add.type = "button";
        add.textContent = "+ Add LoRA";
        add.setAttribute("aria-label", "Add LoRA");
        Object.assign(add.style, { padding: "7px", cursor: "pointer", flexShrink: "0", minHeight: "32px",
            background: "#252525", color: "#f2f2f2", border: "1px solid #777", borderRadius: "4px" });
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
            return [];
        };
        const edit = action => {
            try {
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
            for (const w of [state, catalog]) {
                if (!w) continue;
                w.type = "converted-widget";
                w.computeSize = () => [0, 0];
                w.draw = () => {};
                w.hidden = true;
                if (w.inputEl) w.inputEl.style.display = "none";
                if (w.element) w.element.hidden = true;
            }
            rows.replaceChildren();
            let entries;
            try { entries = read(); }
            catch (error) { status.textContent = error.message; return; }
            let options = catalog?.options?.values ?? ["None"];
            if (typeof options === "function") options = options();
            entries.forEach((entry, index) => {
                const row = document.createElement("div");
                Object.assign(row.style, { display: "grid", gridTemplateColumns: "minmax(0, 1fr) 70px 28px", gap: "5px" });
                const file = searchableLoRA([...new Set([...options, entry.name])], entry.name,
                    `LoRA ${index + 1}`, name => edit(items => { items[index].name = name; }));
                const strength = document.createElement("input");
                strength.type = "number";
                strength.min = "-10";
                strength.max = "10";
                strength.step = "0.05";
                strength.value = String(entry.strength);
                strength.style.width = "100%";
                strength.style.boxSizing = "border-box";
                strength.style.height = "32px";
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
                Object.assign(remove.style, { height: "32px", background: "#252525", color: "#f2f2f2",
                    border: "1px solid #777", borderRadius: "4px", cursor: "pointer" });
                remove.addEventListener("click", () => edit(items => { items.splice(index, 1); }));
                row.append(file, strength, remove);
                rows.append(row);
            });
            const hint = mode.value === "auto" ? "Auto uses preset/source LoRAs. Select Custom or Extension to use this list."
                : mode.value === "disabled" ? "Preset LoRAs disabled. This list is retained."
                : mode.value === "extension" ? "Adds this ordered list to preset/source LoRAs."
                : "Replaces preset LoRAs with this ordered list.";
            status.textContent = hint;
            resize();
            node.graph?.setDirtyCanvas(true, true);
        };
        let height = 180;
        node.addDOMWidget("mmh3_lora_editor", "LoRA list", root,
            { serialize: false, hideOnZoom: false, getMinHeight: () => height, getMaxHeight: () => height });
        const resize = () => {
            // Measure the actual rows and wrapped hint, including the empty list.
            // A fixed estimate squeezed the initial row beneath the Add button.
            // 16px padding + two 8px gaps + the host's 20px DOM inset.
            const measured = Math.ceil(rows.offsetHeight + add.offsetHeight + status.offsetHeight + 52);
            if (!root.isConnected || measured === height) return;
            height = measured;
            // Grow only to fit actual rows. Keep the user's width and spare
            // height when filtering, changing a hint, or removing a row.
            node.setSize?.([node.size?.[0] ?? 400,
                Math.max(node.size?.[1] ?? 0, node.computeSize?.()[1] ?? height + 70)]);
            node.graph?.setDirtyCanvas(true, true);
        };
        const observer = new ResizeObserver(resize);
        for (const element of [rows, add, status]) observer.observe(element);
        const removed = node.onRemoved;
        node.onRemoved = function(...args) { observer.disconnect(); return removed?.apply(this, args); };
        add.addEventListener("click", () => edit(items => { items.push({ name: "None", strength: 1 }); }));
        for (const w of [state, mode]) {
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
