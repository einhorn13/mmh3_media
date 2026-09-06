import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_CLASS = "MMH3BatchInputPlan";
const ROUTE = "/mmh3_media/batch_files";

const naturalKey = (value) => value.split(/(\d+)/).map((part) => /^\d+$/.test(part) ? Number(part) : part.toLocaleLowerCase());
const naturalCompare = (left, right) => {
    const a = naturalKey(left);
    const b = naturalKey(right);
    for (let index = 0; index < Math.max(a.length, b.length); index += 1) {
        if (a[index] === b[index]) continue;
        if (a[index] === undefined) return -1;
        if (b[index] === undefined) return 1;
        return a[index] < b[index] ? -1 : 1;
    }
    return 0;
};

app.registerExtension({
    name: "mmh3.media.batch.input",
    nodeCreated(node) {
        if (node.comfyClass !== NODE_CLASS) return;
        const pathsWidget = node.widgets?.find((widget) => widget.name === "paths_json");
        if (!pathsWidget) return;
        const root = document.createElement("div");
        Object.assign(root.style, { display: "grid", gap: "6px", padding: "8px", fontSize: "12px" });
        const available = document.createElement("select");
        available.multiple = true;
        available.size = 7;
        available.setAttribute("aria-label", "Available media files");
        available.style.width = "100%";
        const orderedList = document.createElement("div");
        Object.assign(orderedList.style, { display: "grid", gap: "3px", maxHeight: "220px", overflow: "auto" });
        const status = document.createElement("div");
        status.setAttribute("role", "status");
        let removed = false;
        let request = null;

        const read = () => {
            const parsed = JSON.parse(pathsWidget.value || "[]");
            if (!Array.isArray(parsed) || !parsed.every(value => typeof value === "string")) {
                throw new Error("Paths must be a JSON array of strings.");
            }
            return parsed;
        };
        const edit = action => {
            try {
                const ordered = read();
                action(ordered);
                node.graph?.beforeChange?.();
                try {
                    pathsWidget.value = JSON.stringify(ordered, null, 2);
                    pathsWidget.callback?.(pathsWidget.value, app.canvas, node, app.canvas?.graph_mouse, {});
                } finally {
                    node.graph?.afterChange?.();
                }
            } catch (error) {
                status.textContent = error.message;
            }
        };
        const render = () => {
            orderedList.replaceChildren();
            let ordered;
            try { ordered = read(); status.textContent = `${ordered.length} files`; }
            catch (error) { status.textContent = error.message; return; }
            ordered.forEach((path, index) => {
                const row = document.createElement("div");
                Object.assign(row.style, { display: "grid", gridTemplateColumns: "1fr auto auto auto", gap: "3px", alignItems: "center" });
                const label = document.createElement("span");
                label.textContent = `${index + 1}. ${path}`;
                label.title = path;
                label.style.overflowWrap = "anywhere";
                const button = (text, title, action, disabled = false) => {
                    const value = document.createElement("button");
                    value.type = "button";
                    value.textContent = text;
                    value.title = title;
                    value.setAttribute("aria-label", title);
                    value.disabled = disabled;
                    value.addEventListener("click", () => edit(action));
                    return value;
                };
                row.append(
                    label,
                    button("↑", "Move up", items => { if (index > 0) [items[index - 1], items[index]] = [items[index], items[index - 1]]; }, index === 0),
                    button("↓", "Move down", items => { if (index + 1 < items.length) [items[index + 1], items[index]] = [items[index], items[index + 1]]; }, index + 1 === ordered.length),
                    button("×", "Remove file", items => { items.splice(index, 1); }),
                );
                orderedList.append(row);
            });
            node.graph?.setDirtyCanvas(true, true);
        };

        const controls = document.createElement("div");
        Object.assign(controls.style, { display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "4px" });
        const add = document.createElement("button");
        add.textContent = "ADD SELECTED";
        add.type = "button";
        add.addEventListener("click", () => edit(ordered => {
            for (const option of available.selectedOptions) if (!ordered.includes(option.value)) ordered.push(option.value);
        }));
        const sort = document.createElement("button");
        sort.textContent = "NATURAL SORT";
        sort.type = "button";
        sort.addEventListener("click", () => edit(ordered => ordered.sort(naturalCompare)));
        const refresh = document.createElement("button");
        refresh.textContent = "REFRESH";
        refresh.type = "button";
        refresh.addEventListener("click", async () => {
            request?.abort();
            request = new AbortController();
            refresh.disabled = true;
            try {
                const response = await api.fetchApi(ROUTE, { signal: request.signal });
                const values = await response.json();
                if (!response.ok || !Array.isArray(values)) throw new Error(values?.error || `HTTP ${response.status}`);
                if (removed) return;
                available.replaceChildren(...values.map((path) => {
                    const option = document.createElement("option");
                    option.value = path;
                    option.textContent = path;
                    return option;
                }));
            } catch (error) {
                if (!removed && error.name !== "AbortError") status.textContent = "File list unavailable: " + error.message;
            } finally {
                refresh.disabled = false;
            }
        });
        controls.append(add, sort, refresh);
        root.append(available, controls, status, orderedList);
        node.addDOMWidget("mmh3_batch_editor", "MMH3 batch editor", root, { serialize: false, hideOnZoom: false });
        const callback = pathsWidget.callback;
        pathsWidget.callback = function(...args) {
            const result = callback?.apply(this, args);
            render();
            return result;
        };
        const configure = node.onConfigure;
        node.onConfigure = function(...args) {
            const result = configure?.apply(this, args);
            render();
            return result;
        };
        const remove = node.onRemoved;
        node.onRemoved = function(...args) {
            removed = true;
            request?.abort();
            return remove?.apply(this, args);
        };
        render();
        void refresh.click();
    },
});
