import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const refreshers = new Map();
let saveRefreshTimer;

function element(tag, className, text) {
    const item = document.createElement(tag);
    item.className = className;
    if (text != null) item.textContent = text;
    return item;
}

function installStyles() {
    if (document.getElementById("mmh3-gallery-style")) return;
    const style = element("style", "");
    style.id = "mmh3-gallery-style";
    style.textContent = `
    .mmh3-gallery { box-sizing:border-box; width:100%; height:320px; overflow:auto;
      padding:8px; color:var(--input-text,#ddd); background:var(--comfy-input-bg,#222);
      border:1px solid var(--border-color,#555); border-radius:7px; font:12px sans-serif; }
    .mmh3-gallery-title { font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .mmh3-gallery-info { color:var(--descrip-text,#aaa); margin:5px 0 9px; line-height:1.4; }
    .mmh3-gallery-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
    .mmh3-gallery-tile { min-width:0; padding:0; border:1px solid var(--border-color,#555);
      background:var(--comfy-menu-bg,#292929); color:inherit; border-radius:5px; overflow:hidden;
      font:inherit; text-align:left; cursor:pointer; }
    .mmh3-gallery-tile:disabled { opacity:0.8; cursor:default; }
    .mmh3-gallery-tile:focus-visible,.mmh3-gallery-more:focus-visible { outline:2px solid #8ab4f8; }
    .mmh3-gallery-thumb { width:100%; height:84px; object-fit:contain; display:block; background:#181818; }
    .mmh3-gallery-placeholder { display:grid; place-items:center; height:84px; background:#181818; color:#aaa; }
    .mmh3-gallery-label { display:block; padding:5px 6px; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }
    .mmh3-gallery-more { width:100%; margin-top:8px; padding:6px; cursor:pointer; }
    .mmh3-gallery-dialog { max-width:90vw; max-height:90vh; padding:16px; color:#eee;
      background:#202020; border:1px solid #666; border-radius:10px; font:14px sans-serif; }
    .mmh3-gallery-dialog::backdrop { background:#000b; }
    .mmh3-gallery-dialog img { display:block; max-width:80vw; max-height:65vh; object-fit:contain; margin:12px auto; }
    .mmh3-gallery-dialog button { float:right; padding:6px 14px; cursor:pointer; }
    `;
    document.head.append(style);
}

app.registerExtension({
    name: "mmh3.media.load.preview",
    refreshComboInNodes() {
        for (const refresh of refreshers.values()) void refresh();
    },
    nodeCreated(node) {
        const isSave = node.comfyClass === "MMH3Save";
        if (!isSave && node.comfyClass !== "MMH3Load") return;
        installStyles();
        const card = element("div", "mmh3-gallery");
        card.setAttribute("aria-label", "MMH3 archive contents");
        let serial = 0, request, dialog, removed = false, savedFile, savedPath;
        const closePreview = () => {
            const current = dialog;
            dialog = null;
            current?.close();
            current?.remove();
        };
        const reserveSpace = () => node.setSize([Math.max(380, node.size[0]), Math.max(470, node.size[1])]);
        const showPreview = (item, url) => {
            closePreview();
            const current = element("dialog", "mmh3-gallery-dialog");
            dialog = current;
            current.setAttribute("aria-label", item.name + " preview");
            const close = element("button", "", "Close");
            close.type = "button";
            close.addEventListener("click", closePreview);
            const image = element("img", "");
            image.src = url;
            image.alt = item.summary;
            current.append(close, element("strong", "", item.name), image, element("p", "", item.summary));
            if (item.preview_source === "paired_decoded_result")
                current.append(element("p", "", "Preview of the paired decoded result — not a new latent decode."));
            current.addEventListener("close", () => {
                current.remove();
                if (dialog === current) dialog = null;
            }, { once: true });
            document.body.append(current);
            current.showModal();
        };
        const updateCard = async () => {
            const current = ++serial;
            request?.abort();
            closePreview();
            const file = isSave ? savedFile : node.widgets?.find(w => w.name === "file")?.value;
            const override = node.widgets?.find(w => w.name === "path_override")?.value;
            card.replaceChildren();
            if (String(override || "").trim()) {
                card.textContent = "Path override is active. Preview is available for files selected from input/output.";
                return;
            }
            if (!file || file === "(none)") {
                card.textContent = isSave
                    ? (savedPath ? "Saved: " + savedPath + " (preview is available inside input/output folders)." : "Run to save the archive and see its preview.")
                    : "Select an MMH3 archive to see its contents.";
                return;
            }
            card.textContent = "Reading archive…";
            request = new AbortController();
            try {
                const response = await api.fetchApi("/mmh3_media/file_info?file=" + encodeURIComponent(file), { signal: request.signal });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || "HTTP " + response.status);
                if (removed || current !== serial) return;
                const title = element("div", "mmh3-gallery-title", data.name);
                title.title = data.prompt_summary || data.name;
                const g = data.geometry || {};
                const info = [isSave ? "Saved" : null, data.task, g.width && g.height ? g.width + "×" + g.height : null,
                    g.frames ? g.frames + " frames" : null, g.duration != null ? Number(g.duration).toFixed(1) + " s" : null].filter(Boolean).join(" · ");
                const grid = element("div", "mmh3-gallery-grid");
                card.replaceChildren(title, element("div", "mmh3-gallery-info", info), grid);
                if (isSave && savedPath) {
                    const copy = element("button", "mmh3-gallery-more", "Copy saved path");
                    copy.type = "button";
                    copy.title = savedPath;
                    copy.addEventListener("click", async () => {
                        try { await navigator.clipboard.writeText(savedPath); copy.textContent = "Copied"; }
                        catch { copy.textContent = savedPath; }
                    });
                    card.insertBefore(copy, grid);
                }
                const items = data.gallery || [];
                if (!items.length) grid.append(element("div", "mmh3-gallery-info", "No media resources."));
                let shown = 0;
                const more = element("button", "mmh3-gallery-more");
                more.type = "button";
                const appendPage = () => {
                    for (const item of items.slice(shown, shown + 4)) {
                        const tile = element("button", "mmh3-gallery-tile");
                        tile.type = "button";
                        tile.title = item.summary;
                        tile.setAttribute("aria-label", item.kind + ": " + item.name);
                        if (item.representation_id || item.can_preview_source) {
                            const url = api.apiURL("/mmh3_media/file_preview?file=" + encodeURIComponent(file)
                                + (item.representation_id ? "&representation_id=" + encodeURIComponent(item.representation_id)
                                    : "&resource_id=" + encodeURIComponent(item.resource_id))
                                + "&revision=" + encodeURIComponent(data.revision));
                            const image = element("img", "mmh3-gallery-thumb");
                            image.loading = "lazy";
                            image.alt = item.summary;
                            image.src = url;
                            image.addEventListener("error", () => {
                                image.replaceWith(element("div", "mmh3-gallery-placeholder", "Preview unavailable"));
                                tile.disabled = true;
                            }, { once: true });
                            tile.append(image);
                            tile.addEventListener("click", () => showPreview(item, url));
                        } else {
                            tile.disabled = true;
                            tile.append(element("div", "mmh3-gallery-placeholder", item.kind.toUpperCase()));
                            tile.title += item.kind === "latent" ? " — No cached preview. Use MMH3 Preview with a compatible VAE." : " — No cached preview in this archive.";
                        }
                        const label = item.primary ? "Result · " + item.kind : item.name + " · " + (item.role || item.kind);
                        tile.append(element("span", "mmh3-gallery-label", label));
                        grid.append(tile);
                    }
                    shown = Math.min(shown + 4, items.length);
                    more.textContent = "Show more (" + (items.length - shown) + ")";
                    more.hidden = shown >= items.length;
                };
                more.addEventListener("click", appendPage);
                card.append(more);
                appendPage();
                if (data.warnings?.length) card.append(element("div", "mmh3-gallery-info", data.warnings.join(" · ")));
            } catch (error) {
                if (removed || current !== serial || error.name === "AbortError") return;
                card.textContent = "Preview unavailable: " + error.message;
            }
            node.graph?.setDirtyCanvas(true, true);
        };
        node.addDOMWidget("mmh3_file_card", "MMH3 file card", card, {
            serialize: false, hideOnZoom: false, getMinHeight: () => 320, getMaxHeight: () => 320,
        });
        refreshers.set(node, updateCard);
        if (isSave) {
            const executed = node.onExecuted;
            node.onExecuted = function(message, ...args) {
                const result = executed?.call(this, message, ...args);
                const saved = message?.mmh3_saved?.[0];
                if (saved) {
                    savedFile = saved.file;
                    savedPath = saved.path;
                    void updateCard();
                    // Coalesce multiple Save nodes; use the same public refresh as R.
                    clearTimeout(saveRefreshTimer);
                    saveRefreshTimer = setTimeout(() => {
                        app.refreshComboInNodes().catch(error => console.warn("[MMH3] File list refresh failed", error));
                    }, 150);
                }
                return result;
            };
        }
        for (const name of ["file", "path_override"]) {
            const widget = node.widgets?.find(w => w.name === name);
            if (!widget) continue;
            const callback = widget.callback;
            widget.callback = function(...args) {
                const result = callback?.apply(this, args);
                void updateCard();
                return result;
            };
        }
        const configure = node.onConfigure;
        node.onConfigure = function(...args) {
            const result = configure?.apply(this, args);
            reserveSpace();
            void updateCard();
            return result;
        };
        const remove = node.onRemoved;
        node.onRemoved = function(...args) {
            removed = true;
            ++serial;
            request?.abort();
            refreshers.delete(node);
            closePreview();
            return remove?.apply(this, args);
        };
        reserveSpace();
        queueMicrotask(() => { if (!removed) void updateCard(); });
    },
});
