import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const element = (tag, text) => { const e = document.createElement(tag); if (text != null) e.textContent = text; return e; };

app.registerExtension({
    name: "mmh3.media.referenceCards",
    nodeCreated(node) {
        if (node.comfyClass !== "MMH3ReferenceCards") return;
        const panel = element("div");
        panel.style.cssText = "display:grid;gap:10px;padding:8px;max-height:500px;overflow:auto;color:var(--input-text);background:var(--comfy-input-bg)";
        panel.append(element("p", "Queue once to discover references. Card edits apply on the next run."));
        node.addDOMWidget("reference_cards", "reference_cards", panel, {serialize:false, hideOnZoom:false});
        const executed = node.onExecuted;
        node.onExecuted = function(message) {
            executed?.apply(this, arguments);
            const report = message.mmh3_reference_cards?.at(-1);
            if (!report) return;
            panel.replaceChildren();
            const cards = report.cards.map(c => ({key:c.key,alias:c.alias,enabled:c.enabled,purposes:[...c.purposes],scenes:[...c.scenes]}));
            const status = element("p");
            const write = () => {
                const widget = node.widgets.find(w => w.name === "cards_json");
                if (widget) widget.value = JSON.stringify(cards);
                status.textContent = "Changes pending. Queue to compile native labels and check aliases.";
                node.graph?.setDirtyCanvas(true, true);
            };
            for (const [index, item] of report.cards.entries()) {
                const row = element("section");
                row.style.cssText = "padding:8px;border:1px solid #596575;border-radius:6px;display:grid;gap:6px";
                row.append(element("strong", `${item.key} · ${item.prompt_tag}`));
                if (item.preview) {
                    const image = element("img"); image.alt = `${item.key} reference`; image.src = api.apiURL(`/view?${new URLSearchParams(item.preview)}`);
                    image.style.cssText = "width:100%;max-height:128px;object-fit:contain";row.append(image);
                }
                const enabled = element("input"); enabled.type="checkbox";enabled.checked=item.enabled;enabled.setAttribute("aria-label", `Enable ${item.key}`);
                enabled.onchange=()=>{cards[index].enabled=enabled.checked;write();};
                const enabledLabel=element("label","Active ");enabledLabel.append(enabled);row.append(enabledLabel);
                const alias=element("input");alias.value=item.alias;alias.placeholder="Alias, e.g. hero_face";alias.setAttribute("aria-label", `Alias ${item.key}`);
                alias.onchange=()=>{cards[index].alias=alias.value.trim().replace(/^@/,"");write();};row.append(alias);
                const roles=element("select");roles.multiple=true;roles.setAttribute("aria-label", `Roles ${item.key}`);
                for(const role of report.purposes_available){const option=new Option(role,role);option.selected=item.purposes.includes(role);roles.add(option);}
                roles.onchange=()=>{cards[index].purposes=[...roles.selectedOptions].map(o=>o.value);write();};row.append(element("label","Roles"),roles);
                const scenes=element("input");scenes.value=item.scenes.join(", ");scenes.placeholder="scene_1, scene_2";scenes.setAttribute("aria-label", `Scenes ${item.key}`);
                scenes.onchange=()=>{cards[index].scenes=scenes.value.split(",").map(x=>x.trim()).filter(Boolean);write();};row.append(scenes);
                panel.append(row);
            }
            const prompt=element("pre",report.prompt);prompt.style.cssText="white-space:pre-wrap;overflow-wrap:anywhere";
            status.textContent=report.conditioning==="encoder_only"?"All references use encoder-only conditioning.":"References use the text encoder and native VAE reference path.";
            panel.append(status,element("strong","Compiled prompt"),prompt);
            node.setSize([Math.max(node.size[0],380),Math.max(node.size[1],620)]);
        };
    },
});
