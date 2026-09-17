import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// Retain saved results while the dialog is closed; backend still validates every import.
const savedResults = new Map();

const el = (tag, text, cls = "") => {
    const node = document.createElement(tag); node.className = cls;
    if (text != null) node.textContent = text;
    return node;
};
const button = (text, action) => { const b = el("button", text); b.type = "button"; b.onclick = action; return b; };
function upstream(node, name = "packet") {
    const id = node?.inputs?.find(i => i.name === name)?.link;
    const link = node?.graph?.links?.get?.(id) ?? node?.graph?.links?.[id];
    return link ? node.graph.getNodeById(link.origin_id) : null;
}
function find(node, kind) {
    const seen = new Set();
    while (node && !seen.has(node.id)) {
        if (node.comfyClass === kind) return node;
        seen.add(node.id); node = upstream(node);
    }
    return null;
}
export function configureWorkflow(origin, handoff) {
    const graph = origin.graph;
    const prepares = (graph?._nodes || []).filter(n => n.comfyClass === "MMH3H3SegmentPrepare");
    const prepare = find(origin, "MMH3H3SegmentPrepare") ??
        prepares.find(n => find(upstream(n), "MMH3Load") === origin) ?? (prepares.length === 1 ? prepares[0] : null);
    const source = find(upstream(prepare), "MMH3Load");
    const review = (graph?._nodes || []).find(n => n.comfyClass === "MMH3SegmentReview" && find(upstream(n), "MMH3H3SegmentPrepare") === prepare);
    const chains = [prepare, review].map(n => find(upstream(n, "chain_packet"), "MMH3Load"));
    if (!prepare || !source || !review || (handoff.target_segment_id && chains.some(n => !n || n === source)))
        throw new Error("Open the F04 Segment Workflow with Source, Prepare, Review and separate chain loaders for reroll.");
    const set = (node, name, value) => {
        const w = node.widgets?.find(w => w.name === name);
        if (!w) throw new Error(`Missing workflow widget: ${name}`);
        if (name === "file" && Array.isArray(w.options?.values) && !w.options.values.includes(value)) w.options.values.push(value);
        w.value = value; w.callback?.(value, app.canvas, node, app.canvas?.graph_mouse, {});
    };
    // Verify all widgets before changing any graph state.
    const changes = [[source,"file",handoff.source_file],[source,"path_override",""],
        [prepare,"action",handoff.action],[prepare,"target_segment_id",handoff.target_segment_id],
        [prepare,"prompts",handoff.prompts],[prepare,"seed",handoff.seed],[review,"decision","Draft"]];
    if (handoff.target_segment_id) for (const chain of new Set(chains))
        changes.push([chain,"file",handoff.chain_file],[chain,"path_override",""]);
    if (changes.some(([node,name]) => !node.widgets?.some(w => w.name === name))) throw new Error("Workflow widgets are incomplete.");
    for (const node of graph._nodes || []) if (node.comfyClass === "MMH3Save" && find(upstream(node), "MMH3SegmentReview") === review) {
        node.properties ??= {}; node.properties.mmh3_project_id = origin.properties?.mmh3_project_id;
    }
    graph.beforeChange?.();
    try { for (const [node,name,value] of changes) set(node,name,value); }
    finally { graph.afterChange?.(); graph.setDirtyCanvas?.(true,true); }
}

export async function openProjectManager(origin, initialFile = "") {
    document.querySelector(".mmh3-project-manager")?.close();
    if (!document.getElementById("mmh3-project-style")) {
        const style = el("style"); style.id = "mmh3-project-style";
        style.textContent = `
        .mmh3-project-manager{width:min(1120px,94vw);max-height:90vh;box-sizing:border-box;overflow:auto;background:#20242b;color:#edf1f5;border:1px solid #596575;border-radius:12px;padding:20px;font:14px/1.5 system-ui}
        .mmh3-project-manager::backdrop{background:#000b}.mmh3-project-manager *{box-sizing:border-box}
        .mmh3-project-manager h2,.mmh3-project-manager h3{margin:0 0 12px}.mmh3-project-manager button,.mmh3-project-manager select,.mmh3-project-manager input,.mmh3-project-manager textarea{background:#303945;color:inherit;border:1px solid #64748b;border-radius:6px;padding:8px;font:inherit;max-width:100%}
        .mmh3-project-manager button{cursor:pointer}.mmh3-project-manager button:disabled{opacity:.45;cursor:default}.mmh3-project-manager button:focus-visible{outline:2px solid #8ac9ff;outline-offset:2px}
        .mmh3-project-toolbar{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px;align-items:center}.mmh3-project-toolbar select{flex:1;min-width:170px}
        .mmh3-project-grid{display:grid;grid-template-columns:240px 1fr;gap:18px}.mmh3-project-list{display:grid;gap:8px;align-content:start}.mmh3-project-list button{text-align:left;overflow-wrap:anywhere}.mmh3-project-list .selected{border-color:#8ac9ff;background:#284665}
        .mmh3-project-previews{display:grid;grid-template-columns:1fr 1fr;gap:10px}.mmh3-project-previews img,.mmh3-project-previews video{width:100%;height:190px;object-fit:contain;background:#10151b;border-radius:6px}
        .mmh3-project-manager label{display:block;margin:10px 0 4px}.mmh3-project-manager textarea{width:100%;height:85px}.mmh3-project-note{color:#b7c7d9;overflow-wrap:anywhere}.mmh3-project-error{color:#ffc09a}.mmh3-project-confirm{padding:14px;margin:12px 0;border:1px solid #d1a266;border-radius:8px;background:#3a3026}
        @media(max-width:700px){.mmh3-project-grid,.mmh3-project-previews{grid-template-columns:1fr}.mmh3-project-manager{padding:12px}.mmh3-project-previews img,.mmh3-project-previews video{height:150px}}
        `; document.head.append(style);
    }
    const dialog = el("dialog", null, "mmh3-project-manager"); dialog.setAttribute("aria-label", "MMH3 Project Manager");
    const title = el("h2", "Project Manager"), status = el("p", "", "mmh3-project-note"); status.setAttribute("role","status");
    const top = el("div", null, "mmh3-project-toolbar"), content = el("div");
    const projects = el("select"); projects.setAttribute("aria-label","Project");
    const files = el("select"); files.setAttribute("aria-label","Saved archive");
    let data = null, busy = false, closed = false, timer, confirmation = null;
    const compose = {target:"",prompts:"",seed:Math.floor(Math.random()*4294967295)};
    const pendingSaves = [];
    const operations = new Map();
    const message = (text, error = false) => { status.textContent = text; status.className = error ? "mmh3-project-error" : "mmh3-project-note"; };
    async function request(action, extra = {}) {
        const response = await api.fetchApi("/mmh3_media/project", {method:"POST", headers:{"Content-Type":"application/json"},
            body:JSON.stringify({action,project_id:data?.id,expected_state_digest:data?.state.state_digest,...extra})});
        const value = await response.json();
        if (!response.ok) { const error = new Error(value.error || `HTTP ${response.status}`); error.status = response.status; throw error; }
        return value;
    }
    async function run(action) {
        if (busy || closed) return;
        busy = true; dialog.querySelectorAll("button,select,input,textarea").forEach(n => n.disabled = true);
        try { await action(); }
        catch (error) {
            if (error.status === 409 && data) { data = await request("state").catch(() => data); render(); }
            confirmation?.remove(); confirmation = null; message(error.message, true);
        } finally { busy = false; if (!closed) {
            dialog.querySelectorAll("button,select,input,textarea").forEach(n => n.disabled = false);
            if (pendingSaves.length) queueMicrotask(importSaved);
        } }
    }
    function importSaved() {
        if (busy || closed || !pendingSaves.length) return;
        const saved = pendingSaves.shift();
        if (saved.project_id !== data?.id) return;
        void run(async () => {
            data = await request("state");
            data = await request("add",{file:saved.file}); render();
            savedResults.set(saved.project_id,(savedResults.get(saved.project_id)||[]).filter(file=>file!==saved.file));
            message("New take added automatically. Select a candidate, render another take, or accept and continue.");
        });
    }
    const onSaved = event => { if(event.detail.project_id === data?.id){pendingSaves.push(event.detail);importSaved();} };
    window.addEventListener("mmh3-project-saved",onSaved);
    async function inventory() {
        const [p,f] = await Promise.all([api.fetchApi("/mmh3_media/projects"),api.fetchApi("/mmh3_media/batch_files")]);
        if (!p.ok || !f.ok) throw new Error("Could not load projects and saved archives");
        projects.replaceChildren(new Option("Choose a project…", ""));
        for (const item of await p.json()) projects.add(new Option(item.name + " · " + item.id.slice(0,8), item.id));
        const previous = files.value || initialFile; files.replaceChildren(new Option("Choose a saved .mmh3…", ""));
        for (const item of (await f.json()).filter(x => x.endsWith(".mmh3") && !x.includes(".mmh3_review/"))) files.add(new Option(item,item));
        if ([...files.options].some(o => o.value === previous)) files.value = previous;
        projects.value = data?.id || origin.properties?.mmh3_project_id || "";
    }
    function previewPane(label, candidateId) {
        const pane = el("div"), media = el("img"); media.alt = label;
        const query = new URLSearchParams({project_id:data.id,v:data.state.state_digest});
        if (candidateId) query.set("candidate_id",candidateId);
        const url = "/mmh3_media/project_media?" + query;
        media.src = api.apiURL?.(url) ?? url;
        media.onerror = () => { media.replaceWith(el("p","No cached image preview.")); };
        const play = button("Play saved video", () => {
            const video = el("video"); video.controls = true; video.preload = "metadata";
            video.src = api.apiURL?.(url + "&kind=video") ?? url + "&kind=video";
            video.onerror = () => video.replaceWith(el("p","No decoded video in this archive."));
            pane.replaceChildren(el("h3",label),video);
        });
        pane.append(el("h3",label),media,play); return pane;
    }
    function render() {
        confirmation = null; content.replaceChildren();
        if (!data) { content.append(el("p","Choose an existing project or create one from a saved accepted segment.")); return; }
        origin.properties ??= {}; origin.properties.mmh3_project_id = data.id;
        projects.value = data.id;
        const grid = el("div",null,"mmh3-project-grid"), timeline = el("div",null,"mmh3-project-list"), detail = el("div");
        timeline.append(el("h3","Segments"));
        const target = el("select"); target.setAttribute("aria-label","Render action"); target.add(new Option("Continue from accepted head", ""));
        for (const s of data.state.segments) {
            const label = `${s.index + 1}. ${s.scene_id} · r${s.active_revision} · ${s.status}`;
            timeline.append(s.status === "accepted" ? button(label,() => {compose.target=s.segment_id;compose.prompts="";render();message("Segment selected for a replacement take. Its saved parent will be located automatically.");}) : el("div",label,"mmh3-project-note"));
            if (s.status === "accepted") target.add(new Option(`Reopen segment ${s.index + 1}`,s.segment_id));
        }
        target.value=compose.target; target.onchange=()=>{compose.target=target.value;};
        const selected = data.candidates.find(c => c.selected && c.status === "review" && !c.stale);
        detail.append(el("h3",data.state.name));
        const previews = el("div",null,"mmh3-project-previews"); previews.append(previewPane("Accepted head"));
        if (selected) previews.append(previewPane("Selected candidate",selected.id));
        detail.append(previews,el("h3","Saved candidates"));
        const candidates = el("div",null,"mmh3-project-list");
        if (!data.candidates.length) candidates.append(el("p","Choose a saved draft above and add it to compare candidates."));
        for (const c of data.candidates) {
            const row = el("div",null,"mmh3-project-toolbar");
            row.append(el("span",`${c.name} · seed ${c.seed} · ${c.stale ? "stale" : c.status}`));
            if (!c.stale && !["accepted","trashed"].includes(c.status)) {
                const choose = button(c.selected ? "Selected" : "Select",() => run(async () => { data = await request("select",{candidate_id:c.id}); render(); }));
                if (c.selected) choose.className = "selected"; row.append(choose);
                if (c.status !== "rejected") row.append(button("Reject",() => run(async () => { data = await request("reject",{candidate_id:c.id}); render(); })));
            }
            candidates.append(row);
        }
        detail.append(candidates);
        const branchName = el("input"); branchName.placeholder = "Branch name (optional)"; branchName.setAttribute("aria-label","Branch name");
        const branchSource = el("select"); branchSource.setAttribute("aria-label","Branch source");
        branchSource.add(new Option("Accepted head", "head"));
        branchSource.add(new Option("Saved historical segment (archive above)", "saved"));
        for (const candidate of data.candidates.filter(c => c.status !== "trashed")) branchSource.add(new Option(`Candidate: ${candidate.name} · ${candidate.status}`, candidate.id));
        const branchTools=el("details");branchTools.append(el("summary","Independent branch"));detail.append(branchTools);
        branchTools.append(branchName,branchSource,button("Preview branch",() => run(async () => {
            const source = branchSource.value;
            const args = {candidate_id:source === "head" || source === "saved" ? "" : source,
                source_file:source === "saved" ? files.value : "", name:branchName.value};
            if (source === "saved" && !args.source_file) throw new Error("Choose a saved historical archive above.");
            const preview = await request("branch_preview",args);
            const expected = data.state.state_digest;
            const operationId = crypto.randomUUID();
            confirmation?.remove(); confirmation = el("section",null,"mmh3-project-confirm");
            confirmation.append(el("strong","Create independent branch"),el("p",`${preview.description} Source archive: ${(preview.source_archive_bytes / 1048576).toFixed(1)} MiB. The copy requires additional disk space.`),
                button("Confirm branch",() => run(async () => {
                    data = await request("branch",{...args,expected_state_digest:expected,source_sha256:preview.source_sha256,operation_id:operationId});
                    await inventory(); render(); message("Independent branch created and opened. Source project preserved.");
                })),button("Cancel",() => { confirmation.remove(); confirmation = null; }));
            detail.append(confirmation);
        })));
        if (selected) detail.append(button("Accept selected candidate",() => run(async () => {
            const impact = await request("impact",{candidate_id:selected.id});
            confirmation?.remove(); confirmation = el("section",null,"mmh3-project-confirm");
            confirmation.append(el("strong","Confirm acceptance"),el("p",impact.invalidated_segment_ids.length
                ? `This replaces the selected segment and invalidates ${impact.invalidated_segment_ids.length} later segment(s). Their archives are preserved.`
                : "This publishes the selected candidate as the accepted head. The previous archive is preserved."));
            confirmation.append(button("Confirm acceptance",() => run(async () => {
                const key = data.id + data.state.state_digest + selected.id;
                if (!operations.has(key)) operations.set(key,crypto.randomUUID());
                data = await request("accept",{candidate_id:selected.id,operation_id:operations.get(key)});
                compose.target=""; compose.prompts=""; compose.seed=Math.floor(Math.random()*4294967295);
                render(); message("Candidate accepted. The workflow can now continue from this head.");
            })),button("Cancel",() => { confirmation.remove(); confirmation = null; })); detail.append(confirmation);
        })));
        const prompt = el("textarea"); prompt.setAttribute("aria-label","Segment prompts"); prompt.placeholder = "Blank inherits the saved prompt plan";
        prompt.value=compose.prompts; prompt.oninput=()=>{compose.prompts=prompt.value;};
        const seed = el("input"); seed.type = "number"; seed.min = "0"; seed.max = "4294967295"; seed.value = String(compose.seed); seed.setAttribute("aria-label","Seed");
        seed.oninput=()=>{compose.seed=Number(seed.value);};
        async function prepare(queue = false, newSeed = false) {
            if (queue && typeof app.queuePrompt !== "function") throw new Error("Queue API is unavailable in this frontend. Use Prepare in workflow and the normal Queue button.");
            if(newSeed){compose.seed=Math.floor(Math.random()*4294967295);seed.value=String(compose.seed);}
            const handoff=await request("handoff",{target_segment_id:target.value,parent_file:files.value,prompts:prompt.value,seed:Number(seed.value)});
            configureWorkflow(origin,handoff);
            if(queue){
                const queued=await app.queuePrompt(0,1);
                if(queued===false)throw new Error("The frontend did not queue this workflow. Check its validation errors.");
                message("Render submitted to the current F04 workflow. A saved draft will appear here automatically; acceptance remains your choice.");
            }
            else message("Workflow prepared in Draft mode. Use Queue to render; its saved draft will be added here automatically.");
        }
        detail.append(el("h3","Render another segment or take"),target,el("label","Prompts"),prompt,el("label","Seed"),seed,
            el("p","Render keeps the accepted chain unchanged. Reopen finds its saved parent automatically; for imported history, attach missing accepted archives below.","mmh3-project-note"),
            button("Prepare in workflow",() => run(()=>prepare())),
            button("Render draft",() => run(()=>prepare(true))),
            button("Render another take · new seed",() => run(()=>prepare(true,true))));
        detail.append(el("h3","Assemble final video"),
            button("Attach saved accepted segment",() => run(async()=>{data=await request("attach_segment",{file:files.value});render();message("Saved segment attached. It is available for automatic parent lookup and assembly.");})),
            button("Preview final assembly",() => run(async()=>{
                const plan=await request("assembly_preview");
                confirmation?.remove(); confirmation=el("section",null,"mmh3-project-confirm");
                confirmation.append(el("strong",`${plan.segments.length} accepted segment(s) · ${plan.duration_seconds.toFixed(2)} seconds`),el("p",plan.policy));
                for(const segment of plan.segments)confirmation.append(el("div",`Segment ${segment.index+1} · r${segment.revision} · ${segment.owned_frames} frames · remove ${segment.video_trim_start_frame} context frames`));
                for(const error of plan.errors)confirmation.append(el("p",error,"mmh3-project-error"));
                if(plan.ready)confirmation.append(button("Assemble and save MP4",()=>run(async()=>{
                    message("Assembling accepted video and audio. Encoding may take several minutes.");
                    const result=await request("export",{expected_state_digest:plan.state_digest,assembly_digest:plan.assembly_digest});
                    const link=el("a","Download final video");link.style.color="#8ac9ff";
                    const url="/mmh3_media/project_export?"+new URLSearchParams({project_id:data.id,export_id:result.export_id});
                    link.href=api.apiURL?.(url)??url;link.download="final.mp4";
                    confirmation.replaceChildren(el("p",`Saved: ${result.filename}`),link);
                    message("Final video saved. The accepted chain and source archives are preserved.");
                })));
                detail.append(confirmation);
            })));
        detail.append(el("h3","Project storage"),button("Inspect storage",() => run(async () => {
            const report = await request("storage");
            confirmation?.remove(); confirmation = el("section",null,"mmh3-project-confirm");
            confirmation.append(el("strong","Storage report"),el("p",report.policy));
            for (const [kind,bytes] of Object.entries(report.bytes)) confirmation.append(el("div",`${kind}: ${(bytes/1048576).toFixed(2)} MiB`));
            for (const item of report.candidates) {
                const row = el("div",null,"mmh3-project-toolbar");
                row.append(el("span",`${item.name}: ${(item.bytes/1048576).toFixed(2)} MiB${item.protected ? " · protected by acceptance or snapshot" : ""}${!item.integrity_ok ? " · missing or changed file" : ""}`));
                if (item.can_trash || item.can_restore) row.append(button(item.can_restore ? "Restore rejected candidate" : "Move rejected candidate to trash",() => run(async () => {
                    data = await request(item.can_restore ? "restore" : "trash",{candidate_id:item.id,
                        expected_state_digest:report.state_digest,storage_digest:report.storage_digest});
                    render(); message(item.can_restore ? "Candidate restored as rejected." : "Candidate moved to trash. Its archive is preserved; disk usage is unchanged.");
                })));
                confirmation.append(row);
            }
            detail.append(confirmation);
        })));
        grid.append(timeline,detail); content.append(grid);
    }
    projects.onchange = () => run(async () => { if (!projects.value) return; data = await request("state",{project_id:projects.value}); render(); message("Project loaded."); });
    top.append(projects,button("Refresh",() => run(async () => { await inventory(); if(data) data=await request("state"); render(); message("Refreshed."); })),button("Close",() => dialog.close()));
    const sourcebar = el("div",null,"mmh3-project-toolbar");
    sourcebar.append(files,button("Create from accepted archive",() => run(async () => { data=await request("create",{file:files.value}); await inventory(); render(); message("Project created. Original archive preserved."); })),
        button("Add saved draft",() => run(async () => { if(!data) throw new Error("Choose a project first."); data=await request("add",{file:files.value}); render(); message("Candidate added. Select it to compare and accept."); })));
    dialog.append(title,top,sourcebar,status,content); document.body.append(dialog); dialog.showModal(); render();
    dialog.addEventListener("close",() => { closed=true;clearInterval(timer);window.removeEventListener("mmh3-project-saved",onSaved);dialog.querySelectorAll("video").forEach(v=>v.pause());dialog.remove(); },{once:true});
    await run(async () => {
        await inventory(); if(projects.value) data=await request("state",{project_id:projects.value}); render();
        for(const file of savedResults.get(data?.id)||[])pendingSaves.push({project_id:data.id,file});
    });
    if (!closed) timer=setInterval(() => { if(!busy && data && !closed) void run(async () => {
        const fresh=await request("state"); if(fresh.state.state_digest!==data.state.state_digest){data=fresh;render();message("Project changed elsewhere. Review the refreshed state before acting.");}
    }); },15000);
}

app.registerExtension({name:"mmh3.media.project_manager",nodeCreated(node){
    if(!["MMH3Load","MMH3Save","MMH3SegmentReview"].includes(node.comfyClass))return;
    let saved=""; const executed=node.onExecuted;
    node.onExecuted=function(message,...args){
        const file=message?.mmh3_saved?.[0]?.file; saved=file||saved;
        if(file && node.comfyClass === "MMH3Save" && node.properties?.mmh3_project_id) {
            const id=node.properties.mmh3_project_id;
            savedResults.set(id,[...new Set([...(savedResults.get(id)||[]),file])]);
            window.dispatchEvent(new CustomEvent("mmh3-project-saved",{detail:{file,project_id:node.properties.mmh3_project_id}}));
        }
        return executed?.call(this,message,...args);
    };
    const open=button("Open Project Manager",()=>void openProjectManager(node,saved||node.widgets?.find(w=>w.name==="file")?.value||""));
    node.addDOMWidget("mmh3_project_manager","button",open,{serialize:false,getMinHeight:()=>34,getMaxHeight:()=>34});
}});
