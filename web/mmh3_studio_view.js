const el=(tag,text)=>{const e=document.createElement(tag);if(text!=null)e.textContent=text;return e;};
const button=(text,fn)=>{const b=el("button",text);b.type="button";b.onclick=fn;return b;};

export function renderStudioProject(container, data, {request, run, update, generate, apiURL, drafts = new Map()}) {
    container.append(el("h3",`Studio · ${data.name}`),el("p",`${data.status} · ${data.scenes.length} scenes · ${data.timeline.duration_seconds.toFixed(2)} s`));
    container.append(el("p","Accepted scenes are locked. Saving a reviewed scene creates a new settings revision and prepares another take. Duration changes reflow only never-rendered scenes."));
    const controls=el("div");controls.className="mmh3-project-toolbar";
    controls.append(button("Generate next",()=>run(()=>generate("generate_next"))),button("Assemble",()=>run(()=>generate("assemble"))));container.append(controls);
    const track=el("div");track.style.cssText="display:flex;gap:4px;overflow:auto;min-height:60px;margin:12px 0";
    for(const scene of data.scenes){const b=button(`${scene.index+1} · ${scene.status}\n${scene.duration_seconds.toFixed(2)} s`,()=>container.querySelector(`[data-studio-scene="${scene.id}"]`)?.scrollIntoView({block:"nearest"}));
        b.style.cssText=`min-width:${Math.max(100,scene.duration_seconds*24)}px;white-space:pre-wrap`;track.append(b);}container.append(track);
    const grid=el("div");grid.style.cssText="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:12px";container.append(grid);
    for(const scene of data.scenes){
        const draftKey=JSON.stringify([data.id,scene.id]);
        const draft=drafts.get(draftKey);
        const card=el("section");card.dataset.studioScene=scene.id;card.style.cssText="padding:12px;border:1px solid #596575;border-radius:8px;min-width:0";
        card.append(el("h4",`Scene ${scene.index+1} · ${scene.status} · revision ${scene.settings.revision}`),
            el("p",`Start ${scene.start_seconds.toFixed(2)} s · output ${scene.duration_seconds.toFixed(2)} s · H3 window ${scene.generation_seconds.toFixed(2)} s`));
        const prompt=el("textarea");prompt.value=draft?.prompt??scene.settings.prompt;prompt.setAttribute("aria-label",`Prompt scene ${scene.index+1}`);prompt.disabled=!scene.editable;prompt.style.cssText="width:100%;box-sizing:border-box;min-height:100px";
        const duration=el("input");duration.type="number";duration.min="0.05";duration.max="15";duration.step="0.05";duration.value=String(scene.duration_seconds);duration.setAttribute("aria-label",`Duration scene ${scene.index+1}`);
        duration.value=String(draft?.duration??scene.duration_seconds);
        duration.disabled=!scene.editable || scene.duration_editable===false || scene.status!=="pending" || scene.candidates.length>0;
        card.append(el("label","Prompt"),prompt,el("label","Delivered duration (seconds)"),duration);
        const references=el("div");references.style.cssText="display:flex;gap:8px;flex-wrap:wrap";
        const choices=[];
        for(const reference of data.references){
            const label=el("label");label.style.cssText="display:grid;gap:4px;max-width:110px";
            const selected=draft?.references??scene.settings.references;
            const check=el("input");check.type="checkbox";check.checked=selected===null ? reference.enabled!==false : selected.includes(reference.key);check.disabled=!scene.editable;
            check.setAttribute("aria-label",`Reference ${reference.key} scene ${scene.index+1}`);choices.push([reference.key,check]);
            if(reference.kind==="image"){const img=el("img");img.alt=reference.name;img.loading="lazy";img.style.cssText="width:90px;height:70px;object-fit:contain";
                img.src=apiURL(`/mmh3_media/studio_reference?${new URLSearchParams({project_id:data.id,key:reference.key})}`);label.append(img);}
            label.append(check,el("span",reference.name||reference.key));references.append(label);
        }
        card.append(el("p","Active references"),references);
        const values=()=>({prompt:prompt.value,duration:Number(duration.value),references:choices.filter(([,c])=>c.checked).map(([key])=>key)});
        card.addEventListener("input",event=>{
            if(!scene.editable || !event.target.matches("input,textarea,select"))return;
            const edited=values();
            const originalReferences=scene.settings.references??data.references.filter(r=>r.enabled!==false).map(r=>r.key);
            const unchanged=edited.prompt===scene.settings.prompt && edited.duration===scene.duration_seconds &&
                JSON.stringify([...edited.references].sort())===JSON.stringify([...originalReferences].sort());
            if(unchanged)drafts.delete(draftKey);else drafts.set(draftKey,edited);
        });
        if(scene.editable)card.append(button("Save scene revision",()=>run(async()=>{
            const edited=values();
            if(Math.abs(edited.duration-scene.duration_seconds)>1/48 && [...drafts.keys()].some(key=>key!==draftKey))
                throw new Error("Save other scene edits before changing a duration; the pending timeline will be replanned.");
            const value=await request("studio_edit",{job_id:scene.id,...edited});
            drafts.delete(draftKey);update(value);
        })));
        for(const take of scene.candidates){
            const box=el("div");box.append(el("p",`Take ${take.attempt} · scene revision ${take.scene_revision??0}`));
            if(take.scene_prompt)box.append(el("p",take.scene_prompt));
            if(take.preview){const video=el("video");video.controls=true;video.preload="metadata";video.style.cssText="width:100%;max-height:220px";video.src=apiURL(`/view?${new URLSearchParams(take.preview)}`);box.append(video);}
            if(scene.status==="review")box.append(button("Accept this take",()=>run(async()=>update(await request("studio_accept",{job_id:scene.id,candidate_id:take.candidate_id})))));
            card.append(box);
        }
        if(scene.status==="review")card.append(button("Reroll",()=>run(async()=>{update(await request("studio_reroll",{job_id:scene.id}));await generate("generate_next");})));
        grid.append(card);
    }
}
