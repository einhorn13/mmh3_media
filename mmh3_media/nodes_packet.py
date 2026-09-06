from __future__ import annotations

from .node_support import (
    Any,
    CATEGORY,
    MMH3,
    MMH3Media,
    MMH3ResourceError,
    _load_options,
    _file_selector,
    _packet,
    _parse_object,
    _resolve_load_path,
    _resolve_save_path,
    build_high_sigma_lora_plan,
    build_lora_reapply_expansion,
    clear_generation_loras,
    describe_media_payload,
    folder_paths,
    get_generation_loras,
    hashlib,
    infer_kind,
    io,
    ui,
    json,
    load_archive,
    lora_provenance_summary,
    save_archive,
    set_generation_loras,
)

from .h3_resource_semantics import make_context_contract, make_reference_contract
from .resource_model import CORE_RESOURCE_ROLES, descriptor_from_media_metadata
from .resolution import _keyframe_mode


def _task_value(task: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
    value = task if isinstance(task, dict) else {"task": task}
    label = str(value.get("task") or "")
    if label == "Video (optional frames)":
        return _keyframe_mode(value.get("first_frame") is not None, value.get("last_frame") is not None), value
    if label == "References to video":
        return "ref2va", value
    raise MMH3ResourceError(f"Unsupported H3 generation task {label!r}")


def _dynamic_media(group: Any) -> list[Any]:
    if not isinstance(group, dict):
        return []
    return [group[key] for key in sorted(group, key=lambda name: int(name.rsplit("_", 1)[-1]))
            if group[key] is not None]


def _require_metadata_action(field: str, action: str) -> None:
    if action not in {"keep", "set", "clear"}:
        raise MMH3ResourceError(f"Unsupported {field} action {action!r}")


def _validate_metadata_merge_patch(patch: dict[str, Any]) -> None:
    conflicts = sorted({"name", "notes", "tags"}.intersection(patch))
    generation_patch = patch.get("generation")
    if "generation" in patch and not isinstance(generation_patch, dict):
        conflicts.append("generation")
    elif isinstance(generation_patch, dict):
        conflicts.extend(
            f"generation.{key}"
            for key in sorted({"task", "prompt", "seed"}.intersection(generation_patch))
        )
    if conflicts:
        raise MMH3ResourceError(
            "custom JSON patch touches protected fields; use MMH3 Create or generation overrides for generation intent: "
            + ", ".join(conflicts)
        )


def _generation_from_video_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    generation: dict[str, Any] = {}
    dimensions = metadata.get("dimensions")
    if isinstance(dimensions, (list, tuple)) and len(dimensions) >= 2:
        try:
            width, height = int(dimensions[0]), int(dimensions[1])
            if width > 0 and height > 0:
                generation.update(width=width, height=height)
        except (TypeError, ValueError):
            pass
    try:
        frames = int(metadata.get("frame_count"))
        if frames > 0:
            generation["frames"] = frames
    except (TypeError, ValueError):
        pass
    try:
        fps = float(metadata.get("fps"))
        if fps > 0:
            generation["fps"] = fps
    except (TypeError, ValueError):
        pass
    if "fps" not in generation and "frames" in generation:
        try:
            duration = float(metadata.get("duration"))
            if duration > 0:
                generation["fps"] = generation["frames"] / duration
        except (TypeError, ValueError):
            pass
    return generation

class MMH3Create(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3Create",
            display_name="MMH3 Create",
            category=CATEGORY,
            description="Describe an H3 generation and attach only the inputs required by the selected task.",
            inputs=[
                io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    dynamic_prompts=False,
                    placeholder="Describe the shot, motion, camera, lighting and sound...",
                    tooltip="Main H3 generation prompt.",
                ),
                io.DynamicCombo.Input(
                    "task",
                    display_name="Generation type",
                    options=[
                        io.DynamicCombo.Option(
                            "Video (optional frames)",
                            [],
                        ),
                        io.DynamicCombo.Option(
                            "References to video",
                            [
                                io.Autogrow.Input(
                                    "reference_images",
                                    template=io.Autogrow.TemplatePrefix(io.Image.Input("image"), prefix="image_", min=0, max=10),
                                    optional=True,
                                    display_name="Reference images",
                                ),
                                io.Autogrow.Input(
                                    "reference_videos",
                                    template=io.Autogrow.TemplatePrefix(io.Video.Input("video"), prefix="video_", min=0, max=10),
                                    optional=True,
                                    display_name="Reference videos",
                                ),
                                io.Autogrow.Input(
                                    "reference_audios",
                                    template=io.Autogrow.TemplatePrefix(io.Audio.Input("audio"), prefix="audio_", min=0, max=10),
                                    optional=True,
                                    display_name="Reference audio",
                                ),
                            ],
                        ),
                    ],
                    tooltip="The selected task controls which media inputs are shown.",
                ),
                io.Int.Input(
                    "seed",
                    display_name="Seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Generation seed. ComfyUI can randomize or increment it after each run.",
                ),
                io.String.Input(
                    "name",
                    default="",
                    advanced=True,
                    tooltip="Optional archive label. If empty, a concise task-based name is used.",
                ),
                io.String.Input(
                    "notes",
                    default="",
                    multiline=True,
                    advanced=True,
                    placeholder="Optional notes...",
                    tooltip="Optional human note stored in the archive.",
                ),
                # Keep the default DynamicCombo branch socket-free: older frontends
                # create its children before creating the combo's input socket.
                io.Image.Input("first_frame", display_name="First frame", optional=True, tooltip="Video mode: optional first frame. Ignored in References mode."),
                io.Image.Input("last_frame", display_name="Last frame", optional=True, tooltip="Video mode: optional last frame. Connect either, both, or neither frame. Ignored in References mode."),
            ],
            outputs=[MMH3.Output("packet", tooltip="New in-memory packet containing the recorded intent and connected resources.")],
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
        task: dict[str, Any] | str,
        seed: int,
        name: str = "",
        notes: str = "",
        first_frame: Any = None,
        last_frame: Any = None,
    ) -> io.NodeOutput:
        task = dict(task) if isinstance(task, dict) else {"task": task}
        if task.get("task") == "Video (optional frames)":
            task.update(first_frame=first_frame, last_frame=last_frame)
        selected_task, task_inputs = _task_value(task)
        generation: dict[str, Any] = {"task": selected_task, "seed": int(seed)}
        if prompt:
            generation["prompt"] = prompt
        packet = MMH3Media.create(name=name.strip() or selected_task.upper(), generation=generation)
        if notes:
            packet = packet.edit_metadata(notes=notes)
        first_frame = task_inputs.get("first_frame")
        last_frame = task_inputs.get("last_frame")
        if first_frame is not None:
            facts = describe_media_payload(first_frame, "image")
            packet = packet.put(
                first_frame, kind="image", role="context", order=0,
                descriptor=descriptor_from_media_metadata("image", facts),
                extensions={"minimax_h3": {"context": make_context_contract(usage="first_frame")}},
                tags=["first-frame"], record_history=False,
            )
        if last_frame is not None:
            facts = describe_media_payload(last_frame, "image")
            packet = packet.put(
                last_frame, kind="image", role="context", order=1,
                descriptor=descriptor_from_media_metadata("image", facts),
                extensions={"minimax_h3": {"context": make_context_contract(usage="last_frame")}},
                tags=["last-frame"], record_history=False,
            )
        reference_order = 0
        for kind, key in (("image", "reference_images"), ("video", "reference_videos"), ("audio", "reference_audios")):
            for media in _dynamic_media(task_inputs.get(key)):
                facts = describe_media_payload(media, kind)
                packet = packet.put(
                    media,
                    kind=kind,
                    role="reference",
                    order=reference_order,
                    descriptor=descriptor_from_media_metadata(kind, facts),
                    extensions={"minimax_h3": {"reference": make_reference_contract(kind=kind)}},
                    tags=["reference"],
                    record_history=False,
                )
                reference_order += 1
        return io.NodeOutput(packet)


class MMH3Load(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3Load",
            display_name="MMH3 Load",
            category=CATEGORY,
            description="Load packet.json and the ZIP directory only. Heavy media/latents remain lazy.",
            inputs=[
                io.Combo.Input(
                    "file",
                    options=_load_options(),
                    tooltip="Archives in ComfyUI input/output. Use the standard ComfyUI Refresh (R), as with LoRA lists, to rescan files.",
                ),
                io.Combo.Input("verify", options=["on_access", "manifest", "full"], default="on_access", advanced=True),
                io.String.Input(
                    "path_override",
                    default="",
                    advanced=True,
                    tooltip="Optional absolute/local path for automation. The file selector is ignored when set.",
                ),
            ],
            outputs=[MMH3.Output("packet")],
        )

    @classmethod
    def execute(cls, file: str, verify: str, path_override: str) -> io.NodeOutput:
        path = _resolve_load_path(file, path_override)
        return io.NodeOutput(load_archive(path, verify=verify))

    @classmethod
    def fingerprint_inputs(cls, file: str, verify: str, path_override: str):
        try:
            path = _resolve_load_path(file, path_override)
            stat = path.stat()
            return (str(path), stat.st_mtime_ns, stat.st_size, verify)
        except Exception:
            return (file, path_override, verify, "missing")


class MMH3Save(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3Save",
            display_name="MMH3 Save",
            category=CATEGORY,
            description="Save a self-contained MMH3 archive with image/video previews. Existing media are byte-copied when possible.",
            inputs=[
                MMH3.Input("packet"),
                io.String.Input(
                    "filename_prefix",
                    display_name="Filename",
                    default="mmh3/MMH3",
                    tooltip="Output-relative folder and filename. Extension is added automatically; existing names get a numeric suffix unless overwrite is enabled.",
                ),
                io.Combo.Input(
                    "target",
                    options=["output", "source (in-place)"],
                    default="output",
                    advanced=True,
                    tooltip="Output creates an archive. Source replaces the original file, ignores Filename and requires overwrite.",
                ),
                io.Boolean.Input(
                    "overwrite",
                    default=False,
                    advanced=True,
                    tooltip="Allow replacing an existing archive at the resolved path. Keep off for safer iterative work.",
                ),
            ],
            outputs=[MMH3.Output("packet"), io.String.Output("path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, packet, filename_prefix: str, target: str, overwrite: bool) -> io.NodeOutput:
        packet = _packet(packet)
        path = _resolve_save_path(packet, filename_prefix, target, overwrite)
        saved, saved_path = save_archive(packet, path)
        segment = saved.manifest.get("extensions", {}).get("mmh3_media", {}).get("segment_plan") or {}
        return io.NodeOutput(saved, saved_path, ui={
            "text": [str(saved_path)],
            "mmh3_saved": [{"file": _file_selector(saved_path), "path": str(saved_path), "size_bytes": path.stat().st_size,
                            "segment_state": segment.get("state")}],
        })


class MMH3Put(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3Put", display_name="MMH3 Put", category=CATEGORY,
            description="Add or replace one canonical v0.3 resource.",
            inputs=[
                MMH3.Input("packet"),
                io.MultiType.Input("resource", types=[io.Latent, io.Image, io.Video, io.Audio, io.Mask]),
                io.Combo.Input("role", options=list(CORE_RESOURCE_ROLES), default="auxiliary"),
                io.Int.Input("order", default=-1, min=-1, max=9999, tooltip="-1 stores no semantic order."),
                io.Combo.Input("mode", options=["add", "upsert", "replace"], default="add"),
                io.Boolean.Input(
                    "primary",
                    default=False,
                    tooltip=(
                        "For video/audio/image/latent/mask, atomically replace and bind the primary resource. "
                        "When enabled, role/order/mode are ignored and the resource is stored as auxiliary with no order."
                    ),
                ),
                io.String.Input("resource_id", default="", advanced=True, tooltip="Required for deterministic replace; blank lets MMH3 select by role/kind/order for upsert."),
                io.String.Input("name", default="", advanced=True),
                io.String.Input("tags", default="", advanced=True, tooltip="Comma-separated canonical tags."),
                io.String.Input("descriptor_json", default="", multiline=True, advanced=True),
                io.String.Input("extensions_json", default="", multiline=True, advanced=True),
            ],
            outputs=[MMH3.Output("packet"), io.String.Output("resource_id")],
        )

    @classmethod
    def execute(cls, packet, resource, role: str, order: int, mode: str, primary: bool, resource_id: str, name: str, tags: str, descriptor_json: str, extensions_json: str) -> io.NodeOutput:
        packet = _packet(packet)
        kind = infer_kind(resource)
        observed = descriptor_from_media_metadata(kind, describe_media_payload(resource, kind))
        supplied = _parse_object(descriptor_json, "descriptor_json")
        observed.update(supplied)
        extensions = _parse_object(extensions_json, "extensions_json")
        parsed_tags = [item.strip() for item in tags.split(",") if item.strip()]
        if primary:
            out, rid = packet.put_primary(
                resource, kind=kind, resource_id=resource_id.strip(), name=name, tags=parsed_tags,
                descriptor=observed, extensions=extensions,
            )
        else:
            out, rid = packet.put_with_id(
                resource, kind=kind, role=role, order=None if int(order) < 0 else int(order), mode=mode,
                resource_id=resource_id.strip(), name=name, tags=parsed_tags, descriptor=observed, extensions=extensions,
            )
        return io.NodeOutput(out, rid)


class MMH3Remove(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3Remove", display_name="MMH3 Remove", category=CATEGORY,
            inputs=[
                MMH3.Input("packet"),
                io.Combo.Input("role", options=list(CORE_RESOURCE_ROLES), default="auxiliary"),
                io.Int.Input("order", default=-1, min=-1, max=9999),
                io.String.Input("resource_id", default="", advanced=True, tooltip="If set, overrides role/order."),
                io.Combo.Input("missing", options=["error", "ignore"], default="error", advanced=True),
            ], outputs=[MMH3.Output("packet")],
        )

    @classmethod
    def execute(cls, packet, role: str, order: int, resource_id: str, missing: str) -> io.NodeOutput:
        return io.NodeOutput(_packet(packet).remove(
            role=role, order=None if int(order) < 0 else int(order), resource_id=resource_id.strip(), missing=missing
        ))


class MMH3Move(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3Move", display_name="MMH3 Move", category=CATEGORY,
            description="Reorder resources within one generic v0.3 role; resource IDs remain stable.",
            inputs=[
                MMH3.Input("packet"),
                io.Combo.Input("role", options=list(CORE_RESOURCE_ROLES), default="reference"),
                io.Int.Input("from_order", default=0, min=0, max=9999),
                io.Int.Input("to_order", default=0, min=0, max=9999),
            ], outputs=[MMH3.Output("packet")],
        )

    @classmethod
    def execute(cls, packet, role: str, from_order: int, to_order: int) -> io.NodeOutput:
        return io.NodeOutput(_packet(packet).move(role=role, from_order=int(from_order), to_order=int(to_order)))


class MMH3Metadata(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3Metadata",
            display_name="MMH3 Metadata",
            category=CATEGORY,
            description="Edit packet name, tags, notes and advanced descriptive metadata without changing generation intent or media resources.",
            inputs=[
                MMH3.Input("packet"),
                io.String.Input("name", default="", tooltip="New packet name. Blank keeps the current name."),
                io.Combo.Input(
                    "tags_action",
                    options=["keep", "set", "clear"],
                    default="keep",
                    advanced=True,
                    tooltip="Keep, replace or clear packet-level tags.",
                ),
                io.String.Input(
                    "tags",
                    default="",
                    advanced=True,
                    tooltip="Comma-separated packet tags. Applied only when tags action is set.",
                ),
                io.Combo.Input(
                    "notes_action",
                    options=["keep", "set", "clear"],
                    default="keep",
                    advanced=True,
                    tooltip="Keep, replace or clear the human-readable packet note.",
                ),
                io.String.Input(
                    "notes",
                    default="",
                    multiline=True,
                    advanced=True,
                    placeholder="Your comment here...",
                    tooltip="Creator/user comment. Applied only when notes_action is set.",
                ),
                io.String.Input(
                    "custom_json_merge_patch",
                    default="",
                    multiline=True,
                    advanced=True,
                    tooltip="Advanced RFC 7396-style patch for descriptive or measured geometry/timing metadata. Name, tags, notes and generation task/prompt/seed are protected.",
                ),
            ],
            outputs=[MMH3.Output("packet", tooltip="Packet with updated metadata and unchanged media resources.")],
        )

    @classmethod
    def execute(
        cls,
        packet,
        name: str,
        tags_action: str,
        tags: str,
        notes_action: str,
        notes: str,
        custom_json_merge_patch: str,
    ) -> io.NodeOutput:
        patch = _parse_object(custom_json_merge_patch, "custom_json_merge_patch") if custom_json_merge_patch.strip() else None
        for field, action in (
            ("tags", tags_action),
            ("notes", notes_action),
        ):
            _require_metadata_action(field, action)
        if patch:
            _validate_metadata_merge_patch(patch)

        out = _packet(packet).edit_metadata(
            name=name,
            notes=(None if notes_action == "keep" else ("" if notes_action == "clear" else notes)),
        )
        if tags_action == "set":
            out = out.edit_metadata(tags=tags)
            if not tags.strip():
                out = out.edit_metadata(merge_patch_json={"tags": []})
        elif tags_action == "clear":
            out = out.edit_metadata(merge_patch_json={"tags": []})
        if patch:
            out = out.edit_metadata(merge_patch_json=patch)
        return io.NodeOutput(out)


class MMH3GenerationLoRAs(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3GenerationLoRAs",
            display_name="MMH3 Source LoRA List",
            category=CATEGORY,
            description="View and edit the ordered LoRA list used to create the source media. The list is provenance for reproducible refinement; this node does not load LoRAs.",
            inputs=[
                MMH3.Input("packet"),
                io.Combo.Input(
                    "action",
                    options=["view current", "save list", "mark no LoRAs", "remove provenance"],
                    default="view current",
                    tooltip="View leaves the packet unchanged. Save replaces the ordered list. Mark no LoRAs records an explicit empty list. Remove provenance makes the source LoRAs unknown.",
                ),
                io.String.Input(
                    "loras_list",
                    display_name="LoRAs list",
                    default="[]",
                    multiline=True,
                    placeholder='[{"name":"example.safetensors","strength_model":1.0,"strength_clip":1.0,"purpose":"style"}]',
                    tooltip="Editable ordered list. Each item needs name and model strength; CLIP strength may be a number or null. Order is preserved.",
                ),
            ],
            outputs=[
                MMH3.Output("packet"),
                io.String.Output("current_loras_list", display_name="current LoRAs list"),
                io.Int.Output("lora_count", display_name="LoRA count"),
                io.Boolean.Output(
                    "provenance_known",
                    display_name="LoRA provenance known",
                    tooltip="True when the packet explicitly records a list, including an empty list meaning no LoRAs were used.",
                ),
                io.String.Output("summary"),
            ],
        )

    @classmethod
    def execute(cls, packet, action: str, loras_list: str) -> io.NodeOutput:
        packet = _packet(packet)
        if action == "save list":
            try:
                value = json.loads(loras_list)
            except json.JSONDecodeError as exc:
                raise MMH3ResourceError(f"LoRAs list is not valid JSON: {exc}") from exc
            if not isinstance(value, list):
                raise MMH3ResourceError("LoRAs list must be an ordered JSON array")
            packet = set_generation_loras(packet, value)
        elif action == "mark no LoRAs":
            packet = set_generation_loras(packet, [])
        elif action == "remove provenance":
            packet = clear_generation_loras(packet)
        elif action != "view current":
            raise MMH3ResourceError(f"Unsupported LoRA provenance action {action!r}")
        loras = get_generation_loras(packet)
        current_list = "null" if loras is None else json.dumps(list(loras), ensure_ascii=False, indent=2)
        summary = lora_provenance_summary(packet)
        if loras:
            rows = []
            for index, entry in enumerate(loras, start=1):
                clip = "null" if entry.get("strength_clip") is None else entry["strength_clip"]
                rows.append(
                    f"{index}. {entry['name']} · model={entry['strength_model']:g} · clip={clip} · purpose={entry.get('purpose', 'unknown')}"
                )
            summary += "\n" + "\n".join(rows)
        return io.NodeOutput(
            packet,
            current_list,
            len(loras or ()),
            loras is not None,
            summary,
            ui=ui.PreviewText(summary),
        )


class MMH3H3RefineLoRAs(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MMH3H3RefineLoRAs",
            display_name="MMH3 H3 Refine Source LoRAs",
            category=CATEGORY,
            description=(
                "Reapply the packet's recorded source-generation LoRAs to standard MODEL/CLIP outputs "
                "in exact order before high-sigma refine. Missing/unknown overrides are explicit; hash mismatch blocks."
            ),
            inputs=[
                MMH3.Input("packet"),
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Combo.Input("unknown_policy", options=["error", "continue_without_source_loras"], default="error"),
                io.Combo.Input("missing_policy", options=["error", "skip_missing"], default="error"),
                io.String.Input("turbo_override", default="", optional=True, advanced=True,
                                tooltip="Replace only source acceleration LoRAs with this model-only LoRA at strength 1."),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Clip.Output("clip"),
                io.Int.Output("applied_count"),
                io.String.Output("summary"),
                io.String.Output("process_info_json"),
                io.String.Output("applied_loras_json"),
            ],
            enable_expand=True,
        )

    @classmethod
    def execute(
        cls,
        packet,
        model,
        clip,
        unknown_policy: str,
        missing_policy: str,
        turbo_override: str = "",
    ) -> io.NodeOutput:
        packet = _packet(packet)
        from .upscale_overrides import replace_upscale_turbo
        turbo_override = turbo_override.strip().replace('\\', '/')
        if turbo_override:
            packet = replace_upscale_turbo(packet, turbo_override)
        recorded = get_generation_loras(packet) or ()
        wanted_hash_names = {
            entry["name"]
            for entry in recorded
            if entry.get("reapply_for_high_sigma", True) and entry.get("sha256")
        }
        inventory: dict[str, str | None] = {}
        for catalog_name in folder_paths.get_filename_list("loras"):
            normalized = str(catalog_name).replace("\\", "/")
            digest = None
            if normalized in wanted_hash_names or normalized == turbo_override:
                full = folder_paths.get_full_path("loras", catalog_name)
                if full:
                    hasher = hashlib.sha256()
                    with open(full, "rb") as handle:
                        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                            hasher.update(chunk)
                    digest = hasher.hexdigest()
            inventory[normalized] = digest
        if turbo_override:
            if not inventory.get(turbo_override):
                raise MMH3ResourceError(f'Upscale Turbo LoRA is unavailable: {turbo_override}')
            packet = replace_upscale_turbo(packet, turbo_override, inventory[turbo_override])
        plan = build_high_sigma_lora_plan(
            packet,
            lora_inventory=inventory,
            allow_unknown=unknown_policy == "continue_without_source_loras",
            allow_missing=missing_policy == "skip_missing",
        )
        if not plan.ready:
            details = "; ".join(item.message for item in plan.diagnostics if item.severity == "error")
            raise MMH3ResourceError(f"{plan.summary()}: {details}")
        expansion = build_lora_reapply_expansion(plan, model=model, clip=clip)
        info = plan.to_dict()
        info["operation"] = "high_sigma_refine_lora_reapply"
        if turbo_override:
            info['turbo_override'] = turbo_override
        applied_loras = [
            {key: value for key, value in entry.items() if key != "source_index"}
            for entry in plan.applied
        ]
        summary = plan.summary()
        return io.NodeOutput(
            expansion.model,
            expansion.clip,
            len(plan.applied),
            summary,
            json.dumps(info, ensure_ascii=False, indent=2),
            json.dumps(applied_loras, ensure_ascii=False, indent=2),
            expand=expansion.graph,
        )
