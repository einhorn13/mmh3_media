from __future__ import annotations

import json

from .node_support import CATEGORY, MMH3, io, folder_paths, MMH3ResourceError
from . import long_video_ui as studio


def _prompt_id():
    from comfy_execution.utils import get_executing_context
    return get_executing_context().prompt_id


def _result(ledger, path=""):
    text = studio.summary(ledger)
    previews = ledger["studio"].get("previews", {})
    candidates = [{**candidate, "preview": previews.get(candidate["packet_path"])} for job in ledger["jobs"] for candidate in job.get("candidates", [])]
    return io.NodeOutput(json.dumps(ledger), str(path), text, ui={"text": [text], "mmh3_studio": [{"candidates": candidates, "path": str(path), "preview": previews.get(str(path)), "status": text}]})


def _queue_state():
    from server import PromptServer
    queue = PromptServer.instance.prompt_queue
    running, pending = queue.get_current_queue()
    active = {str(entry[1]) for entry in running + pending}
    failed = {str(key) for key, value in queue.get_history().items() if (value.get("status") or {}).get("status_str") == "error"}
    return active, failed


class MMH3LongVideoStudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LongVideoStudio", display_name="MMH3 Long Video Studio", category=CATEGORY,
            description="Complete audio-master video workflow. Generate/review scenes, then assemble. Sources and default render settings are frozen; edit individual scene prompts, delivered durations and references in Project Manager. Use a new project for different sources/models.",
            inputs=[
                io.String.Input("project", default="mmh3_studio/my_video"),
                io.Combo.Input("action", options=["generate_next", "accept", "reroll", "assemble", "inspect", "recover"], default="generate_next"),
                io.String.Input("prompt", default="Use <Picture 1> as the identity reference. Follow @master_audio exactly for singing timing and mouth articulation. Stable identity, natural motion.", multiline=True),
                io.Combo.Input("model_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("clip_name", options=folder_paths.get_filename_list("text_encoders")),
                io.Combo.Input("video_vae_name", options=folder_paths.get_filename_list("vae")),
                io.Combo.Input("audio_vae_name", options=folder_paths.get_filename_list("vae")),
                io.Int.Input("width", default=768, min=256, max=2048, step=32),
                io.Int.Input("height", default=448, min=256, max=2048, step=32),
                io.Int.Input("steps", default=20, min=1, max=100),
                io.Float.Input("scene_seconds", default=8.0, min=5.0, max=15.1, step=0.1),
                io.Float.Input("context_seconds", default=0.5, min=0.0, max=4.0, step=0.1),
                io.Boolean.Input("review_takes", default=True),
                io.Boolean.Input("auto_continue", default=False, tooltip="With review off: generate all pending scenes sequentially and assemble automatically."),
                io.String.Input("candidate_id", default="", advanced=True, tooltip="Empty selects the latest take; the take picker fills this automatically."),
                io.Image.Input("image", optional=True), io.Audio.Input("audio", optional=True),
                MMH3.Input("references", optional=True, tooltip="Optional reference library from H3 Reference Cards. Frozen when creating the project; scene selections are edited in Project Manager."),
            ],
            outputs=[io.String.Output("ledger_json"), io.String.Output("path"), io.String.Output("status")],
            is_output_node=True, enable_expand=True,
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # Filesystem ledger actions must run even when the graph's widgets are unchanged.
        return float("nan")

    @classmethod
    def execute(cls, project, action, prompt, model_name, clip_name, video_vae_name, audio_vae_name,
                width, height, steps, scene_seconds, context_seconds, review_takes, auto_continue,
                candidate_id="", image=None, audio=None, references=None):
        from .archive import save_archive
        from .core import MMH3Media
        from .media_metadata import describe_media_payload
        from .resource_model import descriptor_from_media_metadata
        from .h3_resource_semantics import make_reference_contract
        from .nodes_automation import MMH3LongAudioTimelinePlan, MMH3LongVideoAudioSyncSettings
        from .automation_execution import create_execution_ledger
        output_root = folder_paths.get_output_directory()
        directory = studio.project_directory(output_root, project)
        with studio.lock_project(directory):
            if (directory / "ledger.json").exists():
                ledger = studio.read(directory)
            else:
                if action != "generate_next" or (image is None and references is None) or audio is None:
                    raise MMH3ResourceError("New project: connect an identity image and master audio, then choose generate_next")
                if width % 32 or height % 32:
                    raise MMH3ResourceError("H3 width and height must be multiples of 32")
                import torchaudio.functional as AF
                rate = int(audio["sample_rate"])
                waveform = audio["waveform"]
                if rate != 32000:
                    waveform = AF.resample(waveform, rate, 32000)
                master = {"waveform": waveform, "sample_rate": 32000}
                if references is not None:
                    from .node_support import _packet
                    packet = _packet(references).edit_metadata(name=project, task="ref2va", prompt=prompt, seed=0)
                    if not any(r["role"] == "reference" and r["kind"] == "image" for r in packet.manifest["resources"]):
                        raise MMH3ResourceError("Studio reference library needs at least one identity image")
                else:
                    packet = MMH3Media.create(name=project, generation={"task": "ref2va", "prompt": prompt, "seed": 0})
                    packet = packet.put(image, kind="image", role="reference", order=0,
                        descriptor=descriptor_from_media_metadata("image", describe_media_payload(image, "image")),
                        extensions={"minimax_h3": {"reference": make_reference_contract(kind="image")}})
                packet, _ = packet.put_primary(master, kind="audio", descriptor=descriptor_from_media_metadata("audio", describe_media_payload(master, "audio")))
                packet, _ = save_archive(packet, directory / "source.mmh3")
                plan_json = MMH3LongAudioTimelinePlan.execute(packet, scene_seconds, context_seconds)[1]
                settings_json = MMH3LongVideoAudioSyncSettings.execute(packet, plan_json, "lipsync", "[]", source_mode="image_reference", require_candidate_review=review_takes)[0]
                ledger = create_execution_ledger(json.loads(plan_json), operation="long_video_lipsync", effective_settings=json.loads(settings_json))
                ledger["studio"] = {"default_scene_seconds": scene_seconds, "render": {"prompt": prompt, "model_name": model_name, "clip_name": clip_name,
                    "video_vae_name": video_vae_name, "audio_vae_name": audio_vae_name, "width": width, "height": height, "steps": steps}}
                studio.store(directory, ledger)
            if action in {"accept", "reroll"}:
                ledger = studio.review_action(ledger, action, candidate_id)
                studio.store(directory, ledger)
                return _result(ledger)
            if action == "recover":
                active, failed = _queue_state()
                return _result(studio.recover(directory, ledger, output_root, active, failed))
            if action == "inspect":
                return _result(ledger, ledger["studio"].get("final_path", ""))
        return MMH3LongVideoStudioStep.execute(project, auto_continue, action == "assemble")


class MMH3LongVideoStudioStep(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3LongVideoStudioStep", category=CATEGORY,
            inputs=[io.String.Input("project"), io.Boolean.Input("auto_continue", default=False), io.Boolean.Input("assemble", default=False)],
            outputs=[io.String.Output("ledger_json"), io.String.Output("path"), io.String.Output("status")],
            is_output_node=True, enable_expand=True)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def execute(cls, project, auto_continue=False, assemble=False):
        from comfy_execution.graph_utils import GraphBuilder
        output_root = folder_paths.get_output_directory()
        directory = studio.project_directory(output_root, project)
        graph = GraphBuilder()
        with studio.lock_project(directory):
            ledger = studio.read(directory)
            if assemble or ledger["status"] == "completed":
                if ledger["studio"].get("final_path"):
                    return _result(ledger, ledger["studio"]["final_path"])
                workflow = studio.assembly_workflow(ledger, project)
                nodes = studio.expand_api(graph, workflow)
                finish = graph.node("MMH3LongVideoStudioComplete", project=project, packet=nodes["4"].out(0), path=nodes["4"].out(1), video=nodes["3"].out(0), lease_id="", job_id="", auto_continue=False)
            else:
                ledger, lease = studio.acquire(directory, ledger, output_root, _prompt_id())
                nodes = studio.expand_api(graph, studio.job_workflow(ledger, lease))
                finish = graph.node("MMH3LongVideoStudioComplete", project=project, packet=nodes["20"].out(0), path=nodes["20"].out(1),
                    video=nodes["21"].out(0), lease_id=lease["lease_id"], job_id=lease["job_id"], auto_continue=auto_continue)
        return io.NodeOutput(finish.out(0), finish.out(1), finish.out(2), expand=graph.finalize())


class MMH3LongVideoStudioComplete(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3LongVideoStudioComplete", category=CATEGORY,
            inputs=[MMH3.Input("packet"), io.String.Input("path"), io.String.Input("project"),
                    io.String.Input("lease_id"), io.String.Input("job_id"), io.Boolean.Input("auto_continue", default=False), io.Video.Input("video", optional=True)],
            outputs=[io.String.Output("ledger_json"), io.String.Output("path"), io.String.Output("status")],
            is_output_node=True, enable_expand=True)

    @classmethod
    def execute(cls, packet, path, project, lease_id, job_id, auto_continue=False, video=None):
        directory = studio.project_directory(folder_paths.get_output_directory(), project)
        with studio.lock_project(directory):
            ledger = studio.read(directory)
            if job_id:
                ledger = studio.commit_saved(ledger, {"job_id": job_id, "lease_id": lease_id}, path)
                ledger["studio"].pop("submission", None)
            else:
                ledger["studio"]["final_path"] = str(path)
            # SaveVideo has already encoded this file; reuse it for browser preview.
            from pathlib import Path
            if video is not None:
                source = video.get_stream_source() if hasattr(video, "get_stream_source") else None
                if isinstance(source, str):
                    root = Path(folder_paths.get_output_directory()).resolve()
                    file = Path(source).resolve()
                    if file.is_relative_to(root):
                        relative = file.relative_to(root)
                        ledger["studio"].setdefault("previews", {})[str(path)] = {"filename": relative.name, "subfolder": relative.parent.as_posix(), "type": "output"}
            studio.store(directory, ledger)
        if job_id and auto_continue and ledger["effective_settings"].get("review_policy") != "candidate_required":
            from comfy_execution.graph_utils import GraphBuilder
            graph = GraphBuilder()
            next_step = graph.node("MMH3LongVideoStudioStep", project=project, auto_continue=True, assemble=False)
            return io.NodeOutput(next_step.out(0), next_step.out(1), next_step.out(2), expand=graph.finalize())
        return _result(ledger, path)
