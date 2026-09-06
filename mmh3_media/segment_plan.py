from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .chain import start_chain, validate_chain, validate_reroll_source, commit_chain_segment
from .chain import _packet_state_fingerprint
from .core import MMH3Media
from .errors import MMH3ResourceError
from .util import deep_copy_json, json_dumps_canonical


SEGMENT_ACTIONS = ("First segment", "Continue", "Reroll accepted", "New scene")
PLAN_CONTRACT = "mmh3_segment_plan_v1"


def segment_timing(frames: int, context_frames: int = 0) -> dict:
    frames, context_frames = int(frames), int(context_frames)
    if frames < 5 or (frames - 5) % 17:
        raise MMH3ResourceError("Segment frames must use the H3 17k+5 grid")
    if context_frames and (context_frames < 39 or (context_frames - 39) % 51):
        raise MMH3ResourceError("Context must use exact AV boundaries: 39, 90, 141, ...")
    if context_frames < 0 or context_frames >= frames:
        raise MMH3ResourceError("The segment must contain new frames after its context")
    return {
        "generated_frames": frames, "context_frames": context_frames,
        "new_frames": frames - context_frames, "fps": 24,
        "generated_seconds": frames / 24, "context_seconds": context_frames / 24,
        "new_seconds": (frames - context_frames) / 24,
        "prompt_to_output_offset_seconds": -context_frames / 24,
    }


def timing_summary(timing: dict) -> str:
    return (f"{timing['generated_seconds']:.2f}s generated · "
            f"{timing['context_seconds']:.2f}s context · {timing['new_seconds']:.2f}s new")


def _namespace(packet: MMH3Media) -> dict:
    return packet.manifest.get("extensions", {}).get("mmh3_media", {})


def _ledger_digest(chain: object) -> str:
    return hashlib.sha256(json_dumps_canonical(chain).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SegmentPlan:
    packet: MMH3Media
    info: dict

    @property
    def summary(self) -> str:
        return f"Segment {self.info['prompt_index'] + 1} · {timing_summary(self.info['timing'])}"


def prepare_segment(packet: MMH3Media, *, action: str, prompts: str = "", seed: int = -1,
                    frames: int = 124, context_frames: int = 39,
                    chain_packet: MMH3Media | None = None, target_segment_id: str = "") -> SegmentPlan:
    """Select from accepted state only. Queueing a draft never advances the plan."""
    if action not in SEGMENT_ACTIONS:
        raise MMH3ResourceError(f"Unknown segment action {action!r}")
    ledger = chain_packet if chain_packet is not None else packet
    prior = _namespace(ledger).get("segment_plan") or {}
    if prior.get("state") == "draft":
        raise MMH3ResourceError("Load the accepted parent, not a draft, to prepare another segment")
    ledger = start_chain(ledger)
    report = validate_chain(ledger, require_head_state=True)
    if not report.ready:
        raise MMH3ResourceError("Invalid accepted chain: " + "; ".join(report.reasons))
    chain = deep_copy_json(_namespace(ledger)["chain"])
    if not chain["segments"] and "root_state" not in chain:
        chain["root_state"] = {"packet_manifest_id": packet.manifest["id"], "output_resource_ids": {},
                               "packet_state_fingerprint": _packet_state_fingerprint(packet, {})}
    head = next((s for s in chain["segments"] if s["segment_id"] == chain["head_segment_id"]), None)
    target = None
    if action == "First segment":
        if head is not None:
            raise MMH3ResourceError("First segment requires an empty chain; use New scene")
        index, commit_action, operation = 0, "append", "generate"
    elif action == "Reroll accepted":
        if chain_packet is None:
            raise MMH3ResourceError("Reroll requires the accepted chain and its separately loaded parent packet")
        target_segment_id = target_segment_id.strip() or chain["head_segment_id"]
        proof = validate_reroll_source(ledger, packet, target_segment_id=target_segment_id)
        if not proof.ready:
            raise MMH3ResourceError("Reroll parent mismatch: " + "; ".join(proof.reasons))
        target = next(s for s in chain["segments"] if s["segment_id"] == target_segment_id)
        previous = target.get("segment_plan") or {}
        index = int(previous.get("prompt_index", target["index"]))
        commit_action = "reroll"
        operation = "generate" if target.get("handover") in {"initial", "reanchor"} else "continuation"
    else:
        if head is None and action == "Continue" and packet.get_primary("latent") is None:
            raise MMH3ResourceError("Continue requires a source latent; use First segment")
        if chain_packet is not None and chain_packet is not packet:
            # An append must start from the authoritative head, never from an older loaded source.
            if packet.manifest != chain_packet.manifest:
                raise MMH3ResourceError("Continue/New scene must use the accepted head as its source")
        index = int((head.get("segment_plan") or {}).get("prompt_index", head["index"])) + 1 if head else 0
        commit_action = "reanchor" if action == "New scene" else "append"
        operation = "generate" if action == "New scene" else "continuation"

    inherited = (target or {}).get("segment_plan") or prior
    text = prompts.strip() or str(inherited.get("prompts") or "")
    entries = [part.strip() for part in re.split(r"(?m)^\s*---\s*$", text)] if text else []
    if entries and any(not p for p in entries):
        raise MMH3ResourceError("Each prompt between --- separators must be non-empty")
    if len(entries) > 1 and index >= len(entries):
        raise MMH3ResourceError("Prompt plan finished. Add a prompt or use a single prompt for all segments")
    prompt = entries[min(index, len(entries) - 1)] if entries else str(packet.manifest.get("generation", {}).get("prompt") or "")
    if not prompt.strip():
        raise MMH3ResourceError("Write a prompt or load a packet containing one")
    if seed < -1 or seed > 0xFFFFFFFFFFFFFFFF:
        raise MMH3ResourceError("Seed must be -1 or an unsigned 64-bit integer")
    selected_seed = int(seed if seed >= 0 else inherited.get("seed", packet.manifest.get("generation", {}).get("seed", 0)))
    if not 0 <= selected_seed <= 0xFFFFFFFFFFFFFFFF:
        raise MMH3ResourceError("Saved seed is invalid; set an unsigned 64-bit seed")
    timing = segment_timing(frames, context_frames if operation == "continuation" else 0)
    from .resolution import resolve_reference_set
    references = resolve_reference_set(packet, preset="all")
    info = {
        "contract": PLAN_CONTRACT, "state": "draft", "action": commit_action,
        "operation": operation, "target_segment_id": target_segment_id if target else "",
        "prompt_index": index, "prompts": text, "prompt": prompt, "seed": selected_seed,
        "parent_segment_id": (target or {}).get("parent_segment_id", chain["head_segment_id"]),
        "ledger_sha256": _ledger_digest(chain), "timing": timing,
        "reference_ids": [r.resource_id for r in references.resources],
        "reference_revisions": {r.resource_id: packet.get_by_id(r.resource_id)["content"]["revision"]
                                for r in references.resources},
        "source_process_sha256": _ledger_digest(_namespace(packet).get("last_process")),
    }
    out = packet.set_extension_value("mmh3_media", "chain", chain)
    if action == "New scene":
        # Generated boundary frames are output state, not an instruction for the next scene.
        from .h3_resource_semantics import find_context_resource
        for usage in ("first_frame", "last_frame"):
            resource = find_context_resource(out, usage)
            if resource is not None:
                out = out.remove(resource_id=resource["id"], record_history=False)
        out = out.edit_metadata(task="ref2va" if info["reference_ids"] else "t2va")
    out = out.edit_metadata(prompt=prompt, seed=selected_seed)
    out = out.set_extension_value("mmh3_media", "segment_plan", info)
    return SegmentPlan(out, info)


def review_segment(packet: MMH3Media, *, accept: bool = False,
                   chain_packet: MMH3Media | None = None) -> tuple[MMH3Media, str, str]:
    info = deep_copy_json(_namespace(packet).get("segment_plan"))
    if not isinstance(info, dict) or info.get("contract") != PLAN_CONTRACT:
        raise MMH3ResourceError("Connect the result of Prepare H3 Segment")
    if info.get("state") == "accepted":
        report = validate_chain(packet, require_head_state=True)
        if not report.ready:
            raise MMH3ResourceError("Accepted segment no longer matches its recorded state")
        return packet, info["filename_prefix"], "Accepted · ready to continue"
    process = _namespace(packet).get("last_process")
    if not process or _ledger_digest(process) == info.get("source_process_sha256"):
        raise MMH3ResourceError("Generate and pack a new result before reviewing this segment")
    if process.get("operation") != info["operation"]:
        raise MMH3ResourceError("Packed operation does not match the prepared segment")
    generation = packet.manifest.get("generation", {})
    if generation.get("prompt") != info["prompt"] or generation.get("seed") != info["seed"]:
        raise MMH3ResourceError("Packed prompt/seed differs from the segment plan; prepare it again")
    if _ledger_digest(_namespace(packet).get("chain")) != info["ledger_sha256"]:
        raise MMH3ResourceError("Draft chain changed after preparation; prepare the segment again")
    if not accept:
        chain = _namespace(packet)["chain"]
        prefix = f"mmh3/{chain['chain_id']}/draft_{info['prompt_index'] + 1:03d}"
        return packet, prefix, "Draft · repeat from the same parent, or choose Accept"
    if chain_packet is not None:
        chain = _namespace(chain_packet).get("chain")
        if not chain or _ledger_digest(chain) != info["ledger_sha256"]:
            raise MMH3ResourceError("Accepted chain changed after preparation; prepare the segment again")
    elif info["action"] == "reroll":
        raise MMH3ResourceError("Reroll acceptance requires the authoritative accepted chain")
    result = commit_chain_segment(packet, action=info["action"],
                                  target_segment_id=info["target_segment_id"], chain_packet=chain_packet)
    info.update(state="accepted", filename_prefix=result.filename_prefix)
    info.pop("source_process_sha256", None)
    out = result.packet.set_extension_value("mmh3_media", "segment_plan", info)
    chain = deep_copy_json(_namespace(out)["chain"])
    segment = next(s for s in chain["segments"] if s["segment_id"] == result.segment_id)
    segment["segment_plan"] = info
    out = out.set_extension_value("mmh3_media", "chain", chain)
    return out, result.filename_prefix, f"Accepted · segment {info['prompt_index'] + 1} · next prompt on Continue"
