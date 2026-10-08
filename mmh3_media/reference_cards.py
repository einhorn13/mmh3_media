"""Visual reference settings compiled into the existing reference/alias contracts."""
import json
import re

from .errors import MMH3ResourceError
from .reference_management import configure_reference
from .scheduled_references import configure_reference_schedule, compile_scene_references
from .resolution import resolve_reference_set


def reference_slots(packet):
    resources = [r for r in packet.manifest["resources"] if r.get("role") == "reference" and r["kind"] in {"image", "video", "audio"}]
    resources.sort(key=lambda r: (r["kind"], r.get("order") is None, r.get("order") or 0))
    counts, result = {}, []
    for resource in resources:
        kind = resource["kind"]
        index = counts.get(kind, 0)
        counts[kind] = index + 1
        result.append((f"{kind}:{index}", resource))
    return result


def reference_card_settings(packet, scene_id="scene_1"):
    """Global card intent, kept separately from current-scene inclusion."""
    context = packet.manifest.get("extensions", {}).get("minimax_h3", {})
    saved = {c["resource_id"]: c for c in context.get("reference_cards", [])}
    aliases = {item["resource_id"]: (name, item["scenes"])
               for name, item in context.get("reference_schedule", {}).get("aliases", {}).items()}
    cards = []
    for key, resource in reference_slots(packet):
        reference = resource.get("extensions", {}).get("minimax_h3", {}).get("reference", {})
        alias, scenes = aliases.get(resource["id"], ("", [scene_id]))
        defaults = {"alias": alias, "scenes": scenes,
                    "enabled": bool(scenes) if alias else reference.get("enabled", True),
                    "purposes": reference.get("purposes", ["unknown"])}
        cards.append({**defaults, **saved.get(resource["id"], {}), "key": key, "resource_id": resource["id"]})
    return cards


def apply_reference_cards(packet, cards_json="[]", scene_id="scene_1", prompt="", conditioning="encoder_and_vae"):
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise MMH3ResourceError("Choose a nonempty scene ID")
    if conditioning not in {"encoder_and_vae", "encoder_only"}:
        raise MMH3ResourceError("Unknown reference conditioning path")
    try:
        updates = json.loads(cards_json)
    except (ValueError, TypeError) as exc:
        raise MMH3ResourceError("Invalid reference cards JSON") from exc
    if not isinstance(updates, list) or any(not isinstance(c, dict) for c in updates):
        raise MMH3ResourceError("Reference cards must be an array")
    cards = {c["key"]: c for c in reference_card_settings(packet, scene_id)}
    seen = set()
    for update in updates:
        key = update.get("key")
        if not isinstance(key, str) or key not in cards or key in seen:
            raise MMH3ResourceError(f"Unknown or duplicate reference card {key!r}; refresh cards after changing inputs")
        seen.add(key)
        cards[key].update({k: v for k, v in update.items() if k in {"alias", "scenes", "enabled", "purposes"}})

    out, aliases, scenes, normalized = packet, {}, [scene_id], []
    for card in cards.values():
        selected = card["scenes"]
        if (not isinstance(selected, list) or any(not isinstance(s, str) or not s.strip() for s in selected)
                or len(set(selected)) != len(selected)):
            raise MMH3ResourceError("Reference scenes must be unique nonempty scene IDs")
        active = card["enabled"]
        if type(active) is not bool:
            raise MMH3ResourceError("Reference enabled must be boolean")
        if not isinstance(card["purposes"], list):
            raise MMH3ResourceError("Reference purposes must be an array")
        if not isinstance(card["alias"], str):
            raise MMH3ResourceError("Reference alias must be text")
        alias = card["alias"].strip().lstrip("@")
        if alias and (not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", alias) or alias in aliases):
            raise MMH3ResourceError("Reference aliases must be unique names using letters, digits and underscores")
        out = configure_reference(out, card["resource_id"],
            inclusion="include" if active and scene_id in selected else "exclude", purposes=card["purposes"]).packet
        purposes = out.get_by_id(card["resource_id"])["extensions"]["minimax_h3"]["reference"]["purposes"]
        normalized.append({**card, "alias": alias, "purposes": purposes})
        scenes.extend(selected)
        if alias:
            aliases[alias] = {"resource_id": card["resource_id"], "scenes": selected if active else []}
    out = configure_reference_schedule(out, {"scenes": list(dict.fromkeys(scenes)), "aliases": aliases})
    out = out.set_extension_value("minimax_h3", "reference_cards", normalized)
    out = out.set_extension_value("minimax_h3", "reference_conditioning", conditioning)
    context = packet.manifest.get("extensions", {}).get("minimax_h3", {})
    previous = context.get("compiled_references", {})
    stored_prompt = packet.manifest.get("generation", {}).get("prompt", "")
    inherited = previous.get("source_prompt", stored_prompt) if stored_prompt == previous.get("prompt") else stored_prompt
    source_prompt = prompt or inherited
    out, compiled, report = compile_scene_references(out, scene_id, source_prompt)
    resolved = resolve_reference_set(out)
    if not resolved.ready:
        raise MMH3ResourceError("; ".join(d.message for d in resolved.diagnostics if d.severity == "error"))
    tags = {r.resource.resource_id: r.prompt_tag for r in resolved.presentation}
    slots = dict(reference_slots(out))
    rendered = [{**card, "kind": slots[card["key"]]["kind"], "prompt_tag": tags.get(card["resource_id"], "inactive")}
                for card in normalized]
    from .constants import REFERENCE_PURPOSES
    return out, compiled, {"cards": rendered, "scene_id": scene_id, "conditioning": conditioning, "prompt": compiled,
                           "compilation": report, "purposes_available": list(REFERENCE_PURPOSES)}
