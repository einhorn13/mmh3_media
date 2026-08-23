from __future__ import annotations

import json
from typing import Any

from .core import MMH3Media


def _clean_manifest(packet: MMH3Media, *, include_preview: bool) -> dict[str, Any]:
    m = json.loads(json.dumps(packet.manifest))
    if not include_preview:
        m["resources"] = [r for r in m.get("resources", []) if r.get("role") != "preview"]
    # Volatile serialization/runtime fields do not define semantic equality.
    m.pop("updated_at", None)
    for r in m.get("resources", []):
        r.pop("path", None)
        r.pop("size", None)
        r.pop("serializer", None)
        r.pop("media_type", None)
        if r.get("role") == "preview":
            r.pop("sha256", None)
    return m


def compare_packets(a: MMH3Media, b: MMH3Media, *, include_preview: bool = False) -> dict[str, Any]:
    am = _clean_manifest(a, include_preview=include_preview)
    bm = _clean_manifest(b, include_preview=include_preview)

    packet_fields: dict[str, Any] = {}
    for key in sorted(set(am) | set(bm)):
        if key in {"resources", "history"}:
            continue
        av, bv = am.get(key), bm.get(key)
        if av != bv:
            packet_fields[key] = {"a": av, "b": bv}

    def index(m):
        return {(r.get("role"), int(r.get("slot", 0))): r for r in m.get("resources", [])}

    ai, bi = index(am), index(bm)
    resources = []
    for sel in sorted(set(ai) | set(bi), key=lambda x: (x[0], x[1])):
        ar, br = ai.get(sel), bi.get(sel)
        if ar is None:
            resources.append({"selector": [sel[0], sel[1]], "status": "only_b", "b": br})
        elif br is None:
            resources.append({"selector": [sel[0], sel[1]], "status": "only_a", "a": ar})
        elif ar != br:
            resources.append({"selector": [sel[0], sel[1]], "status": "changed", "a": ar, "b": br})

    history_equal = am.get("history", []) == bm.get("history", [])
    equal = not packet_fields and not resources and history_equal
    return {
        "equal": equal,
        "include_preview": include_preview,
        "packet_fields": packet_fields,
        "resources": resources,
        "history_equal": history_equal,
        "summary": (
            "Packets are semantically equal."
            if equal
            else f"Differences: {len(packet_fields)} packet field(s), {len(resources)} resource selector(s), history_equal={history_equal}."
        ),
    }
