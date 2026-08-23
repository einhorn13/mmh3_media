from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from .archive import PACKET_JSON, save_archive
from .core import MMH3Media
from .errors import MMH3ResourceError
from .util import safe_member_path


def export_packet(packet: MMH3Media, directory: str | os.PathLike, *, include_preview: bool = True) -> str:
    """Export an MMH3 packet as an ordinary directory tree.

    Export is an interoperability/debugging utility. The exported packet.json is self-consistent;
    when preview is excluded, its descriptor is removed as well.
    """
    dest = Path(directory).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="mmh3_export_"))
    try:
        temp_archive = work / "packet.mmh3"
        saved, _ = save_archive(packet, temp_archive)
        manifest = json.loads(json.dumps(saved.manifest))
        if not include_preview:
            manifest["resources"] = [r for r in manifest["resources"] if r.get("role") != "preview"]
        with zipfile.ZipFile(temp_archive, "r", allowZip64=True) as zf:
            for res in manifest["resources"]:
                member = safe_member_path(res["path"])
                target = (dest / member).resolve()
                try:
                    target.relative_to(dest)
                except ValueError as e:
                    raise MMH3ResourceError("Export path escapes destination") from e
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        (dest / PACKET_JSON).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(dest)
    finally:
        shutil.rmtree(work, ignore_errors=True)
