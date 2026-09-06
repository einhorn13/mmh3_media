from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

from package_manifest import collect_package_files


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "dist" / "ComfyUI_mmh3_media_source.zip"
ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_export(root: str | Path, output: str | Path) -> dict[str, object]:
    source_root = Path(root).resolve()
    target = Path(output).resolve()
    files = tuple(
        path
        for path in collect_package_files(source_root, mode="export")
        if (source_root / path).resolve() != target
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for relative in files:
                info = zipfile.ZipInfo(relative.as_posix(), ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, (source_root / relative).read_bytes())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(target), "files": len(files), "sha256": _sha256(target)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic MMH3 source ZIP by excluding hidden/internal directories, "
            "dist artifacts and Python cache files."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--list", action="store_true", help="List selected files without creating a ZIP.")
    args = parser.parse_args()
    files = collect_package_files(ROOT, mode="export")
    if args.list:
        print("\n".join(path.as_posix() for path in files))
        return 0
    print(json.dumps(build_export(ROOT, args.output), ensure_ascii=False, indent=2))
    return 0


__all__ = ["build_export", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
