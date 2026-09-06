from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path

from package_manifest import collect_package_files


ROOT = Path(__file__).resolve().parent
DEFAULT_DESTINATION = Path(r"E:\ComfyUI_PE\ComfyUI\custom_nodes\_test_mmh3_media")


def _validate_destination(source_root: Path, target_root: Path) -> None:
    if target_root == Path(target_root.anchor):
        raise ValueError("Deploy destination must not be a filesystem root")
    if target_root == source_root or source_root in target_root.parents or target_root in source_root.parents:
        raise ValueError("Deploy destination must be separate from the source workspace")


def _changed_files(source_root: Path, target_root: Path, files: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(
        relative
        for relative in files
        if not (target_root / relative).is_file()
        or not filecmp.cmp(source_root / relative, target_root / relative, shallow=False)
    )


def deploy(root: str | Path, destination: str | Path) -> dict[str, object]:
    source_root = Path(root).resolve()
    target_root = Path(destination).expanduser().absolute()
    _validate_destination(source_root, target_root.resolve())
    if target_root.exists() and not target_root.is_dir():
        raise ValueError("Deploy destination must be a directory")

    files = collect_package_files(source_root, mode="runtime")
    changed = _changed_files(source_root, target_root, files)

    target_root.mkdir(parents=True, exist_ok=True)
    for relative in changed:
        source = source_root / relative
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    return {
        "destination": str(target_root),
        "files": len(files),
        "changed": len(changed),
        "changed_files": tuple(path.as_posix() for path in changed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy the MMH3 runtime files into ComfyUI.")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    result = deploy(ROOT, args.destination)
    for relative in result["changed_files"]:
        print(f"COPY {relative}")
    print(
        f"Deployed {result['changed']} changed of {result['files']} allowlisted files "
        f"to {result['destination']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
