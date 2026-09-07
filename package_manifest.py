from __future__ import annotations

from pathlib import Path


RUNTIME_ROOT_FILES = ("__init__.py", "requirements.txt", "README.md", "automation_runner.py")
RUNTIME_DOC_FILES = ("docs/MMH3_MEDIA_v0.3.md", "docs/H3_LATENT_CONTRACT_V2.md", "docs/WORKFLOWS.md")
EXPORT_EXCLUDED_DIRECTORIES = {"dist", "trash"}
EXPORT_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _tree(root: Path, directory: str, suffixes: set[str]) -> set[Path]:
    base = root / directory
    if not base.is_dir():
        return set()
    return {
        path.relative_to(root)
        for path in base.rglob("*")
        if path.is_file()
        and path.suffix.lower() in suffixes
        and "__pycache__" not in path.parts
    }


def _collect_export_files(root: Path) -> set[Path]:
    """Collect the source tree by exclusion without following symlinks."""
    files: set[Path] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            if path.is_symlink():
                continue
            if path.is_dir():
                name = path.name
                if name.startswith((".", "_")) or name.casefold() in EXPORT_EXCLUDED_DIRECTORIES:
                    continue
                pending.append(path)
                continue
            if not path.is_file():
                continue
            if path.suffix.casefold() in EXPORT_EXCLUDED_SUFFIXES:
                continue
            files.add(path.relative_to(root))
    return files


def collect_package_files(root: str | Path, *, mode: str) -> tuple[Path, ...]:
    source_root = Path(root).resolve()
    if mode not in {"runtime", "export"}:
        raise ValueError("mode must be runtime or export")

    if mode == "export":
        files = _collect_export_files(source_root)
    else:
        files = {Path(name) for name in (*RUNTIME_ROOT_FILES, *RUNTIME_DOC_FILES)}
        files |= _tree(source_root, "mmh3_media", {".py"})
        files |= _tree(source_root, "web", {".js"})
        files |= _tree(source_root, "schema", {".json"})
        files |= _tree(source_root, "example_workflows", {".json"})
        files |= _tree(source_root, "automation/workflows", {".json", ".md"})

    missing = [path for path in files if not (source_root / path).is_file()]
    if missing:
        raise FileNotFoundError("Required package files are missing: " + ", ".join(map(str, missing)))
    return tuple(sorted(files, key=lambda path: path.as_posix().casefold()))


__all__ = [
    "EXPORT_EXCLUDED_DIRECTORIES",
    "RUNTIME_DOC_FILES",
    "EXPORT_EXCLUDED_SUFFIXES",
    "collect_package_files",
]
