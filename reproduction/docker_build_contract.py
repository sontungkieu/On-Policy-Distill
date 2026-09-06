"""Validation and provenance helpers for the remote Docker build context."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = Path(os.environ.get("DPCA_DOCKER_CONTEXT_ROOT", SOURCE_ROOT)).resolve()
_IMAGE_REF = re.compile(
    r"^(?:docker\.io/)?[a-z0-9]+(?:[._-][a-z0-9]+)*/"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})$"
)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_CACHE_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "wandb",
    "artifacts",
    "checkpoint-final",
}
_WEIGHT_SUFFIXES = {
    ".safetensors",
    ".gguf",
    ".onnx",
    ".ckpt",
    ".pth",
    ".pt",
    ".bin",
    ".pem",
    ".key",
}


def normalize_image_ref(value: str) -> str:
    if not _IMAGE_REF.fullmatch(value):
        raise ValueError("image must be a tagged Docker Hub reference")
    return value if value.startswith("docker.io/") else f"docker.io/{value}"


def validate_secret_name(value: str) -> str:
    if not _SAFE_NAME.fullmatch(value):
        raise ValueError("registry secret name contains unsupported characters")
    return value


def ignore_local_path(path: Path) -> bool:
    candidate = path if path.is_absolute() else LOCAL_ROOT / path
    try:
        relative = candidate.resolve().relative_to(LOCAL_ROOT.resolve())
    except ValueError:
        return True
    lowered_parts = {part.lower() for part in relative.parts}
    lowered_name = relative.name.lower()
    return (
        bool(lowered_parts & _CACHE_PARTS)
        or relative.suffix.lower() in _WEIGHT_SUFFIXES
        or lowered_name == ".env"
        or lowered_name.startswith(".env.")
        or "secret" in lowered_name
        or "credential" in lowered_name
        or "dockerhub" in lowered_name
        or lowered_name in {".netrc", "_netrc"}
    )


def audit_local_context() -> dict[str, int | str]:
    files: list[Path] = []
    for root, directory_names, file_names in os.walk(LOCAL_ROOT, topdown=True):
        root_path = Path(root)
        # Prune ignored trees before enumerating their contents. A Linux-created
        # venv may contain symlinks that Windows cannot dereference even though
        # the entire directory is outside the Docker context contract.
        directory_names[:] = [
            name for name in directory_names if not ignore_local_path(root_path / name)
        ]
        for name in file_names:
            path = root_path / name
            if ignore_local_path(path):
                continue
            if path.is_file():
                if path.stat().st_size > 100 * 1024 * 1024:
                    raise RuntimeError(f"unexpected large Docker input: {path}")
                files.append(path)
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(LOCAL_ROOT).as_posix()):
        relative = path.relative_to(LOCAL_ROOT).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "scanned_files": len(files),
        "large_files": 0,
        "context_sha256": digest.hexdigest(),
    }
