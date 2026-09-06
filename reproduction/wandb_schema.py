"""Canonical W&B metadata for DPCA implementation-validation runs."""

from __future__ import annotations

import re
from typing import Any

SCHEMA_VERSION = "experiment-v1"
IMPLEMENTATION_VALIDATION_JOB_TYPE = "implementation-validation"
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_ORDER = (
    "schema",
    "framework",
    "method",
    "regime",
    "objective",
    "variant",
    "platform",
    "accelerator",
    "budget",
    "stage",
    "student",
    "teacher",
)


def _slug(field: str, value: str) -> str:
    if not isinstance(value, str) or not _SLUG_RE.fullmatch(value):
        raise ValueError(f"{field} must match [a-z0-9][a-z0-9._-]*")
    return value


def _model_slug(repo_id: str) -> str:
    return _slug("model slug", repo_id.rsplit("/", 1)[-1].lower())


def resolve_wandb_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve ordered tags from the effective experiment config only."""

    tracking = config["tracking"]
    dimensions = {
        "schema": SCHEMA_VERSION,
        "framework": tracking["framework"],
        "method": tracking["method"],
        "regime": tracking["regime"],
        "objective": config["opd"]["objective"],
        "variant": config["opd"]["variant"],
        "platform": tracking["platform"],
        "accelerator": tracking["accelerator"],
        "budget": f"{config['optimizer']['steps']}-update",
        "stage": config["evidence_class"].replace("_", "-"),
        "student": _model_slug(config["student"]["repo_id"]),
        "teacher": _model_slug(config["teacher"]["repo_id"]),
    }
    tags = [f"{key}:{_slug(f'tag value for {key}', dimensions[key])}" for key in _ORDER]
    if any(tag.split(":", 1)[0] != key for tag, key in zip(tags, _ORDER, strict=True)):
        raise AssertionError("W&B metadata order drifted")
    return {
        "schema": SCHEMA_VERSION,
        "job_type": IMPLEMENTATION_VALIDATION_JOB_TYPE,
        "tags": tags,
    }
