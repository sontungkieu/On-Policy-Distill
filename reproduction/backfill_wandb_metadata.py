"""Backfill and verify canonical metadata without exposing the W&B API key."""

from __future__ import annotations

import argparse
import json
import netrc
import os
from pathlib import Path

from wandb_schema import resolve_wandb_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-path", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--netrc", type=Path, required=True)
    args = parser.parse_args()

    credentials = netrc.netrc(str(args.netrc))
    auth = credentials.authenticators("api.wandb.ai") or credentials.authenticators(
        "wandb.ai"
    )
    if auth is None or not auth[2]:
        raise RuntimeError("No W&B token found in the supplied netrc")
    os.environ["WANDB_API_KEY"] = auth[2]

    import wandb

    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected = resolve_wandb_metadata(config)
    api = wandb.Api()
    run = api.run(args.run_path)
    original_identity = {"id": run.id, "name": run.name}
    run.tags = expected["tags"]
    run.job_type = expected["job_type"]
    run.config.update({"wandb_metadata": expected}, allow_val_change=True)
    run.update()

    verified = wandb.Api().run(args.run_path)
    actual = {
        "job_type": verified.job_type,
        "tags": list(verified.tags),
        "config_metadata": verified.config.get("wandb_metadata"),
    }
    if (
        verified.id != original_identity["id"]
        or verified.name != original_identity["name"]
    ):
        raise RuntimeError("W&B run identity changed during metadata backfill")
    tags_match = set(actual["tags"]) == set(expected["tags"]) and len(
        actual["tags"]
    ) == len(expected["tags"])
    if (
        actual["job_type"] != expected["job_type"]
        or not tags_match
        or actual["config_metadata"] != expected
    ):
        raise RuntimeError(
            f"W&B metadata verification failed: expected={expected} actual={actual}"
        )
    print(
        "WANDB_METADATA_VERIFIED_JSON="
        + json.dumps(
            {
                "run_id": verified.id,
                "run_name": verified.name,
                "job_type": verified.job_type,
                "canonical_tags": expected["tags"],
                "api_tags": list(verified.tags),
                "api_tag_order_note": "W&B API returns tags in backend-sorted order",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
