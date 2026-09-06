"""Download immutable public model snapshots into the container image."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.output,
    )
    print(
        f"MODEL_SNAPSHOT_READY repo={args.repo_id} revision={args.revision} output={args.output}"
    )


if __name__ == "__main__":
    main()
