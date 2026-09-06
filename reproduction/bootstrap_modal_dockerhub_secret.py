"""Create a Modal registry secret from a two-line username/token file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--secret-name", required=True)
    args = parser.parse_args()

    lines = [
        line.strip()
        for line in args.credential_file.read_text().splitlines()
        if line.strip()
    ]
    if len(lines) != 2:
        raise RuntimeError(
            "Docker Hub credential file must contain username then token"
        )
    username, token = lines
    if not username or not token:
        raise RuntimeError("Docker Hub username/token is empty")
    os.environ["MODAL_PROFILE"] = args.profile

    import modal

    modal.Secret.objects.create(
        args.secret_name,
        {"REGISTRY_USERNAME": username, "REGISTRY_PASSWORD": token},
        allow_existing=False,
    )
    print(
        "MODAL_DOCKERHUB_SECRET_READY "
        f"profile={args.profile} name={args.secret_name} username={username}"
    )


if __name__ == "__main__":
    main()
