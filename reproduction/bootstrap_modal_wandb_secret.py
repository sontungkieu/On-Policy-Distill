"""Create a Modal W&B secret without printing or persisting the API key."""

from __future__ import annotations

import argparse
import netrc
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--netrc", type=Path, required=True)
    parser.add_argument("--secret-name", required=True)
    args = parser.parse_args()

    os.environ["MODAL_PROFILE"] = args.profile
    credentials = netrc.netrc(str(args.netrc))
    selected_host = next(
        (
            host
            for host in ("api.wandb.ai", "wandb.ai")
            if credentials.authenticators(host)
        ),
        None,
    )
    if selected_host is None:
        raise RuntimeError(
            "No W&B credential entry was found in the supplied netrc file"
        )
    api_key = credentials.authenticators(selected_host)[2]
    if not api_key:
        raise RuntimeError("The W&B credential entry has no password/token")

    import modal

    modal.Secret.objects.create(
        args.secret_name,
        {"WANDB_API_KEY": api_key},
        allow_existing=False,
    )
    print(
        "MODAL_SECRET_READY "
        f"profile={args.profile} name={args.secret_name} source_host={selected_host}"
    )


if __name__ == "__main__":
    main()
