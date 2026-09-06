"""Build a pinned DPCA smoke image and qualify it on one Modal B200."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "dpca-breaking-token-barrier-smoke-20260905-r1"
BASE_IMAGE = (
    "pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime@"
    "sha256:f0ca81b440e252399d9954a45b616ee2540959466aacf3dfc3f856691eee66e8"
)
REMOTE_ROOT = Path("/opt/dpca")
LOCAL_ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else REMOTE_ROOT

runtime_image = (
    modal.Image.from_registry(BASE_IMAGE)
    .pip_install("uv==0.10.2")
    .add_local_dir(
        str(LOCAL_ROOT / "reproduction"),
        "/tmp/repro-env",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .run_commands(
        "uv export --project /tmp/repro-env --locked --no-dev --no-emit-project "
        "--no-hashes --output-file /tmp/requirements.lock.txt",
        "env -u UV_INDEX_URL -u UV_DEFAULT_INDEX -u PIP_INDEX_URL "
        "-u PIP_TRUSTED_HOST uv pip install --system --no-cache "
        "--requirements /tmp/requirements.lock.txt",
    )
    .add_local_dir(
        str(LOCAL_ROOT),
        str(REMOTE_ROOT),
        copy=True,
        ignore=[".git", "**/__pycache__", "**/*.pyc"],
    )
    .run_commands(
        "env -u UV_INDEX_URL -u UV_DEFAULT_INDEX -u PIP_INDEX_URL "
        "-u PIP_TRUSTED_HOST uv pip install --system --no-deps /opt/dpca"
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=runtime_image,
    gpu="B200",
    cpu=4,
    memory=32_768,
    timeout=1_800,
    retries=0,
    single_use_containers=True,
)
def smoke() -> dict[str, object]:
    process = subprocess.run(
        [sys.executable, str(REMOTE_ROOT / "reproduction/dpca_contract_smoke.py")],
        cwd=REMOTE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    print(process.stdout, end="")
    if process.stderr:
        print(process.stderr, file=sys.stderr, end="")
    if process.returncode != 0:
        raise RuntimeError(
            f"DPCA contract smoke failed with exit code {process.returncode}"
        )

    marker = "DPCA_SMOKE_JSON="
    payload = next(
        (
            line.removeprefix(marker)
            for line in process.stdout.splitlines()
            if line.startswith(marker)
        ),
        None,
    )
    if payload is None:
        raise RuntimeError("DPCA contract smoke did not emit its JSON marker")
    result = json.loads(payload)
    result["modal"] = {
        "app": APP_NAME,
        "base_image": BASE_IMAGE,
        "gpu_request": "B200:1",
        "retries": 0,
    }
    print("DPCA_MODAL_RESULT_JSON=" + json.dumps(result, sort_keys=True))
    return result


@app.local_entrypoint()
def main() -> None:
    result = smoke.remote()
    print("DPCA_LOCAL_RESULT_JSON=" + json.dumps(result, sort_keys=True))
