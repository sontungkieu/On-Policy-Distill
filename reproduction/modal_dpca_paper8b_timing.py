"""Run a bounded Qwen3-8B -> Llama-3.1-8B DPCA timing canary on 2x B200."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

APP_NAME = "dpca-paper8b-timing-2xb200-10step-20260906-r1"
RUN_ID = APP_NAME
SECRET_NAME = "dpca-wandb-kieusontung8"
VOLUME_NAME = "dpca-breaking-token-barrier-runs"
REMOTE_ROOT = Path("/opt/dpca")
LOCAL_ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else REMOTE_ROOT
CONFIG_PATH = "/opt/dpca/reproduction/configs/dpca_paper8b_2gpu_10step_timing.json"

runtime_image = modal.Image.from_dockerfile(
    LOCAL_ROOT / "reproduction/Dockerfile.dpca-paper8b-timing",
    context_dir=LOCAL_ROOT,
    ignore=[".git", "**/__pycache__", "**/*.pyc", "reproduction/.venv"],
)
run_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
wandb_secret = modal.Secret.from_name(SECRET_NAME)
app = modal.App(APP_NAME)


@app.function(
    image=runtime_image,
    gpu="B200:2",
    cpu=8,
    memory=131_072,
    timeout=7_200,
    retries=0,
    single_use_containers=True,
    secrets=[wandb_secret],
    volumes={"/runs": run_volume},
)
def train() -> dict[str, object]:
    started = time.perf_counter()
    process = subprocess.Popen(
        [
            sys.executable,
            str(REMOTE_ROOT / "reproduction/dpca_2gpu_train.py"),
            "--config",
            CONFIG_PATH,
            "--output-root",
            "/runs",
        ],
        cwd=REMOTE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    run_volume.commit()
    if return_code != 0:
        raise RuntimeError(f"Training subprocess failed with exit code {return_code}")
    summary_path = Path("/runs") / RUN_ID / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Missing training summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["modal"] = {
        "app_name": APP_NAME,
        "gpu_request": "B200:2",
        "retries": 0,
        "single_use_containers": True,
        "volume": VOLUME_NAME,
        "function_wall_seconds": time.perf_counter() - started,
    }
    print(
        "DPCA_PAPER8B_TIMING_RESULT_JSON=" + json.dumps(summary, sort_keys=True),
        flush=True,
    )
    return summary


@app.local_entrypoint()
def main() -> None:
    result = train.remote()
    print("DPCA_PAPER8B_TIMING_LOCAL_JSON=" + json.dumps(result, sort_keys=True))
