"""Build the locked Dockerfile and run exactly ten DPCA steps on 2x B200."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "dpca-breaking-token-barrier-2xb200-10step-20260905-r2"
RUN_ID = "dpca-2xb200-10step-20260905-r2"
SECRET_NAME = "dpca-wandb-kieusontung8"
VOLUME_NAME = "dpca-breaking-token-barrier-runs"
REMOTE_ROOT = Path("/opt/dpca")
LOCAL_ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else REMOTE_ROOT

runtime_image = modal.Image.from_dockerfile(
    LOCAL_ROOT / "reproduction/Dockerfile.dpca-2gpu",
    context_dir=LOCAL_ROOT,
    ignore=[".git", "**/__pycache__", "**/*.pyc"],
)
run_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
wandb_secret = modal.Secret.from_name(SECRET_NAME)
app = modal.App(APP_NAME)


@app.function(
    image=runtime_image,
    cpu=2,
    memory=8_192,
    timeout=900,
    retries=0,
    single_use_containers=True,
)
def preflight() -> dict[str, object]:
    code = """
import json
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
from verl.trainer.ppo.core_algos import compute_policy_loss_opd
from verl.workers.reward_manager.opd import TeacherClient

root = Path('/opt/dpca')
config = json.loads((root / 'reproduction/configs/dpca_2gpu_10step.json').read_text())
for role in ('teacher', 'student'):
    path = config[role]['container_path']
    AutoConfig.from_pretrained(path, local_files_only=True)
    AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
assert callable(TeacherClient._align_chunks)
assert callable(compute_policy_loss_opd)
assert (root / 'reproduction/uv.lock').is_file()
print('DPCA_PREFLIGHT_OK=' + json.dumps({'models': ['teacher', 'student'], 'uv_lock': True}))
"""
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REMOTE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    print(process.stdout, end="")
    if process.stderr:
        print(process.stderr, file=sys.stderr, end="")
    if process.returncode != 0:
        raise RuntimeError(f"CPU preflight failed with exit code {process.returncode}")
    return {"status": "passed", "models": ["teacher", "student"], "uv_lock": True}


@app.function(
    image=runtime_image,
    gpu="B200:2",
    cpu=8,
    memory=65_536,
    timeout=3_600,
    retries=0,
    single_use_containers=True,
    secrets=[wandb_secret],
    volumes={"/runs": run_volume},
)
def train() -> dict[str, object]:
    output_dir = Path("/runs") / RUN_ID
    process = subprocess.Popen(
        [
            sys.executable,
            str(REMOTE_ROOT / "reproduction/dpca_2gpu_train.py"),
            "--config",
            str(REMOTE_ROOT / "reproduction/configs/dpca_2gpu_10step.json"),
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
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Missing training summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["modal"] = {
        "app_name": APP_NAME,
        "gpu_request": "B200:2",
        "retries": 0,
        "single_use_containers": True,
        "volume": VOLUME_NAME,
    }
    print("DPCA_MODAL_RESULT_JSON=" + json.dumps(summary, sort_keys=True), flush=True)
    return summary


@app.local_entrypoint()
def main() -> None:
    preflight_result = preflight.remote()
    print("DPCA_PREFLIGHT_RESULT_JSON=" + json.dumps(preflight_result, sort_keys=True))
    result = train.remote()
    print("DPCA_LOCAL_RESULT_JSON=" + json.dumps(result, sort_keys=True))
