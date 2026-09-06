"""Qualify the exact registry image in one disposable 2x B200 container."""

from __future__ import annotations

import json
import os

import modal

IMAGE = os.environ.get("DPCA_DOCKER_IMAGE", "")
SECRET_NAME = os.environ.get("DPCA_DOCKER_REGISTRY_SECRET", "")
if not IMAGE or not SECRET_NAME:
    raise RuntimeError("DPCA_DOCKER_IMAGE and DPCA_DOCKER_REGISTRY_SECRET are required")

registry_secret = modal.Secret.from_name(SECRET_NAME)
runtime_image = modal.Image.from_registry(IMAGE, secret=registry_secret).env(
    {
        "DPCA_DOCKER_IMAGE": IMAGE,
        "DPCA_DOCKER_REGISTRY_SECRET": SECRET_NAME,
    }
)
app = modal.App("dpca-breaking-token-barrier-docker-2xb200-qualification")


@app.function(
    image=runtime_image,
    gpu="B200:2",
    cpu=4,
    memory=16_384,
    timeout=900,
    retries=0,
    single_use_containers=True,
)
def qualify() -> dict[str, object]:
    import torch
    from transformers import AutoConfig, AutoTokenizer

    from verl.trainer.ppo.core_algos import compute_policy_loss_opd
    from verl.workers.reward_manager.opd import TeacherClient

    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"Expected exactly two visible GPUs, got {torch.cuda.device_count()}"
        )
    devices = []
    for index in range(2):
        name = torch.cuda.get_device_name(index)
        capability = list(torch.cuda.get_device_capability(index))
        if "B200" not in name or capability != [10, 0]:
            raise RuntimeError(
                f"Unexpected GPU {index}: {name}, capability={capability}"
            )
        left = torch.randn((256, 256), device=index, dtype=torch.bfloat16)
        right = torch.randn((256, 256), device=index, dtype=torch.bfloat16)
        product = left @ right
        if not torch.isfinite(product).all():
            raise RuntimeError(f"Non-finite BF16 matrix multiplication on GPU {index}")
        devices.append({"index": index, "name": name, "capability": capability})
    for path in ("/opt/models/teacher", "/opt/models/student"):
        AutoConfig.from_pretrained(path, local_files_only=True)
        AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if not callable(TeacherClient._align_chunks) or not callable(
        compute_policy_loss_opd
    ):
        raise TypeError("DPCA source functions are unavailable")
    result = {
        "status": "passed",
        "image": IMAGE,
        "devices": devices,
        "cuda": torch.version.cuda,
        "torch": str(torch.__version__),
        "model_snapshots": ["teacher", "student"],
        "dpca_imports": True,
        "bf16_matmul_both_gpus": True,
    }
    print("DPCA_DOCKER_QUALIFICATION_JSON=" + json.dumps(result, sort_keys=True))
    return result


@app.local_entrypoint()
def main() -> None:
    print(
        "DPCA_DOCKER_LOCAL_RESULT_JSON=" + json.dumps(qualify.remote(), sort_keys=True)
    )
