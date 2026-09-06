"""Prepare pinned assets and run the configured DPCA student SFT phase on Modal."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = os.environ.get(
    "DPCA_MODAL_APP_NAME", "dpca8b-scientific-sft50-fsdp-20260906-r8-reentrant-gc"
)
CONFIG_NAME = os.environ.get("DPCA_CONFIG_NAME", "dpca_8b_scientific.json")
SCIENTIFIC_ROOT = Path("/opt/dpca/reproduction/scientific")
DATASET_INFO_SOURCE = Path("/opt/dpca/reproduction/scientific/data/dataset_info.json")
WORKSPACE_ROOT = Path("/workspace")
ASSET_ROOT = WORKSPACE_ROOT / "assets"
RUN_ROOT = WORKSPACE_ROOT / "runs"
VOLUME_NAME = "dpca-breaking-token-barrier-scientific"
WANDB_SECRET_NAME = os.environ.get("DPCA_WANDB_SECRET_NAME", "dpca-wandb-kieusontung8")

MODULE_PATH = Path(__file__).resolve()
# Only the locked runtime is baked into the image. Config, runner, and dataset
# metadata are startup mounts, so changing run IDs or step counts does not
# invalidate the dependency image.
if MODULE_PATH.parent.name == "scientific":
    LOCAL_SCIENTIFIC_ROOT = MODULE_PATH.parent
else:
    LOCAL_SCIENTIFIC_ROOT = Path("/opt/dpca/reproduction/scientific")
RUNTIME_ROOT = LOCAL_SCIENTIFIC_ROOT / "runtime"
RUNTIME_DOCKERFILE_NAME = os.environ.get("DPCA_RUNTIME_DOCKERFILE", "Dockerfile")
if Path(RUNTIME_DOCKERFILE_NAME).name != RUNTIME_DOCKERFILE_NAME:
    raise RuntimeError("DPCA_RUNTIME_DOCKERFILE must be a basename")
runtime_image = modal.Image.from_dockerfile(
    RUNTIME_ROOT / RUNTIME_DOCKERFILE_NAME,
    context_dir=RUNTIME_ROOT,
)
image = runtime_image.add_local_dir(
    LOCAL_SCIENTIFIC_ROOT,
    remote_path="/opt/dpca/reproduction/scientific",
    copy=False,
    ignore=["runtime/.venv", "runtime/.venv/**", "**/__pycache__", "**/*.pyc"],
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
wandb_secret = modal.Secret.from_name(WANDB_SECRET_NAME)
app = modal.App(APP_NAME)


def _resolve_config_path(config_name: str) -> Path:
    if Path(config_name).name != config_name or not config_name.endswith(".json"):
        raise RuntimeError("config_name must be a JSON basename")
    return SCIENTIFIC_ROOT / config_name


def _load_config(config_name: str) -> tuple[dict[str, Any], Path]:
    config_path = _resolve_config_path(config_name)
    return json.loads(config_path.read_text(encoding="utf-8")), config_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_source_manifest(run_dir: Path, config_path: Path) -> Path:
    scientific_root = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime_dockerfile = (
        "Dockerfile.flashattn"
        if config["sft"].get("attention_kernel_policy") == "external_fa2"
        else "Dockerfile"
    )
    scientific_paths = [
        config_path,
        scientific_root / "modal_sft.py",
        scientific_root / "llamafactory_sft_entry.py",
        scientific_root / "sft_telemetry.py",
        scientific_root / "data/dataset_info.json",
        scientific_root / "runtime" / runtime_dockerfile,
    ]
    runtime_paths = [
        Path("/opt/scientific-env/pyproject.toml"),
        Path("/opt/scientific-env/uv.lock"),
    ]
    paths = scientific_paths + runtime_paths
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Scientific source manifest inputs are missing: {missing}")
    manifest = {
        "schema_version": 1,
        "dpca_upstream_repo": "https://github.com/ivanniu/On-Policy-Distill.git",
        "dpca_upstream_commit": "927a8264f2e303b7f82c2d331a58fd4240c8805a",
        "files": [
            {
                "path": (
                    str(path.relative_to(scientific_root))
                    if path.is_relative_to(scientific_root)
                    else str(path)
                ),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in paths
        ],
    }
    path = run_dir / "source-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


@app.function(
    image=image,
    cpu=16,
    memory=32_768,
    timeout=7_200,
    retries=0,
    volumes={str(WORKSPACE_ROOT): volume},
)
def prepare_assets(config_name: str) -> dict[str, Any]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, snapshot_download

    config, config_path = _load_config(config_name)
    dataset = config["sft"]["dataset"]
    student = config["models"]["student_base"]
    dataset_root = ASSET_ROOT / "openthoughts3-1.2m"
    model_root = ASSET_ROOT / "models/student-base"
    manifest_path = ASSET_ROOT / "asset-manifest.json"
    dataset_info_path = ASSET_ROOT / "dataset_info.json"

    dataset_root.mkdir(parents=True, exist_ok=True)
    shard_records = []
    row_count = 0
    first = int(dataset["first_shard"])
    count = int(dataset["shard_count"])
    total = int(dataset["total_shards"])
    for shard_index in range(first, first + count):
        filename = f"data/train-{shard_index:05d}-of-{total:05d}.parquet"
        local_path = Path(
            hf_hub_download(
                repo_id=dataset["repo_id"],
                repo_type="dataset",
                revision=dataset["revision"],
                filename=filename,
                local_dir=dataset_root,
            )
        )
        rows = pq.ParquetFile(local_path).metadata.num_rows
        row_count += rows
        shard_records.append(
            {
                "filename": filename,
                "rows": rows,
                "bytes": local_path.stat().st_size,
                "sha256": _sha256(local_path),
            }
        )

    expected_rows = int(dataset["subset_rows"])
    if row_count != expected_rows:
        raise RuntimeError(
            f"Pinned SFT subset row count mismatch: {row_count} != {expected_rows}"
        )

    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_info_pending = dataset_info_path.with_suffix(".json.pending")
    shutil.copyfile(DATASET_INFO_SOURCE, dataset_info_pending)
    dataset_info_pending.replace(dataset_info_path)

    snapshot_download(
        repo_id=student["repo_id"],
        revision=student["revision"],
        local_dir=model_root,
    )
    model_index = model_root / "model.safetensors.index.json"
    if not model_index.is_file():
        raise RuntimeError(
            f"Missing model index after snapshot download: {model_index}"
        )

    manifest = {
        "schema_version": 1,
        "dataset": {
            "repo_id": dataset["repo_id"],
            "revision": dataset["revision"],
            "selection": dataset["selection"],
            "rows": row_count,
            "shards": shard_records,
        },
        "student_base": {
            "repo_id": student["repo_id"],
            "revision": student["revision"],
            "upstream_repo_id": student["upstream_repo_id"],
            "upstream_revision": student["upstream_revision"],
            "model_index_sha256": _sha256(model_index),
        },
        "config_sha256": _sha256(config_path),
        "dataset_info_sha256": _sha256(dataset_info_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pending = manifest_path.with_suffix(".json.pending")
    pending.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pending.replace(manifest_path)
    volume.commit()
    print(
        "DPCA_SFT_ASSET_MANIFEST_JSON=" + json.dumps(manifest, sort_keys=True),
        flush=True,
    )
    return {
        "status": "ready",
        "dataset_rows": row_count,
        "dataset_shards": len(shard_records),
        "manifest": str(manifest_path),
        "model": str(model_root),
    }


def _render_llamafactory_config(
    config: dict[str, Any],
    output_dir: Path,
    *,
    optimizer_steps: int | None = None,
    save_step: int | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    sft = config["sft"]
    distributed = sft["distributed"]
    effective_optimizer_steps = int(optimizer_steps or sft["optimizer_steps"])
    effective_save_step = int(save_step or sft["save_step"])
    if int(sft["global_batch_size"]) != (
        int(sft["gpu_count"])
        * int(sft["per_device_micro_batch_size"])
        * int(sft["gradient_accumulation_steps"])
    ):
        raise RuntimeError(
            "SFT global batch does not match GPU x micro-batch x accumulation"
        )
    return {
        "model_name_or_path": str(ASSET_ROOT / "models/student-base"),
        "trust_remote_code": False,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        "dataset": "openthoughts3_400k_pinned",
        "dataset_dir": str(ASSET_ROOT),
        "template": sft["template"],
        "cutoff_len": int(sft["maximum_sequence_length"]),
        "max_samples": int(sft["dataset"]["subset_rows"]),
        "overwrite_cache": False,
        "tokenized_path": sft["tokenized_path"],
        "data_shared_file_system": True,
        "preprocessing_num_workers": 16,
        "packing": bool(sft["packing"]),
        "flash_attn": sft["attention_backend"],
        "gradient_checkpointing": bool(sft["gradient_checkpointing"]),
        "disable_gradient_checkpointing": bool(sft["disable_gradient_checkpointing"]),
        "use_reentrant_gc": bool(sft["use_reentrant_gc"]),
        "fsdp": "full_shard auto_wrap",
        "fsdp_config": {
            "transformer_layer_cls_to_wrap": distributed[
                "transformer_layer_cls_to_wrap"
            ],
            "activation_checkpointing": bool(distributed["activation_checkpointing"]),
            "cpu_ram_efficient_loading": bool(distributed["cpu_ram_efficient_loading"]),
            "state_dict_type": distributed["state_dict_type"],
        },
        "output_dir": str(output_dir),
        "logging_dir": str(output_dir / "tensorboard"),
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": effective_save_step,
        "save_total_limit": 1,
        "save_only_model": False,
        "save_safetensors": True,
        "overwrite_output_dir": False,
        "per_device_train_batch_size": int(sft["per_device_micro_batch_size"]),
        "gradient_accumulation_steps": int(sft["gradient_accumulation_steps"]),
        "learning_rate": float(sft["learning_rate"]),
        "max_steps": effective_optimizer_steps,
        "lr_scheduler_type": sft["scheduler"],
        "warmup_steps": int(sft["warmup_steps"]),
        "optim": sft["optimizer"],
        "weight_decay": float(sft["weight_decay"]),
        "max_grad_norm": float(sft["gradient_clip"]),
        "bf16": True,
        "ddp_find_unused_parameters": False,
        "ddp_timeout": 180000000,
        "seed": int(sft["seed"]),
        "data_seed": int(sft["seed"]),
        # The rank-aware callback is the single writer for W&B, TensorBoard,
        # and JSONL. Disabling Trainer reporters avoids duplicate/sparse rows.
        "report_to": [],
        "run_name": run_name or config["run_id"],
        "include_num_input_tokens_seen": True,
        "include_tokens_per_second": True,
        "include_effective_tokens_per_second": True,
        "logging_nan_inf_filter": False,
        "skip_memory_metrics": False,
        "tf32": bool(sft["tf32"]),
        "torch_empty_cache_steps": int(sft["torch_empty_cache_steps"]),
    }


@app.function(
    image=image,
    gpu="B200:2",
    cpu=32,
    memory=196_608,
    timeout=28_800,
    retries=0,
    single_use_containers=True,
    secrets=[wandb_secret],
    volumes={str(WORKSPACE_ROOT): volume},
)
def train_sft(
    qualification_steps: int | None = None,
    config_name: str = "dpca_8b_scientific.json",
) -> dict[str, Any]:
    import yaml

    volume.reload()
    config, config_path = _load_config(config_name)
    sft = config["sft"]
    if qualification_steps is not None and qualification_steps < 1:
        raise RuntimeError("qualification_steps must be positive")
    expected_steps = int(qualification_steps or sft["optimizer_steps"])
    save_step = (
        expected_steps if qualification_steps is not None else int(sft["save_step"])
    )
    run_id = (
        f"{config['run_id']}-fit-qualification-{qualification_steps}step"
        if qualification_steps is not None
        else config["run_id"]
    )
    if int(sft["gpu_count"]) != 2:
        raise RuntimeError("This qualification function requires exactly two B200 GPUs")
    run_dir = RUN_ROOT / run_id
    if run_dir.exists():
        raise RuntimeError(
            f"Run directory already exists; refusing duplicate run: {run_dir}"
        )
    manifest_path = ASSET_ROOT / "asset-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Pinned assets are missing; run prepare_assets first")

    run_dir.mkdir(parents=True)
    effective_path = run_dir / "effective_config.json"
    effective_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lf_config = _render_llamafactory_config(
        config,
        run_dir,
        optimizer_steps=expected_steps,
        save_step=save_step,
        run_name=run_id,
    )
    lf_config_path = run_dir / "llamafactory-effective.yaml"
    lf_config_path.write_text(
        yaml.safe_dump(lf_config, sort_keys=False), encoding="utf-8"
    )
    source_manifest_path = _write_source_manifest(run_dir, config_path)
    log_path = run_dir / "sft.log"

    environment = os.environ.copy()
    stage_tag = (
        "scientific-qualification"
        if qualification_steps is not None
        else "scientific-run"
    )
    variant_tag = sft.get("tracking_variant")
    variant_fragment = f"variant:{variant_tag}," if variant_tag else ""
    environment.update(
        {
            "FORCE_TORCHRUN": "1",
            "PYTORCH_ALLOC_CONF": sft["pytorch_alloc_conf"],
            "WANDB_PROJECT": config["tracking"]["wandb_project"],
            "WANDB_ENTITY": config["tracking"]["wandb_entity"],
            "WANDB_NAME": run_id,
            "WANDB_RUN_ID": run_id,
            "WANDB_JOB_TYPE": "sft-initialization",
            "WANDB_TAGS": (
                "schema:experiment-v1,framework:llamafactory,method:sft,"
                "regime:off-policy,objective:causal-cross-entropy,"
                f"{variant_fragment}"
                "platform:modal,accelerator:b200x2,"
                f"budget:{expected_steps}-update,stage:{stage_tag},"
                "student:meta-llama-3.1-8b"
            ),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc_per_node={int(sft['gpu_count'])}",
                str(config_path.parent / "llamafactory_sft_entry.py"),
                "--llamafactory-config",
                str(lf_config_path),
                "--scientific-config",
                str(effective_path),
                "--asset-manifest",
                str(manifest_path),
                "--source-manifest",
                str(source_manifest_path),
            ],
            cwd="/opt/dpca",
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_stream.write(line)
            log_stream.flush()
        return_code = process.wait()
    runtime_seconds = time.perf_counter() - started

    trainer_state_path = run_dir / "trainer_state.json"
    checkpoint_dir = run_dir / f"checkpoint-{save_step}"
    if return_code != 0:
        volume.commit()
        raise RuntimeError(
            f"LLaMA-Factory SFT failed with exit code {return_code}; log={log_path}"
        )
    if not trainer_state_path.is_file():
        candidates = sorted(run_dir.glob("checkpoint-*/trainer_state.json"))
        if candidates:
            trainer_state_path = candidates[-1]
    if not trainer_state_path.is_file():
        raise RuntimeError("SFT finished without trainer_state.json")
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    global_step = int(trainer_state["global_step"])
    if global_step != expected_steps:
        raise RuntimeError(
            f"Expected {expected_steps} optimizer steps; got {global_step}"
        )
    if not checkpoint_dir.is_dir():
        candidates = sorted(run_dir.glob("checkpoint-*"))
        if not candidates:
            raise RuntimeError("SFT finished without a checkpoint directory")
        checkpoint_dir = candidates[-1]
    model_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not model_files:
        raise RuntimeError(f"SFT checkpoint has no safetensors: {checkpoint_dir}")
    required_resume_files = [
        checkpoint_dir / "optimizer.bin",
        checkpoint_dir / "scheduler.pt",
        checkpoint_dir / "trainer_state.json",
        checkpoint_dir / "training_args.bin",
    ]
    missing_resume_files = [
        str(path) for path in required_resume_files if not path.is_file()
    ]
    rng_state_files = sorted(checkpoint_dir.glob("rng_state_*.pth"))
    if missing_resume_files or len(rng_state_files) != int(sft["gpu_count"]):
        raise RuntimeError(
            "SFT checkpoint is not resume-complete: "
            f"missing={missing_resume_files}, rng_states={len(rng_state_files)}"
        )
    event_files = sorted(run_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise RuntimeError("SFT produced no TensorBoard event file")
    metrics_path = run_dir / "metrics.jsonl"
    telemetry_summary_path = run_dir / "telemetry-summary.json"
    runtime_environment_path = run_dir / "runtime-environment.json"
    required_telemetry = [
        metrics_path,
        telemetry_summary_path,
        runtime_environment_path,
    ]
    missing_telemetry = [str(path) for path in required_telemetry if not path.is_file()]
    if missing_telemetry:
        raise RuntimeError(
            f"SFT finished without required telemetry: {missing_telemetry}"
        )
    metric_rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    optimizer_rows = [row for row in metric_rows if "train/loss" in row]
    if len(optimizer_rows) != expected_steps:
        raise RuntimeError(
            f"Expected {expected_steps} optimizer telemetry rows; got {len(optimizer_rows)}"
        )

    log_history = trainer_state.get("log_history", [])
    numeric_rows = [
        row for row in log_history if isinstance(row, dict) and "loss" in row
    ]
    summary = {
        "status": "completed",
        "run_id": run_id,
        "qualification_steps": qualification_steps,
        "optimizer_steps_completed": global_step,
        "global_batch_size": int(sft["global_batch_size"]),
        "maximum_sequence_length": int(sft["maximum_sequence_length"]),
        "attention_backend": sft["attention_backend"],
        "attention_kernel_policy": sft.get("attention_kernel_policy", "auto"),
        "pytorch_alloc_conf": sft["pytorch_alloc_conf"],
        "torch_empty_cache_steps": int(sft["torch_empty_cache_steps"]),
        "gpu_count": int(sft["gpu_count"]),
        "gpu_type": "B200",
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_safetensors": len(model_files),
        "checkpoint_resume_state_files": len(required_resume_files)
        + len(rng_state_files),
        "tensorboard_event_files": len(event_files),
        "logged_loss_rows": len(numeric_rows),
        "unified_metric_rows": len(optimizer_rows),
        "metrics_jsonl": str(metrics_path),
        "telemetry_summary": str(telemetry_summary_path),
        "runtime_environment": str(runtime_environment_path),
        "source_manifest": str(source_manifest_path),
        "runtime_seconds": runtime_seconds,
        "final_log": numeric_rows[-1] if numeric_rows else None,
        "asset_manifest": str(manifest_path),
        "effective_config": str(effective_path),
        "llamafactory_effective_config": str(lf_config_path),
        "log": str(log_path),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    volume.commit()
    print("DPCA_SFT_RESULT_JSON=" + json.dumps(summary, sort_keys=True), flush=True)
    return summary


@app.local_entrypoint()
def main() -> None:
    asset_result = prepare_assets.remote(CONFIG_NAME)
    print("DPCA_SFT_ASSET_RESULT_JSON=" + json.dumps(asset_result, sort_keys=True))
    train_result = train_sft.remote(None, CONFIG_NAME)
    print("DPCA_SFT_LOCAL_RESULT_JSON=" + json.dumps(train_result, sort_keys=True))


@app.local_entrypoint()
def qualify(steps: int = 2) -> None:
    asset_result = prepare_assets.remote(CONFIG_NAME)
    print("DPCA_SFT_ASSET_RESULT_JSON=" + json.dumps(asset_result, sort_keys=True))
    train_result = train_sft.remote(steps, CONFIG_NAME)
    print(
        "DPCA_SFT_QUALIFICATION_RESULT_JSON=" + json.dumps(train_result, sort_keys=True)
    )
