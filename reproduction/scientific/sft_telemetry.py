"""Unified, rank-aware telemetry for the LLaMA-Factory SFT phase.

The module uses LLaMA-Factory's public ``run_exp(..., callbacks=[...])`` hook and
wraps only ``training_step`` to observe the batch that is actually consumed.
It does not modify inputs, loss computation, gradients, optimizer state, or the
checkpoint format.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import wandb
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback

IGNORE_INDEX = -100


def package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_trainer_logs(logs: dict[str, Any] | None) -> dict[str, float]:
    """Return finite numeric Trainer logs under the shared metric namespace."""

    normalized: dict[str, float] = {}
    for key, value in (logs or {}).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            metric = key if "/" in key else f"train/{key}"
            normalized[metric] = number
    return normalized


def _quantile(values: list[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return float(ordered[low])
    weight = index - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _distributed_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def _distributed_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


class StepAccumulator:
    """Collect detached scalar tensors without synchronizing every microbatch."""

    def __init__(self, cutoff_len: int) -> None:
        self.cutoff_len = int(cutoff_len)
        self.eos_token_ids: set[int] = set()
        self.reset()

    def reset(self) -> None:
        self.microbatches = 0
        self.samples = 0
        self._active_tokens: list[torch.Tensor] = []
        self._supervised_tokens: list[torch.Tensor] = []
        self._padded_positions: list[int] = []
        self._eos_targets: list[torch.Tensor] = []
        self._sequences_with_eos: list[torch.Tensor] = []
        self._sequence_lengths: list[torch.Tensor] = []
        self.training_step_seconds = 0.0

    def set_eos_token_ids(self, value: Any) -> None:
        if isinstance(value, (list, tuple, set)):
            self.eos_token_ids = {int(item) for item in value if item is not None}
        elif value is not None:
            self.eos_token_ids = {int(value)}

    def record(
        self, inputs: dict[str, Any], training_step_seconds: float | None = None
    ) -> None:
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        reference = attention_mask if torch.is_tensor(attention_mask) else input_ids
        if not torch.is_tensor(reference) or reference.ndim < 2:
            return

        batch_size = int(reference.shape[0])
        self.microbatches += 1
        self.samples += batch_size
        self._padded_positions.append(int(reference.numel()))

        if torch.is_tensor(attention_mask):
            sequence_lengths = attention_mask.detach().to(torch.int64).sum(dim=-1)
        else:
            sequence_lengths = torch.full(
                (batch_size,),
                int(reference.shape[-1]),
                dtype=torch.int64,
                device=reference.device,
            )
        self._sequence_lengths.append(sequence_lengths)
        self._active_tokens.append(sequence_lengths.sum())

        if torch.is_tensor(labels):
            detached_labels = labels.detach()
            supervised_mask = detached_labels.ne(IGNORE_INDEX)
            self._supervised_tokens.append(supervised_mask.to(torch.int64).sum())
            if self.eos_token_ids:
                eos_mask = torch.zeros_like(supervised_mask)
                for token_id in self.eos_token_ids:
                    eos_mask.logical_or_(detached_labels.eq(token_id))
                eos_mask.logical_and_(supervised_mask)
                self._eos_targets.append(eos_mask.to(torch.int64).sum())
                self._sequences_with_eos.append(
                    eos_mask.any(dim=-1).to(torch.int64).sum()
                )

        if training_step_seconds is not None:
            self.training_step_seconds += float(training_step_seconds)

    def snapshot_local(self) -> dict[str, Any]:
        device = None
        for values in (
            self._active_tokens,
            self._supervised_tokens,
            self._sequence_lengths,
        ):
            if values:
                device = values[0].device
                break

        def tensor_sum(values: list[torch.Tensor]) -> int:
            if not values:
                return 0
            return int(
                torch.stack([value.reshape(()) for value in values]).sum().item()
            )

        if self._sequence_lengths:
            sequence_lengths = (
                torch.cat([value.reshape(-1) for value in self._sequence_lengths])
                .cpu()
                .tolist()
            )
        else:
            sequence_lengths = []

        cuda_metrics: dict[str, int] = {}
        if device is not None and device.type == "cuda":
            index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            cuda_metrics = {
                "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
            }

        return {
            "rank": _distributed_rank(),
            "cutoff_len": self.cutoff_len,
            "microbatches": self.microbatches,
            "samples": self.samples,
            "active_tokens": tensor_sum(self._active_tokens),
            "supervised_tokens": tensor_sum(self._supervised_tokens),
            "padded_positions": sum(self._padded_positions),
            "eos_targets": tensor_sum(self._eos_targets),
            "sequences_with_eos": tensor_sum(self._sequences_with_eos),
            "sequence_lengths": [int(value) for value in sequence_lengths],
            "training_step_seconds": self.training_step_seconds,
            "cuda": cuda_metrics,
        }


def aggregate_rank_snapshots(snapshots: list[dict[str, Any]]) -> dict[str, float]:
    samples = sum(int(item["samples"]) for item in snapshots)
    active = sum(int(item["active_tokens"]) for item in snapshots)
    supervised = sum(int(item["supervised_tokens"]) for item in snapshots)
    padded = sum(int(item["padded_positions"]) for item in snapshots)
    eos_targets = sum(int(item["eos_targets"]) for item in snapshots)
    sequences_with_eos = sum(int(item["sequences_with_eos"]) for item in snapshots)
    lengths = [
        int(length) for item in snapshots for length in item.get("sequence_lengths", [])
    ]
    training_times = [float(item["training_step_seconds"]) for item in snapshots]
    cutoff_len = int(snapshots[0]["cutoff_len"]) if snapshots else 0

    result: dict[str, float] = {
        "telemetry/world_size": float(len(snapshots)),
        "data/microbatches_global": float(
            sum(int(item["microbatches"]) for item in snapshots)
        ),
        "data/samples_global": float(samples),
        "data/active_tokens_global": float(active),
        "data/supervised_tokens_global": float(supervised),
        "data/prompt_tokens_global": float(max(active - supervised, 0)),
        "data/padded_positions_global": float(padded),
        "data/padding_ratio": float(max(padded - active, 0) / max(padded, 1)),
        "data/supervised_token_ratio": float(supervised / max(active, 1)),
        "data/eos_targets_global": float(eos_targets),
        "data/sequences_with_eos_ratio": float(sequences_with_eos / max(samples, 1)),
        "data/sequences_at_cutoff_ratio": float(
            sum(length >= cutoff_len for length in lengths) / max(samples, 1)
        ),
        "data/sequence_length_min": float(min(lengths, default=0)),
        "data/sequence_length_p10": _quantile(lengths, 0.10),
        "data/sequence_length_p50": _quantile(lengths, 0.50),
        "data/sequence_length_p90": _quantile(lengths, 0.90),
        "data/sequence_length_max": float(max(lengths, default=0)),
        "data/sequence_length_mean": float(
            statistics.fmean(lengths) if lengths else 0.0
        ),
        "timing/training_step_seconds_rank_min": min(training_times, default=0.0),
        "timing/training_step_seconds_rank_max": max(training_times, default=0.0),
        "timing/rank_straggler_ratio": float(
            max(training_times, default=0.0)
            / max(min(training_times, default=0.0), 1e-12)
        ),
    }

    for item in snapshots:
        rank = int(item["rank"])
        for key, value in item.get("cuda", {}).items():
            result[f"system/cuda_rank_{rank}/{key}"] = float(value)
    for key in (
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    ):
        values = [float(item.get("cuda", {}).get(key, 0)) for item in snapshots]
        result[f"system/cuda_global/{key}_max"] = max(values, default=0.0)
    return result


def install_training_step_instrumentation(accumulator: StepAccumulator) -> None:
    """Observe executed SFT batches through a transparent method wrapper."""

    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

    original = CustomSeq2SeqTrainer.training_step
    if getattr(original, "_dpca_telemetry_wrapper", False):
        raise RuntimeError("SFT training_step telemetry was installed more than once")

    def instrumented_training_step(self, model, inputs, *args, **kwargs):
        started = time.perf_counter()
        result = original(self, model, inputs, *args, **kwargs)
        accumulator.record(inputs, time.perf_counter() - started)
        return result

    instrumented_training_step._dpca_telemetry_wrapper = True  # type: ignore[attr-defined]
    CustomSeq2SeqTrainer.training_step = instrumented_training_step


class SFTTelemetryCallback(TrainerCallback):
    """Log the same scalar record to W&B, TensorBoard, and JSONL."""

    def __init__(
        self,
        *,
        accumulator: StepAccumulator,
        run_dir: Path,
        run_id: str,
        scientific_config: dict[str, Any],
        effective_config_path: Path,
        llamafactory_config_path: Path,
        asset_manifest_path: Path,
        source_manifest_path: Path,
    ) -> None:
        self.accumulator = accumulator
        self.run_dir = run_dir
        self.run_id = run_id
        self.scientific_config = scientific_config
        self.effective_config_path = effective_config_path
        self.llamafactory_config_path = llamafactory_config_path
        self.asset_manifest_path = asset_manifest_path
        self.source_manifest_path = source_manifest_path
        self.metrics_path = run_dir / "metrics.jsonl"
        self.rank_metrics_path = run_dir / f"metrics-rank-{_distributed_rank()}.jsonl"
        self.telemetry_summary_path = run_dir / "telemetry-summary.json"
        self.runtime_path = run_dir / "runtime-environment.json"
        self.writer: SummaryWriter | None = None
        self._step_started = 0.0
        self._optimizer_started = 0.0
        self._optimizer_seconds = 0.0
        self._pending: dict[str, float] = {}
        self._records = 0
        self._optimizer_records = 0
        self._nonfinite_events = 0
        self._wandb_run = None
        self._last_global_step = 0
        self._finalized = False

    @property
    def is_main_process(self) -> bool:
        return _distributed_rank() == 0

    def _gather(self, local: dict[str, Any]) -> list[dict[str, Any]]:
        if _distributed_world_size() == 1:
            return [local]
        gathered: list[dict[str, Any] | None] = [None] * _distributed_world_size()
        dist.all_gather_object(gathered, local)
        return [item for item in gathered if item is not None]

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def _runtime_record(self, args, model) -> dict[str, Any]:
        devices = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": props.name,
                        "capability": list(torch.cuda.get_device_capability(index)),
                        "total_memory_bytes": int(props.total_memory),
                    }
                )
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "rank": _distributed_rank(),
            "world_size": _distributed_world_size(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch_version": str(torch.__version__),
            "flash_attn_version": package_version("flash-attn"),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "distributed_backend": str(dist.get_backend())
            if dist.is_available() and dist.is_initialized()
            else None,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "attention_implementation": getattr(
                getattr(model, "config", None), "_attn_implementation", None
            ),
            "attention_kernel_policy": self.scientific_config.get("sft", {}).get(
                "attention_kernel_policy", "auto"
            ),
            "sdpa_flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
            "sdpa_math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
            "sdpa_mem_efficient_enabled": bool(
                torch.backends.cuda.mem_efficient_sdp_enabled()
            ),
            "sdpa_cudnn_enabled": bool(torch.backends.cuda.cudnn_sdp_enabled()),
            "bos_token_id": getattr(
                getattr(model, "config", None), "bos_token_id", None
            ),
            "eos_token_id": getattr(
                getattr(model, "config", None), "eos_token_id", None
            ),
            "pad_token_id": getattr(
                getattr(model, "config", None), "pad_token_id", None
            ),
            "max_grad_norm": float(args.max_grad_norm),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF"),
            "devices": devices,
        }

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        eos_ids = getattr(getattr(model, "config", None), "eos_token_id", None)
        self.accumulator.set_eos_token_ids(eos_ids)
        runtime_records = self._gather(self._runtime_record(args, model))
        if self.is_main_process:
            self.runtime_path.write_text(
                json.dumps({"ranks": runtime_records}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.writer = SummaryWriter(
                log_dir=str(self.run_dir / "tensorboard-telemetry")
            )
            tags = [
                item for item in os.environ.get("WANDB_TAGS", "").split(",") if item
            ]
            contract = {
                "scientific": self.scientific_config,
                "provenance": {
                    "scientific_config_sha256": sha256_file(self.effective_config_path),
                    "llamafactory_config_sha256": sha256_file(
                        self.llamafactory_config_path
                    ),
                    "asset_manifest_sha256": sha256_file(self.asset_manifest_path),
                    "source_manifest_sha256": sha256_file(self.source_manifest_path),
                },
                "runtime": runtime_records,
            }
            self._wandb_run = wandb.init(
                entity=os.environ["WANDB_ENTITY"],
                project=os.environ["WANDB_PROJECT"],
                id=self.run_id,
                name=self.run_id,
                job_type=os.environ.get("WANDB_JOB_TYPE", "sft-initialization"),
                tags=tags,
                resume=os.environ.get("WANDB_RESUME", "never"),
                config=contract,
                dir=str(self.run_dir / "wandb"),
            )
            wandb.define_metric("train/global_step")
            wandb.define_metric("*", step_metric="train/global_step")
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self.accumulator.reset()
        self._step_started = time.perf_counter()
        self._optimizer_seconds = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self._optimizer_started = time.perf_counter()
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self._optimizer_started:
            self._optimizer_seconds = time.perf_counter() - self._optimizer_started
        return control

    def on_step_end(self, args, state, control, **kwargs):
        local = self.accumulator.snapshot_local()
        local_record = {
            "schema_version": 1,
            "optimizer_step": int(state.global_step),
            "timestamp_unix": time.time(),
            **local,
        }
        self._append_jsonl(self.rank_metrics_path, local_record)
        snapshots = self._gather(local)
        if self.is_main_process:
            self._pending = aggregate_rank_snapshots(snapshots)
            self._pending["timing/optimizer_step_wall_seconds"] = float(
                time.perf_counter() - self._step_started
            )
            self._pending["timing/optimizer_call_seconds_rank0"] = float(
                self._optimizer_seconds
            )
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.is_main_process:
            return control
        base = normalize_trainer_logs(logs)
        step = int(state.global_step)
        self._last_global_step = step
        record: dict[str, Any] = {
            "schema_version": 1,
            "timestamp_unix": time.time(),
            "train/global_step": float(step),
            **base,
            **self._pending,
        }
        max_grad_norm = float(args.max_grad_norm)
        raw_grad_norm = (logs or {}).get("grad_norm")
        raw_loss = (logs or {}).get("loss")
        grad_norm = (
            float(raw_grad_norm) if isinstance(raw_grad_norm, (int, float)) else None
        )
        loss = float(raw_loss) if isinstance(raw_loss, (int, float)) else None
        record["diagnostic/loss_finite"] = float(loss is None or math.isfinite(loss))
        record["diagnostic/grad_norm_finite"] = float(
            grad_norm is None or math.isfinite(grad_norm)
        )
        record["optimizer/max_grad_norm"] = max_grad_norm
        record["optimizer/grad_clip_triggered"] = float(
            grad_norm is not None
            and math.isfinite(grad_norm)
            and grad_norm > max_grad_norm
        )
        nonfinite = int(not record["diagnostic/loss_finite"]) + int(
            not record["diagnostic/grad_norm_finite"]
        )
        self._nonfinite_events += nonfinite
        record["diagnostic/nonfinite_events_cumulative"] = float(self._nonfinite_events)
        self._append_jsonl(self.metrics_path, record)
        self._records += 1
        if "train/loss" in record:
            self._optimizer_records += 1
        if self._wandb_run is not None:
            self._wandb_run.log(record)
        if self.writer is not None:
            for key, value in record.items():
                if key not in {
                    "schema_version",
                    "timestamp_unix",
                    "train/global_step",
                } and isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)
            self.writer.flush()
        self._pending = {}
        return control

    def on_save(self, args, state, control, **kwargs):
        if self.is_main_process and self._wandb_run is not None:
            self._wandb_run.log(
                {
                    "train/global_step": float(state.global_step),
                    "checkpoint/save_event": 1.0,
                    "checkpoint/global_step": float(state.global_step),
                }
            )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._last_global_step = int(state.global_step)
        return control

    def finalize(self, exit_code: int) -> None:
        """Close telemetry after LLaMA-Factory has emitted its final train log."""

        if self._finalized:
            return
        self._finalized = True
        if not self.is_main_process:
            return
        summary = {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": "completed" if exit_code == 0 else "failed",
            "exit_code": int(exit_code),
            "optimizer_steps_completed": self._last_global_step,
            "metric_records": self._records,
            "optimizer_metric_records": self._optimizer_records,
            "nonfinite_events": self._nonfinite_events,
            "metrics_jsonl": str(self.metrics_path),
            "rank_metric_files": _distributed_world_size(),
            "runtime_environment": str(self.runtime_path),
        }
        self.telemetry_summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        if self._wandb_run is not None:
            artifact = wandb.Artifact(
                f"{self.run_id}-telemetry",
                type="scientific-run-telemetry",
                metadata=summary,
            )
            for path in (
                self.metrics_path,
                self.telemetry_summary_path,
                self.runtime_path,
                self.effective_config_path,
                self.llamafactory_config_path,
                self.asset_manifest_path,
                self.source_manifest_path,
            ):
                if path.is_file():
                    artifact.add_file(str(path))
            for path in sorted(self.run_dir.glob("metrics-rank-*.jsonl")):
                artifact.add_file(str(path))
            self._wandb_run.log_artifact(artifact)
            self._wandb_run.summary.update(summary)
            self._wandb_run.finish(exit_code=exit_code)
