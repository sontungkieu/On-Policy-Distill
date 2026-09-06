"""Ten-step DPCA topology and timing qualification run.

This is intentionally a small public-model run. It executes the authors' DPCA
alignment and OPD policy-loss functions but is not a matched reproduction of
the paper's SFT-initialized 7B/8B experiments. Device placement is declared by
the run config so the same scientific inputs can qualify one or two B200s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
)
from wandb_schema import resolve_wandb_metadata

from verl.trainer.ppo.core_algos import compute_policy_loss_opd
from verl.workers.reward_manager.opd import TeacherClient


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _response_logps(
    model: torch.nn.Module,
    prompt_ids: list[int],
    response_ids: list[int],
    device: torch.device,
    *,
    grad: bool,
) -> torch.Tensor:
    all_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(all_ids)
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        logits = model(
            input_ids=all_ids, attention_mask=attention_mask, use_cache=False
        ).logits
        start = len(prompt_ids) - 1
        response_logits = logits[:, start : start + len(response_ids), :]
        targets = torch.tensor([response_ids], dtype=torch.long, device=device)
        logps = (
            F.log_softmax(response_logits.float(), dim=-1)
            .gather(-1, targets.unsqueeze(-1))
            .squeeze(-1)
        )
    return logps


def _strip_trailing_special_tokens(ids: list[int], tokenizer: Any) -> list[int]:
    special_ids = set(tokenizer.all_special_ids)
    while ids and ids[-1] in special_ids:
        ids.pop()
    return ids


def _project_advantage_for_metrics(
    prior: torch.Tensor,
    teacher_aligned: torch.Tensor,
    chunk_ids: torch.Tensor,
    clamp: float,
) -> torch.Tensor:
    sentinel = torch.isinf(teacher_aligned) | (chunk_ids < 0)
    teacher_clean = torch.where(sentinel, prior, teacher_aligned)
    target = prior.clone()
    valid = ~sentinel
    for chunk_id in torch.unique(chunk_ids[valid]):
        mask = valid & (chunk_ids == chunk_id)
        teacher_sum = teacher_clean[mask].sum()
        student_sum = prior[mask].sum()
        if torch.abs(student_sum) < 1e-8:
            target[mask] = teacher_sum / mask.sum().clamp_min(1)
        else:
            target[mask] = (teacher_sum / student_sum) * prior[mask]
    return torch.clamp(target - prior, min=-clamp, max=clamp)


def _gpu_metrics(prefix: str, device_index: int) -> dict[str, float]:
    return {
        f"gpu/{prefix}_allocated_gib": torch.cuda.memory_allocated(device_index)
        / (1024**3),
        f"gpu/{prefix}_reserved_gib": torch.cuda.memory_reserved(device_index)
        / (1024**3),
        f"gpu/{prefix}_peak_allocated_gib": torch.cuda.max_memory_allocated(
            device_index
        )
        / (1024**3),
    }


def _mean(values: torch.Tensor) -> float:
    return float(values.detach().float().mean().item())


def _run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    execution = config.get("execution", {})
    expected_gpu_count = int(execution.get("gpu_count", 2))
    teacher_device_index = int(execution.get("teacher_device_index", 0))
    student_device_index = int(execution.get("student_device_index", 1))
    actual_gpu_count = torch.cuda.device_count()
    if actual_gpu_count != expected_gpu_count:
        raise RuntimeError(
            f"Expected exactly {expected_gpu_count} CUDA GPU(s); found {actual_gpu_count}"
        )
    if not (
        0 <= teacher_device_index < actual_gpu_count
        and 0 <= student_device_index < actual_gpu_count
    ):
        raise RuntimeError(
            "Invalid device placement: "
            f"teacher={teacher_device_index}, student={student_device_index}, "
            f"cuda_device_count={actual_gpu_count}"
        )
    teacher_device = torch.device(f"cuda:{teacher_device_index}")
    student_device = torch.device(f"cuda:{student_device_index}")
    device_records = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        device_records.append(
            {
                "index": index,
                "name": props.name,
                "capability": list(props.major_minor)
                if hasattr(props, "major_minor")
                else list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": props.total_memory,
            }
        )
    if not all("B200" in record["name"] for record in device_records):
        raise RuntimeError(
            f"Expected {expected_gpu_count} B200 device(s), got {device_records}"
        )

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    output_dir.mkdir(parents=True, exist_ok=False)
    tb_dir = output_dir / "tensorboard"
    checkpoint_dir = output_dir / "checkpoint-final"
    metrics_path = output_dir / "metrics.jsonl"
    effective_config_path = output_dir / "effective_config.json"
    prompts_path = Path(config["prompts_path"])
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    steps = int(config["optimizer"]["steps"])
    if len(prompts) < steps:
        raise RuntimeError(f"Need at least {steps} prompts; found {len(prompts)}")

    source_root = Path(__file__).resolve().parents[1]
    source_manifest_path = source_root / "reproduction/source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    expected_source_hashes = source_manifest.get(
        "effective_sha256", source_manifest["sha256"]
    )
    actual_source_hashes = {
        relative_path: _sha256(source_root / relative_path)
        for relative_path in expected_source_hashes
    }
    if actual_source_hashes != expected_source_hashes:
        raise RuntimeError(
            f"Pinned effective source hash mismatch: expected={expected_source_hashes} "
            f"actual={actual_source_hashes}"
        )
    provenance = {
        "git_commit": source_manifest["commit"],
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_hashes": actual_source_hashes,
        "prompt_sha256": _sha256(prompts_path),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "devices": device_records,
    }
    if provenance["git_commit"] != config["source_commit"]:
        raise RuntimeError(
            f"Source commit mismatch: {provenance['git_commit']} != {config['source_commit']}"
        )
    effective = {"config": config, "provenance": provenance}
    effective_config_path.write_text(
        json.dumps(effective, indent=2, sort_keys=True), encoding="utf-8"
    )

    teacher_path = config["teacher"]["container_path"]
    student_path = config["student"]["container_path"]
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_path, local_files_only=True, use_fast=True
    )
    student_tokenizer = AutoTokenizer.from_pretrained(
        student_path, local_files_only=True, use_fast=True
    )
    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(teacher_device)
    student = AutoModelForCausalLM.from_pretrained(
        student_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(student_device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student.train()
    student.config.use_cache = False

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config["optimizer"]["warmup_steps"]),
    )
    loss_config = OmegaConf.create(
        {
            "clip_ratio": float(config["opd"]["clip_ratio_low"]),
            "clip_ratio_low": float(config["opd"]["clip_ratio_low"]),
            "clip_ratio_high": float(config["opd"]["clip_ratio_high"]),
            "clip_ratio_c": float(config["opd"]["clip_ratio_c"]),
            "policy_loss": {
                "opd_loss_max_clamp": float(config["opd"]["loss_max_clamp"])
            },
            "global_batch_info": {},
        }
    )

    writer = SummaryWriter(log_dir=str(tb_dir), flush_secs=5)
    wandb_metadata = resolve_wandb_metadata(config)
    wb_run = wandb.init(
        project=config["wandb_project"],
        name=config["run_id"],
        job_type=wandb_metadata["job_type"],
        config=effective,
        tags=wandb_metadata["tags"],
        settings=wandb.Settings(init_timeout=180),
    )
    if wb_run.offline:
        raise RuntimeError(
            "W&B initialized offline; online logging is required for this run"
        )
    wandb.define_metric("optimizer_step")
    wandb.define_metric("*", step_metric="optimizer_step")
    writer.add_text("run/evidence_class", config["evidence_class"], 0)
    writer.add_text("run/source_commit", config["source_commit"], 0)

    step_metrics: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    for step_index in range(steps):
        optimizer_step = step_index + 1
        prompt = prompts[step_index]
        student_prompt_text = student_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        teacher_prompt_text = teacher_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        student_prompt_ids = student_tokenizer(
            student_prompt_text, add_special_tokens=False
        )["input_ids"]
        teacher_prompt_ids = teacher_tokenizer(
            teacher_prompt_text, add_special_tokens=False
        )["input_ids"]

        student.eval()
        generation_start = time.perf_counter()
        with torch.no_grad():
            prompt_tensor = torch.tensor(
                [student_prompt_ids], dtype=torch.long, device=student_device
            )
            generated = student.generate(
                input_ids=prompt_tensor,
                attention_mask=torch.ones_like(prompt_tensor),
                do_sample=True,
                temperature=float(config["rollout"]["temperature"]),
                top_p=float(config["rollout"]["top_p"]),
                top_k=int(config["rollout"]["top_k"]),
                min_new_tokens=int(config["rollout"]["min_new_tokens"]),
                max_new_tokens=int(config["rollout"]["max_new_tokens"]),
                pad_token_id=student_tokenizer.pad_token_id,
            )
        torch.cuda.synchronize(student_device)
        generation_seconds = time.perf_counter() - generation_start
        response_ids = _strip_trailing_special_tokens(
            generated[0, len(student_prompt_ids) :].tolist(), student_tokenizer
        )
        if not response_ids:
            raise RuntimeError(
                f"Step {optimizer_step} produced no response content tokens"
            )
        response_text = student_tokenizer.decode(
            response_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        teacher_response_ids = teacher_tokenizer(
            response_text, add_special_tokens=False
        )["input_ids"]
        if not teacher_response_ids:
            raise RuntimeError(
                f"Step {optimizer_step} produced no teacher response tokens"
            )

        score_start = time.perf_counter()
        old_logp = _response_logps(
            student, student_prompt_ids, response_ids, student_device, grad=False
        ).detach()
        teacher_logp = _response_logps(
            teacher,
            teacher_prompt_ids,
            teacher_response_ids,
            teacher_device,
            grad=False,
        ).detach()
        torch.cuda.synchronize(teacher_device)
        torch.cuda.synchronize(student_device)
        score_seconds = time.perf_counter() - score_start

        aligned_cpu, chunk_ids_cpu, chunk_details, teacher_truncated = (
            TeacherClient._align_chunks(
                response_ids,
                teacher_response_ids,
                teacher_logp[0].float().cpu(),
                student_tokenizer,
                teacher_tokenizer,
                debug=True,
                large_chunk_threshold=int(config["opd"]["large_chunk_threshold"]),
            )
        )
        aligned = aligned_cpu.to(student_device).unsqueeze(0)
        chunk_ids = chunk_ids_cpu.to(student_device).unsqueeze(0)
        packed_teacher = torch.cat([aligned, chunk_ids], dim=-1)
        response_mask = torch.ones_like(old_logp, dtype=torch.float32)
        advantage = _project_advantage_for_metrics(
            old_logp[0],
            aligned[0],
            chunk_ids[0],
            clamp=float(config["opd"]["loss_max_clamp"]),
        )

        student.train()
        optimizer.zero_grad(set_to_none=True)
        update_start = time.perf_counter()
        current_logp = _response_logps(
            student, student_prompt_ids, response_ids, student_device, grad=True
        )
        lr_used = float(optimizer.param_groups[0]["lr"])
        loss, upstream_metrics = compute_policy_loss_opd(
            old_log_prob=old_logp,
            log_prob=current_logp,
            advantages=packed_teacher,
            response_mask=response_mask,
            loss_agg_mode=config["opd"]["loss_agg_mode"],
            config=loss_config,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at optimizer step {optimizer_step}: {loss.item()}"
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            student.parameters(), float(config["optimizer"]["gradient_clip"])
        )
        if not torch.isfinite(grad_norm):
            raise RuntimeError(
                f"Non-finite gradient norm at optimizer step {optimizer_step}: {grad_norm.item()}"
            )
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize(student_device)
        update_seconds = time.perf_counter() - update_start

        valid_chunks = [
            detail for detail in chunk_details if detail["type"] == "matched"
        ]
        one_to_one = sum(
            1
            for detail in valid_chunks
            if detail["n_stu"] == 1 and detail["n_tec"] == 1
        )
        multi = len(valid_chunks) - one_to_one
        inf_tokens = int(torch.isinf(aligned).sum().item())
        valid_tokens = len(response_ids) - inf_tokens
        total_seconds = generation_seconds + score_seconds + update_seconds
        metrics: dict[str, Any] = {
            "optimizer_step": optimizer_step,
            "loss/opd_policy": float(loss.detach().item()),
            "loss/ppo_kl": float(upstream_metrics["actor/ppo_kl"]),
            "loss/clip_fraction": float(upstream_metrics["actor/pg_clipfrac"]),
            "loss/clip_fraction_lower": float(
                upstream_metrics["actor/pg_clipfrac_lower"]
            ),
            "opd/inf_tokens": inf_tokens,
            "opd/inf_ratio": inf_tokens / max(len(response_ids), 1),
            "opd/valid_tokens": valid_tokens,
            "opd/chunks": len(valid_chunks),
            "opd/chunks_one_to_one": one_to_one,
            "opd/chunks_multi": multi,
            "opd/multi_chunk_ratio": multi / max(len(valid_chunks), 1),
            "opd/max_student_tokens_per_chunk": max(
                (d["n_stu"] for d in valid_chunks), default=0
            ),
            "opd/max_teacher_tokens_per_chunk": max(
                (d["n_tec"] for d in valid_chunks), default=0
            ),
            "opd/teacher_truncated": int(teacher_truncated),
            "logp/student_mean": _mean(old_logp),
            "logp/student_sum": float(old_logp.detach().sum().item()),
            "logp/teacher_mean": _mean(teacher_logp),
            "logp/teacher_sum": float(teacher_logp.detach().sum().item()),
            "advantage/mean": _mean(advantage),
            "advantage/min": float(advantage.detach().min().item()),
            "advantage/max": float(advantage.detach().max().item()),
            "advantage/std": float(
                advantage.detach().float().std(unbiased=False).item()
            ),
            "tokens/student_prompt": len(student_prompt_ids),
            "tokens/student_response": len(response_ids),
            "tokens/teacher_prompt": len(teacher_prompt_ids),
            "tokens/teacher_response": len(teacher_response_ids),
            "timing/generation_seconds": generation_seconds,
            "timing/scoring_seconds": score_seconds,
            "timing/update_seconds": update_seconds,
            "timing/step_seconds": total_seconds,
            "throughput/student_response_tokens_per_second": len(response_ids)
            / max(total_seconds, 1e-9),
            "optimizer/lr_used": lr_used,
            "optimizer/lr_next": float(scheduler.get_last_lr()[0]),
            "optimizer/grad_norm_before_clip": float(grad_norm.detach().item()),
            **_gpu_metrics("teacher", teacher_device_index),
            **_gpu_metrics("student", student_device_index),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        wandb.log(metrics)
        for name, value in metrics.items():
            if (
                name != "optimizer_step"
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
            ):
                writer.add_scalar(name, value, optimizer_step)
        writer.flush()
        step_metrics.append(metrics)
        print("DPCA_STEP_JSON=" + json.dumps(metrics, sort_keys=True), flush=True)

    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    student.save_pretrained(checkpoint_dir, safe_serialization=True)
    student_tokenizer.save_pretrained(checkpoint_dir)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "optimizer_steps": steps,
        },
        checkpoint_dir / "trainer_state.pt",
    )
    total_runtime = time.perf_counter() - run_start
    summary = {
        "status": "completed",
        "evidence_class": config["evidence_class"],
        "run_id": config["run_id"],
        "optimizer_steps_completed": len(step_metrics),
        "b200_topology_verified": True,
        "two_gpu_b200_verified": expected_gpu_count == 2,
        "gpu_count": expected_gpu_count,
        "device_placement": {
            "teacher_device_index": teacher_device_index,
            "student_device_index": student_device_index,
            "shared_device": teacher_device_index == student_device_index,
        },
        "teacher_device": device_records[teacher_device_index],
        "student_device": device_records[student_device_index],
        "wandb_run_id": wb_run.id,
        "wandb_run_url": wb_run.url,
        "tensorboard_dir": str(tb_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "metrics_path": str(metrics_path),
        "total_runtime_seconds": total_runtime,
        "final_metrics": step_metrics[-1],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    writer.add_text("run/wandb_url", wb_run.url or "", steps)
    writer.close()
    artifact = wandb.Artifact(
        f"{config['run_id']}-logs", type="implementation-validation"
    )
    artifact.add_file(str(summary_path))
    artifact.add_file(str(metrics_path))
    artifact.add_file(str(effective_config_path))
    wb_run.log_artifact(artifact)
    wb_run.summary.update(summary)
    wb_run.finish(exit_code=0)
    print(
        "DPCA_TRAINING_SUMMARY_JSON=" + json.dumps(summary, sort_keys=True), flush=True
    )
    return summary


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_MODE", "online")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/opt/dpca/reproduction/configs/dpca_2gpu_10step.json"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("/runs"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = _run(config, args.output_root / config["run_id"])
    expected_steps = int(config["optimizer"]["steps"])
    if summary["optimizer_steps_completed"] != expected_steps:
        raise RuntimeError(
            f"Expected exactly {expected_steps} optimizer steps, "
            f"got {summary['optimizer_steps_completed']}"
        )


if __name__ == "__main__":
    main()
