"""Phase-bounded DPCA launcher with topology-portable checkpoint contracts.

The scientific schedule remains 500 optimizer steps with a 10-step linear
warmup. Qualification runs stop at step 10, save, then resume to step 20.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "reproduction" / "configs"
PROFILES = {
    "paper_8b": CONFIG_ROOT / "dpca_paper_8b.json",
    "common_4b1b": CONFIG_ROOT / "dpca_common_4b1b.json",
}
PHASES = ("warmup", "post_warmup")
GPU_COUNTS = (1, 2, 4)
REQUIRED_EXEC_ENV = (
    "MODEL_PATH",
    "TEACHER_CKPT_PATH",
    "TRAIN_FILE",
    "TEST_FILE",
    "CKPTS_DIR",
    "TRAIN_MANIFEST_SHA256",
    "VAL_MANIFEST_SHA256",
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_profile(name: str) -> dict[str, Any]:
    profile = json.loads(PROFILES[name].read_text(encoding="utf-8"))
    validate_profile(profile)
    return profile


def validate_profile(profile: dict[str, Any]) -> None:
    schedule = profile["schedule"]
    phases = schedule["qualification_phases"]
    if schedule["planned_total_optimizer_steps"] != 500:
        raise ValueError(
            "DPCA qualification profiles must retain the released 500-step horizon"
        )
    if schedule["warmup_optimizer_steps"] != 10:
        raise ValueError(
            "DPCA qualification profiles must use the released 10-step warmup"
        )
    if phases["warmup"] != {"start_exclusive": 0, "stop_inclusive": 10}:
        raise ValueError("Warmup phase must cover optimizer steps 1..10")
    if phases["post_warmup"] != {"start_exclusive": 10, "stop_inclusive": 20}:
        raise ValueError("Post-warmup phase must cover optimizer steps 11..20")
    if sorted(int(value) for value in profile["topologies"]) != list(GPU_COUNTS):
        raise ValueError("Every profile must define 1, 2, and 4 GPU topologies")
    science = profile["scientific"]
    required_science = {
        "learning_rate": 1e-6,
        "precision": "bfloat16",
        "samples_per_prompt": 1,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "seed": 44,
        "ppo_clip_ratio_low": 0.2,
        "ppo_clip_ratio_high": 0.28,
        "dual_clip_ratio": 10.0,
        "opd_advantage_clamp": 10.0,
    }
    for key, expected in required_science.items():
        if science.get(key) != expected:
            raise ValueError(
                f"DPCA profile {key} must be {expected!r}; got {science.get(key)!r}"
            )


def _checkpoint_path(checkpoint_root: Path, step: int) -> Path:
    return checkpoint_root / f"global_step_{step}"


def build_launch_contract(
    profile: dict[str, Any],
    phase: str,
    gpu_count: int,
    run_id: str,
    base_env: dict[str, str],
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"Unknown phase: {phase}")
    if gpu_count not in GPU_COUNTS:
        raise ValueError(f"GPU count must be one of {GPU_COUNTS}; got {gpu_count}")

    schedule = profile["schedule"]
    phase_spec = schedule["qualification_phases"][phase]
    topology = profile["topologies"][str(gpu_count)]
    science = profile["scientific"]
    checkpoint_root = Path(base_env.get("CKPTS_DIR", "/checkpoints/dpca"))
    start_step = int(phase_spec["start_exclusive"])
    stop_step = int(phase_spec["stop_inclusive"])

    env = {
        "PROJECT_NAME": "DPCA_BREAKING_TOKEN_BARRIER",
        "EXP_NAME": run_id,
        "NNODES": str(topology["nnodes"]),
        "NGPUS_PER_NODE": str(topology["gpus_per_node"]),
        "FSDP_SIZE": str(topology["fsdp_size"]),
        "GEN_TP": str(topology["rollout_tp"]),
        "SP_SIZE": str(topology["sequence_parallel_size"]),
        "TRAIN_PROMPT_BSZ": str(science["global_batch_size"]),
        "TRAIN_PROMPT_MINI_BSZ": str(science["global_batch_size"]),
        "MAX_PROMPT_LENGTH": str(science["prompt_max_tokens"]),
        "MAX_RESPONSE_LENGTH": str(science["response_max_tokens"]),
        "ROLLOUT_TEMPERATURE": str(science["temperature"]),
        "ROLLOUT_TOP_P": str(science["top_p"]),
        "ROLLOUT_TOP_K": str(science["top_k"]),
        "SAMPLES_PER_PROMPT": str(science["samples_per_prompt"]),
        "ACTOR_LR": str(science["learning_rate"]),
        "WEIGHT_DECAY": str(science["weight_decay"]),
        "GRADIENT_CLIP": str(science["gradient_clip"]),
        "DATA_SEED": str(science["seed"]),
        "PPO_CLIP_RATIO_LOW": str(science["ppo_clip_ratio_low"]),
        "PPO_CLIP_RATIO_HIGH": str(science["ppo_clip_ratio_high"]),
        "DUAL_CLIP_RATIO": str(science["dual_clip_ratio"]),
        "OPD_ADVANTAGE_CLAMP": str(science["opd_advantage_clamp"]),
        "TOTAL_TRAINING_STEPS": str(schedule["planned_total_optimizer_steps"]),
        "STOP_AFTER_GLOBAL_STEP": str(stop_step),
        "LR_WARMUP_STEPS": str(schedule["warmup_optimizer_steps"]),
        "SAVE_FREQ": "10",
        "PORTABLE_CHECKPOINT": "True",
        "PORTABLE_SEED": str(science["seed"]),
        "WANDB_RUN_ID": run_id,
        "WANDB_RESUME": "allow" if phase == "warmup" else "must",
        "DPCA_TRACK_ID": profile["track_id"],
        "DPCA_PROFILE_SHA256": canonical_sha256(profile),
    }
    for key in REQUIRED_EXEC_ENV:
        if key in base_env:
            env[key] = base_env[key]

    teacher_revision = profile["models"]["teacher"].get("revision") or base_env.get(
        "TEACHER_REVISION"
    )
    student_revision = profile["models"]["student"].get("revision") or base_env.get(
        "STUDENT_REVISION"
    )
    if teacher_revision:
        env["TEACHER_REVISION"] = teacher_revision
    if student_revision:
        env["STUDENT_REVISION"] = student_revision

    resume_path = None
    if phase == "warmup":
        env["RESUME_MODE"] = "disable"
    else:
        resume_path = _checkpoint_path(checkpoint_root, start_step)
        env["RESUME_MODE"] = "resume_path"
        env["RESUME_FROM_PATH"] = str(resume_path)

    return {
        "schema_version": 1,
        "track_id": profile["track_id"],
        "evidence_class": profile["evidence_class"],
        "profile_sha256": canonical_sha256(profile),
        "phase": phase,
        "start_step_exclusive": start_step,
        "stop_step_inclusive": stop_step,
        "planned_total_optimizer_steps": schedule["planned_total_optimizer_steps"],
        "warmup_optimizer_steps": schedule["warmup_optimizer_steps"],
        "gpu_count": gpu_count,
        "topology": topology,
        "resume_path": str(resume_path) if resume_path else None,
        "expected_checkpoint": str(_checkpoint_path(checkpoint_root, stop_step)),
        "environment": env,
    }


def validate_resume_checkpoint(contract: dict[str, Any]) -> dict[str, Any]:
    resume_path_value = contract["resume_path"]
    if resume_path_value is None:
        return {"status": "not-applicable"}
    resume_path = Path(resume_path_value)
    actor_path = resume_path / "actor"
    manifest_path = actor_path / "portable_manifest.json"
    metadata_path = actor_path / "portable_dcp" / ".metadata"
    root_contract_path = resume_path / "resume_contract.json"
    missing = [
        str(path)
        for path in (manifest_path, metadata_path, root_contract_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Portable resume checkpoint is incomplete: " + ", ".join(missing)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root_contract = json.loads(root_contract_path.read_text(encoding="utf-8"))
    if root_contract["profile_sha256"] != contract["profile_sha256"]:
        raise ValueError(
            "Checkpoint profile hash differs from the requested scientific profile"
        )
    if int(root_contract["global_step"]) != int(contract["start_step_exclusive"]):
        raise ValueError(
            "Checkpoint global step differs from the requested phase boundary"
        )
    if manifest["format"] != "torch-distributed-checkpoint-v1":
        raise ValueError(
            f"Unsupported portable checkpoint format: {manifest['format']}"
        )
    return {
        "status": "passed",
        "saved_world_size": int(manifest["saved_world_size"]),
        "requested_world_size": int(contract["gpu_count"]),
        "topology_change": int(manifest["saved_world_size"])
        != int(contract["gpu_count"]),
        "rng_resume_policy": manifest["rng_policy_on_topology_change"],
    }


def _write_resume_contract(contract: dict[str, Any]) -> Path:
    checkpoint_path = Path(contract["expected_checkpoint"])
    actor_manifest = checkpoint_path / "actor" / "portable_manifest.json"
    dcp_metadata = checkpoint_path / "actor" / "portable_dcp" / ".metadata"
    if not actor_manifest.is_file() or not dcp_metadata.is_file():
        raise FileNotFoundError(
            f"Expected portable actor checkpoint was not completed at {checkpoint_path}"
        )
    payload = {
        "schema_version": 1,
        "track_id": contract["track_id"],
        "profile_sha256": contract["profile_sha256"],
        "global_step": contract["stop_step_inclusive"],
        "saved_world_size": contract["gpu_count"],
        "planned_total_optimizer_steps": contract["planned_total_optimizer_steps"],
        "warmup_optimizer_steps": contract["warmup_optimizer_steps"],
    }
    path = checkpoint_path / "resume_contract.json"
    pending = path.with_suffix(".json.pending")
    pending.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pending.replace(path)
    return path


def execute(contract: dict[str, Any]) -> int:
    if platform.system() == "Windows":
        raise RuntimeError(
            "Execute this launcher inside the Linux training container or WSL, not Windows PowerShell"
        )
    missing_env = [
        key for key in REQUIRED_EXEC_ENV if not contract["environment"].get(key)
    ]
    missing_env.extend(
        key
        for key in ("TEACHER_REVISION", "STUDENT_REVISION")
        if not contract["environment"].get(key)
    )
    if missing_env:
        raise ValueError(
            "Missing required execution environment: " + ", ".join(missing_env)
        )
    resume_audit = validate_resume_checkpoint(contract)
    print(
        "DPCA_RESUME_AUDIT_JSON=" + json.dumps(resume_audit, sort_keys=True), flush=True
    )

    run_root = ROOT / "reproduction" / "runs" / contract["environment"]["EXP_NAME"]
    run_root.mkdir(parents=True, exist_ok=True)
    phase = contract["phase"]
    plan_path = run_root / f"{phase}-launch-contract.json"
    plan_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log_path = run_root / f"{phase}.log"

    environment = os.environ.copy()
    environment.update(contract["environment"])
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            ["bash", str(ROOT / "cross_distill.sh")],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_stream.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"DPCA phase {phase} failed with exit code {return_code}; log={log_path}"
        )
    resume_contract_path = _write_resume_contract(contract)
    print(
        "DPCA_PHASE_SUMMARY_JSON="
        + json.dumps(
            {
                "status": "completed",
                "phase": phase,
                "checkpoint": contract["expected_checkpoint"],
                "resume_contract": str(resume_contract_path),
                "log": str(log_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--gpus", type=int, choices=GPU_COUNTS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    contract = build_launch_contract(
        profile, args.phase, args.gpus, args.run_id, dict(os.environ)
    )
    print(
        "DPCA_LAUNCH_CONTRACT_JSON=" + json.dumps(contract, sort_keys=True), flush=True
    )
    if args.execute:
        raise SystemExit(execute(contract))


if __name__ == "__main__":
    main()
