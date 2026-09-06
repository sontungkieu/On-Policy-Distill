#!/usr/bin/env python3
"""Fail-closed smoke checks for the pinned DPCA implementation.

This is an implementation/configuration qualification. It intentionally loads no
model or dataset and performs no optimizer update.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
import sys
import types
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.platform == "win32":
    resource = types.ModuleType("resource")
    resource.RLIMIT_MEMLOCK = 0
    resource.RLIM_INFINITY = -1
    resource.getrlimit = lambda _kind: (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    resource.setrlimit = lambda _kind, _limits: None
    sys.modules.setdefault("resource", resource)
MANIFEST_PATH = ROOT / "reproduction/source_manifest.json"
CONFIG_PATH = ROOT / "reproduction/config_audit.json"


class PieceTokenizer:
    """Small deterministic tokenizer fixture with exact decoded pieces."""

    def __init__(self, pieces: dict[int, str]) -> None:
        self.pieces = pieces

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(self.pieces[token_id] for token_id in token_ids)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def extract_shell_number(text: str, name: str) -> float:
    match = re.search(rf"(?m)^{re.escape(name)}=([^#\n]+)", text)
    require(match is not None, f"missing launcher variable: {name}")
    raw = match.group(1).strip()
    defaulted = re.fullmatch(r"\$\{[A-Z0-9_]+:-(.*)\}", raw)
    if defaulted is not None:
        raw = defaulted.group(1)
    arithmetic = re.fullmatch(r"\$\(\(([-+*/%() 0-9]+)\)\)", raw)
    if arithmetic is not None:
        expression = arithmetic.group(1)
        require(
            re.fullmatch(r"[-+*/%() 0-9]+", expression) is not None,
            f"unsafe arithmetic: {name}",
        )
        return float(eval(expression, {"__builtins__": {}}, {}))
    return float(raw)


def check_source_and_config() -> dict[str, object]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    actual_hashes: dict[str, str] = {}
    expected_hashes = manifest.get("effective_sha256", manifest["sha256"])
    for relative, expected in expected_hashes.items():
        actual = sha256(ROOT / relative)
        require(actual == expected, f"source hash mismatch: {relative}")
        actual_hashes[relative] = actual

    launcher = (ROOT / "cross_distill.sh").read_text(encoding="utf-8")
    paper = config["paper_reported"]
    launcher_values = {
        "global_batch_size_prompts": extract_shell_number(launcher, "train_prompt_bsz"),
        "max_prompt_tokens": extract_shell_number(launcher, "max_prompt_length"),
        "max_response_tokens": extract_shell_number(launcher, "max_response_length"),
        "temperature": extract_shell_number(launcher, "temperature"),
        "top_p": extract_shell_number(launcher, "top_p"),
        "top_k": extract_shell_number(launcher, "top_k"),
        "learning_rate": extract_shell_number(launcher, "actor_lr"),
        "gradient_clip": extract_shell_number(launcher, "gradient_clip"),
    }
    require(
        launcher_values["global_batch_size_prompts"]
        == paper["global_batch_size_prompts"],
        "batch mismatch",
    )
    require(
        launcher_values["max_prompt_tokens"] == paper["max_prompt_tokens"],
        "prompt length mismatch",
    )
    require(
        launcher_values["max_response_tokens"] == paper["max_response_tokens"],
        "response length mismatch",
    )
    require(
        launcher_values["temperature"] == paper["rollout"]["temperature"],
        "temperature mismatch",
    )
    require(launcher_values["top_p"] == paper["rollout"]["top_p"], "top-p mismatch")
    require(launcher_values["top_k"] == paper["rollout"]["top_k"], "top-k mismatch")
    require(
        launcher_values["learning_rate"] == paper["learning_rate"],
        "learning-rate mismatch",
    )
    require(
        launcher_values["gradient_clip"] == paper["gradient_clip"],
        "gradient-clip mismatch",
    )

    required_fragments = (
        "algorithm.adv_estimator=${adv_estimator}",
        'actor_rollout_ref.actor.policy_loss.loss_mode="opd"',
        "actor_rollout_ref.actor.optim.lr=${actor_lr}",
        "actor_rollout_ref.actor.grad_clip=${gradient_clip}",
        "reward_model.reward_manager=opd",
    )
    for fragment in required_fragments:
        require(fragment in launcher, f"missing launcher contract: {fragment}")

    return {
        "source_commit": manifest["commit"],
        "source_hashes": actual_hashes,
        "upstream_source_hashes": manifest["sha256"],
        "paper_launcher_contract": "pass",
    }


def check_dpca_alignment() -> dict[str, object]:
    from verl.workers.reward_manager.opd import TeacherClient

    student_tokenizer = PieceTokenizer({100: "120"})
    teacher_tokenizer = PieceTokenizer({1: "1", 2: "2", 3: "0"})
    teacher_logps = torch.tensor([-1.81, -1.26, -4.04], dtype=torch.float64)
    aligned, chunk_ids, details, teacher_truncated = TeacherClient._align_chunks(
        [100],
        [1, 2, 3],
        teacher_logps,
        student_tokenizer,
        teacher_tokenizer,
        debug=True,
    )
    require(not teacher_truncated, "teacher unexpectedly marked truncated")
    require(chunk_ids.tolist() == [0.0], "DPCA failed to emit one minimal chunk")
    require(
        len(details) == 1 and details[0]["n_stu"] == 1 and details[0]["n_tec"] == 3,
        "wrong DPCA chunk",
    )
    require(
        torch.allclose(aligned.sum(), teacher_logps.sum()),
        "teacher log-mass was not conserved",
    )

    student_tokenizer_2 = PieceTokenizer({10: "...", 11: "]"})
    teacher_tokenizer_2 = PieceTokenizer({12: "...]"})
    teacher_logps_2 = torch.tensor([-15.91], dtype=torch.float64)
    aligned_2, chunk_ids_2, details_2, truncated_2 = TeacherClient._align_chunks(
        [10, 11],
        [12],
        teacher_logps_2,
        student_tokenizer_2,
        teacher_tokenizer_2,
        debug=True,
    )
    require(not truncated_2, "teacher unexpectedly marked truncated in split case")
    require(
        chunk_ids_2.tolist() == [0.0, 0.0], "student split tokens do not share a chunk"
    )
    require(
        len(details_2) == 1
        and details_2[0]["n_stu"] == 2
        and details_2[0]["n_tec"] == 1,
        "wrong split chunk",
    )
    require(
        torch.allclose(aligned_2.sum(), teacher_logps_2.sum()),
        "split case did not conserve log-mass",
    )

    return {
        "merge_case": {"student_tokens": 1, "teacher_tokens": 3, "chunks": 1},
        "split_case": {"student_tokens": 2, "teacher_tokens": 1, "chunks": 1},
        "teacher_log_mass_conservation": "pass",
        "aligned_split_teacher_logps": aligned_2.tolist(),
    }


def check_semantic_prior_loss() -> dict[str, object]:
    from verl.trainer.ppo.core_algos import compute_policy_loss_opd

    old_log_prob = torch.tensor([[-6.93, -7.65]], dtype=torch.float64)
    current_log_prob = old_log_prob.clone()
    response_mask = torch.ones_like(old_log_prob)

    teacher_budget = torch.tensor(-15.91, dtype=torch.float64)
    repeated_average = torch.full_like(old_log_prob, teacher_budget.item() / 2)
    chunk_ids = torch.zeros_like(old_log_prob)
    packed_teacher_and_chunks = torch.cat([repeated_average, chunk_ids], dim=-1)

    config = OmegaConf.create(
        {
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.28,
            "clip_ratio_c": 10.0,
            "policy_loss": {"opd_loss_max_clamp": 10.0},
            "global_batch_info": {},
        }
    )
    loss, metrics = compute_policy_loss_opd(
        old_log_prob=old_log_prob,
        log_prob=current_log_prob,
        advantages=packed_teacher_and_chunks,
        response_mask=response_mask,
        config=config,
    )

    ratio = teacher_budget / old_log_prob.sum()
    projected = ratio * old_log_prob
    expected_advantage = projected - old_log_prob
    expected_loss = -expected_advantage.mean()
    require(
        torch.allclose(loss, expected_loss, atol=1e-12, rtol=1e-12),
        "semantic-prior loss mismatch",
    )
    require(
        math.isclose(metrics["actor/opd_inf_ratio"], 0.0),
        "unexpected invalid-token ratio",
    )
    required_metrics = {
        "actor/opd_valid_tokens",
        "actor/opd_inf_tokens",
        "actor/opd_inf_ratio",
        "actor/opd_chunk_count",
        "actor/opd_teacher_truncated_sequences",
        "actor/opd_teacher_truncated_ratio",
    }
    require(
        required_metrics.issubset(metrics),
        "DPCA diagnostic metric schema is incomplete",
    )

    return {
        "projected_teacher_logps": projected.flatten().tolist(),
        "advantages": expected_advantage.flatten().tolist(),
        "loss": loss.item(),
        "expected_loss": expected_loss.item(),
        "diagnostic_metrics": {
            name: metrics[name] for name in sorted(required_metrics)
        },
    }


def check_cuda() -> dict[str, object]:
    result: dict[str, object] = {
        "available": torch.cuda.is_available(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    if not torch.cuda.is_available():
        result["status"] = "not_run"
        return result

    device = torch.device("cuda")
    left = torch.randn((512, 512), device=device, dtype=torch.bfloat16)
    right = torch.randn((512, 512), device=device, dtype=torch.bfloat16)
    product = left @ right
    torch.cuda.synchronize()
    require(
        bool(torch.isfinite(product).all()),
        "CUDA bf16 matmul produced non-finite values",
    )
    properties = torch.cuda.get_device_properties(0)
    result.update(
        {
            "status": "pass",
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
            "bf16_matmul": "finite",
        }
    )
    return result


def main() -> None:
    result = {
        "status": "pass",
        "scope": "implementation_config_qualification",
        "scientific_reproduction": False,
        "source": check_source_and_config(),
        "dpca": check_dpca_alignment(),
        "semantic_prior": check_semantic_prior_loss(),
        "cuda": check_cuda(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("ray", "tensordict", "transformers", "pyzmq", "omegaconf")
        },
    }
    print("DPCA_SMOKE_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
