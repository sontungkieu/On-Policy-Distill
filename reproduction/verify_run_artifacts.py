"""Read-only verification for downloaded DPCA run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REQUIRED_METRICS = {
    "loss/opd_policy",
    "opd/inf_ratio",
    "opd/chunks",
    "logp/student_mean",
    "logp/teacher_mean",
    "advantage/mean",
    "optimizer/lr_used",
    "optimizer/grad_norm_before_clip",
    "gpu/teacher_allocated_gib",
    "gpu/student_allocated_gib",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["status"] == "completed"
    assert summary["optimizer_steps_completed"] == 10
    assert summary["two_gpu_b200_verified"] is True
    assert [row["optimizer_step"] for row in rows] == list(range(1, 11))
    assert REQUIRED_METRICS <= rows[-1].keys()

    event_files = list((run_dir / "tensorboard").glob("events.out.tfevents.*"))
    assert len(event_files) == 1
    accumulator = EventAccumulator(str(event_files[0]), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags()["scalars"])
    assert REQUIRED_METRICS <= scalar_tags
    for tag in REQUIRED_METRICS:
        assert [event.step for event in accumulator.Scalars(tag)] == list(range(1, 11))

    checkpoint = run_dir / "checkpoint-final"
    model_path = checkpoint / "model.safetensors"
    trainer_state_path = checkpoint / "trainer_state.pt"
    assert model_path.stat().st_size > 600 * 1024 * 1024
    assert trainer_state_path.stat().st_size > 1024 * 1024 * 1024
    with safe_open(model_path, framework="numpy", device="cpu") as handle:
        tensor_count = len(handle.keys())
    assert tensor_count > 0

    result = {
        "status": "passed",
        "optimizer_steps": len(rows),
        "tensorboard_required_tags": len(REQUIRED_METRICS),
        "tensorboard_scalar_tags_total": len(scalar_tags),
        "checkpoint_tensor_count": tensor_count,
        "model_bytes": model_path.stat().st_size,
        "trainer_state_bytes": trainer_state_path.stat().st_size,
    }
    print("DPCA_ARTIFACT_QA_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
