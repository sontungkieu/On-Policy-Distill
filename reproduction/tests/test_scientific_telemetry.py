from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

SCIENTIFIC_ROOT = Path(__file__).resolve().parents[1] / "scientific"
sys.path.insert(0, str(SCIENTIFIC_ROOT))

import sft_telemetry
from sft_telemetry import (
    SFTTelemetryCallback,
    StepAccumulator,
    aggregate_rank_snapshots,
    normalize_trainer_logs,
)


class ScientificTelemetryTest(unittest.TestCase):
    def test_trainer_logs_use_shared_namespace_and_reject_nonfinite_values(self):
        metrics = normalize_trainer_logs(
            {
                "loss": 1.25,
                "grad_norm": float("inf"),
                "learning_rate": 1e-4,
                "ignored": "text",
                "already/namespaced": 3,
            }
        )
        self.assertEqual(metrics["train/loss"], 1.25)
        self.assertEqual(metrics["train/learning_rate"], 1e-4)
        self.assertEqual(metrics["already/namespaced"], 3.0)
        self.assertNotIn("train/grad_norm", metrics)

    def test_consumed_batch_statistics_use_labels_and_attention_mask(self):
        accumulator = StepAccumulator(cutoff_len=8)
        accumulator.set_eos_token_ids(9)
        accumulator.record(
            {
                "input_ids": torch.tensor([[1, 2, 3, 4, 9, 0, 0, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]),
                "labels": torch.tensor([[-100, -100, 3, 4, 9, -100, -100, -100]]),
            },
            training_step_seconds=2.5,
        )
        snapshot = accumulator.snapshot_local()
        self.assertEqual(snapshot["samples"], 1)
        self.assertEqual(snapshot["active_tokens"], 5)
        self.assertEqual(snapshot["supervised_tokens"], 3)
        self.assertEqual(snapshot["eos_targets"], 1)
        self.assertEqual(snapshot["sequences_with_eos"], 1)
        self.assertEqual(snapshot["sequence_lengths"], [5])
        self.assertEqual(snapshot["training_step_seconds"], 2.5)

    def test_rank_aggregation_produces_exact_length_quantiles_and_ratios(self):
        snapshots = [
            {
                "rank": 0,
                "cutoff_len": 8,
                "microbatches": 2,
                "samples": 2,
                "active_tokens": 12,
                "supervised_tokens": 8,
                "padded_positions": 16,
                "eos_targets": 2,
                "sequences_with_eos": 2,
                "sequence_lengths": [4, 8],
                "training_step_seconds": 3.0,
                "cuda": {"allocated_bytes": 10, "peak_allocated_bytes": 20},
            },
            {
                "rank": 1,
                "cutoff_len": 8,
                "microbatches": 2,
                "samples": 2,
                "active_tokens": 14,
                "supervised_tokens": 10,
                "padded_positions": 16,
                "eos_targets": 1,
                "sequences_with_eos": 1,
                "sequence_lengths": [6, 8],
                "training_step_seconds": 4.0,
                "cuda": {"allocated_bytes": 12, "peak_allocated_bytes": 24},
            },
        ]
        metrics = aggregate_rank_snapshots(snapshots)
        self.assertEqual(metrics["data/samples_global"], 4)
        self.assertEqual(metrics["data/active_tokens_global"], 26)
        self.assertEqual(metrics["data/supervised_tokens_global"], 18)
        self.assertEqual(metrics["data/sequences_at_cutoff_ratio"], 0.5)
        self.assertEqual(metrics["data/sequence_length_p50"], 7.0)
        self.assertEqual(metrics["data/sequence_length_p90"], 8.0)
        self.assertEqual(metrics["system/cuda_global/allocated_bytes_max"], 12)
        self.assertTrue(math.isclose(metrics["timing/rank_straggler_ratio"], 4 / 3))

    def test_callback_writes_unified_offline_wandb_tensorboard_and_jsonl(self):
        class FakeRun:
            def __init__(self):
                self.records = []
                self.summary = {}
                self.artifacts = []
                self.exit_code = None

            def log(self, record):
                self.records.append(record)

            def log_artifact(self, artifact):
                self.artifacts.append(artifact)

            def finish(self, exit_code=0):
                self.exit_code = exit_code

        class FakeArtifact:
            def __init__(self, name, type, metadata):
                self.name = name
                self.type = type
                self.metadata = metadata
                self.files = []

            def add_file(self, path):
                self.files.append(path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            effective = root / "effective.json"
            lf_config = root / "llamafactory.yaml"
            asset = root / "asset-manifest.json"
            source = root / "source-manifest.json"
            for path in (effective, lf_config, asset, source):
                path.write_text("{}\n", encoding="utf-8")

            accumulator = StepAccumulator(cutoff_len=8)
            callback = SFTTelemetryCallback(
                accumulator=accumulator,
                run_dir=root,
                run_id="offline-telemetry-test",
                scientific_config={"schema_version": 1},
                effective_config_path=effective,
                llamafactory_config_path=lf_config,
                asset_manifest_path=asset,
                source_manifest_path=source,
            )
            args = SimpleNamespace(max_grad_norm=1.0, gradient_accumulation_steps=1)
            state = SimpleNamespace(global_step=0)
            control = SimpleNamespace()
            model = SimpleNamespace(
                config=SimpleNamespace(
                    _attn_implementation="sdpa",
                    bos_token_id=1,
                    eos_token_id=9,
                    pad_token_id=0,
                )
            )
            environment = {
                "WANDB_ENTITY": "test-entity",
                "WANDB_PROJECT": "test-project",
                "WANDB_JOB_TYPE": "test",
                "WANDB_TAGS": "schema:experiment-v1,objective:causal-cross-entropy",
            }
            fake_run = FakeRun()
            with (
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(sft_telemetry.wandb, "init", return_value=fake_run),
                mock.patch.object(sft_telemetry.wandb, "define_metric"),
                mock.patch.object(sft_telemetry.wandb, "Artifact", FakeArtifact),
            ):
                callback.on_train_begin(args, state, control, model=model)
                callback.on_step_begin(args, state, control)
                accumulator.record(
                    {
                        "input_ids": torch.tensor([[1, 2, 9, 0]]),
                        "attention_mask": torch.tensor([[1, 1, 1, 0]]),
                        "labels": torch.tensor([[-100, 2, 9, -100]]),
                    },
                    training_step_seconds=0.1,
                )
                state.global_step = 1
                callback.on_step_end(args, state, control)
                callback.on_log(
                    args,
                    state,
                    control,
                    logs={"loss": 1.5, "grad_norm": 2.0, "learning_rate": 1e-4},
                )
                callback.on_train_end(args, state, control)
                callback.on_log(args, state, control, logs={"train_runtime": 1.0})
                callback.finalize(exit_code=0)

            rows = [
                json.loads(line)
                for line in (root / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len([row for row in rows if "train/loss" in row]), 1)
            self.assertEqual(rows[0]["data/active_tokens_global"], 3)
            self.assertEqual(rows[0]["optimizer/grad_clip_triggered"], 1)
            summary = json.loads(
                (root / "telemetry-summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["optimizer_metric_records"], 1)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(fake_run.exit_code, 0)
            self.assertEqual(len(fake_run.artifacts), 1)
            self.assertTrue((root / "runtime-environment.json").is_file())
            self.assertTrue(
                list((root / "tensorboard-telemetry").glob("events.out.tfevents.*"))
            )


if __name__ == "__main__":
    unittest.main()
