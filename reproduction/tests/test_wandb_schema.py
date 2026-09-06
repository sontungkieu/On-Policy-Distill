from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCTION_ROOT))

from wandb_schema import resolve_wandb_metadata


class WandbSchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        config_path = REPRODUCTION_ROOT / "configs/dpca_2gpu_10step.json"
        self.config = json.loads(config_path.read_text(encoding="utf-8"))

    def test_exact_metadata_contract(self) -> None:
        metadata = resolve_wandb_metadata(self.config)
        self.assertEqual(metadata["job_type"], "implementation-validation")
        self.assertEqual(
            metadata["tags"],
            [
                "schema:experiment-v1",
                "framework:pytorch",
                "method:dpca",
                "regime:on-policy",
                "objective:semantic-prior-clipped-pg",
                "variant:upstream-code",
                "platform:modal",
                "accelerator:b200x2",
                "budget:10-update",
                "stage:implementation-validation",
                "student:smollm2-360m-instruct",
                "teacher:qwen2.5-0.5b-instruct",
            ],
        )

    def test_no_flat_or_duplicate_dimensions(self) -> None:
        tags = resolve_wandb_metadata(self.config)["tags"]
        self.assertTrue(all(":" in tag for tag in tags))
        dimensions = [tag.split(":", 1)[0] for tag in tags]
        self.assertEqual(len(dimensions), len(set(dimensions)))

    def test_invalid_slug_fails_closed(self) -> None:
        self.config["opd"]["variant"] = "Not a slug"
        with self.assertRaises(ValueError):
            resolve_wandb_metadata(self.config)

    def test_paper8b_timing_config_is_pinned_and_labeled(self) -> None:
        config = json.loads(
            (
                REPRODUCTION_ROOT / "configs/dpca_paper8b_2gpu_10step_timing.json"
            ).read_text(encoding="utf-8")
        )
        self.assertRegex(config["teacher"]["revision"], r"^[0-9a-f]{40}$")
        self.assertRegex(config["student"]["revision"], r"^[0-9a-f]{40}$")
        self.assertTrue(config["timing_scope"]["microbenchmark"])
        self.assertTrue(config["timing_scope"]["not_an_efficacy_run"])
        metadata = resolve_wandb_metadata(config)
        self.assertIn("stage:timing-qualification", metadata["tags"])
        self.assertIn("student:meta-llama-3.1-8b-instruct", metadata["tags"])
        self.assertIn("teacher:qwen3-8b", metadata["tags"])

    def test_paper8b_1gpu_timing_matches_2gpu_scientific_inputs(self) -> None:
        one_gpu = json.loads(
            (
                REPRODUCTION_ROOT / "configs/dpca_paper8b_1gpu_10step_timing.json"
            ).read_text(encoding="utf-8")
        )
        two_gpu = json.loads(
            (
                REPRODUCTION_ROOT / "configs/dpca_paper8b_2gpu_10step_timing.json"
            ).read_text(encoding="utf-8")
        )
        for key in (
            "teacher",
            "student",
            "prompts_path",
            "optimizer",
            "rollout",
            "opd",
            "precision",
            "seed",
        ):
            self.assertEqual(one_gpu[key], two_gpu[key])
        self.assertEqual(one_gpu["execution"]["gpu_count"], 1)
        self.assertEqual(one_gpu["execution"]["teacher_device_index"], 0)
        self.assertEqual(one_gpu["execution"]["student_device_index"], 0)
        metadata = resolve_wandb_metadata(one_gpu)
        self.assertIn("accelerator:b200x1", metadata["tags"])


if __name__ == "__main__":
    unittest.main()
