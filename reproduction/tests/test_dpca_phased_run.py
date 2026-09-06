from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "reproduction" / "dpca_phased_run.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("dpca_phased_run", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DPCAPhasedRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = load_runner()

    def test_both_profiles_share_released_schedule(self) -> None:
        for name in ("paper_8b", "common_4b1b"):
            profile = self.runner.load_profile(name)
            self.assertEqual(profile["schedule"]["planned_total_optimizer_steps"], 500)
            self.assertEqual(profile["schedule"]["warmup_optimizer_steps"], 10)

    def test_warmup_and_resume_keep_500_step_scheduler(self) -> None:
        profile = self.runner.load_profile("common_4b1b")
        with tempfile.TemporaryDirectory() as temp:
            env = {"CKPTS_DIR": temp}
            warmup = self.runner.build_launch_contract(
                profile, "warmup", 1, "run-a", env
            )
            resumed = self.runner.build_launch_contract(
                profile, "post_warmup", 4, "run-a", env
            )
        self.assertEqual(warmup["environment"]["TOTAL_TRAINING_STEPS"], "500")
        self.assertEqual(warmup["environment"]["STOP_AFTER_GLOBAL_STEP"], "10")
        self.assertEqual(warmup["environment"]["RESUME_MODE"], "disable")
        self.assertEqual(resumed["environment"]["TOTAL_TRAINING_STEPS"], "500")
        self.assertEqual(resumed["environment"]["STOP_AFTER_GLOBAL_STEP"], "20")
        self.assertEqual(resumed["environment"]["RESUME_MODE"], "resume_path")
        self.assertTrue(resumed["resume_path"].endswith("global_step_10"))
        self.assertEqual(resumed["environment"]["PORTABLE_CHECKPOINT"], "True")

    def test_all_supported_topologies_preserve_scientific_batch(self) -> None:
        expected = {
            "paper_8b": {"batch": "128", "prompt": "1024", "response": "16384"},
            "common_4b1b": {"batch": "64", "prompt": "1024", "response": "2048"},
        }
        for profile_name, values in expected.items():
            profile = self.runner.load_profile(profile_name)
            for gpu_count in (1, 2, 4):
                contract = self.runner.build_launch_contract(
                    profile, "warmup", gpu_count, f"{profile_name}-{gpu_count}", {}
                )
                self.assertEqual(
                    contract["environment"]["TRAIN_PROMPT_BSZ"], values["batch"]
                )
                self.assertEqual(
                    contract["environment"]["MAX_PROMPT_LENGTH"], values["prompt"]
                )
                self.assertEqual(
                    contract["environment"]["MAX_RESPONSE_LENGTH"], values["response"]
                )
                self.assertEqual(
                    contract["environment"]["NGPUS_PER_NODE"], str(gpu_count)
                )
                self.assertEqual(contract["environment"]["FSDP_SIZE"], str(gpu_count))

    def test_common_profile_carries_pinned_revisions(self) -> None:
        profile = self.runner.load_profile("common_4b1b")
        contract = self.runner.build_launch_contract(profile, "warmup", 2, "run-c", {})
        self.assertEqual(
            contract["environment"]["TEACHER_REVISION"],
            "1cfa9a7208912126459214e8b04321603b3df60c",
        )
        self.assertEqual(
            contract["environment"]["STUDENT_REVISION"],
            "4e20de362430cd3b72f300e6b0f18e50e7166e08",
        )

    def test_dual_logging_and_dpca_diagnostic_schema_are_enabled(self) -> None:
        launcher = (ROOT / "cross_distill.sh").read_text(encoding="utf-8")
        loss_source = (ROOT / "verl" / "trainer" / "ppo" / "core_algos.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('trainer.logger=\'["console","wandb","tensorboard"]\'', launcher)
        self.assertIn(
            'export TENSORBOARD_DIR=${TENSORBOARD_DIR:-"${CKPTS_DIR}/tensorboard"}',
            launcher,
        )
        for metric in (
            "actor/opd_valid_tokens",
            "actor/opd_inf_tokens",
            "actor/opd_inf_ratio",
            "actor/opd_chunk_count",
            "actor/opd_teacher_truncated_sequences",
            "actor/opd_teacher_truncated_ratio",
        ):
            self.assertIn(metric, loss_source)

    def test_resume_preflight_accepts_world_size_change_only_with_dcp(self) -> None:
        profile = self.runner.load_profile("common_4b1b")
        with tempfile.TemporaryDirectory() as temp:
            checkpoint_root = Path(temp)
            warmup = self.runner.build_launch_contract(
                profile, "warmup", 1, "run-a", {"CKPTS_DIR": temp}
            )
            checkpoint = Path(warmup["expected_checkpoint"])
            actor = checkpoint / "actor"
            (actor / "portable_dcp").mkdir(parents=True)
            (actor / "portable_dcp" / ".metadata").write_bytes(b"test")
            (actor / "portable_manifest.json").write_text(
                json.dumps(
                    {
                        "format": "torch-distributed-checkpoint-v1",
                        "saved_world_size": 1,
                        "rng_policy_on_topology_change": "deterministic-reseed",
                    }
                ),
                encoding="utf-8",
            )
            (checkpoint / "resume_contract.json").write_text(
                json.dumps(
                    {
                        "profile_sha256": warmup["profile_sha256"],
                        "global_step": 10,
                    }
                ),
                encoding="utf-8",
            )
            resumed = self.runner.build_launch_contract(
                profile, "post_warmup", 4, "run-a", {"CKPTS_DIR": str(checkpoint_root)}
            )
            audit = self.runner.validate_resume_checkpoint(resumed)
        self.assertEqual(audit["status"], "passed")
        self.assertTrue(audit["topology_change"])
        self.assertEqual(audit["saved_world_size"], 1)
        self.assertEqual(audit["requested_world_size"], 4)

    def test_resume_preflight_fails_without_portable_payload(self) -> None:
        profile = self.runner.load_profile("paper_8b")
        with tempfile.TemporaryDirectory() as temp:
            contract = self.runner.build_launch_contract(
                profile, "post_warmup", 2, "run-b", {"CKPTS_DIR": temp}
            )
            with self.assertRaises(FileNotFoundError):
                self.runner.validate_resume_checkpoint(contract)


if __name__ == "__main__":
    unittest.main()
