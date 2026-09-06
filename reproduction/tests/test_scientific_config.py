import json
import unittest
from pathlib import Path

REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPRODUCTION_ROOT / "scientific" / "dpca_8b_scientific.json"
FLASH_CONFIG_PATH = (
    REPRODUCTION_ROOT / "scientific" / "dpca_8b_scientific_sdpa_flash.json"
)
FA2_CONFIG_PATH = REPRODUCTION_ROOT / "scientific" / "dpca_8b_scientific_fa2.json"
DATASET_INFO_PATH = REPRODUCTION_ROOT / "scientific" / "data" / "dataset_info.json"


def _walk_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_keys(nested)


class ScientificConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_single_effective_config_has_no_paper_actual_shadow_keys(self):
        keys = list(_walk_keys(self.config))
        self.assertFalse([key for key in keys if key.startswith(("paper_", "actual_"))])

    def test_sft_contract(self):
        sft = self.config["sft"]
        self.assertEqual(sft["optimizer_steps"], 50)
        self.assertEqual(sft["save_step"], 50)
        self.assertTrue(
            self.config["opd"]["student_checkpoint"].endswith("/checkpoint-50")
        )
        self.assertEqual(sft["maximum_sequence_length"], 16_384)
        self.assertEqual(sft["global_batch_size"], 128)
        self.assertEqual(
            sft["gpu_count"]
            * sft["per_device_micro_batch_size"]
            * sft["gradient_accumulation_steps"],
            sft["global_batch_size"],
        )
        self.assertEqual(sft["dataset"]["subset_rows"], 400_000)
        self.assertEqual(sft["distributed"]["strategy"], "fsdp_full_shard")
        self.assertEqual(
            sft["distributed"]["transformer_layer_cls_to_wrap"],
            "LlamaDecoderLayer",
        )
        self.assertFalse(sft["distributed"]["activation_checkpointing"])
        self.assertTrue(sft["gradient_checkpointing"])
        self.assertFalse(sft["disable_gradient_checkpointing"])
        self.assertTrue(sft["use_reentrant_gc"])
        self.assertEqual(sft["attention_backend"], "sdpa")
        self.assertEqual(sft["attention_kernel_policy"], "auto")
        self.assertTrue(sft["tf32"])
        self.assertEqual(sft["pytorch_alloc_conf"], "expandable_segments:True")
        self.assertEqual(sft["torch_empty_cache_steps"], 1)
        self.assertTrue(sft["tokenized_path"].startswith("/workspace/cache/"))

    def test_opd_contract(self):
        opd = self.config["opd"]
        self.assertEqual(opd["optimizer_steps"], 100)
        self.assertEqual(opd["global_batch_size"], 128)
        self.assertEqual(opd["prompt_max_tokens"], 1_024)
        self.assertEqual(opd["response_max_tokens"], 16_384)
        self.assertEqual(opd["temperature"], 1.0)
        self.assertEqual(opd["top_p"], 1.0)
        self.assertEqual(opd["top_k"], -1)

    def test_dataset_info_matches_local_pinned_shards(self):
        dataset_info = json.loads(DATASET_INFO_PATH.read_text(encoding="utf-8"))
        entry = dataset_info["openthoughts3_400k_pinned"]
        self.assertEqual(entry["file_name"], "openthoughts3-1.2m/data")
        self.assertEqual(entry["formatting"], "sharegpt")

    def test_tracking_targets_the_shared_wandb_project(self):
        tracking = self.config["tracking"]
        self.assertEqual(
            tracking["wandb_entity"],
            "kieusontung8-hanoi-university-of-science-and-technology",
        )
        self.assertEqual(
            tracking["wandb_project"],
            "dpca-breaking-token-barrier-reproduction",
        )
        self.assertTrue(tracking["metrics_jsonl"])
        self.assertEqual(
            tracking["telemetry"]["unified_sinks"],
            ["wandb", "tensorboard", "jsonl"],
        )

    def test_flash_variant_changes_only_backend_identity_and_output_lineage(self):
        flash_config = json.loads(FLASH_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertNotEqual(flash_config["run_id"], self.config["run_id"])
        self.assertEqual(flash_config["track_id"], self.config["track_id"])
        self.assertEqual(flash_config["models"], self.config["models"])
        self.assertEqual(flash_config["tracking"], self.config["tracking"])

        flash_sft = flash_config["sft"]
        base_sft = self.config["sft"]
        self.assertEqual(flash_sft["attention_backend"], "sdpa")
        self.assertEqual(flash_sft["attention_kernel_policy"], "flash_only")
        self.assertEqual(flash_sft["tracking_variant"], "sdpa-flash-only")
        for key, value in base_sft.items():
            if key == "attention_kernel_policy":
                continue
            self.assertEqual(flash_sft[key], value)

        base_opd = dict(self.config["opd"])
        flash_opd = dict(flash_config["opd"])
        base_opd.pop("student_checkpoint")
        flash_opd.pop("student_checkpoint")
        self.assertEqual(flash_opd, base_opd)

    def test_external_fa2_variant_changes_only_backend_identity_and_output_lineage(
        self,
    ):
        fa2_config = json.loads(FA2_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertNotEqual(fa2_config["run_id"], self.config["run_id"])
        self.assertEqual(fa2_config["track_id"], self.config["track_id"])
        self.assertEqual(fa2_config["models"], self.config["models"])
        self.assertEqual(fa2_config["tracking"], self.config["tracking"])

        fa2_sft = fa2_config["sft"]
        base_sft = self.config["sft"]
        self.assertEqual(fa2_sft["attention_backend"], "fa2")
        self.assertEqual(fa2_sft["attention_kernel_policy"], "external_fa2")
        self.assertEqual(fa2_sft["tracking_variant"], "flash-attention-2")
        for key, value in base_sft.items():
            if key in {"attention_backend", "attention_kernel_policy"}:
                continue
            self.assertEqual(fa2_sft[key], value)

        base_opd = dict(self.config["opd"])
        fa2_opd = dict(fa2_config["opd"])
        base_opd.pop("student_checkpoint")
        fa2_opd.pop("student_checkpoint")
        self.assertEqual(fa2_opd, base_opd)


if __name__ == "__main__":
    unittest.main()
