from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "reproduction/docker_build_contract.py"


def load_contract():
    spec = importlib.util.spec_from_file_location("docker_build_contract", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DockerBuildContractTest(unittest.TestCase):
    def test_reference_is_tagged_docker_hub_only(self) -> None:
        contract = load_contract()
        self.assertEqual(
            contract.normalize_image_ref(
                "codemaivanngu/dpca-breaking-token-barrier:test"
            ),
            "docker.io/codemaivanngu/dpca-breaking-token-barrier:test",
        )
        for invalid in (
            "image:latest",
            "ghcr.io/user/image:tag",
            "user/image",
            "user/image:bad;tag",
        ):
            with self.assertRaises(ValueError):
                contract.normalize_image_ref(invalid)

    def test_context_excludes_credentials_and_weight_artifacts(self) -> None:
        contract = load_contract()
        excluded = (
            ROOT / "checkpoint-final/model.safetensors",
            ROOT / "wandb/run.json",
            ROOT / ".env.private",
            ROOT / "dockerhub.key",
            ROOT / "dockerhub.txt",
            ROOT / ".netrc",
            ROOT / "reproduction/bootstrap_modal_dockerhub_secret.py",
        )
        self.assertTrue(all(contract.ignore_local_path(path) for path in excluded))
        self.assertFalse(contract.ignore_local_path(ROOT / "verl/models/qwen2.py"))
        audit = contract.audit_local_context()
        self.assertEqual(audit["large_files"], 0)
        self.assertRegex(audit["context_sha256"], r"^[0-9a-f]{64}$")

    def test_dockerfile_pins_uv_source_and_two_gpu_entrypoint(self) -> None:
        dockerfile = (ROOT / "reproduction/Dockerfile.dpca-2gpu").read_text()
        self.assertIn("pip install --no-cache-dir uv==0.10.2", dockerfile)
        self.assertIn('test "$(uv --version)" = "uv 0.10.2"', dockerfile)
        self.assertIn("uv export", dockerfile)
        self.assertIn("--locked", dockerfile)
        self.assertIn("/opt/models/teacher", dockerfile)
        self.assertIn("/opt/models/student", dockerfile)
        self.assertIn("dpca_2gpu_train.py", dockerfile)


if __name__ == "__main__":
    unittest.main()
