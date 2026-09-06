"""Launch LLaMA-Factory SFT with the DPCA scientific telemetry callback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from sft_telemetry import (
    SFTTelemetryCallback,
    StepAccumulator,
    install_training_step_instrumentation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llamafactory-config", type=Path, required=True)
    parser.add_argument("--scientific-config", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    return parser.parse_args()


def configure_sdpa_kernel_policy(scientific_config: dict) -> None:
    """Apply a fail-closed PyTorch SDPA kernel policy before model creation."""
    policy = scientific_config["sft"].get("attention_kernel_policy", "auto")
    if policy in {"auto", "external_fa2"}:
        if (
            policy == "external_fa2"
            and scientific_config["sft"]["attention_backend"] != "fa2"
        ):
            raise RuntimeError("external_fa2 requires attention_backend=fa2")
        return
    if policy != "flash_only":
        raise RuntimeError(f"Unknown attention_kernel_policy: {policy}")
    if scientific_config["sft"]["attention_backend"] != "sdpa":
        raise RuntimeError("flash_only requires attention_backend=sdpa")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("flash_only requires CUDA")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)


def main() -> None:
    args = parse_args()
    lf_config = yaml.safe_load(args.llamafactory_config.read_text(encoding="utf-8"))
    scientific_config = json.loads(args.scientific_config.read_text(encoding="utf-8"))
    configure_sdpa_kernel_policy(scientific_config)
    run_dir = Path(lf_config["output_dir"])
    accumulator = StepAccumulator(cutoff_len=int(lf_config["cutoff_len"]))
    install_training_step_instrumentation(accumulator)

    from llamafactory.train.tuner import run_exp

    callback = SFTTelemetryCallback(
        accumulator=accumulator,
        run_dir=run_dir,
        run_id=str(lf_config["run_name"]),
        scientific_config=scientific_config,
        effective_config_path=args.scientific_config,
        llamafactory_config_path=args.llamafactory_config,
        asset_manifest_path=args.asset_manifest,
        source_manifest_path=args.source_manifest,
    )
    try:
        run_exp(args=lf_config, callbacks=[callback])
    except BaseException:
        callback.finalize(exit_code=1)
        raise
    else:
        callback.finalize(exit_code=0)


if __name__ == "__main__":
    main()
