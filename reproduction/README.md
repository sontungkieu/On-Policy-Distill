# DPCA reproduction environment qualification

## Active 8B scientific pipeline

`scientific/dpca_8b_scientific.json` is the baseline effective scientific
configuration for the current Qwen3-8B to Llama-3.1-8B experiment. It does not
carry parallel `paper_*`, `actual_*`, or timing-only response-length fields.
The SFT and OPD launchers both read their values from this file and save an
unchanged copy beside each checkpoint.

`scientific/dpca_8b_scientific_sdpa_flash.json` is a controlled attention-
kernel variant of that same contract. It changes only the run/output lineage
and sets PyTorch SDPA to `flash_only`; model/data revisions, sample order, batch,
sequence length, optimizer, precision, FSDP, and seed stay matched. The runner
disables math, memory-efficient, and cuDNN SDPA fallbacks before model creation,
so an unsupported shape or GPU fails instead of silently changing kernels. This
is PyTorch's native flash SDPA backend, not the separately installed
`flash-attn` package, and the per-rank runtime manifest records all four backend
enable flags.

`scientific/dpca_8b_scientific_fa2.json` is the corresponding external
FlashAttention-2 variant. It keeps the same scientific fields and changes only
run/output lineage plus `attention_backend: fa2`. Its separate, digest-pinned
CUDA 12.8 devel image builds `flash-attn==2.8.3` for SM100 from the same uv lock;
the runtime manifest records both `_attn_implementation=flash_attention_2` and
the installed package version. This image is separate because compiling the
extension requires NVCC, while the baseline uses a smaller CUDA runtime image.

The first phase currently runs 50 full-SFT optimizer updates with LLaMA-Factory on two
B200 GPUs, using a pinned 400,000-row OpenThoughts3 subset, global batch 128,
maximum sequence length 16,384, AdamW at `1e-4`, bf16, and gradient clipping
at 1. The trainer releases unused CUDA allocator cache at every optimizer-step
boundary. The two workers use FSDP full sharding with LLaMA-Factory's reentrant
model gradient checkpointing; FSDP-level activation checkpointing remains
disabled because its non-reentrant recompute conflicts with LLaMA-Factory
0.9.4's fp32 parameter upcast. This changes memory/computation scheduling but
keeps the SDPA attention backend and scientific batch contract. The run
persists the tokenized dataset cache and saves `checkpoint-50` with model,
optimizer, scheduler and trainer state, plus TensorBoard events, the generated
LLaMA-Factory YAML, W&B metrics, the exact config, and an asset manifest with
dataset-shard hashes. The representative 50-step truncation, constant
scheduler, zero LR warmup, and public ungated mirror of the pinned upstream
Llama base are recorded explicitly in that one config.

The paper specifies LLaMA-Factory for SFT and VeRL for OPD, but the authors'
public DPCA repository ships only the VeRL/teacher-service OPD path: it does not
include the SFT launcher, the selected 400K-sample manifest, or the resulting
Llama-3.1-8B-SFT checkpoint. The SFT wrapper in this overlay therefore fills a
missing released-artifact boundary; it is not a replacement for DPCA code.

The SFT wrapper calls LLaMA-Factory's public callback API through a two-rank
`torchrun` entrypoint. Its telemetry adapter is observational only: it does not
change inputs, loss, gradients, optimizer, or checkpoint state. At each
optimizer step it writes one unified record to W&B, TensorBoard, and
`metrics.jsonl`, plus per-rank JSONL containing the actually consumed batch.
The record includes active/supervised/prompt/padding token counts, exact
sequence-length percentiles, EOS-target coverage, cutoff incidence, non-finite
flags, gradient-clipping incidence, step/rank timing, and CUDA allocator peaks.
Runtime Torch/CUDA/cuDNN, effective SDPA/TF32 state, special-token IDs, full
scientific config, asset hashes, and source hashes are stored in the W&B config
and a small telemetry artifact. W&B's native system monitor remains responsible
for utilization, power, temperature, clocks, and hardware error counters.

The following DPCA phase is already locked in the same file to global batch
128, prompt length 1,024, response length 16,384, one on-policy sample per
prompt, `(temperature=1, top_p=1, top_k=-1)`, AdamW at `1e-6`, bf16, and
gradient clipping at 1. It must consume the saved SFT checkpoint; a launcher
must not substitute a raw instruct/base model or a 64-token timing cap.

The scientific environment is managed separately by
`scientific/runtime/pyproject.toml` and `scientific/runtime/uv.lock`. Only this
locked runtime directory is baked into the Modal image; the config and runner
are startup mounts, so changing a run ID or step count does not rebuild the
dependency image. Build and execution occur
on Modal; the workstation does not start Docker:

```bash
uvx --from modal==1.5.5 modal --profile kieusontung8 run \
  reproduction/scientific/modal_sft.py::main
```

A one-step fail-closed flash-kernel qualification uses the same cached runtime
image and workspace assets:

```bash
DPCA_CONFIG_NAME=dpca_8b_scientific_sdpa_flash.json \
DPCA_MODAL_APP_NAME=dpca8b-sft-sdpa-flashonly-r9-canary \
uvx --from modal==1.5.5 modal --profile kieusontung8 run \
  reproduction/scientific/modal_sft.py::qualify --steps 1
```

That native SDPA qualification is intentionally distinct from an external FA2
qualification. The latter selects its dedicated build and config explicitly:

```bash
DPCA_CONFIG_NAME=dpca_8b_scientific_fa2.json \
DPCA_RUNTIME_DOCKERFILE=Dockerfile.flashattn \
DPCA_MODAL_APP_NAME=dpca8b-sft-fa2-r10-canary \
uvx --from modal==1.5.5 modal --profile kieusontung8 run \
  reproduction/scientific/modal_sft.py::qualify --steps 1
```

For a different authorized Modal workspace, select its profile explicitly and
map the workspace-local W&B secret without changing the scientific config. For
example, the `phamvanvuhoan` workspace exposes its credential as
`wandb-secret`:

```bash
DPCA_WANDB_SECRET_NAME=wandb-secret \
uvx --from modal==1.5.5 modal --profile phamvanvuhoan run \
  reproduction/scientific/modal_sft.py::main
```

Modal Volumes and image caches are workspace-scoped, so the first launch in a
new workspace may rebuild the locked runtime and download the pinned assets.

The earlier 64-token, batch-one timing harnesses below are retained only as
historical implementation-validation evidence. They are not inputs to the
active scientific pipeline.

This overlay qualifies the pinned implementation from
`ivanniu/On-Policy-Distill` commit
`927a8264f2e303b7f82c2d331a58fd4240c8805a`. The upstream DPCA objective is
kept intact; the launcher, trainer stop boundary, and FSDP checkpoint manager
are intentionally extended for phased, topology-portable qualification. The
manifest records both untouched-upstream hashes and effective overlay hashes.

## Evidence boundary

- `config_audit.json` separates values reported in Appendix A, Table 4 from
  values found only in the authors' public `cross_distill.sh`.
- The smoke uses no model, dataset, teacher server, rollout, or optimizer step.
  A passing result is an implementation/environment qualification, not a paper
  reproduction or evidence that DPCA improves an LLM.
- The full paper setup remains a 16-GPU, separate-teacher-service workload with
  SFT-initialized students and 20K-prompt-scale OPD experiments.

## Known source-to-paper qualifications

- The public launcher agrees with the paper on batch size 128, 1024 prompt
  tokens, 16384 response tokens, AdamW learning rate `1e-6`, gradient clip 1,
  bf16, and rollout sampling `(temperature=1, top_p=1, top_k=-1)`.
- Warmup, weight decay, asymmetric PPO clipping, the advantage clamp, seed,
  training steps, epochs, and the exact parallel topology are code-only values,
  not Table 4 disclosures.
- The public implementation normalizes decoded fragments with Unicode NFC and
  suppresses large or failed chunks with a sentinel. These safeguards are not
  specified by the paper's exact-detokenization DPCA definition, so future
  scientific runs should label `upstream-code` and `paper-equation` modes
  separately rather than silently treating them as identical.
- SDPA and TF32 are explicit overlay execution choices rather than settings
  reported by the DPCA paper or its public launcher. They are now present in
  the effective scientific config and runtime manifest so their numerical
  effects cannot be hidden by the orchestration layer.
- `requirements.txt` is a large environment freeze, not a minimal lock. It pins
  Torch 2.9.0/CUDA 12.8-era packages and vLLM 0.12.0, while `setup.py` advertises
  a different optional vLLM upper bound. The overlay now uses `pyproject.toml`
  plus a checked `uv.lock`; the CUDA/PyTorch base is also pinned by digest.

## Modal B200 smoke

From WSL, with an explicit Modal profile:

```bash
uvx --from modal modal --profile kieusontung8 run \
  reproduction/modal_dpca_b200_smoke.py::main
```

The run checks pinned hashes, the launcher contract, two DPCA mismatch
directions, teacher log-mass conservation, semantic-prior projection, and a
finite bf16 matrix multiplication on the requested GPU. The Modal definition
has `retries=0`.

The first qualification passed on 2026-09-05 in Modal app
`ap-ZpKsW23FOFiAfgil8wUkrV`; see
`reports/2026-09-05-modal-b200-smoke.md` and the run directory
`../../modal-runs/dpca-b200-contract-smoke-20260905-r1`.

## Ten-step 2x B200 implementation validation

`configs/dpca_2gpu_10step.json` pins two small public cross-tokenizer models and
ten local prompts. One Modal container receives two B200s: a frozen teacher is
placed on `cuda:0`, while student rollout and AdamW updates run on `cuda:1`.
The run calls the upstream `TeacherClient._align_chunks` implementation and
`compute_policy_loss_opd` for every optimizer step.

The run writes per-step JSONL, TensorBoard events, a final student checkpoint,
and an effective config/provenance record to the Modal volume
`dpca-breaking-token-barrier-runs`. It also requires online W&B and uploads the
small log/provenance bundle there. The Modal function requests `B200:2`, has no
retries, uses a single-use container, and the training entrypoint rejects any
runtime that exposes fewer than two B200 devices.

W&B uses `job_type=implementation-validation` and the shared
`schema:experiment-v1` metadata contract. Tags are deterministic and
namespaced in this order: schema, actual framework, method, regime, objective,
variant, platform, accelerator, update budget, stage, student, teacher. For
this harness the actual framework is `pytorch` (it directly calls the pinned
VeRL functions rather than running the VeRL trainer), the objective is
`semantic-prior-clipped-pg`, and the variant is `upstream-code`. Run
`python -m unittest reproduction.tests.test_wandb_schema` to catch drift.

Run from WSL with the explicit billing profile:

```bash
uvx --from modal modal --profile kieusontung8 run \
  reproduction/modal_dpca_2gpu_train.py::main
```

This is a real ten-optimizer-step integration run, but it is deliberately
classified as `implementation_validation`. It is not a paper reproduction:
the paper's released repository does not provide the reported SFT-initialized
student checkpoints, and this reduced run does not use its 7B/8B models,
16-GPU topology, data scale, response length, or batch size.

## Phase-bounded paper/common DPCA runner

`dpca_phased_run.py` defines two DPCA-only profiles:

- `paper_8b`: Qwen3-8B to the paper's Llama-3.1-8B-SFT initialization,
  batch 128, prompt 1024, response 16384.
- `common_4b1b`: pinned Qwen3-4B to Llama-3.2-1B, batch 64, prompt 1024,
  response 2048.

Both retain the released optimizer schedule: 500 planned steps, learning rate
`1e-6`, and 10 linear warmup steps. Qualification is split without changing
that schedule:

```text
warmup:      steps 1..10, save global_step_10
post_warmup: resume global_step_10, run steps 11..20, save global_step_20
```

Generate and inspect a launch contract without starting training:

```bash
uv run --project reproduction python reproduction/dpca_phased_run.py \
  --profile common_4b1b --phase warmup --gpus 2 \
  --run-id dpca-common-qualification
```

Add `--execute` only inside the Linux training environment after defining
`MODEL_PATH`, `TEACHER_CKPT_PATH`, `TRAIN_FILE`, `TEST_FILE`, `CKPTS_DIR`,
`TRAIN_MANIFEST_SHA256`, and `VAL_MANIFEST_SHA256`. The paper profile also
requires explicit `TEACHER_REVISION` and `STUDENT_REVISION`, because the
released materials do not identify a reproducible SFT checkpoint.

The runner supports actor world sizes 1, 2, and 4. It keeps global batch and
the 500-step scheduler invariant, changes only FSDP placement, and requires a
portable checkpoint before a topology-changing resume. `cross_distill.sh`
enables PyTorch Distributed Checkpoint for model/optimizer resharding and uses
`trainer.stop_after_global_step` to stop at a qualification boundary without
changing `trainer.total_training_steps`.

Same-topology resumes restore rank-local RNG state. When world size changes,
rank-local streams do not have a one-to-one continuation; the checkpoint
manager applies and logs a deterministic seed derived from the pinned seed,
global step, and new rank. This is a reproducible topology transition, not a
claim of bitwise identity with an uninterrupted fixed-topology run.

The `--gpus` value covers the actor/rollout FSDP worker group. Both profiles
retain the paper's external online teacher-service contract; the teacher GPUs
are therefore additional unless a separately qualified colocated deployment
is supplied. Do not interpret `--gpus 1|2|4` as an end-to-end GPU count yet.

Every phase sends the same scalar stream to W&B and TensorBoard (under
`$CKPTS_DIR/tensorboard` by default), including optimizer/loss/gradient and
length metrics plus `actor/opd_valid_tokens`, `actor/opd_inf_tokens`,
`actor/opd_inf_ratio`, `actor/opd_chunk_count`, and teacher-truncation counts
and ratios. `WANDB_RUN_ID` is stable across the two phases; the second phase
uses `WANDB_RESUME=must` so it cannot silently create a different run.

The current local tests exercise contract generation, fail-closed resume
validation, and a single-process DCP model/AdamW round trip. A real Linux/NCCL
qualification must still prove `1 -> 2`, `2 -> 4`, and `4 -> 1` resharding on
the exact B200 image before these transitions are used for a scientific run.

## Remote Docker Hub build and 2x B200 qualification

Docker is never started on the workstation. The portable runtime is built and
pushed inside a disposable Modal VM Sandbox whose nested Docker daemon is
terminated in a `finally` block. The workstation only prepares an audited
context, invokes Modal, and saves the resulting logs.

Create the Modal registry secret once from WSL. The credential stays outside
the repository and its value is never printed:

```bash
uv run --no-project --with modal==1.5.5 \
  python reproduction/bootstrap_modal_dockerhub_secret.py \
  --profile kieusontung8 \
  --credential-file /home/tung/dockerhub.txt \
  --secret-name dpca-dockerhub-kieusontung8
```

Compute `CONTEXT_SHA256` with `audit_local_context()`, then build and push the
immutable tag remotely:

```bash
uvx --from modal==1.5.5 modal --profile kieusontung8 run \
  reproduction/modal_build_push_dockerhub.py::main \
  --image docker.io/<namespace>/dpca-breaking-token-barrier:<immutable-tag> \
  --registry-secret dpca-dockerhub-kieusontung8 \
  --source-commit 927a8264f2e303b7f82c2d331a58fd4240c8805a \
  --context-sha256 "$CONTEXT_SHA256"
```

The builder uses BuildKit registry output with eStargz compression, provenance,
and an SBOM. Qualify the exact pushed tag in one single-use Modal container:

```bash
export DPCA_DOCKER_IMAGE=docker.io/<namespace>/dpca-breaking-token-barrier:<immutable-tag>
export DPCA_DOCKER_REGISTRY_SECRET=dpca-dockerhub-kieusontung8
uvx --from modal==1.5.5 modal --profile kieusontung8 run \
  reproduction/modal_qualify_dockerhub.py::main
```

Qualification requires exactly two B200s, checks compute capability 10.0 and a
finite bf16 matrix multiplication on both devices, loads both pinned model
snapshots offline, and imports the upstream DPCA alignment/loss functions. The
image is still not a full multi-node training image: vLLM, FlashAttention, Ray
multi-node startup, the separate teacher service, and FSDP are outside this
ten-step implementation validation.
