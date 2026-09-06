## Material Passport

- Origin Skill: experiment-agent, modal-gpu-ops
- Origin Mode: run
- Origin Date: 2026-09-06
- Verification Status: VERIFIED
- Version Label: exp_result_v1

## Experiment Result

- **ID**: `dpca8b-sft50-b200x2-fsdp-20260906-r4`
- **Type**: training
- **Status**: crashed
- **Modal app**: `ap-9CIGra4XYmc61dbh2R0w69` (stopped, zero tasks)
- **Working Directory**: `/opt/dpca`
- **Requested topology**: 2 x NVIDIA B200, FSDP full shard
- **Requested optimizer steps**: 50
- **Completed optimizer steps**: 1
- **Actual Modal cost**: USD 1.73737423

### Output Files

| File | Purpose |
|------|---------|
| `D:/dev/codex/research_vdt/modal-runs/dpca8b-sft50-b200x2-fsdp-20260906-r4/modal.log` | Complete local launch and remote stdout/stderr capture |
| `/workspace/runs/dpca8b-sft50-b200x2-fsdp-20260906-r4/sft.log` | Training log retained on the Modal volume |

### Output Summary

Step 1 completed in 270.54 seconds with loss 1.6321, gradient norm
5.928466796875, learning rate 1e-4, 1,955,184 input tokens, and 7,165.031
tokens/second. The next forward pass failed before optimizer step 2 while
trying to allocate another 442 MiB on rank 0. At failure, GPU 0 reported
178.35 GiB total, 393.88 MiB free, 175.75 GiB allocated by PyTorch, and
417.21 MiB reserved but unallocated.

No resume-complete checkpoint exists because the only configured save boundary
was step 50. W&B retained the step-1 metrics under run
`dpca8b-sft50-b200x2-fsdp-20260906-r4`.

### Anomalies Detected

- `RESOURCE_ANOMALY`: steady-state CUDA memory after Adam-state initialization
  exceeded the memory envelope for the next 16K-token micro-batch.
- The preceding one-step qualification could not expose this boundary because
  it terminated immediately after initializing optimizer state.
- No automatic retry was performed.

### Proposed Recovery Gate

Keep the scientific values unchanged and first add only
`PYTORCH_ALLOC_CONF=expandable_segments:True`. Require a two-step qualification
to pass before relaunching 50 steps. If the two-step gate still OOMs, activate
a separately recorded memory-topology adaptation instead of silently reducing
sequence length, global batch size, dataset, or optimizer budget.
