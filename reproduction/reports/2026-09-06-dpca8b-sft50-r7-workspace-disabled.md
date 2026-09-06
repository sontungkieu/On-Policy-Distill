# DPCA 8B SFT50 r7 failure: Modal workspace disabled

## Outcome

The fresh 50-step SFT run did not complete an optimizer step. Modal disabled
the workspace while the remote call was active, and the client terminated with:

```text
ConflictError: workspace ac-XiPnbl6XwQGY7Ikl7P4Ba0 is disabled
```

This is an external Modal workspace-state failure, not a CUDA OOM or a
LLaMA-Factory training exception. No retry was launched.

## Run identity and terminal state

- Run ID: `dpca8b-sft50-b200x2-fsdp-20260906-r7-reentrant-gc`
- Modal app: `ap-3d6nd6OHmi5bjWB1B6yK6E`
- App terminal state: `stopped`, `0` tasks
- Created: `2026-09-06 19:38:50+07:00`
- Stopped: `2026-09-06 19:41:25+07:00`
- W&B initialized, but no optimizer-step metrics were produced.
- TensorBoard event file contains zero scalar tags.

## Artifact inventory

The Modal Volume contains only the effective JSON config, generated
LLaMA-Factory YAML, a 12 KiB training log, and a 5.9 KiB TensorBoard event.
There is no `checkpoint-*`, model shard, optimizer state, scheduler state,
trainer state, or RNG state. Therefore this run cannot be resumed.

Local evidence is retained at
`D:/dev/codex/research_vdt/modal-runs/dpca8b-sft50-b200x2-fsdp-20260906-r7-reentrant-gc`:

| Artifact | SHA-256 |
|---|---|
| `modal.log` | `f561091e32fdde984a72905dac31524e1a95d4262d98810d5f2317122366fb77` |
| `remote-sft.log` | `b092ef565abb08e6acdde00876c43a9ae5874f45da78d34ee9336b6a54955f44` |
| `tensorboard` | `0de1f93125a338b4e3e91125283f511e56b97a4b22a1e39009fa1e95512a5cf6` |

## Billing

The post-failure billing snapshot reports `$25.16005377` calendar-month usage.
The shared r7 app description totals `$6.42893537`; subtracting the completed
two-step qualifier (`$5.90145689`) attributes `$0.52747848` to this failed
50-step launch. The ledger records the run as failed with label
`workspace-disabled-before-step-1`.

## Evidence boundary

The separate r7 two-step qualification remains valid environment evidence: it
completed two steps and saved a resume-complete checkpoint. It is not a partial
checkpoint for this fresh 50-step run, and neither run is evidence of DPCA
efficacy.

## Material Passport

- Origin skill: `modal-gpu-ops`
- Source workspace: `D:/dev/codex/research_vdt/dpca-breaking-token-barrier`
- Execution environment: Modal profile `kieusontung8`, 2 x B200
- Scientific config hash recorded by the launcher:
  `52d17ee4469dedbcb656b9224956c5d40fca0faf50cbba311e61ae5527d8eab9`
- Verification status: terminal state, Volume inventory, local logs,
  TensorBoard tags, and billing checked
- Restart status: none
