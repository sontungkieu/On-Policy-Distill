# DPCA 8B SFT FSDP qualification

## Outcome

The authoritative 8B configuration now contains one set of scientific values:
400,000 pinned OpenThoughts3 rows, maximum sequence length 16,384, global batch
128, AdamW learning rate `1e-4`, bf16 compute, gradient clipping 1, and a
100-optimizer-step stop. There are no `paper_*` or `actual_*` shadow fields.

The two-B200 execution topology uses FSDP full sharding. A one-step fit and
checkpoint qualification completed in Modal app
`ap-BddnT4LYMuRXjshGEryCCU` before launching the costly 100-step run.

## Evidence

- Run: `dpca8b-sft100-b200x2-fsdp-20260906-r3-fit-qualification-1step`
- Dataset gate: 400,000 rows from 40 pinned parquet shards.
- Tokenized cache: 400,000 examples, 157 Arrow shards, persisted on the Modal
  Volume at `/workspace/cache/openthoughts3-400k-llama3-cutoff16384`.
- Distributed contract: 2 B200 workers, per-device batch 1, gradient
  accumulation 64, effective global batch 128.
- Step 1: 260.79 seconds compute, loss 1.6321, gradient norm 5.9886384,
  learning rate `1e-4`, 1,955,184 input tokens, 7,432.022 tokens/second.
- Complete run wall time inside the trainer: 640.05 seconds, including the
  full-state checkpoint.
- Checkpoint: seven safetensors plus `optimizer.bin`, `scheduler.pt`,
  `trainer_state.json`, `training_args.bin`, and one RNG state per rank.
- Tracking: one W&B run and one TensorBoard event file.

## Failed preflights retained for diagnosis

- `r1`: DDP replicated full Adam state and ran out of 178 GiB per GPU before
  optimizer step 1.
- `r2`: FSDP activation checkpointing conflicted with LLaMA-Factory 0.9.4's
  fp32 trainable-parameter upcast (`saved bf16`, `recomputed fp32`). The
  tokenized cache was nevertheless completed and retained.
- `r3`: FSDP full shard with model and FSDP activation checkpointing disabled
  completed optimizer step 1 and the full resume checkpoint.

## Budget gate

The Modal billing snapshot after `r3` reported $13.90855491 used against a
local $30 budget and $29 hard limit. At 260.79 seconds per optimizer step, the
100-step compute-only projection is about 7.24 hours; the conservative $100
run estimate makes projected usage $113.90855491. The billing guard therefore
returned `BLOCK`, so the 100-step run was not launched.
