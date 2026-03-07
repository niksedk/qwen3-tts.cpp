# Qwen3-TTS Development Plan

Last updated: 2026-03-07

## Purpose

Maintain a single, current plan that separates:

- What is already implemented
- What is partially implemented or unverified
- What should be implemented next

## Current State (Implemented)

| Area | Status | Notes |
|---|---|---|
| Windows build workflow | Implemented | `build.ps1` is the primary entrypoint, including Ninja mode. |
| Windows regression workflow | Implemented | `scripts/run_all_tests.ps1` is the primary test entrypoint. |
| Test harness robustness | Implemented | Preflight checks, clearer PASS/FAIL/SKIP summary, better failure tails. |
| Test asset preparation | Implemented | `scripts/prepare_test_assets.ps1` supports local `.venv`, install, generate, and force-regenerate flows. |
| Deterministic reference workflow | Implemented | Determinism gate is documented (`git diff --exit-code -- reference/*.json`). |
| Decoder snake path | Implemented | Decoder uses `ggml_snake(...)`; ggml has `GGML_OP_SNAKE`. |
| Static KV update pattern | Implemented | `tts_transformer.cpp` uses `ggml_set_rows(...)` in attention cache update paths. |
| Python-style dynamic prefill construction | Implemented | `build_prefill_graph(...)` constructs prefill from projected role, codec overlay, first text token, and EOS trailing logic. |

## Current State (Open / Needs Verification)

| Area | Status | Notes |
|---|---|---|
| 1.7B speech quality and stopping behavior | Open | Historical report shows mumbling/overgeneration and no EOS within max tokens under deterministic debug runs. |
| 0.6B deterministic regression under same debug settings | Open | Historical handoff noted similar overgeneration behavior under specific deterministic settings. |
| M-RoPE position handling consistency | Partial | Some prefill paths use `(p,p,p,0)` style, but some step paths still show `(p,p,p,p)` and should be aligned/verified. |
| CUDA throughput claim alignment | Needs verification | Older CPU baseline and later CUDA throughput notes differ; benchmark protocol should be unified and rerun. |

## Performance Baselines and Targets

### Historical baselines

- CPU-only historical report: about 1.94 RTF (slower than real-time), with vocoder and encoder as major costs.
- Later CUDA report claims approximately 1.07 internal throughput on modern laptop GPU class hardware.

### Working targets

1. Functional correctness first:
   - 1.7B and 0.6B should emit EOS reliably under deterministic settings on short prompts.
   - Remove obvious degeneration patterns before throughput optimization.
2. Throughput second:
   - Reach stable real-time or better (`RTF <= 1.0`) for target hardware profiles.
3. Parity stretch goal:
   - Continue toward higher-throughput parity goals only after correctness and reproducibility gates are stable.

## Milestones

### M0: Correctness Gate (Highest priority)

Scope:

- Validate EOS and code generation behavior for 0.6B and 1.7B with deterministic settings.
- Confirm M-RoPE position tensors are consistent in all prefill and step paths.

Exit criteria:

- Deterministic smoke prompts emit EOS before max tokens in expected scenarios.
- No known "mumbling + endless generation" reproduction on baseline prompts.

### M1: Reproducible Benchmarking Gate

Scope:

- Standardize one benchmark script and one reporting format (CPU and CUDA variants).
- Reconcile historical and current metrics in one table with date and hardware fields.

Exit criteria:

- One benchmark command per profile is documented and repeatable.
- Metrics are published in this file with date, commit, model, and hardware.

### M2: Throughput Optimization

Scope:

- Continue CUDA-centric optimizations for encoder, predictor orchestration, and vocoder.
- Prioritize wins that preserve output quality and determinism gates.

Exit criteria:

- RTF target achieved for defined hardware tier.
- Regression suite remains green.

## Immediate Next Actions

1. Run deterministic short-prompt checks for 0.6B and 1.7B with `temperature=0` and `top-k=1`, capturing whether/when EOS appears.
2. Audit all M-RoPE position writes in `tts_transformer.cpp` and normalize to one intended layout per path.
3. Add a small regression test/assertion for position tensor layout where practical.
4. Re-run baseline CPU and CUDA benchmarks with identical prompts, token limits, and reporting fields.
5. Update this document with measured results and promote resolved items from "Open" to "Implemented".

## Ownership and Update Rule

- This file is the source of truth for implementation status and roadmap.
- When status changes, update this file first, then link to supporting PRs/commits.
