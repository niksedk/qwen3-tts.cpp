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
| 1.7B code predictor hidden-size path | Implemented | Code predictor now uses its own config dimensions (hidden/intermediate/heads/KV/head_dim/rope/eps), independent of talker dimensions. |
| 1.7B `small_to_mtp` projection support | Implemented | GGUF conversion and C++ loading now include `code_pred.small_to_mtp.{weight,bias}` and apply projection in predictor prefill/step paths. |
| Missing-projection safety guard | Implemented | Loader now fails fast with a clear message when a model requires `small_to_mtp` but GGUF is missing those tensors. |
| Python vs C++ trace tooling | Implemented | `scripts/dump_python_trace.py` and `scripts/debug_trace_report.py` allow frame/step-level parity checks. |

## Current State (Open / Needs Verification)

| Area | Status | Notes |
|---|---|---|
| 1.7B regression coverage in automated tests | Open | Current CI/regression flows are still centered on 0.6B artifacts; add at least one deterministic 1.7B gate. |
| Cross-speaker/perceptual validation for 1.7B | Open | Validate multiple built-in speakers and prompts after projection fix to guard against voice-specific regressions. |
| M-RoPE position handling consistency | Partial | Remaining path consistency should still be audited and documented with explicit expected layouts per path. |
| CUDA throughput benchmark refresh | Needs verification | Re-run and publish comparable CPU/CUDA benchmark data after the 1.7B predictor changes. |

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

- Lock in 1.7B fix quality with reproducible checks across speakers/prompts.
- Add at least one deterministic 1.7B regression check alongside existing 0.6B checks.
- Confirm M-RoPE position tensors are consistent in all prefill and step paths.

Exit criteria:

- No known 1.7B "mumbling + divergence-at-codepred-step00" reproduction on baseline prompts.
- Deterministic smoke prompts pass for both 0.6B and 1.7B under documented settings.
- Regression job includes a 1.7B check (or an explicitly documented lightweight substitute).

### M1: Reproducible Benchmarking Gate

Scope:

- Standardize one benchmark script and one reporting format (CPU and CUDA variants).
- Reconcile historical and current metrics in one table with date, commit, model, and hardware fields.

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

1. Ensure all developers regenerate `qwen3-tts-1.7b-f16.gguf` with the current converter before debugging/runtime comparison.
2. Add one lightweight 1.7B deterministic regression gate (trace comparison or token-level smoke check).
3. Audit and document M-RoPE position writes in `tts_transformer.cpp`; add assertions where practical.
4. Re-run baseline CPU and CUDA benchmarks with identical prompts, token limits, and reporting fields.
5. Update this document with measured results and move verified items from "Open" to "Implemented".

## Ownership and Update Rule

- This file is the source of truth for implementation status and roadmap.
- When status changes, update this file first, then link to supporting PRs/commits.
- Latest major correctness milestone: commit `8880e4b` (1.7B code predictor projection fix and trace tooling).
