# Handoff: 1.7B TTS Debug (March 6, 2026)

## Context
User issue:
- 1.7B model runs without crash but output is mumbling / wrong speech.
- It overgenerates and does not stop properly.
- Root docs were read first (`chatgpt_info.md`, `GEMINI.md`, `STATUS_REPORT.md`, `gemini_chat.md`).

## What was verified

### Root findings from docs + code
- `codec_eos_id` should be `2150`.
- 1.7B expected to use M-RoPE, but 0.6B config also has `rope_scaling.mrope_section`, so `use_mrope=true` for both in current loader logic.
- Previous code had a hard-coded 1.7B prefill branch (fixed-size assumptions, fragile alignment), and known divergence remained.

### Runtime reproduction (current binary before final rebuild)
Command used:
- `build-cuda-ninja\qwen3-tts-cli.exe -m .\models --model-name qwen3-tts-1.7b-f16.gguf -t "Hello." --max-tokens 40 --temperature 0 --top-k 1 -o out_1.7b_debug.wav`

Observed:
- No EOS within 40 frames (hits max tokens).
- Repetitive top CB0 tokens; no `2150` emission.

Also tested 0.6B with same deterministic settings; same overgeneration behavior happened and log showed the same prefill path banner, confirming shared problematic pathing.

## Files changed in this session

### Modified
- `src/tts_transformer.cpp`

### Already dirty before this session (not reverted)
- `ggml` (modified)
- `src/tts_transformer.h` (modified)
- `models/` (untracked)

Current branch:
- `feat/kotlin-multiplatform`

## Exact code changes applied (in `src/tts_transformer.cpp`)

1. `build_prefill_graph(...)` updated to dynamic Python-style prefill construction (no hard-coded 26-token contract in this file now):
- signature now includes:
  - `std::vector<float> & tts_eos_embed`
  - `int32_t * p_codec_input_len`
- exports `codec_input_len` via pointer when provided.
- keeps `instruct` prefill concatenation for all models.
- builds prefill as:
  - `[optional instruct] + [role 3] + [codec_plus_overlay] + [first_text + codec_bos]`
- builds trailing as Python non-streaming path:
  - `text tokens [4 .. -5)` projected + final `tts_eos_embed`

2. M-RoPE position bug fixes:
- In talker `forward_step`: 4th M-RoPE stream now `0` (not `n_past`).
- In code predictor prefill: `{0,1,0,1,0,1,0,0}` (4th stream zeroed).
- In code predictor step: `{n_past,n_past,n_past,0}` (4th stream zeroed).

3. Generate path wiring:
- Added `tts_eos_embed` local in `generate(...)` and passed it into `build_prefill_graph(...)`.

## Important note
- Debug top-5 logging in generate loop is still present (`if (true) { ... Top tokens ... }`).
- This existed before and remains enabled in current file; keep for debugging or remove once behavior stabilizes.

## Build status
- Build was started and then interrupted by user:
  - `cmake --build build-cuda-ninja -j 12`
- No final compile result was captured after code edits due interruption.

## Next steps after restart (recommended)

1. Build
- `cmake --build build-cuda-ninja -j 12`

2. Deterministic check: 1.7B
- `build-cuda-ninja\qwen3-tts-cli.exe -m .\models --model-name qwen3-tts-1.7b-f16.gguf -t "Hello." --max-tokens 100 --temperature 0 --top-k 1 -o out_1.7b_test.wav`
- Confirm whether token `2150` appears before max tokens.

3. Regression check: 0.6B
- `build-cuda-ninja\qwen3-tts-cli.exe -m .\models --model-name qwen3-tts-0.6b-f16.gguf -t "Hello." --max-tokens 100 --temperature 0 --top-k 1 -o out_0.6b_test.wav`
- Ensure no new breakage.

4. If still diverging
- Compare first-step logits with Python reference for same prompt/speaker/language.
- Verify talker prefill token count and first decode `n_past == prefill_len`.
- Verify M-RoPE position tensors written at prefill and step match intended `(p,p,p,0)` layout.

## Why `apply_patch` failed earlier
- `apply_patch` tool calls failed in this environment due path/workspace resolution mismatch; direct PowerShell file rewrite was used successfully instead.
