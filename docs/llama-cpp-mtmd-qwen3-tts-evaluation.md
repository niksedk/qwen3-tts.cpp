# Evaluation: Qwen3 TTS Integration into llama.cpp / mtmd

Date: 2026-04-15

Original issue: [ggml-org/llama.cpp#21956](https://github.com/ggml-org/llama.cpp/issues/21956)

Scope: this note evaluates the current `qwen3-tts.cpp` repository against the design directions in llama.cpp issue `#21956` ("Support audio output in mtmd"), using the issue text as provided and the code in this repository.

## Executive Summary

`qwen3-tts.cpp` is a strong reference implementation for Qwen3 TTS, but it is not a natural fit for the normal llama.cpp token decode loop.

Best integration path:

- Implement Qwen3 TTS as a dedicated mtmd/libmtmd backend, not as a small extension of the standard text-generation path.
- Keep the Qwen3 vocoder/audio detokenizer in mtmd as a non-causal, model-specific component.
- Treat speaker/style conditioning as explicit embedding inputs.
- Keep the transformer-to-vocoder boundary token-based for Qwen3 TTS, because the vocoder consumes discrete codec IDs, not embeddings.
- Add streaming later; the current repository is batch-oriented and decodes audio only after all codec frames are generated.

The current repo already separates the major components cleanly in CMake and source layout, which makes selective porting practical: `tts_transformer`, `audio_tokenizer_encoder`, and `audio_tokenizer_decoder` are distinct libraries (`CMakeLists.txt:113`, `CMakeLists.txt:130`, `CMakeLists.txt:146`).

## What the Repository Actually Implements

The repository implements a full TTS pipeline, not just an audio post-processor:

- Text tokenization
- Optional speaker embedding extraction from reference audio
- A Qwen3 TTS transformer ("talker" + nested code predictor)
- A separate vocoder/audio decoder model

This is documented in the repo architecture summary:

- `README.md:338`
- `README.md:340`
- `README.md:342`
- `README.md:363`
- `README.md:364`

The runtime loads two GGUF model files:

- Main TTS GGUF containing tokenizer, speaker encoder, talker, and code predictor (`src/pipeline/pipeline_models.cpp:22`)
- Separate tokenizer/vocoder GGUF (`src/pipeline/pipeline_models.cpp:22`, `src/decoder/decoder_loader.cpp:108`, `src/decoder/decoder_loader.cpp:109`)

That matches the repo's documented conversion flow:

- Main model conversion: `README.md:186`
- Vocoder conversion: `README.md:194`

## Evaluation Against Issue #21956

### 1. Model file

Issue direction:

- Normal causal audio detokenizer model -> could be managed as a normal text model
- Fundamentally different audio detokenizer -> implement in `libmtmd`

Assessment for Qwen3 TTS:

- The Qwen3 vocoder is clearly in the second category. It is not a normal causal decoder. It is a graph-built decoder that looks up 16 codebooks, projects them, runs pre-transformer layers, then runs convolutional / upsampling decoder blocks to produce waveform samples (`src/decoder/decoder_graph.cpp:12`, `src/decoder/decoder_runtime.cpp:6`).
- Therefore the vocoder should live in `libmtmd`, not in the normal llama text-model path.

More importantly, the main Qwen3 TTS transformer is also not a clean fit for the ordinary llama.cpp sampling loop:

- It constructs a custom prefill embedding sequence rather than feeding plain token IDs (`src/transformer/transformer_embeddings.cpp:218`).
- It injects speaker embeddings directly into that prefill (`src/transformer/transformer_embeddings.cpp:306`, `src/transformer/transformer_embeddings.cpp:316`).
- It generates frame codebook 0 with the talker, then runs a second autoregressive model to generate codebooks 1-15 (`src/transformer/transformer_generate.cpp:259`, `src/transformer/transformer_runtime_code_pred.cpp:238`).
- That second model has its own KV cache and its own graph family (`src/transformer/transformer_state_internal.h:129`, `src/transformer/transformer_runtime_code_pred.cpp:267`, `src/transformer/transformer_runtime_code_pred.cpp:346`, `src/transformer/transformer_runtime_code_pred.cpp:459`).

Conclusion:

- For Qwen3 TTS, "only put the vocoder in mtmd and treat the rest as a normal llama model" is the wrong split.
- The better split is: implement the whole Qwen3 TTS runtime as a model-specific mtmd backend, while reusing ggml / backend / scheduler infrastructure from llama.cpp where possible.

### 2. Data passing

Issue direction:

- Prefer always passing embeddings from the main model to the audio generation model

Assessment for Qwen3 TTS:

This repository shows that Qwen3 TTS has two different boundaries, and they should not be collapsed into one abstraction.

Boundary A: conditioning inputs

- Reference audio is turned into a speaker embedding (`src/pipeline/pipeline_synthesize.cpp:151`)
- Named speakers are resolved into embedding rows from model metadata (`src/transformer/transformer_embeddings.cpp:386`)
- That embedding is injected directly into the transformer's prefill construction (`src/transformer/transformer_embeddings.cpp:306`, `src/transformer/transformer_embeddings.cpp:316`)

Boundary B: audio generation output

- The transformer emits discrete speech codes as `[n_frames, 16 codebooks]` (`src/pipeline/pipeline_synthesize.cpp:291`, `src/pipeline/pipeline_synthesize.cpp:305`)
- The vocoder consumes integer code IDs, not embeddings (`src/audio_tokenizer_decoder.h:50`, `src/decoder/decoder_runtime.cpp:6`)

This means the issue's "always pass embeddings" proposal is too broad for Qwen3 TTS.

Recommended interpretation for llama.cpp:

- Support embeddings as a first-class conditioning/input path.
- Do not require the transformer-to-vocoder handoff to be embedding-based.
- Use a typed payload boundary in mtmd:
  - conditioning embeddings for speaker/style/reference state
  - discrete audio codes for vocoder input

For Qwen3 TTS specifically, forcing the vocoder interface to embeddings would add work and remove information about codebook structure that the current decoder explicitly depends on.

### 3. Generation state

Issue direction:

- Introduce a generic generation state tracker like `TEXT`, `AUDIO`, etc.

Assessment for Qwen3 TTS:

This repository is a dedicated TTS pipeline, not an interleaved text/audio assistant. Its generation loop:

- tokenizes text input once
- builds a TTS-specific prefill
- generates codec frames until codec EOS
- decodes the resulting speech codes to waveform

Relevant points:

- EOS for codec generation is explicit (`src/transformer/transformer_generate.cpp:226`)
- "Thinking" codec tokens are filtered rather than emitted as output (`src/transformer/transformer_generate.cpp:236`, `src/transformer/transformer_generate.cpp:280`)
- There is no path that alternates back to text output after audio begins

Conclusion:

- A generic interleaved `TEXT <-> AUDIO` public API is not required to support Qwen3 TTS in a first iteration.
- For Qwen3 TTS, a simpler model-specific state machine is enough:
  - prompt / conditioning
  - audio frame generation
  - codec EOS
  - optional vocoder decode

If llama.cpp wants a generic future-proof API, it should still support richer states, but Qwen3 TTS should not be forced through that more complex path in v1.

## Important Gap: Streaming

The largest practical gap between this repository and a likely mtmd user experience is streaming audio output.

Current behavior:

- `synthesize_internal()` generates all `speech_codes` first (`src/pipeline/pipeline_synthesize.cpp:291`)
- It then decodes all frames in one decoder call (`src/pipeline/pipeline_synthesize.cpp:341`)
- The decoder graph is cached per `n_frames`, which reinforces full-buffer decoding rather than chunked streaming (`src/decoder/decoder_cache.cpp:24`, `src/decoder/decoder_cache.cpp:60`)

Implication:

- This repository is a good correctness and architecture reference, but not yet a reference for low-latency incremental audio output.
- If mtmd wants "speak while generating", llama.cpp will need an additional chunked or streaming decode design for the vocoder path.

## Recommended llama.cpp Integration Shape

### Recommended v1 architecture

1. Add a dedicated Qwen3 TTS backend under mtmd/libmtmd.
2. Load:
   - main Qwen3 TTS GGUF
   - separate Qwen3 tokenizer/vocoder GGUF
3. Keep these model-specific responsibilities inside the Qwen3 backend:
   - text tokenization for TTS
   - prefill embedding construction
   - speaker embedding extraction / speaker lookup
   - talker loop
   - nested code predictor loop
   - vocoder invocation
4. Expose model capabilities similar to this repo:
   - supports voice clone
   - supports named speakers
   - supports instruction/style prompting
   - speaker count / embedding dimension
   (`src/qwen3_tts.h:87`, `src/pipeline/pipeline_models.cpp:153`)

### Recommended API direction

For Qwen3 TTS, the useful public surface is closer to:

- load Qwen3 TTS model set
- optionally load or compute speaker embedding
- synthesize text to audio
- optionally enumerate named speakers / model capabilities

This repository's C API is a reasonable reference for that kind of surface:

- model load (`src/qwen3_tts_c.h:67`)
- synthesize (`src/qwen3_tts_c.h:74`)
- synthesize with speaker embedding (`src/qwen3_tts_c.h:87`)
- extract speaker embedding (`src/qwen3_tts_c.h:94`)
- get model capabilities (`src/qwen3_tts_c.h:100`)
- get available speakers (`src/qwen3_tts_c.h:106`)

### What not to do

- Do not force Qwen3 TTS through the ordinary llama token sampler as if it were just "text plus audio tokens".
- Do not force an embedding-only interface between transformer and vocoder.
- Do not assume the current repo already solves streaming output.

## Practical Reuse from This Repository

High-value pieces to port or adapt:

- Qwen3 TTS transformer runtime and GGUF metadata handling
- speaker encoder runtime
- vocoder runtime
- model capability / speaker discovery logic
- test assets and deterministic reference strategy

Lower-value pieces to carry over directly:

- repo-specific top-level wrapper API
- current batch-only synthesis orchestration
- platform-specific extras like JNI/CoreML glue unless llama.cpp explicitly wants them

## Final Recommendation

Use `qwen3-tts.cpp` as a source-level reference and donor for a dedicated Qwen3 TTS mtmd backend.

The cleanest integration is:

- Qwen3 TTS runtime stays model-specific
- non-causal audio decoder stays in mtmd
- speaker/style conditioning is embedding-based
- vocoder input stays discrete-code-based
- streaming is a separate follow-up step, not assumed from the current repo

That path aligns with the issue's overall direction, but it argues against an embedding-only handoff model and against treating Qwen3 TTS as a normal llama decode loop with a small audio extension.
