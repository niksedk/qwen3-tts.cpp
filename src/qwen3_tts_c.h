#ifndef QWEN3_TTS_C_H
#define QWEN3_TTS_C_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  ifdef QWEN3_TTS_EXPORT
#    define QWEN3_TTS_API __declspec(dllexport)
#  else
#    define QWEN3_TTS_API __declspec(dllimport)
#  endif
#else
#  define QWEN3_TTS_API __attribute__((visibility("default")))
#endif

typedef struct qwen3_tts_context qwen3_tts_context_t;

typedef struct {
    int32_t max_audio_tokens;
    float temperature;
    float top_p;
    int32_t top_k;
    int32_t n_threads;
    bool print_progress;
    bool print_timing;
    float repetition_penalty;
    int32_t language_id;
} qwen3_tts_params_t;

typedef struct {
    float* audio;
    int32_t audio_len;
    int32_t sample_rate;
    bool success;
    char* error_msg;
    int64_t t_total_ms;
} qwen3_tts_result_t;

typedef void (*qwen3_tts_progress_callback)(int tokens_generated, int max_tokens, void* user_data);

QWEN3_TTS_API qwen3_tts_context_t* qwen3_tts_init();
QWEN3_TTS_API void qwen3_tts_free(qwen3_tts_context_t* ctx);

QWEN3_TTS_API bool qwen3_tts_load_models(qwen3_tts_context_t* ctx, const char* model_dir);

QWEN3_TTS_API qwen3_tts_result_t qwen3_tts_synthesize(
    qwen3_tts_context_t* ctx, 
    const char* text, 
    qwen3_tts_params_t params
);

QWEN3_TTS_API qwen3_tts_result_t qwen3_tts_synthesize_with_voice(
    qwen3_tts_context_t* ctx, 
    const char* text, 
    const char* reference_audio, 
    qwen3_tts_params_t params
);

QWEN3_TTS_API void qwen3_tts_free_result(qwen3_tts_result_t result);

QWEN3_TTS_API void qwen3_tts_set_progress_callback(
    qwen3_tts_context_t* ctx, 
    qwen3_tts_progress_callback callback, 
    void* user_data
);

#ifdef __cplusplus
}
#endif

#endif // QWEN3_TTS_C_H
