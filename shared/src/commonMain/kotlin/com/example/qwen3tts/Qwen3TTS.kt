package com.example.qwen3tts

/**
 * Common interface for Qwen3 TTS across all platforms.
 */
expect class Qwen3TTS() {
    fun loadModels(modelDir: String): Boolean
    fun synthesize(text: String): Result
    fun close()

    class Result(
        val audio: FloatArray?,
        val sampleRate: Int,
        val success: Boolean,
        val errorMsg: String?,
        val timeMs: Long
    )
}
