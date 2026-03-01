#include <jni.h>
#include "qwen3_tts_c.h"
#include <string>
#include <vector>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_example_qwen3tts_Qwen3TTS_nativeInit(JNIEnv* env, jobject thiz) {
    return reinterpret_cast<jlong>(qwen3_tts_init());
}

JNIEXPORT void JNICALL Java_com_example_qwen3tts_Qwen3TTS_nativeFree(JNIEnv* env, jobject thiz, jlong ctx_ptr) {
    qwen3_tts_free(reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr));
}

JNIEXPORT jboolean JNICALL Java_com_example_qwen3tts_Qwen3TTS_nativeLoadModels(JNIEnv* env, jobject thiz, jlong ctx_ptr, jstring model_dir) {
    const char* c_model_dir = env->GetStringUTFChars(model_dir, nullptr);
    bool result = qwen3_tts_load_models(reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr), c_model_dir);
    env->ReleaseStringUTFChars(model_dir, c_model_dir);
    return result;
}

JNIEXPORT jobject JNICALL Java_com_example_qwen3tts_Qwen3TTS_nativeSynthesize(JNIEnv* env, jobject thiz, jlong ctx_ptr, jstring text, jobject params) {
    // Note: In a real implementation, we would extract fields from the 'params' object.
    // For brevity, we'll use default params here or assume a simplified bridge.
    qwen3_tts_params_t c_params = {4096, 0.9f, 1.0f, 50, 4, false, true, 1.05f, 2050};
    
    const char* c_text = env->GetStringUTFChars(text, nullptr);
    qwen3_tts_result_t c_result = qwen3_tts_synthesize(reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr), c_text, c_params);
    env->ReleaseStringUTFChars(text, c_text);

    // Create a result object in Kotlin
    jclass result_class = env->FindClass("com/example/qwen3tts/Qwen3TTS$Result");
    jmethodID constructor = env->GetMethodID(result_class, "<init>", "([FIZLjava/lang/String;J)V");

    jfloatArray audio_array = nullptr;
    if (c_result.audio_len > 0) {
        audio_array = env->NewFloatArray(c_result.audio_len);
        env->SetFloatArrayRegion(audio_array, 0, c_result.audio_len, c_result.audio);
    }

    jstring error_msg = nullptr;
    if (c_result.error_msg) {
        error_msg = env->NewStringUTF(c_result.error_msg);
    }

    jobject result_obj = env->NewObject(result_class, constructor, audio_array, c_result.sample_rate, c_result.success, error_msg, c_result.t_total_ms);

    qwen3_tts_free_result(c_result);
    return result_obj;
}

}
