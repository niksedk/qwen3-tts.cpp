#pragma once

#include <cstdint>
#include <string>

namespace qwen3_tts {
namespace transformer_internal {

struct debug_trace_config {
    bool enabled = false;
    std::string dir;
    int32_t max_frames = 1;
    int32_t max_code_steps = 15;
};

std::string normalize_speaker_name(const std::string & name);
int32_t parse_env_i32(const char * name, int32_t default_value, int32_t min_value, int32_t max_value);
debug_trace_config init_debug_trace_config();
const debug_trace_config & get_debug_trace_config();
bool debug_trace_should_dump_frame(const debug_trace_config & cfg, int32_t frame);

} // namespace transformer_internal
} // namespace qwen3_tts
