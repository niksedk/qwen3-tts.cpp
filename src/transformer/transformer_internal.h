#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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
void debug_trace_write_bin(const debug_trace_config & cfg,
                           const std::string & name,
                           const float * data,
                           size_t count,
                           const char * dtype,
                           const std::vector<int64_t> & shape);
void debug_trace_write_bin(const debug_trace_config & cfg,
                           const std::string & name,
                           const int32_t * data,
                           size_t count,
                           const char * dtype,
                           const std::vector<int64_t> & shape);
void debug_trace_write_text_line(const debug_trace_config & cfg, const std::string & line);

} // namespace transformer_internal
} // namespace qwen3_tts
