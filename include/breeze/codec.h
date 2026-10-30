#pragma once

#include "breeze/model.h"

#include <string>
#include <memory>
#include <vector>

namespace breeze {

struct CodecStreamState;
struct MimiCodec {
    BreezeModel * m = nullptr;
    std::unique_ptr<Graph> cached_graph;
    std::vector<ggml_tensor *> cached_ids;
    ggml_tensor * cached_audio = nullptr;
    int cached_T = 0, cached_cb = 0;
    bool cached_fast = false, cached_precise = false;
    void clear_cache() { cached_graph.reset(); cached_ids.clear(); cached_audio = nullptr; }
    std::shared_ptr<CodecStreamState> stream_state, primed_state;
    std::vector<int> primed_codes;
    bool primed_fast = false, primed_precise = false;
    void trim_workspace();
    void reset_stream() { stream_state.reset(); clear_cache(); trim_workspace(); }
    void init(BreezeModel & model) { clear_cache(); reset_stream(); primed_state.reset(); primed_codes.clear(); m = &model; }
    std::vector<float> decode_streaming(const std::vector<int> & codes, int frames);
    void prime_stream(const std::vector<int> & reference);

    // codes frame-major [f * n_cb + cb]; n_cb 0 means the full set. fewer codebooks drops detail,
    // 1 leaves only the semantic stage
    std::vector<float> decode(const std::vector<int> & codes, int n_frames, int n_cb = 0);
    // audio waveform in, returns codes frame-major, sets n_frames
    std::vector<int> encode(const std::vector<float> & audio, int & n_frames);
};

namespace codec_detail {

ggml_tensor * conv1d_causal(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                            ggml_tensor * x, int stride, int dilation);
ggml_tensor * convtr1d_causal(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                              ggml_tensor * x, int stride);
ggml_tensor * depthwise1d_causal(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                                 ggml_tensor * x, int kernel);
ggml_tensor * resnet_block(ggml_context * ctx, BreezeModel & m, const std::string & prefix, ggml_tensor * x);
ggml_tensor * seanet_encoder(ggml_context * ctx, BreezeModel & m, ggml_tensor * x);
ggml_tensor * mimi_transformer(ggml_context * ctx, BreezeModel & m, Graph & g, ggml_tensor * x,
                               const std::string & prefix, int seq_len);
ggml_tensor * vocoder_transformer(ggml_context * ctx, BreezeModel & m, Graph & g, ggml_tensor * x, int seq_len);
ggml_tensor * vocoder_decode(ggml_context * ctx, BreezeModel & m, Graph & g,
                             const std::vector<int> & codes, int n_codebooks, int seq_len);

}

}
