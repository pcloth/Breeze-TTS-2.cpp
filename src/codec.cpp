#include "breeze/codec.h"
#include "codec_stream.h"

#include <cfloat>
#include <cstdlib>
#include <stdexcept>
#include <cmath>
#include <string>

namespace breeze {

using namespace codec_detail;
static bool precise_codec() {
    const char * value = std::getenv("BREEZE_CODEC_F32_ACC");
    return value && std::string(value)=="1";
}

static ggml_tensor * transpose_cont(ggml_context * ctx, ggml_tensor * x) {
    return ggml_cont(ctx, ggml_transpose(ctx, x));
}

std::vector<float> MimiCodec::decode(const std::vector<int> & codes, int T, int n_cb) {
    if (n_cb <= 0) n_cb = m->cfg.num_codebooks;
    const char * option = std::getenv("BREEZE_CODEC_CACHE");
    if (option && std::string(option) == "1") {
        const char * fast_opt = std::getenv("BREEZE_FASTCONV");
        bool fast = fast_opt && std::string(fast_opt) == "1";
        // One exact-shape slot only: never cache unbounded 4GB GPU scratch.
        if (!cached_graph || cached_T != T || cached_cb != n_cb || cached_fast != fast || cached_precise != precise_codec()) {
            clear_cache();
            // The shared allocator can retain a previous vocoder-sized arena.
            // Release it before owning a cached arena; otherwise 4GB WDDM
            // devices spill to host memory even though there is only one slot.
            ggml_gallocr_free(m->backend.alloc);
            m->backend.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend.backend));
            cached_graph = std::make_unique<Graph>(32768);
            auto & g = *cached_graph;
            g.precise_matmul = precise_codec();
            auto * x = vocoder_decode(g.ctx, *m, g, codes, n_cb, T);
            cached_audio = ggml_cont(g.ctx, ggml_reshape_1d(g.ctx, x, x->ne[0]));
            // Quantizer inputs precede the positional and mask constants.
            for (int i = 0; i < n_cb; ++i) {
                auto * ids = g.pending.at(i).first;
                if (ids->type != GGML_TYPE_I32 || ggml_nelements(ids) != T)
                    throw std::runtime_error("codec graph input layout changed");
                cached_ids.push_back(ids);
            }
            g.prepare(m->backend, cached_audio);
            cached_T = T; cached_cb = n_cb; cached_fast = fast; cached_precise = precise_codec();
        }
        std::vector<int32_t> ids(T);
        for (int cb = 0; cb < n_cb; ++cb) {
            for (int t = 0; t < T; ++t) ids[t] = codes[(size_t)t * n_cb + cb];
            ggml_backend_tensor_set(cached_ids[cb], ids.data(), 0, ids.size()*sizeof(int32_t));
        }
        cached_graph->replay(m->backend);
        return tensor_to_f32(cached_audio);
    }
    clear_cache();
    Graph g(32768);
    g.precise_matmul = precise_codec();
    ggml_tensor * x = vocoder_decode(g.ctx, *m, g, codes, n_cb, T);
    ggml_tensor * audio = ggml_cont(g.ctx, ggml_reshape_1d(g.ctx, x, x->ne[0]));
    g.compute(m->backend, audio);
    return tensor_to_f32(audio);
}

void MimiCodec::trim_workspace() {
    if (m && m->backend.backend) {
        ggml_gallocr_free(m->backend.alloc);
        m->backend.alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend.backend));
    }
}

std::vector<float> MimiCodec::decode_streaming(const std::vector<int> & codes, int T) {
    if (T < 1 || codes.size() != (size_t)T*m->cfg.num_codebooks)
        throw std::runtime_error("invalid incremental codec input shape");
    clear_cache();
    if (!stream_state) stream_state = std::make_shared<CodecStreamState>();
    Graph g(32768);
    g.precise_matmul = precise_codec();
    auto & state = *stream_state;
    state.graph = &g;
    // <=8 columns selects a different Vulkan matvec path. Causal right
    // padding keeps the GEMM path stable; no padded sample or state is emitted.
    int compute_T = m->backend.is_gpu && T <= 8 ? 16 : T;
    std::vector<int> padded = codes;
    padded.resize((size_t)compute_T*m->cfg.num_codebooks,0);
    state.valid_frames = T; state.padded_frames = compute_T;
    CodecStreamScope scope(&state);
    try {
        auto * x = vocoder_decode(g.ctx, *m, g, padded, m->cfg.num_codebooks, compute_T);
        auto * audio = ggml_cont(g.ctx, ggml_reshape_1d(g.ctx, x, x->ne[0]));
        state.pack();
        g.compute(m->backend, audio);
        auto result = tensor_to_f32(audio);
        if (result.size() != (size_t)compute_T*m->cfg.samples_per_frame)
            throw std::runtime_error("incremental decoder lost PCM samples");
        result.resize((size_t)T*m->cfg.samples_per_frame);
        state.commit(T); state.graph = nullptr;
        return result;
    } catch (...) { state.pending.clear(); state.graph = nullptr; reset_stream(); throw; }
}

void MimiCodec::prime_stream(const std::vector<int> & reference) {
    const int nc = m->cfg.num_codebooks;
    const int keep = std::min(m->cfg.voc.sliding_window+16, (int)reference.size()/nc);
    if (!keep) { reset_stream(); return; }
    std::vector<int> tail(reference.end()-(size_t)keep*nc,reference.end());
    const char * flag = std::getenv("BREEZE_FASTCONV");
    bool fast = flag && std::string(flag)=="1";
    if (!primed_state || primed_codes!=tail || primed_fast!=fast || primed_precise!=precise_codec()) {
        reset_stream();
        decode_streaming(tail,keep);
        // Reference prefill needs a larger arena than steady-state chunks.
        // Do not retain a large/host-spilled arena for every small decode.
        trim_workspace();
        // Only host-held histories survive; no Graph/tensor pointer is retained.
        primed_state = std::make_shared<CodecStreamState>(*stream_state);
        primed_codes = std::move(tail); primed_fast = fast; primed_precise = precise_codec();
    }
    stream_state = std::make_shared<CodecStreamState>(*primed_state);
}

static int nearest(const std::vector<float> & book, const float * v, int dim, int n) {
    int best = 0;
    float best_d = FLT_MAX;
    for (int c = 0; c < n; c++) {
        const float * b = &book[(size_t) c * dim];
        float d = 0.0f;
        for (int k = 0; k < dim; k++) {
            float diff = v[k] - b[k];
            d += diff * diff;
        }
        if (d < best_d) { best_d = d; best = c; }
    }
    return best;
}

static std::vector<float> read_book(BreezeModel & m, const std::string & name) {
    return tensor_to_f32(m.w(name));
}

std::vector<int> MimiCodec::encode(const std::vector<float> & audio, int & n_frames) {
    const int nc = m->cfg.num_codebooks;
    const int n_sem = m->cfg.codec.num_semantic;
    const int n_ac = nc - n_sem;
    const int dim = m->cfg.codec.codebook_dim;
    const int book = m->cfg.codec_codebook_size;

    Graph g(16384);
    ggml_tensor * x = g.input_f32(audio, (int) audio.size(), 1); // [L, 1]
    x = seanet_encoder(g.ctx, *m, x);                            // [T, hidden]
    x = transpose_cont(g.ctx, x);                                // [hidden, T]
    x = mimi_transformer(g.ctx, *m, g, x, "codec.enct", (int) x->ne[1]);
    x = transpose_cont(g.ctx, x);                                // [T, hidden]
    x = conv1d_causal(g.ctx, m->w("codec.downsample.conv.weight"), nullptr, x, 2, 1); // [T2, hidden]
    ggml_tensor * emb = transpose_cont(g.ctx, x);                // [hidden, T2]
    ggml_tensor * rs = linear(g.ctx, m->w("codec.sq.in_proj.weight"), emb); // [dim, T2]
    ggml_tensor * ra = linear(g.ctx, m->w("codec.aq.in_proj.weight"), emb); // [dim, T2]
    ggml_set_output(rs);
    g.write(rs);
    g.compute(m->backend, ra);

    std::vector<float> rs_h = tensor_to_f32(rs);
    std::vector<float> ra_h = tensor_to_f32(ra);
    const int T = (int) ra->ne[1];
    n_frames = T;

    std::vector<float> sbook = read_book(*m, "codec.sq.0.embed");
    std::vector<std::vector<float>> abook(n_ac);
    for (int i = 0; i < n_ac; i++) abook[i] = read_book(*m, "codec.aq." + std::to_string(i) + ".embed");

    std::vector<int> codes((size_t) T * nc);
    std::vector<float> res(dim);
    for (int f = 0; f < T; f++) {
        codes[(size_t) f * nc + 0] = nearest(sbook, &rs_h[(size_t) f * dim], dim, book);
        for (int k = 0; k < dim; k++) res[k] = ra_h[(size_t) f * dim + k];
        for (int i = 0; i < n_ac; i++) {
            int c = nearest(abook[i], res.data(), dim, book);
            codes[(size_t) f * nc + (n_sem + i)] = c;
            const float * b = &abook[i][(size_t) c * dim];
            for (int k = 0; k < dim; k++) res[k] -= b[k];
        }
    }
    return codes;
}

}
