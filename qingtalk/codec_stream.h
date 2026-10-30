#pragma once
#include "breeze/codec.h"
#include <map>
#include <string>
#include <vector>
namespace breeze {
// CPU-held tails are small; no second copy of model weights or unbounded GPU cache.
struct CodecStreamState {
    struct History { std::vector<float> data; int n0=0,n1=0,n2=0; };
    struct Pending { History *history; ggml_tensor *tensor; };
    Graph *graph = nullptr;
    ggml_tensor *packed = nullptr;
    int frames = 0;
    int valid_frames = 0, padded_frames = 0;
    std::map<std::string,History> histories;
    std::vector<Pending> pending;
    ggml_tensor * prefix(ggml_context *,ggml_tensor *,ggml_tensor *,int,const char *);
    void remember(const std::string &,ggml_tensor *);
    void pack();
    void commit(int);
    ggml_tensor * conv(ggml_context *,ggml_tensor *,ggml_tensor *,ggml_tensor *,int);
    ggml_tensor * convtr(ggml_context *,ggml_tensor *,ggml_tensor *,ggml_tensor *,int);
    ggml_tensor * depthwise(ggml_context *,ggml_tensor *,ggml_tensor *,ggml_tensor *,int);
    ggml_tensor * kv(ggml_context *,ggml_tensor *,int,bool,int);
};
extern thread_local CodecStreamState * active_codec_stream;
struct CodecStreamScope {
    CodecStreamState *previous;
    explicit CodecStreamScope(CodecStreamState *s):previous(active_codec_stream){active_codec_stream=s;}
    ~CodecStreamScope(){active_codec_stream=previous;}
};
}
