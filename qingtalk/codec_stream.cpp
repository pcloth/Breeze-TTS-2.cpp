// Incremental causal codec state. Apache-2.0.
#include "codec_stream.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
namespace breeze {
thread_local CodecStreamState *active_codec_stream=nullptr;
static std::string key_for(ggml_tensor*w,const char *kind) {
    return std::string(kind)+std::to_string(reinterpret_cast<std::uintptr_t>(w));
}
void CodecStreamState::remember(const std::string &key,ggml_tensor *tensor) {
    auto *saved=ggml_cont(graph->ctx,tensor);
    ggml_set_output(saved);graph->write(saved);
    pending.push_back({&histories[key],saved});
}
void CodecStreamState::pack() {
    std::vector<ggml_tensor*> level;
    for(auto &p:pending) level.push_back(ggml_reshape_1d(graph->ctx,p.tensor,ggml_nelements(p.tensor)));
    // Balanced concatenation: a single device-to-host transfer, O(log N)
    // copies per state rather than a separate synchronous transfer per layer.
    while(level.size()>1) {
        std::vector<ggml_tensor*> next;
        for(size_t i=0;i<level.size();i+=2)
            next.push_back(i+1<level.size()?ggml_concat(graph->ctx,level[i],level[i+1],0):level[i]);
        level=std::move(next);
    }
    packed=level.empty()?nullptr:level.front();
    if(packed){ggml_set_output(packed);graph->write(packed);}
}
void CodecStreamState::commit(int count) {
    auto values=packed?tensor_to_f32(packed):std::vector<float>();size_t offset=0;
    for(auto &p:pending) {
        size_t n=(size_t)ggml_nelements(p.tensor);
        p.history->data.assign(values.begin()+offset,values.begin()+offset+n);offset+=n;
        p.history->n0=(int)p.tensor->ne[0];p.history->n1=(int)p.tensor->ne[1];p.history->n2=(int)p.tensor->ne[2];
    }
    if(offset!=values.size())throw std::runtime_error("packed codec state size mismatch");
    packed=nullptr;pending.clear();frames+=count;
}
ggml_tensor *CodecStreamState::prefix(ggml_context *ctx,ggml_tensor*w,ggml_tensor*x,int n,const char *kind) {
    if(!n)return x;
    auto key=key_for(w,kind);auto &h=histories[key];int C=(int)x->ne[1];
    if(h.data.empty()){h.data.assign((size_t)n*C,0);h.n0=n;h.n1=C;h.n2=1;}
    if(h.n0!=n||h.n1!=C)throw std::runtime_error("codec history shape changed without reset");
    auto *past=graph->input_f32(h.data,n,C);
    auto *joined=ggml_concat(ctx,past,x,0);
    // Ignore padded future frames when retaining causal histories.
    const int64_t valid=x->ne[0]*valid_frames/padded_frames;
    const int64_t offset=valid*joined->nb[0];
    remember(key,ggml_view_2d(ctx,joined,n,C,joined->nb[1],offset));
    return joined;
}
ggml_tensor *CodecStreamState::conv(ggml_context *ctx,ggml_tensor*w,ggml_tensor*b,ggml_tensor*x,int dilation) {
    int n=((int)w->ne[0]-1)*dilation;
    auto *joined=prefix(ctx,w,x,n,"conv");
    auto *y=ggml_conv_1d(ctx,w,ggml_cont(ctx,joined),1,0,dilation);
    if(y->ne[0]!=x->ne[0])throw std::runtime_error("incremental conv output length mismatch");
    if(b)y=ggml_add(ctx,y,ggml_reshape_2d(ctx,b,1,b->ne[0]));return y;
}
ggml_tensor *CodecStreamState::convtr(ggml_context *ctx,ggml_tensor*w,ggml_tensor*b,ggml_tensor*x,int stride) {
    int n=((int)w->ne[0]-1)/stride;
    auto *joined=prefix(ctx,w,x,n,"convtr");
    auto *all=ggml_conv_transpose_1d(ctx,w,ggml_cont(ctx,joined),stride,0,1);
    int64_t keep=x->ne[0]*stride, offset=(int64_t)n*stride*all->nb[0];
    auto *y=ggml_cont(ctx,ggml_view_2d(ctx,all,keep,all->ne[1],all->nb[1],offset));
    if(b)y=ggml_add(ctx,y,ggml_reshape_2d(ctx,b,1,b->ne[0]));return y;
}
ggml_tensor *CodecStreamState::depthwise(ggml_context *ctx,ggml_tensor*w,ggml_tensor*b,ggml_tensor*x,int K) {
    auto *joined=prefix(ctx,w,x,K-1,"dw");
    int64_t L=x->ne[0],C=x->ne[1];ggml_tensor *acc=nullptr;
    for(int k=0;k<K;k++) {
        auto *seg=ggml_cont(ctx,ggml_view_2d(ctx,joined,L,C,joined->nb[1],(size_t)k*joined->nb[0]));
        auto *wk=ggml_cont(ctx,ggml_transpose(ctx,ggml_view_2d(ctx,w,C,1,w->nb[1],(size_t)k*w->nb[1])));
        auto *product=ggml_mul(ctx,seg,wk);acc=acc?ggml_add(ctx,acc,product):product;
    }
    if(b)acc=ggml_add(ctx,acc,ggml_reshape_2d(ctx,b,1,b->ne[0]));return acc;
}
ggml_tensor *CodecStreamState::kv(ggml_context *ctx,ggml_tensor*cur,int layer,bool is_key,int window) {
    if(window<1)throw std::runtime_error("incremental codec requires finite attention window");
    std::string key=(is_key?"key":"value")+std::to_string(layer);auto &h=histories[key];
    ggml_tensor *all=cur;
    if(!h.data.empty()) {
        if(h.n0!=cur->ne[0]||h.n1!=cur->ne[1])throw std::runtime_error("KV dimensions changed without reset");
        auto *past=graph->input_f32(h.data,h.n0,h.n1,h.n2);all=ggml_concat(ctx,past,cur,2);
    }
    const int old_frames=h.data.empty()?0:h.n2;
    const int end=old_frames+valid_frames;
    int keep=std::min(window,end);
    remember(key,ggml_view_3d(ctx,all,all->ne[0],all->ne[1],keep,all->nb[1],all->nb[2],
        (size_t)(end-keep)*all->nb[2]));return all;
}
}
