// Fixed-code validation harness for the pinned Breeze runtime. Apache-2.0.
#include "breeze/generation.h"
#include "breeze/audio.h"
#include "breeze/voice.h"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
using namespace breeze;
static void flag(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
static double millis(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
}
static bool parity_failed=false;
static void check(bool ok,const char *why) { if(!ok) throw std::runtime_error(why); }
static void save_codes(const std::string &path,const std::vector<int> &codes,int nc) {
    std::ofstream f(path,std::ios::binary); int header[2]={nc,(int)codes.size()/nc};
    f.write((char*)header,sizeof(header));f.write((char*)codes.data(),codes.size()*sizeof(int)); check(bool(f),"codes write failed");
}
static std::vector<int> read_codes(const std::string &path,int nc) {
    std::ifstream f(path,std::ios::binary);int header[2]={};f.read((char*)header,sizeof(header));
    check(f && header[0]==nc && header[1]>0 && header[1]<100000,"invalid code fixture");
    std::vector<int> v((size_t)nc*header[1]);f.read((char*)v.data(),v.size()*sizeof(int));check(bool(f),"truncated fixture"); return v;
}
static std::vector<float> windowed(MimiCodec &codec,const std::vector<int> &codes,int nc,int max_chunk,
    const std::vector<int> &prefix={}) {
    const int context=codec.m->cfg.voc.sliding_window+16, spf=codec.m->cfg.samples_per_frame;
    const int prefix_n=std::min(context,(int)prefix.size()/nc), total=(int)codes.size()/nc;
    std::vector<int> all(prefix.end()-(size_t)prefix_n*nc,prefix.end());all.insert(all.end(),codes.begin(),codes.end());
    std::vector<float> audio;int emitted=0, chunk=4;
    while(emitted<total) {
        int n=std::min(chunk,total-emitted),start=prefix_n+emitted,lo=std::max(0,start-context);
        std::vector<int> sub(all.begin()+(size_t)lo*nc,all.begin()+(size_t)(start+n)*nc);
        auto part=codec.decode(sub,start+n-lo);int skip=(start-lo)*spf;
        check((int)part.size()==(start+n-lo)*spf,"decode changed sample count");
        audio.insert(audio.end(),part.begin()+skip,part.end());emitted+=n;chunk=std::min(chunk+chunk/3+1,max_chunk);
    }
    return audio;
}
static std::vector<float> incremental(MimiCodec &codec,const std::vector<int>&codes,int nc,int step,const std::vector<int>&prefix={}) {
    codec.reset_stream();if(!prefix.empty())codec.prime_stream(prefix);
    std::vector<float> out;
    for(int t=0;t<(int)codes.size()/nc;t+=step) {
        int n=std::min(step,(int)codes.size()/nc-t);
        std::vector<int> part(codes.begin()+(size_t)t*nc,codes.begin()+(size_t)(t+n)*nc);
        auto audio=codec.decode_streaming(part,n);out.insert(out.end(),audio.begin(),audio.end());
    }
    return out;
}
static double compare(const std::vector<float>&a,const std::vector<float>&b,const std::string &name) {
    check(a.size()==b.size(),"output duration changed");double power=0,error=0,maxd=0;
    for(size_t i=0;i<a.size();i++) {
        check(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite output");
        double d=double(a[i])-b[i];error+=d*d;power+=double(a[i])*a[i];maxd=std::max(maxd,std::abs(d));
    }
    std::cout<<name<<" samples="<<a.size()<<" max_abs="<<maxd<<" snr_db="<<(error?10*log10(power/error):999)<<std::endl;
    return maxd;
}
int main(int argc,char**argv) { try {
    check(argc>=5,"usage: probe model capture|replay fixture-or-text output-dir [voice.breeze]");
    std::filesystem::create_directories(argv[4]); std::string out=argv[4];
    BreezeModel m;check(m.load(argv[1],true),"model load failed");std::cout<<"backend="<<m.backend.name()<<std::endl;
    // Keep model alive until all codec-owned graph caches have been released.
    { MimiCodec codec; codec.init(m);const int nc=m.cfg.num_codebooks;
    if(std::string(argv[2])=="capture") {
        check(argc>=6,"capture needs reference voice");Voice voice;check(load_voice(argv[5],voice),"voice load failed");
        std::ifstream f(argv[3],std::ios::binary);std::string text((std::istreambuf_iterator<char>(f)),{});check(!text.empty(),"empty text");
        GenRequest req;req.text=text;req.ref_codes=voice.codes;req.ref_frames=voice.frames;req.ref_text=voice.text;
        req.max_new_tokens = argc>=7 ? std::stoi(argv[6]) : 80;
        req.instruction="";req.chunk_first=4;req.chunk_max=40;req.split_chars=600;req.pace=false;
        std::vector<int> codes;req.on_codes=[&](const auto &v){codes.insert(codes.end(),v.begin(),v.end());};
        std::vector<float> audio;GenTimings tm;
        flag("BREEZE_FASTCONV","0");
        generate(m,codec,req,[&](const float*p,int n){audio.insert(audio.end(),p,p+n);return true;},&tm);
        save_codes(out+"/generated.codes",codes,nc);save_codes(out+"/reference.codes",voice.codes,nc);
        check(write_wav(out+"/capture.wav",audio,m.cfg.sample_rate),"wav write failed");
        std::cout<<"captured frames="<<codes.size()/nc<<" first_ms="<<tm.first_audio<<std::endl;
    } else {
        auto codes=read_codes(argv[3],nc);
        flag("BREEZE_FASTCONV","0");auto t=std::chrono::steady_clock::now();auto base=windowed(codec,codes,nc,40);
        std::cout<<"baseline_ms="<<millis(t)<<std::endl;write_wav(out+"/baseline.wav",base,m.cfg.sample_rate);
        flag("BREEZE_FASTCONV","1");t=std::chrono::steady_clock::now();auto fast=windowed(codec,codes,nc,40);
        std::cout<<"fastconv_ms="<<millis(t)<<std::endl;compare(base,fast,"fastconv");write_wav(out+"/fastconv.wav",fast,m.cfg.sample_rate);
        flag("BREEZE_CODEC_CACHE","1");
        t=std::chrono::steady_clock::now();auto cached=windowed(codec,codes,nc,40);
        std::cout<<"cached_windows_ms="<<millis(t)<<std::endl;compare(fast,cached,"cache_windows");
        // Repeated exact-shape replay, interleaved with another allocator.
        int test_T=std::min(64,(int)codes.size()/nc);
        std::vector<int> fixed(codes.begin(),codes.begin()+(size_t)test_T*nc);
        flag("BREEZE_CODEC_CACHE","0");auto expected=codec.decode(fixed,test_T);
        flag("BREEZE_CODEC_CACHE","1");
        for(int pass=0;pass<3;pass++) {
            Graph scratch(64);auto *v=scratch.input_f32(std::vector<float>(4096,3),4096);
            scratch.compute(m.backend,ggml_sqr(scratch.ctx,v));
            t=std::chrono::steady_clock::now();auto result=codec.decode(fixed,test_T);
            std::cout<<"cache_replay_"<<pass<<"_ms="<<millis(t)<<std::endl;compare(expected,result,"cache_replay");
        }
        fixed[0]=(fixed[0]+1)%m.cfg.codec_codebook_size;
        auto changed=codec.decode(fixed,test_T);
        flag("BREEZE_CODEC_CACHE","0");auto changed_expected=codec.decode(fixed,test_T);
        compare(changed_expected,changed,"cache_changed_input");
        flag("BREEZE_CODEC_CACHE","0");
        flag("BREEZE_CODEC_F32_ACC","1");
        auto precise_windowed=windowed(codec,codes,nc,40);
        auto full=codec.decode(codes,(int)codes.size()/nc);
        compare(full,precise_windowed,"precise_windows_vs_full");
        compare(full,fast,"legacy_windows_vs_full");
        write_wav(out+"/full.wav",full,m.cfg.sample_rate);
        for(int step: {4,13,40}) {
            t=std::chrono::steady_clock::now();auto result=incremental(codec,codes,nc,step);
            std::cout<<"incremental_"<<step<<"_ms="<<millis(t)<<std::endl;
            if(compare(full,result,"incremental_full_parity")>=0.003)parity_failed=true;
            compare(fast,result,"incremental_vs_legacy_windows");
            write_wav(out+"/incremental-"+std::to_string(step)+".wav",result,m.cfg.sample_rate);
        }
        if(argc>=6) {
            auto prefix=read_codes(argv[5],nc);flag("BREEZE_FASTCONV","0");t=std::chrono::steady_clock::now();
            auto with_ref=windowed(codec,codes,nc,40,prefix);
            std::cout<<"reference_context_ms="<<millis(t)<<std::endl;compare(base,with_ref,"reference_context");
            write_wav(out+"/reference-context.wav",with_ref,m.cfg.sample_rate);
            flag("BREEZE_FASTCONV","1");
            int ctx=m.cfg.voc.sliding_window+16;
            int keep=std::min(ctx,(int)prefix.size()/nc);
            int n=std::min(40,(int)codes.size()/nc);
            std::vector<int> joined(prefix.end()-(size_t)keep*nc,prefix.end());
            joined.insert(joined.end(),codes.begin(),codes.begin()+(size_t)n*nc);
            auto full_prefix=codec.decode(joined,keep+n);
            full_prefix.erase(full_prefix.begin(),full_prefix.begin()+(size_t)keep*m.cfg.samples_per_frame);
            t=std::chrono::steady_clock::now();auto result=incremental(codec,codes,nc,4,prefix);
            std::cout<<"incremental_reference_cold_ms="<<millis(t)<<std::endl;
            std::vector<float> head(result.begin(),result.begin()+full_prefix.size());
            if(compare(full_prefix,head,"incremental_reference_parity")>=0.003)parity_failed=true;
            write_wav(out+"/incremental-reference.wav",result,m.cfg.sample_rate);
            t=std::chrono::steady_clock::now();auto repeated=incremental(codec,codes,nc,4,prefix);
            std::cout<<"incremental_reference_warm_ms="<<millis(t)<<std::endl;
            check(compare(result,repeated,"reference_state_reuse")==0,"reference state reused incorrectly");
        }
    } } m.free();check(!parity_failed,"incremental numerical tolerance failed; diagnostics retained");return 0;
} catch(const std::exception&e) {std::cerr<<e.what()<<std::endl;return 1;} }
