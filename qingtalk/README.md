# QingTalk streaming codec changes

Based on Breeze-TTS-2.cpp v0.1.3, commit 2904b9ad126c07363e8a6c9afa08f73e3309075b.
Upstream: https://github.com/HoppouAI/Breeze-TTS-2.cpp.git

This checkout contains all native runtime changes, including FASTCONV, bounded graph caching,
stateful causal codec, reference-context onset correction, and GPU startup warmup.
The three codec helper files are maintained here; CMake uses this directory by default.
No QingTalkServer source directory is needed to compile the runtime.
Code retains the repository Apache-2.0 license.

On the current Windows machine, build/package via the sibling QingTalkServer project:

```powershell
cd ../QingTalkServer
./scripts/build-breeze-native.ps1
```

That script supplies locally cached Vulkan headers/compiler tools and puts artifacts in
QingTalkServer/runtime/breeze-native. Upstream CMake options remain available for standalone builds.
The ggml submodule remains pinned at 36da57138425487184aa1da2eee2cde155909c6f.

Local branch: qingtalk-streaming. The original remote has been renamed upstream.
After creating your GitHub fork, add its URL as origin, commit all local changes (including
this directory), and push qingtalk-streaming. This migration has not committed or pushed anything.
