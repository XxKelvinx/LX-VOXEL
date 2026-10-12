#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "dsp/cpu_dispatch.hpp"

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
extern "C" int lxvoxel_encoder_sse41_main(int argc,char** argv);
extern "C" int lxvoxel_encoder_avx2_main(int argc,char** argv);
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
extern "C" int lxvoxel_encoder_neon_main(int argc,char** argv);
#endif

namespace {

int dispatch(int argc,char** argv){
    const lx_cpu::Path path=lx_cpu::best_path();
    const char* path_name=lx_cpu::name(path);
#if defined(_WIN32)
    (void)_putenv_s("LXVOXEL_CPU_PATH",path_name);
#else
    (void)setenv("LXVOXEL_CPU_PATH",path_name,1);
#endif
    if(argc==2 && std::strcmp(argv[1],"--cpu-info")==0){
        std::cout<<path_name<<"\n";
        return 0;
    }
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    return path==lx_cpu::Path::AVX2_FMA
        ? lxvoxel_encoder_avx2_main(argc,argv)
        : lxvoxel_encoder_sse41_main(argc,argv);
#else
    return lxvoxel_encoder_neon_main(argc,argv);
#endif
}

} // namespace

#if defined(_WIN32)
int wmain(int argc,wchar_t** wide_argv){
    std::vector<std::string> utf8_args;
    std::vector<char*> argv;
    utf8_args.reserve(size_t(argc));
    argv.reserve(size_t(argc));
    for(int i=0;i<argc;++i){
        const int bytes=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,
            wide_argv[i],-1,nullptr,0,nullptr,nullptr);
        if(bytes<=0){std::cerr<<"error: invalid Unicode command line\n";return 2;}
        std::string value(size_t(bytes),'\0');
        if(WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,wide_argv[i],-1,
            value.data(),bytes,nullptr,nullptr)<=0){
            std::cerr<<"error: cannot convert Unicode command line\n";
            return 2;
        }
        value.resize(size_t(bytes-1));
        utf8_args.push_back(std::move(value));
    }
    for(auto& value:utf8_args) argv.push_back(value.data());
#else
int main(int argc,char** argv){
#endif
    try {
#if defined(_WIN32)
        return dispatch(argc,argv.data());
#else
        return dispatch(argc,argv);
#endif
    }catch(const std::exception& e){
        std::cerr<<"error: "<<e.what()<<"\n";
        return 2;
    }catch(...){
        std::cerr<<"error: unknown fatal exception\n";
        return 3;
    }
}
