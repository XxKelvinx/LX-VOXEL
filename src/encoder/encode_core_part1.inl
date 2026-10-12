Result encode(const float*stereo,size_t ns,float bpc,std::vector<float>*recon=nullptr,vox_beta::Writer* vox=nullptr,float physical_target_kbps=0.0f,float allocation_target_kbps=0.0f,size_t max_superframes=std::numeric_limits<size_t>::max(),double startup_pilot_entropy_bits=0.0,double startup_pilot_overhead_bits=0.0,PilotStats* pilot_stats=nullptr,const PilotStats* startup_pilots=nullptr){
    // R220 low-rate profile.  Keep the frozen 20-kHz/48-ERB syntax, but at
    // <140 kbit/s encode only source support below 16.59 kHz.  Fast-FBR is first
    // solved in its calibrated full-band domain; a constrained post-solve then
    // recycles the analytically predicted HF entropy below the cutoff.
    const bool lowrate128 = physical_target_kbps > 0.0f && physical_target_kbps < 140.0f;
    // R301: HF hybrid coding is rate-independent.  Above the 16.59-kHz /
    // band-45 boundary reconstruction is the default at every lossy bitrate;
    // direct D_n coding is admitted only by the perceptual RD escape below.
    const bool hf_hybrid_profile = physical_target_kbps > 0.0f;
    const bool lowrate_cutoff=[](){const char* p=::lxvoxel_release_env("LX_LOW_RATE_CUTOFF");return !p || std::atoi(p)!=0;}();
    const bool lowrate_tone_floor=[](){const char* p=::lxvoxel_release_env("LX_LOW_RATE_TONE_FLOOR");return !p || std::atoi(p)!=0;}();
    const int lowrate_floor_mode=[](){const char* p=::lxvoxel_release_env("LX_LOW_RATE_FLOOR_MODE");return p?std::atoi(p):5;}();
    const double LOWRATE_CUTOFF_HZ=[](){const char* p=::lxvoxel_release_env("LX_LOW_RATE_CUTOFF_HZ");return p?std::atof(p):16590.0;}();
    const auto lowrate_cut_bins=[&](int k){
        if(!hf_hybrid_profile || !lowrate_cutoff) return k;
        return std::clamp(int(std::llround(LOWRATE_CUTOFF_HZ*(2.0*double(k))/double(SR))),0,k);
    };
    bool rate_saturated=false;
    auto& g_tables=encoder_tables();
    auto&eL=g_tables.eL;auto&e1024=g_tables.e1024;auto&eS=g_tables.eS;auto&e256=g_tables.e256;
    auto&fL=g_tables.fL;auto&f1024=g_tables.f1024;auto&fS=g_tables.fS;auto&f256=g_tables.f256;
    auto&fear4_static=g_tables.fear4;
    const double fear4_loga_gain=[](){const char* p=::lxvoxel_release_env("LX_FEAR_V4_LOGA_GAIN");return p?std::atof(p):1.0;}();
    const double fear4_beta_alpha=[](){const char* p=::lxvoxel_release_env("LX_FEAR_V4_BETA_ALPHA");return p?std::atof(p):0.0;}();
    auto&lap=g_tables.lap;auto&dLong=g_tables.dLong;auto&d1024=g_tables.d1024;auto&dShort=g_tables.dShort;auto&d256=g_tables.d256;
    auto&dLongR=g_tables.dLongR;auto&d1024R=g_tables.d1024R;auto&dShortR=g_tables.dShortR;auto&d256R=g_tables.d256R;
    const int cbL=eL.coded_bins(),cb1024=e1024.coded_bins(),cbS=eS.coded_bins(),cb256=e256.coded_bins();

    auto edges_for=[&](BlockSize bs)->const std::array<int,ERB_BANDS+1>&{
        switch(bs){
            case BlockSize::Long2048:   return eL.edge;
            case BlockSize::Medium1024: return e1024.edge;
            case BlockSize::Short512:   return eS.edge;
            case BlockSize::Micro256:   return e256.edge;
        }
        return eL.edge;
    };
    auto ath_for=[&](BlockSize bs)->const std::array<float,ERB_BANDS>&{
        switch(bs){
            case BlockSize::Long2048:   return fL.ath;
            case BlockSize::Medium1024: return f1024.ath;
            case BlockSize::Short512:   return fS.ath;
            case BlockSize::Micro256:   return f256.ath;
        }
        return fL.ath;
    };
    auto spread_for=[&](BlockSize bs)->const std::array<std::array<float,ERB_BANDS>,ERB_BANDS>&{
        switch(bs){
            case BlockSize::Long2048:   return fL.spread;
            case BlockSize::Medium1024: return f1024.spread;
            case BlockSize::Short512:   return fS.spread;
            case BlockSize::Micro256:   return f256.spread;
        }
        return fL.spread;
    };
    // LX-FEAR Arith-5: precompute the five-tap normalized adjoint once per encode
    // for each transform geometry.  No signal-dependent work is added here.
    using W5Table=std::array<std::array<double,5>,ERB_BANDS>;
    W5Table w5L{},w51024{},w5S{},w5256{};
    auto build_w5=[&](const auto& fear,W5Table& out){
        std::array<double,ERB_BANDS> rowInv{};
        for(int i=0;i<ERB_BANDS;++i){double sum=0.0;for(int j=0;j<ERB_BANDS;++j)sum+=double(fear.spread[i][j]);rowInv[i]=1.0/sum;}
        for(int b=0;b<ERB_BANDS;++b){
            double s5=0.0;
            for(int o=-2;o<=2;++o){const int i=b+o;if(i<0||i>=ERB_BANDS)continue;const double p=double(fear.spread[i][b])*rowInv[i];out[b][o+2]=p;s5+=p;}
            for(int k=0;k<5;++k)out[b][k]/=s5;
        }
    };
    build_w5(fL,w5L);build_w5(f1024,w51024);build_w5(fS,w5S);build_w5(f256,w5256);
    auto w5_for=[&](BlockSize bs)->const W5Table&{
        switch(bs){
            case BlockSize::Long2048: return w5L;
            case BlockSize::Medium1024: return w51024;
            case BlockSize::Short512: return w5S;
            case BlockSize::Micro256: return w5256;
        }
        return w5L;
    };

    const size_t frame_capacity = ns/size_t(K_MIN) + 128;
    std::vector<FramePlan> plan;
    plan.reserve(frame_capacity);
    const size_t coeff_capacity=ns+size_t(K_MIN);
    std::vector<float> cL(coeff_capacity),cR(coeff_capacity);
    std::vector<TemporalGbrCell> cells;
    cells.reserve(frame_capacity*ERB_BANDS);
    std::vector<uint8_t> modes;
    modes.reserve(frame_capacity);
    std::vector<float> statStructMotion, statStereoNorm;
    statStructMotion.reserve(frame_capacity); statStereoNorm.reserve(frame_capacity);
    std::vector<std::array<float,ERB_BANDS>> statEL,statER,statCR,statL1LR,statL1MS,statMask,statRelMask,statEta,statNv,statLf,statFear4,statFearTonal,statLineC4,statLineCrest,statLineProm,statLineFreq,statLineWidth,statLinePeak,statLinePeakFreq;
    statEL.reserve(frame_capacity);statER.reserve(frame_capacity);statCR.reserve(frame_capacity);
    statL1LR.reserve(frame_capacity);statL1MS.reserve(frame_capacity);statMask.reserve(frame_capacity);statRelMask.reserve(frame_capacity);statEta.reserve(frame_capacity);statNv.reserve(frame_capacity);statLf.reserve(frame_capacity);
    statFear4.reserve(frame_capacity);statFearTonal.reserve(frame_capacity);statLineC4.reserve(frame_capacity);statLineCrest.reserve(frame_capacity);statLineProm.reserve(frame_capacity);statLineFreq.reserve(frame_capacity);statLineWidth.reserve(frame_capacity);statLinePeak.reserve(frame_capacity);statLinePeakFreq.reserve(frame_capacity);
    std::vector<FbrFrameFeature> fbr_feat;
    fbr_feat.reserve(frame_capacity);

    // Detector FEAR keeps exactly the old causal chronology, but its long
    // spectral input is derived from the prospective codec L/R transform.
    // On this track this produces the exact same 16 switch positions.
    FearTemporalVariable48 detTm;
    detTm.init(fL,SR);
    std::array<float,ERB_BANDS> longEL{},longER{},longCross{},longL1LR{},longL1MS{};
    // Structural block-switch state: normalized spectral centroid trajectory per ERB.
    // This tracks ridge motion without signed-MDCT phase dependence.
    std::array<float,ERB_BANDS> prev_long_centroid{};
    bool prev_long_centroid_valid=false;

    FearTemporalVariable48 tmL,tmR;
    tmL.init(fL,SR);tmR.init(fL,SR);

    size_t pos=0,coff=0,qoff=0;
    double target_dims=0.0;
    alignas(64) std::array<float,K_LONG+OVL+64> pcmL{},pcmR{},pcmM{};
    alignas(64) std::array<float,K_LONG> cMid{};
    float* frameL=pcmL.data()+64;
    float* frameR=pcmR.data()+64;
    float* frameM=pcmM.data()+64;

    struct StereoMdctWorker {
        DCT4Ortho<1024>& d2048; DCT4Ortho<768>& d1024; DCT4Ortho<512>& d512; DCT4Ortho<384>& d256;
        const float* c; const float* sw;
        const float* in=nullptr; float* out=nullptr; bool left=false,right=false; int pending=0;
#if !defined(LXVOXEL_SINGLE_THREAD)
        std::atomic<int> cmd{0};
        std::thread th;
#endif
        StereoMdctWorker(DCT4Ortho<1024>& a,DCT4Ortho<768>& b,DCT4Ortho<512>& d,DCT4Ortho<384>& e,
                         const float* cc,const float* ss)
            :d2048(a),d1024(b),d512(d),d256(e),c(cc),sw(ss)
#if !defined(LXVOXEL_SINGLE_THREAD)
            ,th([this]{run();})
#endif
        {}
        inline void execute(int n){
            switch(n){
                case K_LONG:  d2048.apply_lapped(in,left,right,c,sw,out); break;
                case K_1024:  d1024.apply_lapped(in,left,right,c,sw,out); break;
                case K_SHORT: d512 .apply_lapped(in,left,right,c,sw,out); break;
                case K_256:   d256 .apply_lapped(in,left,right,c,sw,out); break;
                default: std::abort();
            }
        }
#if !defined(LXVOXEL_SINGLE_THREAD)
        void run(){
            for(;;){
                int n; while((n=cmd.load(std::memory_order_acquire))==0) _mm_pause();
                if(n<0) break;
                execute(n);
                cmd.store(0,std::memory_order_release);
            }
        }
#endif
        inline void start(int n,const float* pi,float* po,bool l,bool r){
            in=pi;out=po;left=l;right=r;
#if defined(LXVOXEL_SINGLE_THREAD)
            pending=n;
#else
            cmd.store(n,std::memory_order_release);
#endif
        }
        inline void wait(){
#if defined(LXVOXEL_SINGLE_THREAD)
            if(pending){const int n=pending; pending=0; execute(n);}
#else
            while(cmd.load(std::memory_order_acquire)!=0) _mm_pause();
#endif
        }
        void stop(){
#if defined(LXVOXEL_SINGLE_THREAD)
            wait();
#else
            wait();cmd.store(-1,std::memory_order_release);th.join();
#endif
        }
    } mdctR(dLongR,d1024R,dShortR,d256R,lap.c.data(),lap.s.data());

    // FEAR-v4 runs concurrently with the legacy ERB/masking pass.  Its state is
    // causal and owned by this persistent worker, so there is no thread creation
    // in the frame loop and no race between successive analysis frames.
    struct Fear4Worker {
        const lx_fear_v4::Static& st;
        lx_fear_v4::State state;
        lx_fear_v4::Scratch scratch;
        const float* l=nullptr; const float* r=nullptr; int k=0,cb=0; bool pending=false;
        std::array<float,ERB_BANDS>* out=nullptr;
        std::array<float,ERB_BANDS>* tonal_out=nullptr;
#if !defined(LXVOXEL_SINGLE_THREAD)
        std::atomic<int> cmd{0};
        std::thread th;
#endif
        explicit Fear4Worker(const lx_fear_v4::Static& s):st(s)
#if !defined(LXVOXEL_SINGLE_THREAD)
            ,th([this]{run();})
#endif
        {}
        inline void execute(){
            const int n=lx_fear_v4::build_grid(l,r,k,cb,scratch);
            lx_fear_v4::analyse_features(st,state,scratch,n);
            lx_fear_v4::reduce_to_erb(st,scratch,n,*out,tonal_out);
        }
#if !defined(LXVOXEL_SINGLE_THREAD)
        void run(){
            for(;;){
                int c; while((c=cmd.load(std::memory_order_acquire))==0) _mm_pause();
                if(c<0) break;
                execute();
                cmd.store(0,std::memory_order_release);
            }
        }
#endif
        inline void start(const float* ll,const float* rr,int kk,int cbb,std::array<float,ERB_BANDS>* o,std::array<float,ERB_BANDS>* t=nullptr){
            wait();l=ll;r=rr;k=kk;cb=cbb;out=o;tonal_out=t;
#if defined(LXVOXEL_SINGLE_THREAD)
            pending=true;
#else
            cmd.store(1,std::memory_order_release);
#endif
        }
        inline void wait(){
#if defined(LXVOXEL_SINGLE_THREAD)
            if(pending){pending=false;execute();}
#else
            while(cmd.load(std::memory_order_acquire)!=0) _mm_pause();
#endif
        }
        void stop(){
#if defined(LXVOXEL_SINGLE_THREAD)
            wait();
#else
            wait();cmd.store(-1,std::memory_order_release);th.join();
#endif
        }
    } fear4w(fear4_static);

    Rfft512LocalTR rfft_local_tr;
    const bool tr_analysis_enabled=[](){
        auto nz=[](const char* n){const char* p=std::getenv(n);return p && std::atoi(p)!=0;};
        return nz("LX_TR_LOCAL") || nz("LX_TR_TEMP") || nz("LX_TR_TEMP2");
    }();
    std::vector<std::array<float,ERB_BANDS>> trTensor,trRidge,trJoint,trPersist;
    if(tr_analysis_enabled){
        trTensor.reserve(ns/K_LONG+128); trRidge.reserve(ns/K_LONG+128);
        trJoint.reserve(ns/K_LONG+128); trPersist.reserve(ns/K_LONG+128);
    }

    // R219: encode configuration is immutable for the lifetime of one encode.
    // Keep libc/environment lookup entirely out of the per-frame hot path.
    const char* const gain32_dump_path=::lxvoxel_release_env("LX_GAIN32_DUMP");
    const double log_k_long =std::log(double(K_LONG));
    const double log_k_1024 =std::log(double(K_1024));
    const double log_k_short=std::log(double(K_SHORT));
    const double log_k_256  =std::log(double(K_256));
    const double block_nominal_kbps=(allocation_target_kbps>0.0f)
        ? double(allocation_target_kbps)
        : ((physical_target_kbps>0.0f)?double(physical_target_kbps):0.0);
    const double block_rch=(block_nominal_kbps>0.0)
        ? block_nominal_kbps*1000.0/(2.0*double(SR)) : 1e9;
    const double block_nu=std::exp2(-2.0*block_rch);

    while(pos<ns){
        const bool can_long=pos+K_LONG+OVL<=ns;
        const int max_k = can_long ? K_LONG
                        : (pos+K_1024+OVL<=ns) ? K_1024
                        : (pos+K_SHORT+OVL<=ns) ? K_SHORT
                                               : K_256;
        const bool has_left=!plan.empty();
        const size_t look=size_t(max_k+OVL);
        if(has_left){
            const size_t start=pos-64,need=look+64,avail=std::min(need,ns-start);
            deinterleave_local(stereo+2*start,avail,pcmL.data(),pcmR.data());
            if(avail<need){std::fill(pcmL.data()+avail,pcmL.data()+need,0.0f);std::fill(pcmR.data()+avail,pcmR.data()+need,0.0f);}
        }else{
            const size_t need=look,avail=std::min(need,ns-pos);
            deinterleave_local(stereo+2*pos,avail,frameL,frameR);
            if(avail<need){std::fill(frameL+avail,frameL+need,0.0f);std::fill(frameR+avail,frameR+need,0.0f);}
        }
        BlockSize chosen = max_k==K_LONG ? BlockSize::Long2048
                         : max_k==K_1024 ? BlockSize::Medium1024
                         : max_k==K_SHORT ? BlockSize::Short512
                                         : BlockSize::Micro256;
        double frame_struct_motion=0.0;
        static const bool disable_256=[](){const char* p=::lxvoxel_release_env("LX_DISABLE_256");return p && std::atoi(p)!=0;}();
        if(can_long){
            const bool left=!plan.empty();
            const bool right_if_long=(pos+K_LONG+K_MIN+OVL<=ns);

            // short certificate from pure temporal geometry.  Unity is the
            // natural evidence boundary; >2 means one full evidence octave
            // beyond it.  This region was a strict subset of R81 short decisions
            // across the 5x30 s validation corpus.
            const LXTransientFeatures lx_tf=lx_transient_features_2048(frameL,frameR);
            const float pgeom_pcm=std::sqrt(lx_tf.peak*lx_tf.hp_peak);
            const float fall_pcm=std::max(2.0f*lx_tf.tv-lx_tf.rise,0.0f);
            const float bal_pcm=2.0f*fall_pcm/(fall_pcm+lx_tf.rise+std::numeric_limits<float>::min());
            const double cert_hom=std::sqrt(std::max(double(lx_tf.duty)*double(lx_tf.hp_duty),
                                                     std::numeric_limits<double>::min()));
            const double temporal_cert=(double(lx_tf.coarse_mask)*double(pgeom_pcm)*double(lx_tf.rise)
                                      *std::sqrt(std::max(0.0,1.0-double(lx_tf.hp_duty))))
                                     /(cert_hom*std::max(double(bal_pcm),
                                                        std::sqrt(std::numeric_limits<double>::min())));
            const bool certified_short=temporal_cert>2.0;

            if(certified_short){
                // Preserve the most important part of R81's prospective memory
                // at half the transform cost: by MDCT linearity, transforming
                // M=(L+R)/sqrt(2) gives exactly the detector's prospective Mid
                // spectrum without computing separate prospective L and R MDCTs.
                const int i0=left ? -64 : 0;
                for(int i=i0;i<K_LONG+OVL;++i) frameM[i]=(frameL[i]+frameR[i])*INV_SQRT2;
                dLong.apply_lapped(frameM,left,right_if_long,lap.c.data(),lap.s.data(),cMid.data());
                std::array<float,ERB_BANDS> eM{},sl{},nv{};
                for(int b=0;b<ERB_BANDS;++b){
                    double em=0.0;
                    for(int i=eL.edge[b];i<eL.edge[b+1];++i){const double v=cMid[i];em+=v*v;}
                    eM[b]=float(em);
                }
                detTm.analyze(eM,BlockSize::Long2048,sl,nv);

                // Optional structural-memory update from the same Mid spectrum.
                // Mode 0 freezes structural memory on certified events; mode 1
                // updates its normalized ridge centroid from Mid only.
                static const int cert_state=[](){const char* p=::lxvoxel_release_env("LX_CERT_STATE");return p?std::atoi(p):1;}();
                if(cert_state==1){
                    std::array<float,ERB_BANDS> mid_centroid{};
                    double sn=0.0,sd=0.0;
                    for(int b=0;b<ERB_BANDS;++b){
                        const int lo=eL.edge[b],hi=eL.edge[b+1],nn=std::max(hi-lo,1);
                        double ce=0.0,cn=0.0,l1=0.0;
                        for(int i=lo;i<hi;++i){const double v=cMid[i],z=v*v,u=(2.0*(double(i-lo)+0.5)/double(nn))-1.0;ce+=z;cn+=u*z;l1+=std::abs(v);}
                        const double c=cn/(ce+std::numeric_limits<double>::min()); mid_centroid[b]=float(c);
                        if(prev_long_centroid_valid){
                            const double ncoord=double(nn);
                            const double iso=std::clamp((l1*l1)/(ncoord*ce+std::numeric_limits<double>::min()),1.0/ncoord,1.0);
                            const double tau=(1.0-iso)/std::max(1.0-1.0/ncoord,std::numeric_limits<double>::min());
                            const double dc=c-double(prev_long_centroid[b]);
                            const double fc=(0.5*double(lo+hi))/double(std::max(cbL,1)),w=fc*fc;
                            sn+=w*tau*dc*dc; sd+=w;
                        }
                    }
                    frame_struct_motion=prev_long_centroid_valid?sn/std::max(sd,std::numeric_limits<double>::min()):0.0;
                    prev_long_centroid=mid_centroid; prev_long_centroid_valid=true;
                }
                int level=2;
                if(!disable_256 && double(K_LONG)*double(lx_tf.hp_duty)<=double(K_256)) level=3;
                chosen=block_from_level(level);
            } else {
            // Ambiguous region: retain the original R81 prospective path exactly.
            mdctR.start(K_LONG,frameR,cR.data()+coff,left,right_if_long);
            dLong.apply_lapped(frameL,left,right_if_long,lap.c.data(),lap.s.data(),cL.data()+coff);
            mdctR.wait();

            const float*pl=cL.data()+coff,*pr=cR.data()+coff;
            std::array<float,ERB_BANDS>eM{},sl{},nv{},raw{},imp{},occ{};
            // These arrays live across the frame-analysis loop so the selected
            // long block can reuse the prospective transform below.  Reset them
            // before every prospective long analysis.  The AVX2 path assigns
            // each band, but the portable scalar path accumulates with +=; without
            // this reset it incorrectly carried all earlier long-frame energy
            // forward and could drive anti-collapse to full scale in quiet audio.
            longEL.fill(0.0f);
            longER.fill(0.0f);
            longCross.fill(0.0f);
            longL1LR.fill(0.0f);
            longL1MS.fill(0.0f);
            // One coefficient pass supplies detector-mid energy plus the L/R
            // energy and cross terms later reused by codec FEAR/stereo mode.
            for(int b=0;b<ERB_BANDS;++b){
                const int lo=eL.edge[b],hi=eL.edge[b+1];
#if defined(__AVX2__)
                __m256 aL=_mm256_setzero_ps(),aR=_mm256_setzero_ps(),aC=_mm256_setzero_ps(),aM=_mm256_setzero_ps();
                __m256 aLR1=_mm256_setzero_ps(),aMS1=_mm256_setzero_ps();
                const __m256 g=_mm256_set1_ps(INV_SQRT2),sgn=_mm256_set1_ps(-0.0f);int i=lo;
                for(;i+8<=hi;i+=8){
                    const __m256 vl=_mm256_loadu_ps(pl+i),vr=_mm256_loadu_ps(pr+i);
                    const __m256 vm=_mm256_mul_ps(_mm256_add_ps(vl,vr),g);
                    const __m256 vs=_mm256_mul_ps(_mm256_sub_ps(vl,vr),g);
                    aL=_mm256_fmadd_ps(vl,vl,aL);aR=_mm256_fmadd_ps(vr,vr,aR);
                    aC=_mm256_fmadd_ps(vl,vr,aC);aM=_mm256_fmadd_ps(vm,vm,aM);
                    aLR1=_mm256_add_ps(aLR1,_mm256_add_ps(_mm256_andnot_ps(sgn,vl),_mm256_andnot_ps(sgn,vr)));
                    aMS1=_mm256_add_ps(aMS1,_mm256_add_ps(_mm256_andnot_ps(sgn,vm),_mm256_andnot_ps(sgn,vs)));
                }
                alignas(32)float tl[8],tr[8],tc[8],tm[8],tlr1[8],tms1[8];
                _mm256_store_ps(tl,aL);_mm256_store_ps(tr,aR);_mm256_store_ps(tc,aC);_mm256_store_ps(tm,aM);
                _mm256_store_ps(tlr1,aLR1);_mm256_store_ps(tms1,aMS1);
                float el=tl[0]+tl[1]+tl[2]+tl[3]+tl[4]+tl[5]+tl[6]+tl[7];
                float er=tr[0]+tr[1]+tr[2]+tr[3]+tr[4]+tr[5]+tr[6]+tr[7];
                float ec=tc[0]+tc[1]+tc[2]+tc[3]+tc[4]+tc[5]+tc[6]+tc[7];
                float em=tm[0]+tm[1]+tm[2]+tm[3]+tm[4]+tm[5]+tm[6]+tm[7];
                float l1lr=tlr1[0]+tlr1[1]+tlr1[2]+tlr1[3]+tlr1[4]+tlr1[5]+tlr1[6]+tlr1[7];
                float l1ms=tms1[0]+tms1[1]+tms1[2]+tms1[3]+tms1[4]+tms1[5]+tms1[6]+tms1[7];
                for(;i<hi;++i){const float l0=pl[i],r0=pr[i],m0=(l0+r0)*INV_SQRT2,s0=(l0-r0)*INV_SQRT2;el+=l0*l0;er+=r0*r0;ec+=l0*r0;em+=m0*m0;l1lr+=std::fabs(l0)+std::fabs(r0);l1ms+=std::fabs(m0)+std::fabs(s0);}
                longEL[b]=el;longER[b]=er;longCross[b]=ec;longL1LR[b]=l1lr;longL1MS[b]=l1ms;eM[b]=em;
#else
                for(int i=lo;i<hi;++i){const float l0=pl[i],r0=pr[i],m0=(l0+r0)*INV_SQRT2,s0=(l0-r0)*INV_SQRT2;longEL[b]+=l0*l0;longER[b]+=r0*r0;longCross[b]+=l0*r0;eM[b]+=m0*m0;longL1LR[b]+=std::fabs(l0)+std::fabs(r0);longL1MS[b]+=std::fabs(m0)+std::fabs(s0);}
#endif
            }
            detTm.analyze(eM,BlockSize::Long2048,sl,nv);
            // Xref and raw importance are fused: no detector xref array/pass.
            for(int b=0;b<ERB_BANDS;++b){
#if defined(__AVX2__)
                __m256 acc=_mm256_setzero_ps();
                for(int s=0;s<ERB_BANDS;s+=8)acc=_mm256_fmadd_ps(_mm256_loadu_ps(fL.spread[b].data()+s),_mm256_loadu_ps(eM.data()+s),acc);
                alignas(32)float t[8];_mm256_store_ps(t,acc);
                const float xr=fL.ath[b]+t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
#else
                float xr=fL.ath[b];for(int s=0;s<ERB_BANDS;++s)xr+=fL.spread[b][s]*eM[s];
#endif
                const double self=double(fL.spread[b][b])*eM[b];
                const double th=std::max(double(fL.ath[b]),double(xr)-self);
                raw[b]=float(fear_v3_error_importance(eM[b],sl[b],th,nv[b]));
                const double persistent=std::sqrt(std::max(0.0,double(eM[b])*double(sl[b])));
                occ[b]=float(persistent/(persistent+th+std::numeric_limits<double>::min()));
            }
            for(int b=0;b<ERB_BANDS;++b){
                const int lo=std::max(0,b-1),hi=std::min(ERB_BANDS-1,b+1);
                const float sm=(raw[lo]+2.f*raw[b]+raw[hi])*.25f;
                imp[b]=raw[b]+nv[b]/(1.f+nv[b])*(sm-raw[b]);
            }
            // Reuse the long-window MDCT to estimate which bands need frequency
            // resolution. Concentration distinguishes tonal peaks from broad energy;
            // curvature measures local changes in spectral magnitude. The factor 16
            // is the squared maximum gain of the [1, -2, 1] difference filter.
            std::array<float,ERB_BANDS> mdct_conc{}, mdct_curv{};
            for(int b=0;b<ERB_BANDS;++b){
                const int lo=eL.edge[b], hi=eL.edge[b+1], nn=std::max(hi-lo,1);
                double se=0.0,se2=0.0,sc=0.0;
                // Keep neighboring magnitudes as the stencil advances. This saves
                // two square roots per bin without changing the accumulation order.
                auto magnitude = [&](int k){
                    return std::sqrt(double(pl[k])*pl[k]+double(pr[k])*pr[k]);
                };
                double prev=magnitude(lo), current=prev;
                for(int ii=lo;ii<hi;++ii){
                    const double z=double(pl[ii])*pl[ii]+double(pr[ii])*pr[ii];
                    se+=z; se2+=z*z;
                    const double next=magnitude(ii+1<hi ? ii+1 : hi-1);
                    const double d2=prev-2.0*current+next;
                    sc+=d2*d2;
                    prev=current;
                    current=next;
                }
                const double den=se*se+std::numeric_limits<double>::min();
                const double nipr=double(nn)*se2/den;
                mdct_conc[b]=float(nn>1 ? std::clamp((nipr-1.0)/double(nn-1),0.0,1.0) : 1.0);
                mdct_curv[b]=float(std::clamp(sc/(16.0*se+std::numeric_limits<double>::min()),0.0,1.0));
            }

            double wa=0.0,wimp=0.0,wuimp=0.0,wuenergy=0.0,wenergy=0.0;
            for(int b=0;b<ERB_BANDS;++b){
                const double ib=double(imp[b]), ub=double(occ[b]), eb=double(eM[b]), nb=double(nv[b]);
                wa+=ib*nb; wimp+=ib;
                wuimp+=ib*ub;
                wuenergy+=eb*ub; wenergy+=eb;
            }
            const double dtiny=std::numeric_limits<double>::min();
            const float aud=1.0f+float(wa/(wimp+dtiny));
            const float uimp=float(wuimp/(wimp+dtiny));
            const float uenergy=float(wuenergy/(wenergy+dtiny));
            const float ugeom=std::sqrt(std::max(0.0f,uimp*uenergy));
            double md_w=0.0,md_wp=0.0,md_c=0.0,md_cp=0.0,md_q=0.0,md_n=0.0,md_p=0.0;
            for(int b=0;b<ERB_BANDS;++b){
                const double w=double(imp[b]);
                const double po=std::clamp(double(occ[b]),0.0,1.0);
                const double nw=double(nv[b])/(1.0+double(nv[b]));
                md_w+=w; md_wp+=w*po;
                md_c+=w*double(mdct_conc[b]); md_cp+=w*po*double(mdct_conc[b]);
                md_q+=w*po*double(mdct_curv[b]); md_n+=w*nw; md_p+=w*po;
            }
            const double md_conc=std::clamp(md_c/(md_w+dtiny),0.0,1.0);
            const double md_pconc=std::clamp(md_cp/(md_wp+dtiny),0.0,1.0);
            const double md_curv=std::clamp(md_q/(md_wp+dtiny),0.0,1.0);
            const double md_novel=std::clamp(md_n/(md_w+dtiny),0.0,1.0);
            const double md_persist=std::clamp(md_p/(md_w+dtiny),0.0,1.0);
            // Continuous tonal-ridge motion from the prospective long spectrum.
            // Within each ERB the energy centroid is normalized to [-1,+1],
            // so centroid displacement is dimensionless and independent of ERB width.
            // Tonality tau suppresses diffuse/noise motion.  Frequency weighting is
            // the physical normalized center frequency, not a hard HF threshold.
            std::array<float,ERB_BANDS> long_centroid{};
            double struct_num=0.0,struct_den=0.0;
            for(int b=0;b<ERB_BANDS;++b){
                const int lo=eL.edge[b],hi=eL.edge[b+1],nn=std::max(hi-lo,1);
                double ce=0.0,cn=0.0;
                for(int ii=lo;ii<hi;++ii){
                    const double x2=double(pl[ii])*pl[ii]+double(pr[ii])*pr[ii];
                    const double u=(2.0*(double(ii-lo)+0.5)/double(nn))-1.0;
                    ce+=x2; cn+=u*x2;
                }
                const double c=cn/(ce+std::numeric_limits<double>::min());
                long_centroid[b]=float(c);
                if(prev_long_centroid_valid){
                    const double ncoord=2.0*double(nn);
                    const double l1=double(longL1LR[b]);
                    const double iso=std::clamp((l1*l1)/(ncoord*ce+std::numeric_limits<double>::min()),1.0/ncoord,1.0);
                    const double tau=(1.0-iso)/std::max(1.0-1.0/ncoord,std::numeric_limits<double>::min());
                    const double dc=c-double(prev_long_centroid[b]);
                    const double fc=(0.5*double(lo+hi))/double(std::max(cbL,1));
                    const double w=fc*fc;
                    struct_num += w*tau*dc*dc;
                    struct_den += w;
                }
            }
            const double struct_motion=prev_long_centroid_valid?struct_num/std::max(struct_den,std::numeric_limits<double>::min()):0.0;
            frame_struct_motion=struct_motion;
            prev_long_centroid=long_centroid; prev_long_centroid_valid=true;

            double sum_l1=0.0,sum_e=0.0;
            for(int b=0;b<ERB_BANDS;++b){sum_l1+=double(longL1LR[b]);sum_e+=double(longEL[b])+double(longER[b]);}
            const float flat=float((sum_l1*sum_l1)/(sum_e*double(2*cbL)+std::numeric_limits<double>::min()));
            // LX-TD M9 production detector (balanced-rise).  M8's directed onset evidence is
            // balanced against signal-derived stationarity evidence; there is no
            // track-tuned or fixed scalar decision threshold.
            const float pgeom=pgeom_pcm;
            const float temporal_exposure=std::sqrt(std::max(0.0f,1.0f-lx_tf.hp_duty));
            const float score=aud*lx_tf.coarse_mask*pgeom*lx_tf.rise*temporal_exposure;

            const float sflat=std::sqrt(std::max(0.0f,flat));
            const float shpduty=std::sqrt(std::max(0.0f,lx_tf.hp_duty));
            const float socc=std::sqrt(std::max(0.0f,ugeom));
            const float hflat=2.0f*sflat/(1.0f+sflat);
            const float hhpduty=2.0f*shpduty/(1.0f+shpduty);
            const float hocc=2.0f*socc/(1.0f+socc);

            const float bal=bal_pcm;
            const float threshold=std::sqrt(std::max(0.0f,hhpduty*hocc))
                                 *bal*(1.0f-hflat);
            static const float td_scale=[](){const char* p=::lxvoxel_release_env("LX_TD_THRESHOLD_SCALE");return p?std::strtof(p,nullptr):0.75f;}();
            static const double td_struct_alpha=[](){const char* p=::lxvoxel_release_env("LX_TD_STRUCT_ALPHA");return p?std::atof(p):20.0;}();
            const double structural_scale=std::exp(-td_struct_alpha*struct_motion);
            const double base_threshold=std::max(double(threshold)*double(td_scale),
                                                   std::numeric_limits<double>::min());
            const double effective_threshold=std::max(base_threshold*structural_scale,
                                                      std::numeric_limits<double>::min());
            const double pressure=double(score)/effective_threshold;

            // Compare temporal novelty with spectral detail using the same
            // long-window MDCT analysis. The neighboring duration boundaries
            // follow directly from A*x*x + B/(x*x).
            const double p=std::max(pressure,1.0);
            const double attack=std::log1p(p-1.0)*(1.0+block_nu);
            const double stationary=
                std::sqrt(std::max(0.0,1.0-double(lx_tf.rise))) *
                std::sqrt(std::max(0.0,double(lx_tf.hp_duty))) *
                (1.0+struct_motion);
            double A=std::max(attack,std::numeric_limits<double>::min());
            double B=std::max(stationary,std::numeric_limits<double>::min());
            const double novelty=md_novel*(1.0-md_curv);
            A/=std::sqrt(std::max(1.0-novelty,std::numeric_limits<double>::min()));
            B/=std::max(1.0-md_curv,std::numeric_limits<double>::min());
            const double nineA=9.0*A;
            int level=(nineA < 16.0*B) ? 0
                            :(nineA < 64.0*B) ? 1
                            :(nineA < 256.0*B) ? 2 : 3;
            if(disable_256 && level==3) level=2;
            chosen=block_from_level(level);
            }
        }

        // Never request a transform that cannot be fully supported at the tail.
        if(pos+size_t(block_samples(chosen))+OVL>ns){
            chosen = max_k==K_LONG ? BlockSize::Long2048
                   : max_k==K_1024 ? BlockSize::Medium1024
                   : max_k==K_SHORT ? BlockSize::Short512
                                   : BlockSize::Micro256;
        }
        FramePlan fr;
        fr.sample_pos=pos;
        fr.block=chosen;
        fr.k=uint16_t(block_samples(chosen));
        fr.coded_bins=uint16_t(edges_for(chosen)[ERB_BANDS]);
        fr.coeff_off=coff;fr.q_off=qoff;

        if(chosen!=BlockSize::Long2048){
            const bool left=!plan.empty();
            const bool right_if_block=(pos+size_t(fr.k)+K_MIN+OVL<=ns);
            mdctR.start(fr.k,frameR,cR.data()+coff,left,right_if_block);
            switch(chosen){
                case BlockSize::Medium1024: d1024.apply_lapped(frameL,left,right_if_block,lap.c.data(),lap.s.data(),cL.data()+coff); break;
                case BlockSize::Short512:   dShort.apply_lapped(frameL,left,right_if_block,lap.c.data(),lap.s.data(),cL.data()+coff); break;
                case BlockSize::Micro256:   d256.apply_lapped(frameL,left,right_if_block,lap.c.data(),lap.s.data(),cL.data()+coff); break;
                case BlockSize::Long2048:   break;
            }
            mdctR.wait();
        }

        plan.push_back(fr);
        const size_t fi=plan.size()-1;
        modes.push_back(0);
        std::array<float,ERB_BANDS> _trT{},_trR{},_trJ{},_trP{};
        if(tr_analysis_enabled) rfft_local_tr.analyse(stereo,ns,fr.sample_pos+size_t(fr.k/2),_trT,_trR,_trJ,_trP);
        if(tr_analysis_enabled){
            trTensor.push_back(_trT); trRidge.push_back(_trR);
            trJoint.push_back(_trJ); trPersist.push_back(_trP);
        }
        statStructMotion.push_back(float(frame_struct_motion));
        cells.resize((fi+1)*ERB_BANDS);

        float*l=cL.data()+coff,*r=cR.data()+coff;
        std::array<float,ERB_BANDS> enL{},enR{},cross{},l1LR{},l1MS{},slL{},slR{},nvL{},nvR{},rawL{},rawR{},imL{},imR{},maskERB{},relMaskERB{};
        std::array<float,ERB_BANDS> fear4_pressure{}, fear4_tonal{};
        fear4w.start(l,r,fr.k,fr.coded_bins,&fear4_pressure,&fear4_tonal);
        const auto& frame_edge=edges_for(fr.block);
        const auto& frame_ath=ath_for(fr.block);
        const auto& frame_spread=spread_for(fr.block);
        const auto& frame_w5=w5_for(fr.block);

        if(fr.block==BlockSize::Long2048){
            enL=longEL;enR=longER;cross=longCross;l1LR=longL1LR;l1MS=longL1MS;
        }else{
            for(int b=0;b<ERB_BANDS;++b){
                const int lo=frame_edge[b],hi=frame_edge[b+1];
#if defined(__AVX2__)
                __m256 al=_mm256_setzero_ps(),ar=_mm256_setzero_ps(),ac=_mm256_setzero_ps(),alr1=_mm256_setzero_ps(),ams1=_mm256_setzero_ps();
                const __m256 gg=_mm256_set1_ps(INV_SQRT2),sgn=_mm256_set1_ps(-0.0f);int i=lo;
                for(;i+8<=hi;i+=8){const __m256 vl=_mm256_loadu_ps(l+i),vr=_mm256_loadu_ps(r+i);const __m256 vm=_mm256_mul_ps(_mm256_add_ps(vl,vr),gg),vs=_mm256_mul_ps(_mm256_sub_ps(vl,vr),gg);al=_mm256_fmadd_ps(vl,vl,al);ar=_mm256_fmadd_ps(vr,vr,ar);ac=_mm256_fmadd_ps(vl,vr,ac);alr1=_mm256_add_ps(alr1,_mm256_add_ps(_mm256_andnot_ps(sgn,vl),_mm256_andnot_ps(sgn,vr)));ams1=_mm256_add_ps(ams1,_mm256_add_ps(_mm256_andnot_ps(sgn,vm),_mm256_andnot_ps(sgn,vs)));}
                alignas(32)float tl[8],tr[8],tc[8],tlr1[8],tms1[8];_mm256_store_ps(tl,al);_mm256_store_ps(tr,ar);_mm256_store_ps(tc,ac);_mm256_store_ps(tlr1,alr1);_mm256_store_ps(tms1,ams1);
                float el=tl[0]+tl[1]+tl[2]+tl[3]+tl[4]+tl[5]+tl[6]+tl[7],er=tr[0]+tr[1]+tr[2]+tr[3]+tr[4]+tr[5]+tr[6]+tr[7],ec=tc[0]+tc[1]+tc[2]+tc[3]+tc[4]+tc[5]+tc[6]+tc[7];
                float lrl=tlr1[0]+tlr1[1]+tlr1[2]+tlr1[3]+tlr1[4]+tlr1[5]+tlr1[6]+tlr1[7],msl=tms1[0]+tms1[1]+tms1[2]+tms1[3]+tms1[4]+tms1[5]+tms1[6]+tms1[7];
                for(;i<hi;++i){const float ll=l[i],rr=r[i],mm=(ll+rr)*INV_SQRT2,ss=(ll-rr)*INV_SQRT2;el+=ll*ll;er+=rr*rr;ec+=ll*rr;lrl+=std::fabs(ll)+std::fabs(rr);msl+=std::fabs(mm)+std::fabs(ss);}enL[b]=el;enR[b]=er;cross[b]=ec;l1LR[b]=lrl;l1MS[b]=msl;
#else
                for(int i=lo;i<hi;++i){const float ll=l[i],rr=r[i],mm=(ll+rr)*INV_SQRT2,ss=(ll-rr)*INV_SQRT2;enL[b]+=ll*ll;enR[b]+=rr*rr;cross[b]+=ll*rr;l1LR[b]+=std::fabs(ll)+std::fabs(rr);l1MS[b]+=std::fabs(mm)+std::fabs(ss);}
#endif
            }
        }

        if(const char* gp=gain32_dump_path) {
            static std::ofstream gout(gp);
            if(gout && gout.tellp()==std::streampos(0)) gout << "frame,band,center_hz,ch,gain_db,q32\n";
            const double normN = double(fr.k);
            constexpr double GMIN_DB = -70.0;
            constexpr double GMAX_DB = 0.0;
            constexpr double GSTEP_DB = (GMAX_DB-GMIN_DB)/31.0;
            for(int gb=0; gb<ERB_BANDS; ++gb){
                const int lo=frame_edge[gb];
                const int hi=frame_edge[gb+1];
                const double df=48000.0/(2.0*double(fr.k));
                const double center=0.5*double(lo+hi)*df;
                const double gl=std::sqrt(std::max(0.0,double(enL[gb]))/normN);
                const double gr=std::sqrt(std::max(0.0,double(enR[gb]))/normN);
                const double dl=20.0*std::log10(std::max(gl,1e-30));
                const double dr=20.0*std::log10(std::max(gr,1e-30));
                const int ql=std::clamp(int(std::nearbyint((dl-GMIN_DB)/GSTEP_DB)),0,31);
                const int qr=std::clamp(int(std::nearbyint((dr-GMIN_DB)/GSTEP_DB)),0,31);
                gout << fi << ',' << gb << ',' << center << ",L," << dl << ',' << ql << '\n';
                gout << fi << ',' << gb << ',' << center << ",R," << dr << ',' << qr << '\n';
            }
        }

        // research: bin-local line descriptors.  All are scale-invariant in [0,1].
        // c4    = inverse participation ratio sum(z^2)/sum(z)^2
        // crest = strongest-bin energy share max(z)/sum(z)
        // prom  = strongest locally isolated energy share; no thresholds.
        std::array<float,ERB_BANDS> lineC4,lineCrest,lineProm,lineFreq,lineWidth,linePeak,linePeakFreq;
        const double line_df=double(SR)/(2.0*double(fr.k));
        const double line_tiny=std::numeric_limits<double>::min();
        for(int b=0;b<ERB_BANDS;++b){
            const int lo=frame_edge[b], hi=frame_edge[b+1];
            double e=0.0,e4=0.0,mx=0.0,pr=0.0,pr_num=0.0,sw=0.0,sf=0.0,sf2=0.0; int imax=lo;
            const double tiny=line_tiny;
            auto add_energy=[&](int i) __attribute__((always_inline)) {
                const double z=double(l[i])*l[i]+double(r[i])*r[i];
                e+=z; e4+=z*z; if(z>mx){mx=z;imax=i;}
                return z;
            };
            auto add_shape=[&](int i,double zm,double z,double zp) __attribute__((always_inline)) {
                const double isolation=z/(z+0.5*(zm+zp)+tiny);
                pr_num=std::max(pr_num,z*isolation);
                const double w=z*z*isolation;
                const double hz=(double(i)+0.5)*line_df;
                sw+=w; sf+=w*hz; sf2+=w*hz*hz;
            };
            if(lo<hi){
                double z0=add_energy(lo);
                if(lo+1==hi){
                    add_shape(lo,z0,z0,z0);
                }else{
                    double z1=add_energy(lo+1);
                    add_shape(lo,z0,z0,z1);
                    double zm=z0,z=z1;
                    for(int i=lo+2;i<hi;++i){
                        const double zp=add_energy(i);
                        add_shape(i-1,zm,z,zp);
                        zm=z; z=zp;
                    }
                    add_shape(hi-1,zm,z,z);
                }
            }
            const double inv_e=1.0/(e+tiny);
            pr=pr_num*inv_e;
            const double mu=sf/(sw+tiny);
            const double var=std::max(0.0,sf2/(sw+tiny)-mu*mu);
            lineC4[b]=float(std::clamp(e4/(e*e+tiny),0.0,1.0));
            lineCrest[b]=float(std::clamp(mx/(e+tiny),0.0,1.0));
            lineProm[b]=float(std::clamp(pr,0.0,1.0));
            lineFreq[b]=float(mu);
            lineWidth[b]=float(std::sqrt(var));
            linePeak[b]=float(mx);
            linePeakFreq[b]=float((double(imax)+0.5)*line_df);
        }
        statLineC4.push_back(lineC4); statLineCrest.push_back(lineCrest); statLineProm.push_back(lineProm); statLineFreq.push_back(lineFreq); statLineWidth.push_back(lineWidth); statLinePeak.push_back(linePeak); statLinePeakFreq.push_back(linePeakFreq);

        tmL.analyze(enL,fr.block,slL,nvL);tmR.analyze(enR,fr.block,slR,nvR);
        std::array<double,ERB_BANDS+4> baseThL{},baseThR{},baseRelL{},baseRelR{};
        std::array<double,ERB_BANDS> isoPre{};
        // Pass 1: the production auditory spread.  This is the only dense 48x48 pass.
        for(int b=0;b<ERB_BANDS;++b){
            float xl=frame_ath[b],xr=xl;
#if defined(__AVX2__)
            __m256 al=_mm256_setzero_ps(),ar=_mm256_setzero_ps();
            const float*sp=frame_spread[b].data();
            for(int ss=0;ss<ERB_BANDS;ss+=8){const __m256 k=_mm256_loadu_ps(sp+ss);al=_mm256_fmadd_ps(k,_mm256_loadu_ps(enL.data()+ss),al);ar=_mm256_fmadd_ps(k,_mm256_loadu_ps(enR.data()+ss),ar);}
            alignas(32)float tl[8],tr[8];_mm256_store_ps(tl,al);_mm256_store_ps(tr,ar);
            xl+=tl[0]+tl[1]+tl[2]+tl[3]+tl[4]+tl[5]+tl[6]+tl[7];xr+=tr[0]+tr[1]+tr[2]+tr[3]+tr[4]+tr[5]+tr[6]+tr[7];
#else
            for(int j=0;j<ERB_BANDS;++j){const float k=frame_spread[b][j];xl+=k*enL[j];xr+=k*enR[j];}
#endif
            const double ath=double(frame_ath[b]);
            const double diag=double(frame_spread[b][b]);
            baseThL[b+2]=std::max(ath,double(xl)-diag*double(enL[b]));
            baseThR[b+2]=std::max(ath,double(xr)-diag*double(enR[b]));
            // Relative external masking only: remove the absolute ATH term and
            // the band's own diagonal contribution.  This is decoder-independent
            // psychoacoustic evidence the file itself actually contains.
            baseRelL[b+2]=std::max(0.0,double(xl)-diag*double(enL[b])-ath);
            baseRelR[b+2]=std::max(0.0,double(xr)-diag*double(enR[b])-ath);
            const int n=frame_edge[b+1]-frame_edge[b];
            const double d=2.0*double(n),ee=double(enL[b])+double(enR[b]),l1=double(l1LR[b]);
            isoPre[b]=std::clamp((l1*l1)/(d*ee+std::numeric_limits<double>::min()),1.0/d,1.0);
        }
        // Pass 2: LX-FEAR Arith-5 production path. Five fixed taps, branchless on signal values.
        for(int b=0;b<ERB_BANDS;++b){
            const auto& W=frame_w5[b];
            const double* l=baseThL.data()+b;
            const double* r=baseThR.data()+b;
            const double ath=double(frame_ath[b]);
            const double thL=std::max(ath,W[0]*l[0]+W[1]*l[1]+W[2]*l[2]+W[3]*l[3]+W[4]*l[4]);
            const double thR=std::max(ath,W[0]*r[0]+W[1]*r[1]+W[2]*r[2]+W[3]*r[3]+W[4]*r[4]);
            const double* rl=baseRelL.data()+b;
            const double* rr=baseRelR.data()+b;
            const double relL=std::max(0.0,W[0]*rl[0]+W[1]*rl[1]+W[2]*rl[2]+W[3]*rl[3]+W[4]*rl[4]);
            const double relR=std::max(0.0,W[0]*rr[0]+W[1]*rr[1]+W[2]*rr[2]+W[3]*rr[3]+W[4]*rr[4]);
            maskERB[b]=float(thL+thR);
            relMaskERB[b]=float(relL+relR);
            rawL[b]=float(fear_v3_error_importance(enL[b],slL[b],thL,nvL[b]));
            rawR[b]=float(fear_v3_error_importance(enR[b],slR[b],thR,nvR[b]));
        }
        for(int b=0;b<ERB_BANDS;++b){
            const int lo=std::max(0,b-1),hi=std::min(ERB_BANDS-1,b+1);
            const float smL=(rawL[lo]+2.f*rawL[b]+rawL[hi])*.25f,smR=(rawR[lo]+2.f*rawR[b]+rawR[hi])*.25f;
            imL[b]=rawL[b]+nvL[b]/(1.f+nvL[b])*(smL-rawL[b]);
            imR[b]=rawR[b]+nvR[b]/(1.f+nvR[b])*(smR-rawR[b]);
        }
        // Frame-level psycho state.  FEAR already owns the fast/slow states;
        // preserve their sign instead of keeping only novelty=(fast-slow)^2/...
        // so attacks and decays remain distinguishable to the frame allocator.
        FbrFrameFeature ff{};
        double psycho_dim_sum=0.0;
        double psycho_eta_float_sum=0.0;
        // Signed temporal direction per ERB. FEAR's historical novelty keeps
        // eta^2 and therefore discards attack/decay direction. Keep eta itself
        // here; its dimension-weighted frame common mode is removed below so
        // this term only redistributes the already assigned frame budget.
        std::array<float,ERB_BANDS> local_signed_eta{};
        {
            const double tiny=std::numeric_limits<double>::min();
            double dsum=0.0, flog=0.0, nov=0.0, seta=0.0;
            for(int b=0;b<ERB_BANDS;++b){
                const int nb=frame_edge[b+1]-frame_edge[b];
                const double d=2.0*double(nb);
                const double eb=double(enL[b])+double(enR[b]);
                const double il=double(imL[b]), ir=double(imR[b]);
                const double fl=double(tmL.fast[b]), sl=double(tmL.slow[b]);
                const double frx=double(tmR.fast[b]), sr=double(tmR.slow[b]);
                const double etaL=(fl-sl)/std::sqrt(fl*fl+sl*sl+tiny);
                const double etaR=(frx-sr)/std::sqrt(frx*frx+sr*sr+tiny);
                local_signed_eta[b]=float(0.5*(etaL+etaR));
                psycho_eta_float_sum += d*double(local_signed_eta[b]);

                dsum+=d;
                flog+=d*std::log(il+ir+tiny);
                nov+=d*0.5*(double(nvL[b])+double(nvR[b]));
                seta+=d*0.5*(etaL+etaR);
            }
            psycho_dim_sum=dsum;
            ff.dim=float(dsum);
            ff.fear_loggeo=float(flog/std::max(dsum,1.0));
            ff.fear_novelty=float(nov/std::max(dsum,1.0));
            ff.fear_signed_novelty=float(seta/std::max(dsum,1.0));
        }
        fbr_feat.push_back(ff);

        double stereo_score=0.0, stereo_abs=0.0;constexpr double stiny=std::numeric_limits<double>::min();
        const int sdet=stereo_detector_cfg();
        double l1lr_sum=0.0,l1ms_sum=0.0;
        for(int b=0;b<ERB_BANDS;++b){
            const double el=enL[b],er=enR[b],cr=cross[b],w=double(imL[b])+double(imR[b]);
            const int nb=std::max(1,frame_edge[b+1]-frame_edge[b]);
            const double sum=el+er+stiny, em=0.5*(sum+2.0*cr), es=0.5*(sum-2.0*cr);
            const double lrprod=el*er+stiny, msprod=std::max(em*es,stiny);
            const double lrl=double(l1LR[b])+stiny, msl=double(l1MS[b])+stiny;
            l1lr_sum+=lrl;l1ms_sum+=msl;
            double u=0.0;
            switch(sdet){
                case 0: u=w*(std::abs(2.0*cr)-std::abs(el-er))/sum; break;
                case 1: u=-w*std::log(msl/lrl); break;
                case 2: u=-double(nb)*std::log(msl/lrl); break;
                case 3: u=-w*std::log(msprod/lrprod); break;
                case 4: u=-double(nb)*std::log(msprod/lrprod); break;
                case 5: u=(std::abs(2.0*cr)-std::abs(el-er))/sum; break;
                case 6: u=w*((4.0*cr*cr)-(el-er)*(el-er))/(sum*sum); break;
                case 7: u=double(nb)*((4.0*cr*cr)-(el-er)*(el-er))/(sum*sum); break;
                case 8: u=w*((std::abs(2.0*cr)>std::abs(el-er))?1.0:-1.0); break;
                case 9: u=double(nb)*((std::abs(2.0*cr)>std::abs(el-er))?1.0:-1.0); break;
                case 10:{const double norm=std::sqrt(sum/double(2*nb));u=-w*std::log((msl/norm+stiny)/(lrl/norm+stiny));break;}
                case 11: u=-w*std::log((msl*msl/(msprod+stiny)+stiny)/(lrl*lrl/(lrprod+stiny)+stiny)); break;
                default:u=w*(std::abs(2.0*cr)-std::abs(el-er))/sum;break;
            }
            stereo_score+=u;stereo_abs+=std::abs(u);
        }
        if(sdet==12){stereo_score=l1lr_sum-l1ms_sum;stereo_abs=std::abs(stereo_score);}
        modes[fi]=uint8_t(stereo_score>0.0?StereoMode::MS:StereoMode::LR);
        statStereoNorm.push_back(float(stereo_score/std::max(stereo_abs,stiny)));
        statEL.push_back(enL);statER.push_back(enR);statCR.push_back(cross);statL1LR.push_back(l1LR);statL1MS.push_back(l1MS);statMask.push_back(maskERB);statRelMask.push_back(relMaskERB);
        std::array<float,ERB_BANDS> nvMean{}, lfNow{};
        std::array<double,ERB_BANDS> lfmx{};
        double fear_lf_matrix_sum=0.0;
        for(int b=0;b<ERB_BANDS;++b){
            nvMean[b]=0.5f*(nvL[b]+nvR[b]);
            const double lfv=std::log(double(imL[b])+double(imR[b])+std::numeric_limits<double>::min());
            lfmx[b]=lfv;
            lfNow[b]=float(lfv);
            const int n=frame_edge[b+1]-frame_edge[b];
            fear_lf_matrix_sum += (2.0*double(n))*lfv;
        }
        statEta.push_back(local_signed_eta); statNv.push_back(nvMean); statLf.push_back(lfNow);
        // Local signed-FEAR: apply only the zero-common-mode residual
        //   eta_tilde[f,b] = eta[f,b] - <eta[f,*]>_dimension.
        // The unit gain is parameter-free in log-utility space. Because the
        // weighted residual sums to zero, this cannot request extra frame rate;
        // it only moves precision between ERBs inside the frame.
        const double fear_lf_matrix_mean=fear_lf_matrix_sum/std::max(psycho_dim_sum,1.0);

        // W35 production warp: put stationary FEAR importance and signed
        // temporal pressure on the same ERB scale before the A2 self-dual warp.
        // S3 is the normalized Pascal/binomial kernel [1 2 1]/4.
        std::array<double,ERB_BANDS> lf3{};
        for(int b=0;b<ERB_BANDS;++b){
            const int bm1=std::max(0,b-1), bp1=std::min(ERB_BANDS-1,b+1);
            lf3[b]=0.25*(lfmx[bm1]+2.0*lfmx[b]+lfmx[bp1]);
        }

        const double local_eta_mean=psycho_eta_float_sum/std::max(psycho_dim_sum,1.0);

        std::array<double,ERB_BANDS> eta0{}, eta3{};
        for(int b=0;b<ERB_BANDS;++b)
            eta0[b]=double(local_signed_eta[b])-local_eta_mean;
        for(int b=0;b<ERB_BANDS;++b){
            const auto et=[&](int k){return eta0[std::clamp(k,0,ERB_BANDS-1)];};
            eta3[b]=0.25*(et(b-1)+2.0*et(b)+et(b+1));
        }

        fear4w.wait();
        statFear4.push_back(fear4_pressure);
        statFearTonal.push_back(fear4_tonal);

        for(int b=0;b<ERB_BANDS;++b){
            const int n=frame_edge[b+1]-frame_edge[b];
            auto&c=cells[fi*ERB_BANDS+b];
            // Same D_n isotropy was already computed in the masking pass.
            const double iso=isoPre[b];

            c.loga=float(lx_fear_loga(lf3[b],iso,fear_lf_matrix_mean)+eta3[b]+fear4_loga_gain*double(fear4_pressure[b]));
            c.dim=uint16_t(2*n);
            target_dims+=2*n;
        }

        pos+=fr.k;coff+=fr.k;qoff+=fr.coded_bins;
    }


    // 100 local Tensor x Ridge distribution laws.
    // 5 feature definitions x 5 local/temporal maps x 4 uses = 100.
    // Every log-utility path is dimension-weighted zero-common-mode per frame,
    // so it redistributes precision locally without changing the frame budget.
    const int tr_local_cfg=[](){const char* p=::lxvoxel_release_env("LX_TR_LOCAL");return p?std::atoi(p):0;}();
    std::vector<double> tr_beta_signal;
    if(tr_local_cfg>0 && tr_local_cfg<=100){
        tr_beta_signal.assign(cells.size(),0.0);
        const int z=tr_local_cfg-1, feat=z/20, rem=z%20, map=rem/4, use=rem%4;
        for(size_t f=0;f<plan.size();++f){
            std::array<double,ERB_BANDS> x{}, y{};
            for(int b=0;b<ERB_BANDS;++b){
                const double T=std::clamp(double(trTensor[f][b]),0.0,1.0);
                const double R=std::max(0.0,double(trRidge[f][b]));
                const double J=std::max(0.0,double(trJoint[f][b]));
                const double P=std::clamp(double(trPersist[f][b]),0.0,1.0);
                switch(feat){
                    case 0: x[b]=std::sqrt(std::max(0.0,T*R)); break;
                    case 1: x[b]=J; break;
                    case 2: x[b]=T*R; break;
                    case 3: x[b]=std::sqrt(T)*R; break;
                    default:x[b]=std::sqrt(std::max(0.0,T*R))*P; break;
                }
            }
            double frame_sum=0.0,frame_dim=0.0;
            for(int b=0;b<ERB_BANDS;++b){ const double d=double(cells[f*ERB_BANDS+b].dim); frame_sum+=d*x[b]; frame_dim+=d; }
            const double fm=frame_sum/std::max(frame_dim,1.0);
            for(int b=0;b<ERB_BANDS;++b){
                const int bm=std::max(0,b-1),bp=std::min(ERB_BANDS-1,b+1);
                const double sm=0.25*(x[bm]+2.0*x[b]+x[bp]);
                double prev=x[b];
                if(f){ const double Tp=std::clamp(double(trTensor[f-1][b]),0.0,1.0), Rp=std::max(0.0,double(trRidge[f-1][b])), Jp=std::max(0.0,double(trJoint[f-1][b])), Pp=std::clamp(double(trPersist[f-1][b]),0.0,1.0); switch(feat){case 0:prev=std::sqrt(std::max(0.0,Tp*Rp));break;case 1:prev=Jp;break;case 2:prev=Tp*Rp;break;case 3:prev=std::sqrt(Tp)*Rp;break;default:prev=std::sqrt(std::max(0.0,Tp*Rp))*Pp;break;} }
                switch(map){
                    case 0:y[b]=x[b];break;
                    case 1:y[b]=sm;break;
                    case 2:y[b]=x[b]/(x[b]+sm+1e-12);break;
                    case 3:y[b]=x[b]/(x[b]+fm+1e-12);break;
                    default:y[b]=std::sqrt(std::max(0.0,x[b]*prev));break;
                }
            }
            double sum=0.0,dim=0.0;
            for(int b=0;b<ERB_BANDS;++b){const double d=double(cells[f*ERB_BANDS+b].dim);sum+=d*y[b];dim+=d;}
            const double common=sum/std::max(dim,1.0);
            for(int b=0;b<ERB_BANDS;++b){
                const size_t ci=f*ERB_BANDS+b;
                const double q=y[b]-common;
                if(use==0) cells[ci].loga += float(0.25*q);
                else if(use==1) cells[ci].loga += float(0.5*q);
                else if(use==2) tr_beta_signal[ci]=q;
                else { cells[ci].loga += float(0.25*q); tr_beta_signal[ci]=0.5*q; }
            }
        }
    }

    // give the *intra-frame* FEAR allocator the same centered temporal
    // geometry that FBR already uses at frame scale.  This is deliberately
    // zero-common-mode per frame: it cannot change the frame budget, only move
    // rate between ERBs.  All terms are encoder-side analysis only.
#if LX_FEAR_TC_MODE != 0
    {
        const size_t nf=plan.size();
        auto ix=[&](long i)->size_t{return size_t(std::clamp<long>(i,0,long(nf)-1));};
        for(size_t f=0;f<nf;++f){
            const size_t fm=ix(long(f)-1), fp=ix(long(f)+1);
            std::array<double,ERB_BANDS> press{};
            double psum=0.0, dsum=0.0;
            for(int b=0;b<ERB_BANDS;++b){
                const double eta=double(statEta[f][b]);
#if LX_FEAR_TC_MODE == 1
                const double p=-0.5*phi+0.25*(fc-nc);
#elif LX_FEAR_TC_MODE == 2
                const double p=0.5*(-0.5*phi+0.25*(fc-nc));
#elif LX_FEAR_TC_MODE == 3
                const double p=2.0*(-0.5*phi+0.25*(fc-nc));
#elif LX_FEAR_TC_MODE == 4
                const double p=0.25*(fc-nc);
#elif LX_FEAR_TC_MODE == 5
                const double p=0.25*fc;
#elif LX_FEAR_TC_MODE == 6
                const double p=-0.5*phi;
#elif LX_FEAR_TC_MODE == 7
                const double p=0.5*fc-0.25*nc;
#elif LX_FEAR_TC_MODE == 8
                const double p=-0.0625*phi;
#elif LX_FEAR_TC_MODE == 9
                const double p=-0.125*phi;
#elif LX_FEAR_TC_MODE == 10
                const double p=-0.25*phi;
#elif LX_FEAR_TC_MODE == 11
                const double p=-0.375*phi;
#elif LX_FEAR_TC_MODE == 12
                const double p=0.0625*(fc-nc)-0.125*phi;
#elif LX_FEAR_TC_MODE == 13
                const double p=0.03125*(fc-nc)-0.125*phi;
#elif LX_FEAR_TC_MODE == 14
                const double p=(b>=40)?(-0.0625*phi):0.0;
#elif LX_FEAR_TC_MODE == 15
                const double p=(b>=40)?(-0.125*phi):0.0;
#elif LX_FEAR_TC_MODE == 16
                const double p=(b>=40)?(-0.0625*phi+0.015625*(fc-nc)):0.0;
#else
                const double p=0.0;
#endif
                press[b]=p;
                const double d=double(cells[f*ERB_BANDS+b].dim);
                psum+=d*p; dsum+=d;
            }
#if LX_FEAR_TC_MODE >= 14 && LX_FEAR_TC_MODE <= 16
            double hsum=0.0, hdim=0.0;
            for(int b=40;b<ERB_BANDS;++b){const double d=double(cells[f*ERB_BANDS+b].dim);hsum+=d*press[b];hdim+=d;}
            const double common=hsum/std::max(hdim,1.0);
            for(int b=40;b<ERB_BANDS;++b) cells[f*ERB_BANDS+b].loga += float(press[b]-common);
#else
            const double common=psum/std::max(dsum,1.0);
            for(int b=0;b<ERB_BANDS;++b) cells[f*ERB_BANDS+b].loga += float(press[b]-common);
#endif
        }
    }
#endif

#if LX_FEAR_TC_MODE >= 17 && LX_FEAR_TC_MODE <= 20
    {
        const size_t nf=plan.size();
        auto ix=[&](long i)->size_t{return size_t(std::clamp<long>(i,0,long(nf)-1));};
        std::vector<float> base_loga(cells.size());
        for(size_t i=0;i<cells.size();++i) base_loga[i]=cells[i].loga;
        for(size_t f=0;f<nf;++f){
            const size_t fm=ix(long(f)-1), fp=ix(long(f)+1);
            std::array<double,ERB_BANDS> delta{};
            double sum=0.0, dim=0.0;
            for(int b=0;b<ERB_BANDS;++b){
                const double c=base_loga[f*ERB_BANDS+b];
                const double m=base_loga[fm*ERB_BANDS+b];
                const double p=base_loga[fp*ERB_BANDS+b];
#if LX_FEAR_TC_MODE == 17
                const double tc=0.25*(m+2.0*c+p);
#elif LX_FEAR_TC_MODE == 18
                const double tc=0.125*(m+6.0*c+p);
#elif LX_FEAR_TC_MODE == 19
                const double tc=(b>=40)?0.25*(m+2.0*c+p):c;
#elif LX_FEAR_TC_MODE == 20
                const double tc=0.5*c+0.25*(m+p);
#endif
                delta[b]=tc-c;
                const double d=double(cells[f*ERB_BANDS+b].dim); sum+=d*delta[b]; dim+=d;
            }
            const double common=sum/std::max(dim,1.0);
            for(int b=0;b<ERB_BANDS;++b) cells[f*ERB_BANDS+b].loga += float(delta[b]-common);
        }
    }
#endif

    // baseline: global FEAR extension formula 2, alpha=0.375.
    // phi = tau^2 * ATH_share * relative-mask audibility.
    // Applied to all ERBs and shared by allocation / SFC guard.
    std::vector<float> hf_guard_bpc(cells.size(),0.0f);
    std::vector<double> rd_tau(cells.size()), rd_athf(cells.size()), rd_urr(cells.size()), rd_qsat(cells.size()), rd_ur(cells.size()), rd_satmask(cells.size());
    for(size_t f=0;f<plan.size();++f){
        for(int b=0;b<ERB_BANDS;++b){
            const auto& fr=plan[f];
            const auto& ed=edges_for(fr.block);
            const int n=ed[b+1]-ed[b];
            const double d=2.0*double(n);
            const double e=double(statEL[f][b])+double(statER[f][b]);
            const double l1=double(statL1LR[f][b]);
            const double tiny=std::numeric_limits<double>::min();
            const double iso=std::clamp((l1*l1)/(d*e+tiny),1.0/d,1.0);
            const double tau=(1.0-iso)/std::max(1.0-1.0/d,tiny);
            const double mt=std::max(double(statMask[f][b]),tiny);
            const double mr=std::max(double(statRelMask[f][b]),tiny);
            const double ath=std::max(0.0,mt-mr);
            const double athf=ath/(mt+tiny);
            const double ur=e/(e+mt);
            const double urr=e/(e+mr);
            const double q=std::log1p(e/(mr+tiny));
            const double qsat=q/(1.0+q);
            // the V6 HF extension is useful while the source is exposed,
            // but dense material must not receive the same extra HF pressure.
            // Use total audibility E/(E+M_total), not only the external-mask
            // audibility E/(E+M_rel). This continuously removes the extra HF
            // prior as masking rises, with no threshold or signal-value branch.
            const double boost=0.375*tau*tau*athf*ur;
            const size_t ci=f*ERB_BANDS+b;
            cells[ci].loga += float(boost);
            rd_tau[ci]=tau; rd_athf[ci]=athf; rd_urr[ci]=urr; rd_qsat[ci]=qsat; rd_ur[ci]=ur; rd_satmask[ci]=mr/(e+mr+tiny);
            if(b>=lx_sfc::SFC_FIRST_BAND) hf_guard_bpc[ci]=float(0.5*std::max(boost,0.0)/LN2);
        }
    }

    // experimental router: 54 threshold-free analytical line-preservation laws.
    // cfg = feature(6) x temporal(3) x exposure-map(3). cfg 0 = exact R54.
    constexpr int line_cfg=42;
    constexpr int line_use=6;
    constexpr int line_map=0;
    std::vector<double> line_psi(cells.size(),0.0);
    if(line_cfg>0){
        const int z=line_cfg-1, feat=z/9, rem=z%9, tm=rem/3, em=rem%3;
        auto raw_line=[&](size_t f,int b)->double{
            const double a=double(statLineC4[f][b]), c=double(statLineCrest[f][b]), p=double(statLineProm[f][b]);
            switch(feat){
                case 0:return a; case 1:return c; case 2:return p;
                case 3:return std::sqrt(a*p); case 4:return std::sqrt(c*p);
                default:return std::cbrt(std::max(a*c*p,0.0));
            }
        };
        for(size_t f=0;f<plan.size();++f){
            double sum=0.0,dim=0.0;
            for(int b=0;b<ERB_BANDS;++b){
                const size_t ci=f*ERB_BANDS+b; const double q0=std::clamp(raw_line(f,b),0.0,1.0);
                const size_t fm=f?f-1:f, fp=(f+1<plan.size())?f+1:f;
                constexpr int time2=22;
                auto rb=[&](size_t ff,int bb)->double{return std::clamp(raw_line(ff,std::clamp(bb,0,ERB_BANDS-1)),0.0,1.0);};
                const double pm1=rb(fm,b), pp1=rb(fp,b);
                const double qn=std::sqrt(pm1*pp1);
                double q=q0;
                if(time2==0){
                    if(tm==1) q=std::sqrt(q0*qn);
                    else if(tm==2) q=2.0*q0*qn/(q0+qn+std::numeric_limits<double>::min());
                }else{
                    const size_t fm2=f>1?f-2:fm, fp2=(f+2<plan.size())?f+2:fp;
                    const double nprev1=0.25*rb(fm,b-1)+0.5*pm1+0.25*rb(fm,b+1);
                    const double nnext1=0.25*rb(fp,b-1)+0.5*pp1+0.25*rb(fp,b+1);
                    const double nprev2=0.125*rb(fm,b-2)+0.25*rb(fm,b-1)+0.25*pm1+0.25*rb(fm,b+1)+0.125*rb(fm,b+2);
                    const double nnext2=0.125*rb(fp,b-2)+0.25*rb(fp,b-1)+0.25*pp1+0.25*rb(fp,b+1)+0.125*rb(fp,b+2);
                    const double g1=std::sqrt(nprev1*nnext1);
                    const double g2=std::sqrt(nprev2*nnext2);
                    const double g5=std::sqrt(std::sqrt(rb(fm2,b)*pm1)*std::sqrt(pp1*rb(fp2,b)));
                    switch(time2){
                        case 1:q=std::sqrt(q0*std::sqrt(pm1*pp1));break; // current 3-frame geometric
                        case 2:q=std::cbrt(std::max(q0*pm1*pp1,0.0));break;
                        case 3:q=std::sqrt(q0*g1);break; // ERB-neighbor tolerant
                        case 4:q=std::cbrt(std::max(q0*nprev1*nnext1,0.0));break;
                        case 5:q=std::sqrt(q0*g2);break; // wider ERB tolerance
                        case 6:q=std::sqrt(q0*g5);break; // 5-frame persistence
                        case 7:q=std::sqrt(q0*nprev1);break; // causal continuity
                        case 8:q=2.0*q0*g1/(q0+g1+std::numeric_limits<double>::min());break;
                        case 9:q=std::sqrt(q0*std::max({pm1,pp1,nprev1,nnext1}));break;
                        case 10:q=std::sqrt(q0*(0.5*g1+0.5*g5));break;
                        case 11:q=std::sqrt(q0*std::sqrt(std::max(nprev1*nnext1,0.0)));break;
                        case 12:q=std::sqrt(q0*std::sqrt(std::max(g1*g5,0.0)));break;
                        case 13:q=q0+qn-q0*qn;break; // continuous temporal union, same band
                        case 14:q=q0+g1-q0*g1;break; // frequency-tolerant temporal union
                        case 15:q=1.0-std::sqrt(std::max((1.0-q0)*(1.0-g1),0.0));break;
                        case 16:q=std::sqrt(std::max(q0*q0+g1*g1-q0*q0*g1*g1,0.0));break;
                        case 17:q=q0+(1.0-q0)*nprev1;break; // causal carry-through
                        case 18:q=q0+(1.0-q0)*std::sqrt(std::max(nprev1*nnext1,0.0));break;
                        case 19:q=q0+(1.0-q0)*g5;break; // longer-memory carry
                        case 20:q=1.0-(1.0-q0)*(1.0-0.5*g1-0.5*g5);break;
                        case 21:{const double gg=std::sqrt(q0*qn),hh=2.0*q0*qn/(q0+qn+std::numeric_limits<double>::min());q=0.75*gg+0.25*hh;}break;
                        case 22:{const double gg=std::sqrt(q0*qn),hh=2.0*q0*qn/(q0+qn+std::numeric_limits<double>::min());q=0.5*(gg+hh);}break;
                        case 23:{const double gg=std::sqrt(q0*qn),hh=2.0*q0*qn/(q0+qn+std::numeric_limits<double>::min());q=0.25*gg+0.75*hh;}break;
                        default:q=std::sqrt(q0*qn);break;
                    }
                }
                const double u=std::clamp(rd_urr[ci],0.0,1.0), af=std::clamp(rd_athf[ci],0.0,1.0);
                double psi=q*u;
                if(em==1) psi=std::sqrt(q*u);                 // geometric line/audibility
                else if(em==2) psi=q*u*af;                    // explicitly ATH-exposed line
                line_psi[ci]=std::clamp(psi,0.0,1.0);
                double qmap=line_psi[ci];
                if(line_map==1) qmap=-std::log(std::max(1.0-qmap,std::numeric_limits<double>::min()));
                else if(line_map==2) qmap=std::log1p(qmap);
                const double d=double(cells[ci].dim); sum+=d*qmap; dim+=d;
            }
            constexpr int line_scope=0;
            double common=sum/std::max(dim,1.0);
            if(line_scope==1){
                double hs=0.0,hd=0.0;
                for(int b=lx_sfc::SFC_FIRST_BAND;b<ERB_BANDS;++b){const size_t ci=f*ERB_BANDS+b;double qm=line_psi[ci];if(line_map==1)qm=-std::log(std::max(1.0-qm,std::numeric_limits<double>::min()));else if(line_map==2)qm=std::log1p(qm);const double d=double(cells[ci].dim);hs+=d*qm;hd+=d;}
                common=hs/std::max(hd,1.0);
            }
            // Unit log-utility redistribution: exactly zero dimension-weighted common mode.
            if(line_use&1){
                const int b0=line_scope==1?lx_sfc::SFC_FIRST_BAND:0;
                for(int b=b0;b<ERB_BANDS;++b){const size_t ci=f*ERB_BANDS+b; double qmap=line_psi[ci]; if(line_map==1) qmap=-std::log(std::max(1.0-qmap,std::numeric_limits<double>::min())); else if(line_map==2) qmap=std::log1p(qmap); cells[ci].loga += float(qmap-common);}
            }
        }
    }

    // research: physical CPM tone confidence.  Keep structural evidence
    // separate from FEAR exposure, then fuse continuously.  All state is derived
    // from source-domain physical frequency/amplitude descriptors; it is independent
    // of LR/MS and long/short transform representation.
    const int cpm_tone_model=[](){const char* p=::lxvoxel_release_env("LX_CPM_TONE_MODEL");return p?std::atoi(p):1;}();
    std::vector<double> cpm_tone_conf(cells.size(),0.0);
    for(size_t f=0;f<plan.size();++f){
        const size_t fm=f?f-1:f, fp=(f+1<plan.size())?f+1:f;
        for(int b=0;b<ERB_BANDS;++b){
            const size_t ci=f*ERB_BANDS+b;
            const double eps=std::numeric_limits<double>::min();
            const double C=std::clamp(double(statLineCrest[f][b]),0.0,1.0);
            const double Pr=std::clamp(double(statLineProm[f][b]),0.0,1.0);
            const double N=std::sqrt(C*Pr);
            const double f0=double(statLineFreq[f][b]), fm0=double(statLineFreq[fm][b]), fp0=double(statLineFreq[fp][b]);
            const double w0=double(statLineWidth[f][b]), wm=double(statLineWidth[fm][b]), wp=double(statLineWidth[fp][b]);
            const double ferr=fp0-2.0*f0+fm0;
            const double fscale2=w0*w0+wm*wm+wp*wp+eps;
            const double F=1.0/(1.0+ferr*ferr/fscale2);
            const double a0=std::sqrt(std::max(double(statLinePeak[f][b]),0.0));
            const double am=std::sqrt(std::max(double(statLinePeak[fm][b]),0.0));
            const double ap=std::sqrt(std::max(double(statLinePeak[fp][b]),0.0));
            auto asim=[&](double a,double q){return 2.0*a*q/(a*a+q*q+eps);};
            const double PA=std::sqrt(std::clamp(asim(a0,am),0.0,1.0)*std::clamp(asim(a0,ap),0.0,1.0));
            const double u=std::clamp(rd_urr[ci],0.0,1.0), af=std::clamp(rd_athf[ci],0.0,1.0);
            const double E=u*af;
            const double persistent=std::sqrt(std::max(N*F*PA,0.0));
            double t=line_psi[ci];
            if(cpm_tone_model==1){
                t=persistent*E;
            }else{
                // Uniform energy across a band has sigma^2 = BW^2/12.  These
                // quantities are research-only for CPM modes 2..7; CPM1 does
                // not evaluate them in the production hot path.
                const double df=SR/(2.0*double(plan[f].k));
                const auto& pedge=edges_for(plan[f].block);
                const int elo=pedge[b];
                const int ehi=pedge[b+1];
                const double BW=std::max(double(ehi-elo)*df,eps);
                const double diffuse_width=12.0*w0*w0/(BW*BW);
                const double W=1.0/(1.0+diffuse_width);
                const double onset=N*N;
                switch(cpm_tone_model){
                case 2: t=std::cbrt(std::max(N*F*PA,0.0))*E; break;
                case 3:{ // odds/product fusion, computed in log-odds to stay stable
                    const double n=std::clamp(N,0.0,1.0-std::numeric_limits<double>::epsilon()), ff=std::clamp(F,0.0,1.0-std::numeric_limits<double>::epsilon()), pa=std::clamp(PA,0.0,1.0-std::numeric_limits<double>::epsilon());
                    const double o=(n/(1.0-n+eps))*(ff/(1.0-ff+eps))*(pa/(1.0-pa+eps));
                    t=(o/(1.0+o))*E;
                } break;
                case 4:{ // onset/persistent probabilistic union
                    const double p=std::clamp(persistent*E,0.0,1.0), o=std::clamp(onset*E,0.0,1.0);
                    t=1.0-(1.0-p)*(1.0-o);
                } break;
                case 5:{ // harmonic evidence: one weak cue limits confidence
                    const double a=std::max(N,eps), q=std::max(F,eps), r=std::max(PA,eps), w=std::max(W,eps);
                    t=(4.0/(1.0/a+1.0/q+1.0/r+1.0/w))*E;
                } break;
                case 6: t=std::sqrt(std::max(N*F*PA*W,0.0))*E; break;
                case 7:{ // conservative persistent + narrow onset union
                    const double p=std::clamp(std::sqrt(std::max(N*F*PA*W,0.0))*E,0.0,1.0);
                    const double o=std::clamp(N*N*W*E,0.0,1.0);
                    t=1.0-(1.0-p)*(1.0-o);
                } break;
                default: break;
                }
            }
            cpm_tone_conf[ci]=std::clamp(t,0.0,1.0);
        }
    }

    // FEAR V5 HF structural prior: preserve the validated family-7 gain.
    // The union of persistent tonal confidence and line evidence is weighted by
    // source audibility, then applied only to SFC/HF bands.  A dimension-weighted
    // common mode is removed per frame so total requested rate is unchanged.
    // This prior is intentionally separate from the saturation law below.
    {
        constexpr double v5_alpha=0.125;
        for(size_t f=0; f<plan.size(); ++f){
            double sum=0.0, dim=0.0;
            std::array<double,ERB_BANDS> gv{};
            for(int b=0;b<ERB_BANDS;++b){
                const size_t ci=f*ERB_BANDS+b;
                const double p=std::clamp(cpm_tone_conf[ci],0.0,1.0);
                const double l=std::clamp(line_psi[ci],0.0,1.0);
                const double q=std::clamp(rd_qsat[ci],0.0,1.0);
                const double u=1.0-(1.0-p)*(1.0-l);
                const double g=(b>=lx_sfc::SFC_FIRST_BAND)?(v5_alpha*u*q):0.0;
                gv[b]=g;
                const double d=double(cells[ci].dim); sum+=d*g; dim+=d;
            }
            const double common=sum/std::max(dim,1.0);
            for(int b=0;b<ERB_BANDS;++b){
                const size_t ci=f*ERB_BANDS+b;
                cells[ci].loga += float(gv[b]-common);
            }
        }
    }

    // CPM Tone Protection: top-priority tonal budget before global waterfill.
    // The detector is the already frozen temporal/exposure line score in [0,1].
    // We redistribute log-utility with exactly zero dimension-weighted common mode,
    // so CPM reserves a larger share of the fixed frame/song budget rather than
    // increasing the target rate.  The same protection later removes these bits
    // from the rate that SFC is allowed to claim.
    const int cpm_protect_formula=[](){const char* p=::lxvoxel_release_env("LX_CPM_PROTECT_FORMULA");return p?std::atoi(p):5;}();
    // Runtime research controls are process constants. Read them once outside
    // every frame/band hot loop; getenv() must never sit in SFC evaluation.
    const double fear_deficit_sfc=[](){const char* p=::lxvoxel_release_env("LX_FEAR_DEFICIT_SFC");return p?std::atof(p):0.0;}();
    const double cpm_protect_alpha=[](){const char* p=::lxvoxel_release_env("LX_CPM_PROTECT_ALPHA");return p?std::atof(p):2.0;}();
    std::vector<double> cpm_protect(cells.size(),0.0);
    if(cpm_protect_formula>0){
        const double eps=std::numeric_limits<double>::epsilon();
        for(size_t f=0;f<plan.size();++f){
            double sum=0.0,dim=0.0;
            for(int b=0;b<ERB_BANDS;++b){
                const size_t ci=f*ERB_BANDS+b;
                const double p0=std::clamp(cpm_tone_conf[ci],0.0,1.0-eps);
                double g;
                if(cpm_protect_formula==5) g=p0*p0;
                else switch(cpm_protect_formula){
                    case 1: g=p0; break;                                  // linear probability
                    case 2: g=-std::log1p(-p0); break;                     // information / log-survival
                    case 3: g=p0/(1.0-p0+eps); break;                     // odds, strong rare-tone priority
                    case 4: g=std::sqrt(p0); break;                       // broad weak-line protection
                    case 6: g=-std::log1p(-p0*p0); break;                 // selective information barrier
                    default:g=p0;break;
                }
                const double q=cpm_protect_alpha*g;
                cpm_protect[ci]=q;
                const double d=double(cells[ci].dim);sum+=d*q;dim+=d;
            }
            const double common=sum/std::max(dim,1.0);
            for(int b=0;b<ERB_BANDS;++b){const size_t ci=f*ERB_BANDS+b;cells[ci].loga+=float(cpm_protect[ci]-common);}
        }
    }

    // research: tonal/audibility-dependent RD decay.  The baseline high-rate
    // model is D=a*exp(-beta*r), beta=2 ln 2.  beta is now allowed to vary
    // continuously per cell.  Configs are runtime-only experimental controls.
    // cfg 0 means exact R38 baseline. cfg 1..100 = 5 features x 4 maps x 5 strengths.
    const int rd_cfg=[](){const char* p=::lxvoxel_release_env("LX_RD_CFG"); return p?std::atoi(p):0;}();
    const int cpm_warp_mode=[](){const char* p=::lxvoxel_release_env("LX_CPM_WARP_MODE");return p?std::atoi(p):0;}();
    const double cpm_warp_alpha=[](){const char* p=::lxvoxel_release_env("LX_CPM_WARP_ALPHA");return p?std::atof(p):1.0;}();
    std::vector<double> rd_beta(cells.size(),2.0*LN2);
    if(tr_local_cfg>0 && tr_local_cfg<=100){
        const int use=(tr_local_cfg-1)%4;
        if(use==2 || use==3) for(size_t ci=0;ci<rd_beta.size();++ci){
            const double q=tr_beta_signal[ci];
            rd_beta[ci]=(2.0*LN2)*std::exp(-0.25*q);
        }
    }
    // Modes 2/3 alter the high-rate RD slope rather than only the amplitude prior.
    // A stable tone then keeps marginal utility for additional bits for longer.
    if(cpm_warp_mode==2 || cpm_warp_mode==3){
        for(size_t ci=0;ci<cells.size();++ci){
            const double p=std::clamp(cpm_tone_conf[ci],0.0,1.0);
            const double g=p*p;
            rd_beta[ci]=(2.0*LN2)/(1.0+cpm_warp_alpha*g);
        }
    }
    if(rd_cfg>0){
        static constexpr double A[5]={0.125,0.25,0.5,1.0,2.0};
        const int z=rd_cfg-1, feat=z/20, rem=z%20, map=rem/5, ai=rem%5;
        const double alpha=A[ai];
        for(size_t ci=0;ci<cells.size();++ci){
            const double t=rd_tau[ci], af=rd_athf[ci], u=rd_urr[ci], qs=rd_qsat[ci];
            double psi=0.0;
            switch(feat){
                case 0: psi=t*u; break;
                case 1: psi=t*t*u; break;
                case 2: psi=t*af*u; break;
                case 3: psi=t*t*af*u; break;
                default: psi=t*qs*(0.25+0.75*af); break;
            }
            psi=std::clamp(psi,0.0,1.0);
            double g=1.0;
            switch(map){
                case 0: g=std::exp(-alpha*psi); break;              // slower decay for tonal/exposed
                case 1: g=1.0/(1.0+alpha*psi); break;               // rational slower decay
                case 2: g=std::exp(+alpha*psi); break;              // control: faster decay
                default: g=1.0+alpha*psi; break;                     // control: linear faster decay
            }
            rd_beta[ci]=(2.0*LN2)*g;
        }
    }

    // FEAR-v4 structural fragility on the RD slope.  The same zero-common-mode
    // pressure used for the amplitude prior now changes how long marginal utility
    // persists with rate.  Positive structural evidence slows the exponential
    // distortion decay (smaller beta), so waterfilling spends more rate before
    // declaring that auditory object saturated.  This is continuous and adds no
    // signal-dependent branch to the hot loop.
    if(fear4_beta_alpha!=0.0){
        for(size_t f=0;f<plan.size();++f){
            for(int b=0;b<ERB_BANDS;++b){
                const size_t ci=f*ERB_BANDS+b;
                const double p4=double(statFear4[f][b]);
                rd_beta[ci] *= std::exp(-fear4_beta_alpha*p4);
            }
        }
    }

    // Tone-aware psychoacoustic warp on the distortion prior.  This is a
    // zero-common-mode redistribution, hence no extra frame rate request.
    if(cpm_warp_mode==1 || cpm_warp_mode==3){
        for(size_t f=0;f<plan.size();++f){
            double sum=0.0,dim=0.0;
            for(int b=0;b<ERB_BANDS;++b){const size_t ci=f*ERB_BANDS+b;const double g=cpm_warp_alpha*cpm_tone_conf[ci]*cpm_tone_conf[ci];const double d=double(cells[ci].dim);sum+=d*g;dim+=d;}
            const double common=sum/std::max(dim,1.0);
            for(int b=0;b<ERB_BANDS;++b){const size_t ci=f*ERB_BANDS+b;const double g=cpm_warp_alpha*cpm_tone_conf[ci]*cpm_tone_conf[ci];cells[ci].loga+=float(g-common);}
        }
    }


    // research: perceptual residual equalization priors.
    // These maps do not add rate. They only redefine the relative perceptual
    // error surface seen by the single global KKT solve. Mode 0 is exact V5.
    // The aim is to make "the next bit" follow remaining audible error rather
    // than absolute low/mid-band precision.
    const int eq_mode=[](){const char* p=::lxvoxel_release_env("LX_EQ_MODE");return p?std::atoi(p):0;}();
    const double eq_alpha=[](){const char* p=::lxvoxel_release_env("LX_EQ_ALPHA");return p?std::atof(p):0.0;}();
    if(eq_mode>0 && eq_alpha!=0.0){
        const double eps=std::numeric_limits<double>::min();
        const int eq_norm=[](){const char* p=::lxvoxel_release_env("LX_EQ_NORM");return p?std::atoi(p):0;}();
        for(size_t f=0;f<plan.size();++f){
            double sum=0.0, dim=0.0;
            std::array<double,ERB_BANDS> zz{}, gg{};
            for(int b=0;b<ERB_BANDS;++b){
                const size_t ci=f*ERB_BANDS+b;
                const double e=double(statEL[f][b])+double(statER[f][b]);
                const double mr=std::max(double(statRelMask[f][b]),eps);
                const double mt=std::max(double(statMask[f][b]),eps);
                const double aud=e/(e+mr);
                const double q=std::log1p(e/mr);                 // exposed-error headroom, unbounded
                const double qt=std::log1p(e/mt);
                const double m=mr/(e+mr);                       // relative masking budget
                const double ma=mt/(e+mt);
                const double A=std::exp(double(cells[ci].loga));
                const double imp=(b>=lx_sfc::SFC_FIRST_BAND)
                    ? std::clamp((1.0-(1.0-std::clamp(cpm_tone_conf[ci],0.0,1.0))*(1.0-std::clamp(line_psi[ci],0.0,1.0)))*std::clamp(rd_qsat[ci],0.0,1.0),0.0,1.0)
                    : 0.0;
                double z=0.0;
                switch(eq_mode){
                    case 1: z=-std::log(std::max(m,eps)); break;                         // relative-mask JND equalization
                    case 2: z=q; break;                                                  // log exposed-headroom
                    case 3: z=std::sqrt(std::max(q,0.0)); break;                         // compressed headroom
                    case 4: z=std::log1p(q); break;                                      // doubly-log headroom
                    case 5: z=aud; break;                                                 // bounded audibility
                    case 6: z=aud*aud; break;                                             // selective audibility
                    case 7: z=-std::log(std::max(ma,eps)); break;                         // total-mask JND equalization
                    case 8: z=qt; break;                                                  // total-mask exposed headroom
                    case 9: z=q*(0.5+0.5*imp); break;                                     // exposed + HF structure
                    case 10:z=q + imp; break;                                              // additive HF object deficit
                    case 11:z=q*(1.0+imp); break;                                          // multiplicative HF object deficit
                    case 12:z=std::log1p(A/(mr+eps)); break;                               // predicted-distortion / relative mask
                    case 13:z=std::log1p(A/(mt+eps)); break;                               // predicted-distortion / total mask
                    case 14:z=std::log1p((A*aud)/(mr+eps)); break;                         // audible predicted residual
                    case 15:z=std::log1p((A*(0.5+0.5*imp))/(mr+eps)); break;               // residual + HF structure
                    default:z=0.0;break;
                }
                zz[b]=z; const double d=double(cells[ci].dim); sum+=d*z; dim+=d;
            }
            const double mean=sum/std::max(dim,1.0);
            double var=0.0;
            if(eq_norm){for(int b=0;b<ERB_BANDS;++b){const double d=double(cells[f*ERB_BANDS+b].dim);const double q=zz[b]-mean;var+=d*q*q;}}
            const double scale=eq_norm?1.0/std::sqrt(var/std::max(dim,1.0)+std::numeric_limits<double>::min()):1.0;
            double gsum=0.0;
            for(int b=0;b<ERB_BANDS;++b){const size_t ci=f*ERB_BANDS+b;const double g=eq_alpha*(zz[b]-mean)*scale;gg[b]=g;gsum+=double(cells[ci].dim)*g;}
            const double common=gsum/std::max(dim,1.0);
            for(int b=0;b<ERB_BANDS;++b) cells[f*ERB_BANDS+b].loga += float(gg[b]-common);
        }
    }

    // stable allocation invariant:
    //   1) Build the complete FEAR/RD surface.
    //   2) Solve KKT exactly once at the nominal allocation target.
    //   3) Freeze every relative frame/band rate.
    //   4) Rate control may move only the common quantizer scale.
    // Entropy prediction and outer physical-rate probes therefore cannot feed back
    // into FEAR geometry or re-rank cells.
    // Global FBR Waterfall: frame-scale pressure is folded into the
    // same global RD surface BEFORE waterfilling.  This is the continuous
    // equivalent of the old post-waterfill delta-scale offset:
    //   ds=(8/ln2)*p/sqrt(1+p^2), Delta octave = ds/16
    // and with beta=2 ln2 this maps exactly to a log-distortion shift
    //   d(log a) = -p/sqrt(1+p^2).
    // The selected experimental formula only changes p_f; the global common mode
    // is removed dimension-weighted, so it can only redistribute rate.
    std::vector<double> r81_fbr_frame_g(plan.size(),0.0);
    {
        // quality FBR, bug-fixed without changing its production geometry.
        // The historical log_qabs term was dead here (never populated before
        // this pre-waterfill stage), therefore formula 5 actually reduced to
        //   p = -1/2 phi + 1/2(cF-cN).
        // retunes the same bounded zero-common-mode pressure after the C44 reservoir exposed a severe Adrenalina perceptual allocation failure in the historical weighting.
        const size_t nf=plan.size();
        auto ix=[&](long i)->size_t{return size_t(std::clamp<long>(i,0,long(nf)-1));};
        std::vector<double> g(nf,0.0), pressure(nf,0.0);
        double gsum=0.0, dsum=0.0, p2sum=0.0;
        for(size_t i=0;i<nf;++i){
            const auto& f=fbr_feat[i]; const auto& fm=fbr_feat[ix(long(i)-1)]; const auto& fp=fbr_feat[ix(long(i)+1)];
            const double phi=2.0*double(f.fear_signed_novelty)-0.5*(double(fm.fear_signed_novelty)+double(fp.fear_signed_novelty));
            const double fc=double(f.fear_loggeo)-0.5*(double(fm.fear_loggeo)+double(fp.fear_loggeo));
            const double nc=double(f.fear_novelty)-0.5*(double(fm.fear_novelty)+double(fp.fear_novelty));
            const double pr=0.5*((fc-nc)-phi);
            const double d=double(f.dim); pressure[i]=pr; p2sum+=d*pr*pr; dsum+=d;
        }
        // Analytic pressure normalization: the saturation scale is the track's
        // own dimension-weighted RMS pressure.  This preserves the FEAR-v4
        // semantics of phi vs (cF-cN), but removes the former arbitrary unit
        // saturation radius.  Duplicating a track leaves the scale unchanged.
        const double p2mean=p2sum/std::max(dsum,1.0);
        const double inv_p2=1.0/std::max(p2mean,std::numeric_limits<double>::min());
        for(size_t i=0;i<nf;++i){
            const double pr=pressure[i];
            const double gi=-pr/std::sqrt(1.0+inv_p2*pr*pr);
            g[i]=gi; gsum+=double(fbr_feat[i].dim)*gi;
        }
        const double common=gsum/std::max(dsum,1.0);
        for(size_t f=0;f<nf;++f){
            const double gc=g[f]-common;
            r81_fbr_frame_g[f]=gc;
            const float dg=float(gc);
            for(int b=0;b<ERB_BANDS;++b) cells[f*ERB_BANDS+b].loga += dg;
        }
    }

    // non-causal Fast-FBR perceptual redistribution; geometric multi-objective consensus is the default.
    // All formulas are dimensionless and frame-common-mode-free, so they alter
    // only relative marginal utility; no hidden bitrate boost is possible.
    //
    // mode 0: exact R148.
    // mode 1: geometric consensus of exposure, structure, tonality, unmasking.
    // mode 2: harmonic consensus (strict weakest-link protection).
    // mode 3: RMS consensus (allows one strongly fragile cue to dominate).
    // mode 4: probabilistic union of independent fragility cues.
    // mode 5: information-geometric log-odds of consensus vs masking.
    const int fbr_rd_mode=[](){
        const char* p=::lxvoxel_release_env("LX_FBR_RD_MODE");
        return p?std::atoi(p):0;
    }();

    if(fbr_rd_mode>=1 && fbr_rd_mode<=8){
        const double tiny=std::numeric_limits<double>::min();
        std::vector<double> phi(cells.size(),0.0);
        for(size_t fi=0;fi<plan.size();++fi){
            for(int b=0;b<ERB_BANDS;++b){
                const size_t ci=fi*ERB_BANDS+size_t(b);
                const double a=std::clamp(rd_urr[ci],0.0,1.0);
                const double st=std::clamp(rd_tau[ci],0.0,1.0);
                const double to=std::clamp(double(statFearTonal[fi][b]),0.0,1.0);
                const double um=std::clamp(1.0-rd_satmask[ci],0.0,1.0);
                double q=0.0;
                if(fbr_rd_mode==1){
                    q=std::pow(std::max(tiny,a*st*to*um),0.25);
                }else if(fbr_rd_mode==2){
                    q=4.0/(1.0/(a+tiny)+1.0/(st+tiny)+
                           1.0/(to+tiny)+1.0/(um+tiny));
                }else if(fbr_rd_mode==3){
                    q=0.5*std::sqrt(a*a+st*st+to*to+um*um);
                }else if(fbr_rd_mode==4){
                    q=1.0-(1.0-a)*(1.0-st)*(1.0-to)*(1.0-um);
                }else if(fbr_rd_mode==5){
                    const double c=std::pow(std::max(tiny,a*st*to),1.0/3.0);
                    q=std::log((c+tiny)/(1.0-c+tiny))
                      - std::log((rd_satmask[ci]+tiny)/
                                 (1.0-rd_satmask[ci]+tiny));
                }else{
                    const double x=double(plan[fi].k)/double(K_LONG);
                    if(fbr_rd_mode==6){
                        // Short-block amplification of the robust RMS cue.
                        const double q0=0.5*std::sqrt(a*a+st*st+to*to+um*um);
                        q=q0/std::sqrt(std::max(x,tiny));
                    }else if(fbr_rd_mode==7){
                        // Short-block amplification of four-way consensus.
                        const double q0=std::pow(std::max(tiny,a*st*to*um),0.25);
                        q=q0/std::sqrt(std::max(x,tiny));
                    }else{
                        // Structure/exposure product: preserve shape only where
                        // it is both structured and externally exposed.
                        q=std::sqrt(std::max(tiny,st*to))*a*um;
                    }
                }
                phi[ci]=q;
            }
        }

        // Pure redistribution: remove weighted common-mode from every frame.
        for(size_t fi=0;fi<plan.size();++fi){
            double mean=0.0, dims=0.0;
            for(int b=0;b<ERB_BANDS;++b){
                const size_t ci=fi*ERB_BANDS+size_t(b);
                const double d=double(cells[ci].dim);
                mean += d*phi[ci];
                dims += d;
            }
            mean/=std::max(dims,1.0);
            for(int b=0;b<ERB_BANDS;++b){
                const size_t ci=fi*ERB_BANDS+size_t(b);
                cells[ci].loga += float(phi[ci]-mean);
            }
        }
    }

    double rd_budget_bpc=double(bpc);
    double fbr_physical_kbps=double(physical_target_kbps);
    const double fbr_nominal_kbps=(allocation_target_kbps>0.0f)
        ? double(allocation_target_kbps) : double(physical_target_kbps);
    if(physical_target_kbps>0.0f){
        const double nominal_bits=fbr_nominal_kbps*1000.0*(double(ns)/double(SR));
        rd_budget_bpc=nominal_bits/std::max(target_dims,1.0);
    }
    double fbr_debug_kkt_mu=0.0, fbr_debug_kkt_C=0.0;
    struct KktCell {
        double water_height;
        double active_weight; // dim / beta; zero once inactive
        double inverse_beta;
    };
    std::vector<KktCell> kkt_workspace(cells.size());
    auto solve_kkt=[&](double budget_bpc,std::vector<float>& out){
        // Exact water-filling solution of
        //   r_i = max(0, (h_i - mu) / beta_i)
        // under the global bit budget. A full sort is unnecessary: start with
        // all cells active, remove cells below the current water level, and
        // recompute. The water level can only rise, so every cell leaves once.
        const double min_positive=std::numeric_limits<double>::min();
        const double budget_bits=budget_bpc*target_dims;
        const size_t count=cells.size();
        auto& kkt_cells=kkt_workspace;
        double weighted_height_sum=0.0;
        double active_weight_sum=0.0;
        for(size_t i=0;i<count;++i){
            const double beta=std::max(rd_beta[i],min_positive);
            const double inverse_beta=1.0/beta;
            const double weight=double(cells[i].dim)*inverse_beta;
            const double height=double(cells[i].loga)+std::log(beta);
            kkt_cells[i]={height,weight,inverse_beta};
            weighted_height_sum+=weight*height;
            active_weight_sum+=weight;
        }
        double water_level=(weighted_height_sum-budget_bits)/std::max(active_weight_sum,min_positive);
        for(;;){
            double removed_height_sum=0.0;
            double removed_weight_sum=0.0;
            size_t removed_count=0;
            for(auto& cell:kkt_cells){
                const bool remove=(cell.active_weight>0.0) && (cell.water_height<=water_level);
                if(remove){
                    removed_height_sum+=cell.active_weight*cell.water_height;
                    removed_weight_sum+=cell.active_weight;
                    cell.active_weight=0.0;
                    ++removed_count;
                }
            }
            if(removed_count==0) break;
            weighted_height_sum-=removed_height_sum;
            active_weight_sum-=removed_weight_sum;
            water_level=(weighted_height_sum-budget_bits)/std::max(active_weight_sum,min_positive);
        }
        fbr_debug_kkt_mu=water_level;
        fbr_debug_kkt_C=active_weight_sum;
        out.resize(count);
        for(size_t i=0;i<count;++i){
            const auto& cell=kkt_cells[i];
            out[i]=(cell.active_weight>0.0)
                ? float((cell.water_height-water_level)*cell.inverse_beta)
                : 0.0f;
        }
    };
    // First global KKT pass at the caller's nominal track rate.
    std::vector<float> allocation_bpc;
    solve_kkt(rd_budget_bpc,allocation_bpc);


    // R300 research: dynamic HF direct-coding escape above the low-rate
    // reconstruction boundary. Reconstruction remains the default. A high-HF
    // cell receives true D_n coordinates only when the lower-band folding
    // predictor leaves a structured innovation whose normalized perceptual
    // distortion exceeds the exact KKT opportunity cost of direct coding.
    //
    // For D(r)=A exp(-beta r), KKT gives lambda=beta D(r). Replacing direct
    // coding by a zero-rate predictor is favorable iff
    //     g <= exp(-beta r) * (1 + beta r),
    // where g is predictor distortion normalized by A. We estimate g from the
    // source-domain fold residual multiplied by an existing [0,1] structural
    // confidence; no bitrate threshold is introduced.
    // Mode 20 is the production default: line-coherence weighted,
    // energy-weighted log-amplitude innovation.  Environment override is kept
    // only for research/ablation.
    const int hf_escape_mode=[](){const char* p=::lxvoxel_release_env("LX_HF_ESCAPE_MODE");return p?std::atoi(p):20;}();
    std::vector<uint8_t> lowrate_hf_direct(cells.size(),uint8_t(1));
    if(hf_hybrid_profile && lowrate_cutoff && hf_escape_mode>0){
        const double tiny=std::numeric_limits<double>::min();
        for(size_t fi=0;fi<plan.size();++fi){
            const auto& fr=plan[fi]; const auto& ed=edges_for(fr.block);
            const int cb=lowrate_cut_bins(fr.k);
            const float* L=cL.data()+fr.coeff_off; const float* R=cR.data()+fr.coeff_off;
            for(int b=0;b<ERB_BANDS;++b){
                const int lo=ed[b],hi=ed[b+1];
                if(lo<cb) continue; // below/boundary support always direct
                const int n=hi-lo, src_lo=lo-n;
                const size_t ci=fi*ERB_BANDS+size_t(b);
                double et=0.0,es=0.0,cross=0.0;
                if(src_lo>=0 && n>0){
                    for(int j=0;j<n;++j){
                        const double tl=L[lo+j],tr=R[lo+j],sl=L[src_lo+j],sr=R[src_lo+j];
                        et+=tl*tl+tr*tr; es+=sl*sl+sr*sr; cross+=tl*sl+tr*sr;
                    }
                }
                // Exact least-squares gain with the same positive-energy domain
                // as folding. This is an optimistic predictor: if even this best
                // scaled neighbor cannot explain the line, decoder folding cannot.
                const double gscale=(es>tiny)?std::sqrt(et/es):0.0;
                double err=0.0;
                if(et>tiny && es>tiny){
                    for(int j=0;j<n;++j){
                        const double dl=double(L[lo+j])-gscale*double(L[src_lo+j]);
                        const double dr=double(R[lo+j])-gscale*double(R[src_lo+j]);
                        err+=dl*dl+dr*dr;
                    }
                }else err=et;
                const double innovation=err/std::max(et,tiny);
                const double lp=std::clamp(line_psi[ci],0.0,1.0);
                const double cp=std::clamp(cpm_tone_conf[ci],0.0,1.0);
                double structure=lp;
                switch(hf_escape_mode){
                    case 1: structure=lp; break;
                    case 2: structure=std::sqrt(lp); break;
                    case 3: structure=cp; break;
                    case 4: structure=std::sqrt(lp*cp); break;
                    case 5: structure=1.0-(1.0-lp)*(1.0-cp); break;
                    case 6: structure=lp*cp; break;
                    case 7: structure=std::min(lp,cp); break;
                    case 8: structure=std::sqrt(lp*std::clamp(double(statLineProm[fi][b]),0.0,1.0)); break;
                    case 9: structure=std::sqrt(lp*std::clamp(double(statLineCrest[fi][b]),0.0,1.0)); break;
                    case 10: structure=std::sqrt(lp*std::clamp(double(statFearTonal[fi][b]),0.0,1.0)); break;
                    case 11: structure=std::sqrt(lp*std::clamp(double(trTensor[fi][b]),0.0,1.0)); break;
                    default: structure=lp; break;
                }
                double g=structure*innovation;
                if(hf_escape_mode>=12 && hf_escape_mode<=15){
                    // Line-innovation RD: broadband folding mismatch is irrelevant
                    // for diffuse HF. Measure only survival of the strongest
                    // coherent source line under the same energy-matched fold.
                    const double df=double(SR)/(2.0*double(fr.k));
                    const int ip=std::clamp(int(std::floor(double(statLinePeakFreq[fi][b])/df)),lo,hi-1);
                    const int j=ip-lo;
                    const double at=std::hypot(double(L[ip]),double(R[ip]));
                    const double af=(src_lo>=0 && j<n)?gscale*std::hypot(double(L[src_lo+j]),double(R[src_lo+j])):0.0;
                    const double line_innov=(af-at)*(af-at)/(at*at+tiny);
                    double sc=lp;
                    if(hf_escape_mode==13) sc=std::sqrt(lp);
                    else if(hf_escape_mode==14) sc=cp;
                    else if(hf_escape_mode==15) sc=std::sqrt(lp*cp);
                    g=sc*line_innov;
                }
                if(hf_escape_mode>=16 && hf_escape_mode<=19){
                    const double df=double(SR)/(2.0*double(fr.k));
                    const int ip=std::clamp(int(std::floor(double(statLinePeakFreq[fi][b])/df)),lo,hi-1);
                    const int j=ip-lo;
                    const double at=std::hypot(double(L[ip]),double(R[ip]));
                    const double af=(src_lo>=0 && j<n)?gscale*std::hypot(double(L[src_lo+j]),double(R[src_lo+j])):0.0;
                    const double lr=std::log((af+tiny)/(at+tiny));
                    const double line_innov=lr*lr;
                    double sc=lp;
                    if(hf_escape_mode==17) sc=std::sqrt(lp);
                    else if(hf_escape_mode==18) sc=cp;
                    else if(hf_escape_mode==19) sc=std::sqrt(lp*cp);
                    g=sc*line_innov;
                }
                if(hf_escape_mode>=20 && hf_escape_mode<=23){
                    double wsum=0.0, lsum=0.0;
                    if(src_lo>=0 && n>0){
                        for(int j=0;j<n;++j){
                            const double at=std::hypot(double(L[lo+j]),double(R[lo+j]));
                            const double af=gscale*std::hypot(double(L[src_lo+j]),double(R[src_lo+j]));
                            const double w=at*at;
                            const double lr=std::log((af+tiny)/(at+tiny));
                            wsum+=w; lsum+=w*lr*lr;
                        }
                    }
                    const double line_innov=lsum/std::max(wsum,tiny);
                    double sc=lp;
                    if(hf_escape_mode==21) sc=std::sqrt(lp);
                    else if(hf_escape_mode==22) sc=cp;
                    else if(hf_escape_mode==23) sc=std::sqrt(lp*cp);
                    g=sc*line_innov;
                }
                const double r=std::max(0.0,double(allocation_bpc[ci]));
                const double br=std::max(0.0,rd_beta[ci])*r;
                const double direct_equiv=std::exp(-br)*(1.0+br);
                lowrate_hf_direct[ci]=uint8_t(g>direct_equiv);
            }
        }
    }

    // T-FBR: track-level constrained Fast-FBR.  The inner FEAR/KKT
    // geometry is untouched; only the scalar total track budget moves.
    // Local KKT sensitivity is exact inside the active set:
    //   dmu/dB = -1/C,  B in bits.
    // Therefore C*(mu-mu_ref) is the first-order number of bits needed to
    // bring this track to the corpus reference marginal RD level.  tanh gives
    // a continuous hard-bounded product envelope without signal-value branches.
    const bool tfbr_on=[](){const char* p=::lxvoxel_release_env("LX_TFBR");return p&&std::atoi(p)!=0;}();
    if(tfbr_on && physical_target_kbps>0.0f && fbr_nominal_kbps>0.0){
        const double duration_s=double(ns)/double(SR);
        const double span_frac=[](){const char* p=::lxvoxel_release_env("LX_TFBR_SPAN");return p?std::max(0.0,std::atof(p)):0.02;}();
        const double span_kbps=fbr_nominal_kbps*span_frac;
        if(span_kbps>0.0 && duration_s>0.0){
            const double mu_ref=tfbr_mu_ref(fbr_nominal_kbps);
            const double raw_delta_kbps=fbr_debug_kkt_C*(fbr_debug_kkt_mu-mu_ref)/(1000.0*duration_s);
            const double delta_kbps=span_kbps*std::tanh(raw_delta_kbps/span_kbps);
            fbr_physical_kbps=std::max(std::numeric_limits<double>::min(),fbr_nominal_kbps+delta_kbps);
            rd_budget_bpc=(fbr_physical_kbps*1000.0*duration_s)/std::max(target_dims,1.0);
            solve_kkt(rd_budget_bpc,allocation_bpc);
        }
    }
    const double fbr_target_bits=(fbr_physical_kbps>0.0)
        ? fbr_physical_kbps*1000.0*(double(ns)/double(SR)) : 0.0;

    // second-pass Fast-FBR families.  These are genuinely local-rate
    // corrections: the first KKT solution supplies each cell's predicted
    // operating point r0.  The correction therefore vanishes continuously as
    // the local cell enters the high-rate regime, even inside the same song.
    //
    // mode 9  : q * 2^(-r0)     (amplitude-error asymptote)
    // mode 10 : q * 2^(-2 r0)   (MSE high-rate asymptote)
    // mode 11 : q /(1+2^r0)     (logistic occupancy/scarcity)
    if(fbr_rd_mode>=9 && fbr_rd_mode<=11){
        const double tiny=std::numeric_limits<double>::min();
        std::vector<double> corr(cells.size(),0.0);

        for(size_t fi=0;fi<plan.size();++fi){
            for(int b=0;b<ERB_BANDS;++b){
                const size_t ci=fi*ERB_BANDS+size_t(b);
                const double a=std::clamp(rd_urr[ci],0.0,1.0);
                const double st=std::clamp(rd_tau[ci],0.0,1.0);
                const double to=std::clamp(double(statFearTonal[fi][b]),0.0,1.0);
                const double um=std::clamp(1.0-rd_satmask[ci],0.0,1.0);
                const double q=0.5*std::sqrt(a*a+st*st+to*to+um*um);
                const double r0=std::max(0.0,double(allocation_bpc[ci]));
                double scarcity=0.0;
                if(fbr_rd_mode==9) scarcity=std::exp2(-r0);
                else if(fbr_rd_mode==10) scarcity=std::exp2(-2.0*r0);
                else scarcity=1.0/(1.0+std::exp2(r0));
                corr[ci]=q*scarcity;
            }
        }

        // Pure within-frame redistribution; no hidden bitrate offset.
        for(size_t fi=0;fi<plan.size();++fi){
            double mean=0.0,dims=0.0;
            for(int b=0;b<ERB_BANDS;++b){
                const size_t ci=fi*ERB_BANDS+size_t(b);
                const double d=double(cells[ci].dim);
                mean+=d*corr[ci];
                dims+=d;
            }
            mean/=std::max(dims,1.0);
            for(int b=0;b<ERB_BANDS;++b){
                const size_t ci=fi*ERB_BANDS+size_t(b);
                cells[ci].loga+=float(corr[ci]-mean);
            }
        }
        solve_kkt(rd_budget_bpc,allocation_bpc);
    }

    if(fbr_physical_kbps>0.0){
        // Closed-form high-rate normalization.  For a zero-mean Laplace source,
        // h2 = log2(2 e E|X|).  With the FEAR relative scale surface
        // Delta_i = qbase * 2^(rmax-r_i), the common qbase follows directly
        // from the requested entropy rate.  We then apply one exact sparse-
        // regime correction using the entropy of a rounded Laplace quantizer.
        float allocation_rmax=0.f; for(float r:allocation_bpc) allocation_rmax=std::max(allocation_rmax,r);
        double hsum=0.0,hdim=0.0;
        for(size_t fi=0;fi<plan.size();++fi){
            const auto& pedge=edges_for(plan[fi].block);
            const bool smode=modes[fi]!=0;
            for(int band=0;band<ERB_BANDS;++band){
                const int n=pedge[band+1]-pedge[band];
                const double d=2.0*double(n);
                const double l1=double(smode?statL1MS[fi][band]:statL1LR[fi][band]);
                const double meanabs=l1/std::max(d,1.0);
                if(meanabs>std::numeric_limits<double>::min()){
                    const double rel_log2=double(allocation_rmax)-double(allocation_bpc[fi*ERB_BANDS+band]);
                    hsum += d*(std::log2(2.0*std::numbers::e*meanabs)-rel_log2);
                    hdim += d;
                }
            }
        }
        // Degenerate zero-entropy track: no quantizer scale can manufacture
        // source entropy. Preserve the caller's finite common-scale gauge and
        // let physical rate control observe the true syntax/entropy floor.
        if(!(hdim>0.0)){
            rate_saturated=true;
            for(size_t i=0;i<cells.size();++i) cells[i].bpc=allocation_bpc[i];
        }else{
        const double target_bpd=fbr_target_bits/hdim;
        double log2_qbase=hsum/hdim-target_bpd;
        // Compute rounded-Laplace entropy and its tangent in one traversal.
        // The old path evaluated the same exp/log terms twice per TF cell.
        auto laplace_HS=[](double meanabs,double delta)->std::pair<double,double>{
            if(!(meanabs>0.0) || !(delta>0.0)) return {0.0,0.0};
            const double dd=delta/meanabs;
            const double a=std::exp(-0.5*dd);
            if(a<1e-15) return {0.0,-1.0};
            if(a>1.0-1e-12)
                return {std::max(0.0,std::log2(2.0*std::numbers::e*meanabs/delta)),-1.0};
            const double r=a*a, om=1.0-a, omr=1.0-r;
            const double H=-om*std::log2(om)
                          -a*std::log2(0.5*omr)
                          -a*(1.0+r)/omr*std::log2(a);
            const double gp=(1.0+r)/omr + 4.0*r/(omr*omr);
            const double dHnat_da=(std::log(1.0-a)+1.0)
                +(-std::log(0.5*omr)+2.0*r/omr)
                +(-gp*std::log(a)-(1.0+r)/omr);
            const double dHdx=-0.5*dd*a*dHnat_da;
            return {std::max(H,0.0),dHdx};
        };
        const double q0=std::exp2(log2_qbase);
        double Hsum=0.0,Hdim=0.0,dHsum=0.0;
        for(size_t fi=0;fi<plan.size();++fi){
            const auto& pedge=edges_for(plan[fi].block);
            const bool smode=modes[fi]!=0;
            for(int band=0;band<ERB_BANDS;++band){
                const int n=pedge[band+1]-pedge[band];
                const double dim=2.0*double(n);
                const double l1=double(smode?statL1MS[fi][band]:statL1LR[fi][band]);
                const double meanabs=l1/std::max(dim,1.0);
                const double rel=std::exp2(double(allocation_rmax)-double(allocation_bpc[fi*ERB_BANDS+band]));
                const auto [H,dH]=laplace_HS(meanabs,q0*rel);
                Hsum += dim*H; dHsum += dim*dH; Hdim += dim;
            }
        }
        const double Havg=Hsum/std::max(Hdim,1.0);
        // One feed-forward tangent correction using the exact derivative of the
        // rounded-Laplace entropy.  This is analysis algebra only: no quantizer
        // trial and no encode pass.
        const double dHavg=dHsum/std::max(Hdim,1.0);
        log2_qbase += (target_bpd-Havg)/std::min(dHavg,-1e-6);

        // Exact representability bound for the int32 lattice coordinates.
        // For either LR or the orthonormal MS/IS basis, every coordinate is
        // bounded by sqrt(L^2+R^2).  Use the already measured per-band peak and
        // the exact 1/16-octave relative scale that will be serialized.  This
        // prevents fine-rate requests on sparse signals from driving |q| past
        // the normative int32 alphabet (and therefore removes conversion UB).
        constexpr double QMAX=double(std::numeric_limits<int32_t>::max()-1);
        // max(sqrt(peak)/rel) == sqrt(max(peak/(rel*rel))).  Delay the
        // sqrt until after the reduction: one sqrt per song instead of one per cell.
        double qbase_min_sq=0.0;
        for(size_t fi=0;fi<plan.size();++fi){
            for(int band=0;band<ERB_BANDS;++band){
                const double dr=std::max(0.0,double(allocation_rmax)-double(allocation_bpc[fi*ERB_BANDS+band]));
                const int sk=std::clamp(int(std::nearbyint(16.0*dr)),0,30*16);
                const double rel=std::ldexp(double(LX_DELTA_FRAC16[sk&15]),sk>>4);
                const double peak=std::max(double(statLinePeak[fi][band]),0.0);
                qbase_min_sq=std::max(qbase_min_sq,peak/(rel*rel));
            }
        }
        const double qbase_min=std::sqrt(qbase_min_sq)/QMAX;
        if(qbase_min>0.0){const double lmin=std::log2(qbase_min);if(log2_qbase<lmin){log2_qbase=lmin;rate_saturated=true;}}
        const double C=std::sqrt(12.0/double(K_LONG));
        bpc=float(std::log2(C)-log2_qbase);

        // Commit the one FEAR KKT solution unchanged.  Entropy normalization
        // above changes only global qbase/bpc and never re-ranks FEAR cells.
        for(size_t i=0;i<cells.size();++i) cells[i].bpc=allocation_bpc[i];
        }
    }
    // Micro256 low-band merge.  A 256 transform has too little
    // frequency resolution to justify thirty independent low/mid-band rates.
    // Bands [0,30) therefore share one cell-volume-preserving bpc whenever a
    // Micro256 frame is used.  The dimension-weighted mean preserves exactly
    // sum(n_b * bpc_b), so this is a redistribution, not a hidden rate change.
    static const bool merge_256_low30=[](){const char* p=::lxvoxel_release_env("LX_256_MERGE30");return !p || std::atoi(p)!=0;}();
    for(size_t fi=0;fi<plan.size();++fi){
        if(!merge_256_low30 || plan[fi].block!=BlockSize::Micro256) continue;
        double num=0.0,den=0.0;
        for(int b=0;b<30;++b){
            const double n=double(e256.edge[b+1]-e256.edge[b]);
            num += n*double(cells[fi*ERB_BANDS+b].bpc);
            den += n;
        }
        const float common=float(num/std::max(den,1.0));
        for(int b=0;b<30;++b) cells[fi*ERB_BANDS+b].bpc=common;
    }
    float maxb=0.f;for(const auto&c:cells)maxb=std::max(maxb,c.bpc);
    // Global SFC-band masking level.  The selector must not renormalize away
    // quiet exposed frames: FBR is global, therefore the SFC rate price is
    // referenced to the track-global geometric masking level as well.
    std::vector<double> sfc_frame_logmask(plan.size(),0.0);
    double sfc_global_logmask=0.0;
    for(size_t f=0;f<plan.size();++f){
        double lm=0.0;
        for(int j=0;j<lx_sfc::SFC_BANDS;++j){
            const int b=lx_sfc::SFC_FIRST_BAND+j;
            lm+=std::log(std::max(double(statMask[f][b]),std::numeric_limits<double>::min()));
        }
        lm/=double(lx_sfc::SFC_BANDS);
        sfc_frame_logmask[f]=lm;
        sfc_global_logmask+=lm;
    }
    sfc_global_logmask/=double(std::max<size_t>(1,plan.size()));

    // Explicit 9-bit FEAR-delta scale index per ERB/frame.
    // q carries only the true D_n lattice coordinate; scale is no longer
    // hidden by left-shifting q, so coarser cells now actually reduce entropy.
    std::vector<uint16_t> delta_scale(plan.size()*ERB_BANDS);
    // Parameter-free ridge survival cap, populated below after CPM confidence is
    // available.  30*16 means unconstrained.
    std::vector<uint16_t> tonal_scale_cap(plan.size()*ERB_BANDS,uint16_t(30*16));
    constexpr int line_qcap=0;
    constexpr double line_qcap_alpha=0.9;
    const double qbase=std::sqrt(12.0/double(K_LONG))*std::exp2(-double(bpc));
    for(size_t fi=0;fi<plan.size();++fi){
        const auto* fc=cells.data()+fi*ERB_BANDS;
        for(int b=0;b<ERB_BANDS;++b){
            int sk=std::clamp(int(std::nearbyint(16.0*(double(maxb)-double(fc[b].bpc)))),0,30*16);
            if(line_qcap>0){
                const size_t ci=fi*ERB_BANDS+b;
                const double s0=std::clamp(line_psi[ci],0.0,1.0);
                const double tinyq=std::numeric_limits<double>::epsilon();
                const double odds=s0/(1.0-s0+tinyq);
                const double rho=std::exp(-line_qcap_alpha*odds);
                // For an orthonormal stereo basis, at least one coordinate of a
                // stereo line with energy z_peak has magnitude >= sqrt(z_peak/2).
                // Nearest-integer quantization has coordinate error <= Delta/2.
                // Thus Delta <= rho*sqrt(2*z_peak) enforces a relative line-error
                // envelope rho without any absolute level threshold.
                const double peak=std::max(double(statLinePeak[fi][b]),tinyq);
                const double dmax=rho*std::sqrt(2.0*peak);
                if(std::isfinite(dmax) && dmax>qbase){
                    const int cap=std::clamp(int(std::floor(16.0*std::log2(dmax/qbase))),0,30*16);
                    sk=std::min(sk,cap);
                }else if(std::isfinite(dmax)){
                    sk=0;
                }
            }
            delta_scale[fi*ERB_BANDS+b]=uint16_t(sk);
        }
    }


    if(lowrate128 && lowrate_tone_floor){
        // Only strict local maxima of the persistent tonal-confidence field are
        // protected.  This defines actual ridges without a trained confidence
        // threshold.  For a D_n coordinate, nearest-integer quantization can erase
        // a line when Delta/2 exceeds its coordinate amplitude.  Stereo line energy
        // z guarantees max|x| >= sqrt(z/2); confidence c relaxes uncertain ridges,
        // giving the continuous survival bound c*Delta <= sqrt(2 z).
        for(size_t fi=0;fi<plan.size();++fi){
            for(int b=0;b<ERB_BANDS;++b){
                const size_t ci=fi*ERB_BANDS+b;
                const auto ridge_of=[&](const std::vector<double>& v)->bool{
                    const double c=v[ci],fl=(b>0)?v[ci-1]:-1.0,fr=(b+1<ERB_BANDS)?v[ci+1]:-1.0;
                    return c>0.0 && c>fl && c>=fr;
                };
                const bool rc=ridge_of(cpm_tone_conf);
                const bool rl=ridge_of(line_psi);
                double c=0.0; bool ridge=false;
                switch(lowrate_floor_mode){
                    case 1: ridge=rc; c=std::clamp(cpm_tone_conf[ci],0.0,1.0); break;
                    case 2: ridge=rl; c=std::clamp(line_psi[ci],0.0,1.0); break;
                    case 3:{
                        const double cc=std::clamp(cpm_tone_conf[ci],0.0,1.0),ll=std::clamp(line_psi[ci],0.0,1.0);
                        c=std::sqrt(cc*ll);
                        const double l0=(b>0)?std::sqrt(std::clamp(cpm_tone_conf[ci-1],0.0,1.0)*std::clamp(line_psi[ci-1],0.0,1.0)):-1.0;
                        const double r0=(b+1<ERB_BANDS)?std::sqrt(std::clamp(cpm_tone_conf[ci+1],0.0,1.0)*std::clamp(line_psi[ci+1],0.0,1.0)):-1.0;
                        ridge=c>0.0 && c>l0 && c>=r0;
                    } break;
                    case 4:{ c=std::min(std::clamp(cpm_tone_conf[ci],0.0,1.0),std::clamp(line_psi[ci],0.0,1.0)); ridge=rc||rl; } break;
                    default: ridge=rc&&rl; c=std::sqrt(std::clamp(cpm_tone_conf[ci],0.0,1.0)*std::clamp(line_psi[ci],0.0,1.0)); break;
                }
                if(!ridge || !(c>0.0)) continue;
                const auto& frp=plan[fi]; const auto& edp=edges_for(frp.block);
                const double df=double(SR)/(2.0*double(frp.k));
                const int ip=std::clamp(int(std::floor(double(statLinePeakFreq[fi][b])/df)),edp[b],edp[b+1]-1);
                const float L=cL[frp.coeff_off+size_t(ip)],R=cR[frp.coeff_off+size_t(ip)];
                const double A=(modes[fi]!=0)?double((L+R)*INV_SQRT2):double(L);
                const double B=(modes[fi]!=0)?double((L-R)*INV_SQRT2):double(R);
                const double dmax=2.0*std::max(std::abs(A),std::abs(B));
                if(std::isfinite(dmax)){
                    if(dmax<=qbase) tonal_scale_cap[ci]=0;
                    else tonal_scale_cap[ci]=uint16_t(std::clamp(int(std::floor(16.0*std::log2(dmax/qbase))),0,30*16));
                }
            }
        }
    }


    // Normal Intensity Stereo profile. The 128-kbit/s reference placed the useful
    // operating point in the highest four FEAR ERBs.  Those four bands use one
    // KLT principal-axis shape plus a 2-bit signed angle each; lower bands keep
    // the frame's ordinary LR/MS basis.  No per-band enable map is required.
    //
    // This is intentionally a profile/tool decision, not a hidden bitrate
    // threshold.  The current v0.2.10 encoder enables the low-rate IS tool for
    // this branch; R75 keeps it disabled by default because the 160-kbit/s validation
    // showed a consistent perceptual loss even for the conservative rank-1 selector.
    // final stereo refinement.  The frame-wide LR/MS basis is selected from
    // the same quantizer geometry that will actually be entropy coded.  For a
    // D_n cell, L1/delta is the shell occupancy proxy, so cfg=1 chooses the
    // orthonormal basis with the smaller resolved shell population.  cfg=2 is
    // the corresponding smooth log-volume proxy; cfg=0 preserves R74 exactly.
    // keep cfg=9 as the product stereo selector. A forced-MS ablation
    // helps ALLES DOPE but regresses other validation tracks, so no global MS bias
    // is introduced without a decoder-synchronous RD justification.
    const int stereo_cfg=[](){const char* p=::lxvoxel_release_env("LX_STEREO_CFG");return p?std::atoi(p):10;}();
    if(stereo_cfg!=0){
        for(size_t fi=0;fi<plan.size();++fi){
            const int bl=block_level(plan[fi].block);
            if(stereo_cfg==4 && bl>=2) continue; // entropy model only where ERB cells retain enough frequency resolution
            if(stereo_cfg==5 && bl<2) continue;  // complementary diagnostic ablation
            if(stereo_cfg==6 && bl!=0) continue; // long-only ablation
            if(stereo_cfg==7 && bl!=1) continue; // 1024-only ablation
            if((stereo_cfg>=8 && stereo_cfg<=12) && bl>=2) continue; // coupled entropy model on resolved blocks only
            double score=0.0, score_abs=0.0;
            for(int b=0;b<ERB_BANDS;++b){
                const size_t ci=fi*ERB_BANDS+b;
                const int sk=int(delta_scale[ci]);
                const int oct=sk>>4, frac=sk&15;
                const double delta=std::ldexp(qbase*double(LX_DELTA_FRAC16[frac]),oct);
                const double a=double(statL1LR[fi][b]);
                const double m=double(statL1MS[fi][b]);
                const int n=edges_for(plan[fi].block)[b+1]-edges_for(plan[fi].block)[b];
                if(stereo_cfg==1){
                    score += (a-m)/std::max(delta,std::numeric_limits<double>::min());
                }else if(stereo_cfg==2){
                    const double floor=double(2*n)*delta;
                    score += double(n)*std::log((a+floor)/(m+floor));
                }else{
                    // Mid-tread uniform quantization of a zero-mean Laplace source.
                    // The mean absolute value is the Laplace scale b.  This closed
                    // form is the exact entropy of round(x/delta) under that model,
                    // so the frame basis minimizes predicted symbol entropy directly.
                    const auto qlap_h=[&](double l1)->double{
                        const double dim=double(2*n);
                        const double bs=l1/std::max(dim,1.0);
                        if(!(bs>0.0)) return 0.0;
                        const double d=delta/bs;
                        if(d>80.0) return 0.0;
                        const double r=std::exp(-d), q=std::exp(-0.5*d);
                        const double p0=1.0-q;
                        const double B=0.5*(1.0-r)*std::exp(0.5*d);
                        const double S=0.5*q;
                        const double T=S/std::max(1.0-r,std::numeric_limits<double>::min());
                        const double invln2=1.0/LN2;
                        double H=0.0;
                        if(p0>0.0) H-=p0*std::log(p0)*invln2;
                        if(S>0.0 && B>0.0 && r>0.0)
                            H-=2.0*(S*std::log(B)+T*std::log(r))*invln2;
                        return std::max(H,0.0);
                    };
                    const double u=double(2*n)*(qlap_h(a)-qlap_h(m));
                    score += u; score_abs += std::abs(u);
                }
            }
            if(stereo_cfg>=8 && stereo_cfg<=12){
                const double ent=score/std::max(score_abs,std::numeric_limits<double>::min());
                static const double sa=[](){const char* p=::lxvoxel_release_env("LX_TD_STRUCT_ALPHA");return p?std::atof(p):20.0;}();
                const double old=double(statStereoNorm[fi]);
                const double stationary=std::exp(-sa*double(statStructMotion[fi]));
                const double unc=1.0-std::abs(old);
                const double joint = (stereo_cfg==8)
                    ? ((1.0-stationary)*old + stationary*ent)
                    : (stereo_cfg==9)
                        ? (old + stationary*unc*ent)
                        : (stereo_cfg==10)
                            ? (old + stationary*unc*unc*ent)
                            : (stereo_cfg==11)
                                ? (old + stationary*unc*unc*unc*ent)
                                : (old + stationary*unc*unc*unc*unc*ent);
                modes[fi]=uint8_t(joint>0.0?StereoMode::MS:StereoMode::LR);
            }else{
                modes[fi]=uint8_t(score>0.0?StereoMode::MS:StereoMode::LR);
            }
        }
    }


    // PSQ: per-band stereo-basis detector.  The decoder-facing research
    // syntax is one basis bit per ERB (0=L/R, 1=M/S).  The spatial precision
    // curve itself is deterministic from band index and therefore costs no bits.
    std::vector<std::array<uint8_t,ERB_BANDS>> psq_mode(plan.size());
    for(size_t fi=0;fi<plan.size();++fi){
        psq_mode[fi].fill(modes[fi]);
        if(!psq_enable_cfg()) continue;
        const auto& fr=plan[fi]; const float* L=cL.data()+fr.coeff_off; const float* R=cR.data()+fr.coeff_off;
        const auto& ed=edges_for(fr.block);
        for(int b=0;b<ERB_BANDS;++b){
            const int lo=ed[b],hi=ed[b+1],n=hi-lo;
            if(n<=0){psq_mode[fi][b]=modes[fi];continue;}
            const double el=statEL[fi][b],er=statER[fi][b],cr=statCR[fi][b];
            const int sk=int(delta_scale[fi*ERB_BANDS+b]),oct=sk>>4,frac=sk&15;
            const double d=std::ldexp(qbase*double(LX_DELTA_FRAC16[frac]),oct);
            const double g=double(psq_side_gain(b));
            const double dM=d*g,dS=d/g;
            auto exact_cost=[&](bool ms,int kind)->double{
                double et=0.0,es=0.0,sideE=0.0;
                double qel=0.0,qer=0.0,qcr=0.0;
                for(int i=lo;i<hi;++i){
                    const double l=L[i],r=R[i]; double lq,rq;
                    if(!ms){lq=std::nearbyint(l/d)*d;rq=std::nearbyint(r/d)*d;}
                    else{
                        const double m=(l+r)*double(INV_SQRT2),sd=(l-r)*double(INV_SQRT2);
                        const double mq=std::nearbyint(m/dM)*dM,sq=std::nearbyint(sd/dS)*dS;
                        lq=(mq+sq)*double(INV_SQRT2);rq=(mq-sq)*double(INV_SQRT2);
                    }
                    const double dl=l-lq,dr=r-rq,ss=l-r,ssq=lq-rq;
                    et+=dl*dl+dr*dr;es+=(ss-ssq)*(ss-ssq);sideE+=ss*ss;
                    qel+=lq*lq;qer+=rq*rq;qcr+=lq*rq;
                }
                const double tiny=std::numeric_limits<double>::min();
                const double rel=et/(el+er+tiny), srel=es/(sideE+tiny);
                if(kind==3) return et;
                if(kind==4) return rel+srel;
                const double rho=cr/std::sqrt(el*er+tiny),rhoq=qcr/std::sqrt(qel*qer+tiny);
                if(kind==5) return rel+(rho-rhoq)*(rho-rhoq);
                const double ild=std::log((el+tiny)/(er+tiny)),ildq=std::log((qel+tiny)/(qer+tiny));
                if(kind==6) return rel+(rho-rhoq)*(rho-rhoq)+(ild-ildq)*(ild-ildq);
                if(kind==7){const double x=double(b)/double(ERB_BANDS-1);return x*rel+(1.0-x)*srel;}
                return rel;
            };
            auto refine_metric=[&](bool refined,int kind)->double{
                const bool frame_ms=(modes[fi]!=0);
                double et=0.0,es=0.0,tw=0.0,ang=0.0,sideE=0.0,totE=0.0;
                double qel=0.0,qer=0.0,qcr=0.0;
                for(int i=lo;i<hi;++i){
                    const double l=L[i],r=R[i]; double lq,rq;
                    if(refined){
                        const double m=(l+r)*double(INV_SQRT2),sd=(l-r)*double(INV_SQRT2);
                        const double mq=std::nearbyint(m/dM)*dM,sq=std::nearbyint(sd/dS)*dS;
                        lq=(mq+sq)*double(INV_SQRT2);rq=(mq-sq)*double(INV_SQRT2);
                    }else if(frame_ms){
                        const double m=(l+r)*double(INV_SQRT2),sd=(l-r)*double(INV_SQRT2);
                        const double mq=std::nearbyint(m/d)*d,sq=std::nearbyint(sd/d)*d;
                        lq=(mq+sq)*double(INV_SQRT2);rq=(mq-sq)*double(INV_SQRT2);
                    }else{lq=std::nearbyint(l/d)*d;rq=std::nearbyint(r/d)*d;}
                    const double dl=l-lq,dr=r-rq,ss=l-r,ssq=lq-rq,x2=l*l+r*r,e2=dl*dl+dr*dr;
                    et+=e2;es+=(ss-ssq)*(ss-ssq);sideE+=ss*ss;totE+=x2;tw+=x2*e2;
                    const double a0=std::atan2(r,l),a1=std::atan2(rq,lq);double da=std::remainder(a0-a1,2.0*double(LX_PI));ang+=x2*da*da;
                    qel+=lq*lq;qer+=rq*rq;qcr+=lq*rq;
                }
                const double tiny=std::numeric_limits<double>::min();
                const double rel=et/(totE+tiny),srel=es/(sideE+tiny),trel=tw/(totE*totE/double(n)+tiny),arel=ang/(totE+tiny);
                const double rho=cr/std::sqrt(el*er+tiny),rhoq=qcr/std::sqrt(qel*qer+tiny);
                const double ild=std::log((el+tiny)/(er+tiny)),ildq=std::log((qel+tiny)/(qer+tiny));
                switch(kind){
                    case 10:return et;
                    case 11:return srel;
                    case 12:return rel+srel;
                    case 13:return rel+(rho-rhoq)*(rho-rhoq);
                    case 14:return rel+(rho-rhoq)*(rho-rhoq)+(ild-ildq)*(ild-ildq);
                    case 15:{const double x=double(b)/double(ERB_BANDS-1);return x*rel+(1.0-x)*srel;}
                    case 16:return trel;
                    case 17:return trel+srel;
                    case 18:return arel;
                    case 19:return arel+rel;
                    default:return rel;
                }
            };
            bool use_ms=false;
            switch(psq_detector_cfg()){
                case 0: use_ms=(std::abs(2.0*cr)>std::abs(el-er)); break;
                case 1: use_ms=(double(statL1MS[fi][b])<double(statL1LR[fi][b])); break;
                case 2:{
                    auto qlap_h=[&](double l1)->double{
                        const double dim=double(2*n),bs=l1/std::max(dim,1.0); if(!(bs>0.0))return 0.0;
                        const double dd=d/bs;if(dd>80.0)return 0.0;const double rr=std::exp(-dd),q=std::exp(-0.5*dd),p0=1.0-q;
                        const double B=0.5*(1.0-rr)*std::exp(0.5*dd),S=0.5*q,T=S/std::max(1.0-rr,std::numeric_limits<double>::min());
                        double H=0.0;if(p0>0)H-=p0*std::log2(p0);if(S>0&&B>0&&rr>0)H-=2.0*(S*std::log2(B)+T*std::log2(rr));return H;
                    };
                    use_ms=qlap_h(double(statL1MS[fi][b]))<qlap_h(double(statL1LR[fi][b]));break;
                }
                case 3: use_ms=exact_cost(true,3)<exact_cost(false,3);break;
                case 4: use_ms=exact_cost(true,4)<exact_cost(false,4);break;
                case 5: use_ms=exact_cost(true,5)<exact_cost(false,5);break;
                case 6: use_ms=exact_cost(true,6)<exact_cost(false,6);break;
                case 7: use_ms=exact_cost(true,7)<exact_cost(false,7);break;
                case 8:{
                    // KLT-axis proximity: choose the canonical basis whose first
                    // axis lies closest to the covariance principal axis.
                    const double th=0.5*std::atan2(2.0*cr,el-er);
                    const double dlr=std::abs(std::remainder(th,LX_PI));
                    const double dms=std::abs(std::remainder(th-0.25*LX_PI,LX_PI));
                    use_ms=dms<dlr;break;
                }
                case 10: case 11: case 12: case 13: case 14:
                case 15: case 16: case 17: case 18: case 19:
                    // Here the bit means "PSQ refinement active", not a raw basis
                    // selection. Bit 0 reproduces the frame baseline exactly;
                    // bit 1 switches this ERB to M/S and applies the deterministic
                    // side-protection curve.
                    use_ms=refine_metric(true,psq_detector_cfg()) < refine_metric(false,psq_detector_cfg()); break;
                default:use_ms=(modes[fi]!=0);break;
            }
            psq_mode[fi][b]=uint8_t(use_ms);
        }
    }

    std::vector<uint8_t> is_start(plan.size(),uint8_t(ERB_BANDS));
    std::vector<std::array<uint8_t,ERB_BANDS>> is_angle(plan.size());
    std::vector<std::array<float,ERB_BANDS>> is_src_l(plan.size()),is_src_r(plan.size());
    const int is_cfg=2;
    const int is_style=is_style_cfg(), is_metric=is_metric_cfg(), is_bits=is_bits_cfg();
    const double is_qratio=[](){const char* p=::lxvoxel_release_env("LX_IS_QRATIO");return p?std::atof(p):1.0;}();
    const double tiny_is=std::numeric_limits<double>::min();
    for(size_t fi=0;fi<plan.size();++fi){
        // Pooling statistics used by the frame-wide and two-band-group geometries.
        double pel=0.0,per=0.0,pcr=0.0;
        const bool is_metric24_fast = (is_metric==24);
        for(int b=IS_FIRST_BAND;b<ERB_BANDS;++b){pel+=statEL[fi][b];per+=statER[fi][b];pcr+=statCR[fi][b];}
        std::array<double,IS_MAX_BANDS/2> gel{},ger{},gcr{};
        for(int g=0;g<IS_MAX_BANDS/2;++g)for(int b=IS_FIRST_BAND+2*g;b<IS_FIRST_BAND+2*g+2;++b){gel[g]+=statEL[fi][b];ger[g]+=statER[fi][b];gcr[g]+=statCR[fi][b];}

        double struct_l2=0.0,struct_mask=0.0,qsave_l2=0.0,qsave_mask=0.0;
        double rank2_l2=0.0,rank2_mask=0.0,rank2_excess_mask=0.0,rank2_axis_excess_mask=0.0,rank2_axis_q_mask=0.0; bool rank2_all_below_q=true,rank2_all_below_axis_q=true;
        double norm_struct=0.0,norm_qsave=0.0,rate_full=0.0,rate_is=0.0;
        double exact_full_mask=0.0,exact_is_mask=0.0,lambda_num=0.0,lambda_den=0.0;
        double rel_full=0.0,rel_is=0.0,rel_lambda_num=0.0;
        double int_rate_full=0.0,int_rate_is=0.0,exact_oldfull_mask=0.0;
        std::array<double,ERB_BANDS> is_full_lr_gain{}; is_full_lr_gain.fill(1.0);
        if((is_metric==8 || is_metric==9 || is_metric==10 || is_metric==11) && modes[fi]==uint8_t(StereoMode::LR)){
            auto fill_full_lr_gain=[&](const auto& groups){
                for(const auto gg:groups){
                    const int b0=int(gg.b0),b1=int(gg.b1);
                    double elg=0.0,erg=0.0; int qlo=-480,qhi=480;
                    for(int bb=b0;bb<b1;++bb){
                        elg+=double(statEL[fi][bb]); erg+=double(statER[fi][bb]);
                        const int common=int(delta_scale[fi*ERB_BANDS+size_t(bb)]);
                        qlo=std::max(qlo,std::max(-common,common-480));
                        qhi=std::min(qhi,std::min(480-common,common));
                    }
                    elg=std::max(elg,std::numeric_limits<double>::min());
                    erg=std::max(erg,std::numeric_limits<double>::min());
                    const int q=std::clamp(int(std::nearbyint(4.0*std::log2(elg/erg))),qlo,qhi);
                    const double g=std::exp2(double(q)*(1.0/16.0));
                    for(int bb=b0;bb<b1;++bb) is_full_lr_gain[size_t(bb)]=g;
                }
            };
            switch(plan[fi].block){
                case BlockSize::Long2048:fill_full_lr_gain(EGV_LONG);break;
                case BlockSize::Medium1024:fill_full_lr_gain(EGV_1024);break;
                case BlockSize::Short512:fill_full_lr_gain(EGV_SHORT);break;
                case BlockSize::Micro256:fill_full_lr_gain(EGV_256);break;
            }
        }
        for(int b=IS_FIRST_BAND;b<ERB_BANDS;++b){
            const double el=statEL[fi][b],er=statER[fi][b],cr=statCR[fi][b],T=el+er;
            double ael=el,aer=er,acr=cr;
            if(is_style==2||is_style==4||is_style==6){ael=pel;aer=per;acr=pcr;}
            else if(is_style==9||is_style==10){const int g=(b-IS_FIRST_BAND)>>1;ael=gel[g];aer=ger[g];acr=gcr[g];}
            float theta;
            if(is_style==1||is_style==2||is_style==8||is_style==9){
