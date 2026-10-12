                theta=0.5f*std::atan2(float(2.0*acr),float(ael-aer));
            }else{
                const double sg=std::copysign(1.0,acr==0.0?1.0:acr);
                theta=float(std::atan2(sg*std::sqrt(std::max(aer,0.0)),std::sqrt(std::max(ael,0.0))));
            }
            const uint8_t qa=quantize_is_angle(theta); is_angle[fi][b]=qa;
            const auto axis=intensity_axis_d(qa); const double ct=axis.cosine,st=axis.sine;
            double sa=ct,sb=st;
            if(is_style==11){
                // Energy-normalized shared shape.  Normalize each channel by its
                // own band RMS, align correlation sign, then renormalize the
                // merged shape to total band energy.  The transmitted projective
                // angle still carries the L/R energy ratio.
                const double sg=std::copysign(1.0,cr==0.0?1.0:cr);
                const double il=1.0/std::sqrt(std::max(el,tiny_is));
                const double ir=1.0/std::sqrt(std::max(er,tiny_is));
                const double e0=std::max(2.0+2.0*sg*cr*il*ir,tiny_is);
                const double g=std::sqrt(std::max(T,0.0)/e0);
                sa=g*il; sb=sg*g*ir;
            }else if(is_style==5||is_style==6||is_style==10){
                const double sg=std::copysign(1.0,st==0.0?1.0:st);
                const double ez=std::max(el+er+2.0*sg*cr,tiny_is);
                const double g=std::sqrt(std::max(T,0.0)/ez); sa=g;sb=sg*g;
            }else if(is_style==7){
                const double sg=std::copysign(1.0,st==0.0?1.0:st);
                if(el>=er){const double g=std::sqrt(std::max(T,0.0)/std::max(el,tiny_is));sa=g;sb=0.0;}
                else{const double g=std::sqrt(std::max(T,0.0)/std::max(er,tiny_is));sa=0.0;sb=sg*g;}
            }else if(is_style==8){
                const double kept=std::max(ct*ct*el+st*st*er+2.0*ct*st*cr,tiny_is);
                const double g=std::sqrt(std::max(T,0.0)/kept);sa=g*ct;sb=g*st;
            }
            is_src_l[fi][b]=float(sa);is_src_r[fi][b]=float(sb);
            const double Ezz=sa*sa*el+sb*sb*er+2.0*sa*sb*cr;
            const double xdot=ct*(sa*el+sb*cr)+st*(sa*cr+sb*er);
            const double lost=std::max(0.0,T+Ezz-2.0*xdot);
            const size_t ci=fi*ERB_BANDS+b;
            const int sk=int(delta_scale[ci]),oct=sk>>4,frac=sk&15;
            const double delta=std::ldexp(qbase*double(LX_DELTA_FRAC16[frac]),oct);
            const auto&ed=edges_for(plan[fi].block); const int lo=ed[b],hi=ed[b+1],n=hi-lo;
            const double mask=std::max(double(statMask[fi][b]),tiny_is);
            const double qsave=double(n)*delta*delta/12.0;
            // Rank-1 irreducible stereo loss: the second eigenvalue of the
            // physical L/R covariance.  This is basis invariant and is the
            // minimum energy any one-dimensional stereo representation must discard.
            if(!is_metric24_fast){
            const double disc=std::sqrt(std::max(0.0,(el-er)*(el-er)+4.0*cr*cr));
            const double lambda2=0.5*std::max(0.0,T-disc);
            rank2_l2+=lambda2; rank2_mask+=lambda2/mask;
            rank2_excess_mask+=std::max(0.0,lambda2-qsave)/mask;
            rank2_all_below_q = rank2_all_below_q && (lambda2<=qsave);
            // Expected full-stereo quantization noise projected onto the second
            // covariance eigenvector.  This is the relevant water level for
            // deciding whether rank-1 stereo actually discards information that
            // R332 full stereo would have preserved.
            const double phi=0.5*std::atan2(2.0*cr,el-er);
            const double sp=std::sin(phi),cp=std::cos(phi);
            double qL=qsave,qR=qsave;
            if(modes[fi]==uint8_t(StereoMode::LR)){ const double gg=is_full_lr_gain[size_t(b)]; qL=qsave*gg*gg; qR=qsave/(gg*gg); }
            const double q2=sp*sp*qL+cp*cp*qR;
            rank2_axis_excess_mask += std::max(0.0,lambda2-q2)/mask;
            rank2_axis_q_mask += q2/mask;
            rank2_all_below_axis_q = rank2_all_below_axis_q && (lambda2<=q2);
            }
            struct_l2+=lost; qsave_l2+=qsave; struct_mask+=lost/mask; qsave_mask+=qsave/mask;
            norm_struct+=lost/std::max(T,tiny_is); norm_qsave+=qsave/std::max(T,tiny_is);
            const bool smode=modes[fi]!=0;
            const double l1full=double(smode?statL1MS[fi][b]:statL1LR[fi][b]);
            double l1z=0.0;
            const float*pl=cL.data()+plan[fi].coeff_off; const float*pr=cR.data()+plan[fi].coeff_off;
            double efull=0.0,eis=0.0,eold=0.0,efL=0.0,efR=0.0,eiL=0.0,eiR=0.0;
            for(int i=lo;i<hi;++i){
                const double L=pl[i],R=pr[i],z=sa*L+sb*R;if(!is_metric24_fast)l1z+=std::abs(z);
                if(is_metric==5 || is_metric==7 || is_metric==8 || is_metric==9 || is_metric==10 || is_metric==11 || is_metric==13 || is_metric==14 || is_metric==15 || is_metric==22 || is_metric==23 || is_metric==24 || is_metric==25){
                    if((is_metric==8 || is_metric==9 || is_metric==10 || is_metric==11 || is_metric==13 || is_metric==14 || is_metric==15 || is_metric==22 || is_metric==23 || is_metric==24 || is_metric==25) && !smode){
                        const double g=is_full_lr_gain[size_t(b)],dL=delta*g,dR=delta/g;
                        const double qLi=std::nearbyint(L/dL),qRi=std::nearbyint(R/dR);
                        const double Lq=qLi*dL,Rq=qRi*dR;
                        if(is_metric==23 || is_metric==25) int_rate_full += shadow_signed_integer_bits(qLi)+shadow_signed_integer_bits(qRi);
                        const double dfl=L-Lq,dfr=R-Rq; if(!is_metric24_fast){efL+=dfl*dfl;efR+=dfr*dfr;}
                        efull+=dfl*dfl+dfr*dfr;
                        if(is_metric==24 || is_metric==25){ const double Lo=std::nearbyint(L/delta)*delta,Ro=std::nearbyint(R/delta)*delta; eold+=(L-Lo)*(L-Lo)+(R-Ro)*(R-Ro); }
                    }else{
                        const double A=smode?(L+R)*double(INV_SQRT2):L;
                        const double B=smode?(L-R)*double(INV_SQRT2):R;
                        const double qAi=std::nearbyint(A/delta),qBi=std::nearbyint(B/delta);
                        const double Aq=qAi*delta,Bq=qBi*delta;
                        if(is_metric==23 || is_metric==25) int_rate_full += shadow_signed_integer_bits(qAi)+shadow_signed_integer_bits(qBi);
                        if(smode){ const double Lq=(Aq+Bq)*double(INV_SQRT2),Rq=(Aq-Bq)*double(INV_SQRT2); const double dfl=L-Lq,dfr=R-Rq;if(!is_metric24_fast){efL+=dfl*dfl;efR+=dfr*dfr;} }
                        else { const double dfl=L-Aq,dfr=R-Bq;if(!is_metric24_fast){efL+=dfl*dfl;efR+=dfr*dfr;} }
                        efull+=(A-Aq)*(A-Aq)+(B-Bq)*(B-Bq);
                    }
                    const double qzi=std::nearbyint(z/delta),zq=qzi*delta;
                    if(is_metric==23 || is_metric==25) int_rate_is += shadow_signed_integer_bits(qzi);
                    const double dl=L-ct*zq,dr=R-st*zq;if(!is_metric24_fast){eiL+=dl*dl;eiR+=dr*dr;}eis+=dl*dl+dr*dr;
                }
            }
            if(!is_metric24_fast){
            if((is_metric==8 || is_metric==9 || is_metric==10 || is_metric==11 || is_metric==13 || is_metric==14 || is_metric==15 || is_metric==22 || is_metric==23 || is_metric==24 || is_metric==25) && !smode){
                const double g=is_full_lr_gain[size_t(b)]; double l1eff=0.0;
                for(int i=lo;i<hi;++i) l1eff+=std::abs(double(pl[i])/g)+std::abs(double(pr[i])*g);
                rate_full += double(2*n)*qlap_entropy_pc(l1eff,2*n,delta);
            }else rate_full += double(2*n)*qlap_entropy_pc(l1full,2*n,delta);
            rate_is   += double(n)*qlap_entropy_pc(l1z,n,delta);
            if(is_metric==5 || is_metric==7 || is_metric==8 || is_metric==9 || is_metric==10 || is_metric==11 || is_metric==13 || is_metric==14 || is_metric==15 || is_metric==22 || is_metric==23 || is_metric==24 || is_metric==25){exact_full_mask+=efull/mask;exact_is_mask+=eis/mask;}
            if(is_metric==24 || is_metric==25) exact_oldfull_mask += (smode?efull:eold)/mask;
            if(is_metric==22){
                double qL=qsave,qR=qsave; if(!smode){const double gg=is_full_lr_gain[size_t(b)];qL=qsave*gg*gg;qR=qsave/(gg*gg);}
                const double dLden=std::max(el+qL,tiny_is),dRden=std::max(er+qR,tiny_is);
                rel_full += efL/dLden + efR/dRden;
                rel_is   += eiL/dLden + eiR/dRden;
                rel_lambda_num += qL/dLden + qR/dRden;
            }
            if(is_metric==11 && !smode){ const double g=is_full_lr_gain[size_t(b)]; lambda_num += double(n)*(delta*delta/12.0)*(g*g+1.0/(g*g))/mask; }
            else lambda_num += double(2*n)*(delta*delta/12.0)/mask;
            lambda_den += double(2*n);
            }
        }
        bool enable=false;
        if(is_cfg==2 && is_metric24_fast){
            const bool base_ok=(struct_mask<=is_qratio*qsave_mask);
            const bool lr=modes[fi]==uint8_t(StereoMode::LR);
            const bool flipped=(exact_oldfull_mask>=exact_is_mask)&&(exact_full_mask<exact_is_mask);
            enable=base_ok && !(lr&&flipped);
        }else{
        const double side_bits=double(is_code_count_cfg()*is_bits);
        const double rate_save=rate_full-rate_is-side_bits;
        // Global marginal RD pool: IS savings are returned to Fast-FBR, not
        // confined to the HF bands.  Estimate the frame's current water level
        // from all coded ERBs using the actual R332 stereo cell geometry.
        double global_lambda_num=0.0,global_lambda_den=0.0;
        {
            const auto& ged=edges_for(plan[fi].block);
            const bool gsmode=modes[fi]!=0;
            for(int gb=0;gb<ERB_BANDS;++gb){
                const size_t gci=fi*ERB_BANDS+size_t(gb);
                const int gsk=int(delta_scale[gci]),goct=gsk>>4,gfrac=gsk&15;
                const double gd=std::ldexp(qbase*double(LX_DELTA_FRAC16[gfrac]),goct);
                const int gn=ged[gb+1]-ged[gb];
                const double gm=std::max(double(statMask[fi][gb]),tiny_is);
                double qsum=2.0*double(gn)*gd*gd/12.0;
                if(!gsmode){
                    const double gg=is_full_lr_gain[size_t(gb)];
                    qsum=double(gn)*gd*gd*(gg*gg+1.0/(gg*gg))/12.0;
                }
                global_lambda_num += qsum/gm;
                global_lambda_den += double(2*gn);
            }
        }
        // Exact finite-rate high-resolution RD gain for redistributing the
        // symbols saved by rank-1 stereo over the same weighted coefficient pool.
        // D(R) \propto 2^(-2R/N), so unlike the old tangent lambda*dR this remains
        // valid for large IS rate jumps and needs no empirical safety factor.
        const double rd_gain=(rate_save>0.0 && lambda_den>0.0)
            ? lambda_num*(1.0-std::exp2(-2.0*rate_save/lambda_den)) : 0.0;
        const double rd_gain_global=(rate_save>0.0 && global_lambda_den>0.0)
            ? global_lambda_num*(1.0-std::exp2(-2.0*rate_save/global_lambda_den)) : 0.0;
        const double rd_gain_rel=(rate_save>0.0 && lambda_den>0.0)
            ? rel_lambda_num*(1.0-std::exp2(-2.0*rate_save/lambda_den)) : 0.0;
        const double int_rate_save=int_rate_full-int_rate_is-side_bits;
        const double rd_gain_int=(int_rate_save>0.0 && lambda_den>0.0)
            ? lambda_num*(1.0-std::exp2(-2.0*int_rate_save/lambda_den)) : 0.0;
        if(is_cfg==1) enable=true;
        else if(is_cfg==3){
            // Current Opus CELT intensity-band hysteresis: at >=134 kb/s equiv.
            // the result is the off/end state.  At our 160 kb/s benchmark this
            // branch is therefore an exact no-IS reference control.
            enable = fbr_physical_kbps>0.0 && fbr_physical_kbps<134.0;
        }else if(is_cfg==2){
            switch(is_metric){
                case 1: enable=struct_l2<=is_qratio*qsave_l2;break;
                case 2: enable=struct_mask<=is_qratio*qsave_mask;break;
                case 3: enable=norm_struct<=is_qratio*norm_qsave;break;
                case 4: enable=(rate_save>0.0)&&((struct_mask-qsave_mask)<=rd_gain);break;
                case 5: enable=(rate_save>0.0)&&(exact_is_mask<=exact_full_mask);break;
                case 6: enable=(norm_struct<=norm_qsave)&&(struct_mask<=qsave_mask);break;
                case 7: enable=(rate_save>0.0)&&((exact_is_mask-exact_full_mask)<=rd_gain);break;
                case 8: enable=(rate_save>0.0)&&((exact_is_mask-exact_full_mask)<=rd_gain);break;
                case 9: enable=(struct_mask<=is_qratio*qsave_mask)&&(rate_save>0.0)&&((exact_is_mask-exact_full_mask)<=rd_gain);break;
                case 10:{ const bool base_ok=(struct_mask<=is_qratio*qsave_mask); enable=(modes[fi]!=uint8_t(StereoMode::LR))?base_ok:(base_ok&&(rate_save>0.0)&&((exact_is_mask-exact_full_mask)<=rd_gain)); }break;
                case 11:{ const bool base_ok=(struct_mask<=is_qratio*qsave_mask); enable=(modes[fi]!=uint8_t(StereoMode::LR))?base_ok:(base_ok&&(rate_save>0.0)&&((exact_is_mask-exact_full_mask)<=rd_gain)); }break;
                // 12: Gaussian reverse-waterfilling admissibility.  A rank-1
                // model may drop the second stereo eigenmode only if its total
                // irreducible energy lies below the existing quantizer floor.
                case 12: enable=(rate_save>0.0)&&(rank2_l2<=qsave_l2); break;
                // 13: Exact R332 full-stereo distortion versus the *best possible*
                // rank-1 distortion, with finite-rate credit for the saved symbols.
                case 13: enable=(rate_save>0.0)&&((rank2_mask-exact_full_mask)<=rd_gain); break;
                // 14: Preserve the old masking admissibility, then require the
                // basis-invariant rank-1 floor to be RD-profitable on L/R frames.
                case 14:{ const bool base_ok=(struct_mask<=is_qratio*qsave_mask); enable=(modes[fi]!=uint8_t(StereoMode::LR))?base_ok:(base_ok&&(rate_save>0.0)&&((rank2_mask-exact_full_mask)<=rd_gain)); }break;
                // 15: Strict reverse-waterfilling: every eligible HF band must
                // have its second eigenmode below that band's quantizer noise floor.
                case 15: enable=(rate_save>0.0)&&rank2_all_below_q; break;
                // 16: R332-aware reverse-waterfilling.  Only the second-mode
                // energy above the full-stereo quantization water level counts
                // as an IS penalty; saved bits may repay exactly that excess.
                case 16:{ const bool base_ok=(struct_mask<=is_qratio*qsave_mask); enable=(modes[fi]!=uint8_t(StereoMode::LR))?base_ok:(base_ok&&(rate_save>0.0)&&(rank2_axis_excess_mask<=rd_gain)); }break;
                // 17: strict per-band form of the same criterion (diagnostic).
                case 17:{ const bool base_ok=(struct_mask<=is_qratio*qsave_mask); enable=(modes[fi]!=uint8_t(StereoMode::LR))?base_ok:(base_ok&&(rate_save>0.0)&&rank2_all_below_axis_q); }break;
                // 18: pooled perceptual reverse-waterfilling across the IS bands.
                case 18:{ const bool base_ok=(struct_mask<=is_qratio*qsave_mask); enable=(modes[fi]!=uint8_t(StereoMode::LR))?base_ok:(base_ok&&(rate_save>0.0)&&(rank2_mask<=rank2_axis_q_mask)); }break;
                // 19: allow rank-1 only when its excess second-mode loss is no
                // larger than the distortion already present in full R332 stereo.
                case 19:{ const bool base_ok=(struct_mask<=is_qratio*qsave_mask); enable=(modes[fi]!=uint8_t(StereoMode::LR))?base_ok:(base_ok&&(rate_save>0.0)&&(rank2_axis_excess_mask<=exact_full_mask)); }break;
                // 20: same comparison against the analytical full quantizer floor.
                case 20:{ const bool base_ok=(struct_mask<=is_qratio*qsave_mask); enable=(modes[fi]!=uint8_t(StereoMode::LR))?base_ok:(base_ok&&(rate_save>0.0)&&(rank2_axis_excess_mask<=qsave_mask)); }break;
                // 21: true R332 Lagrangian selector.  Full-vs-IS distortion is
                // local, but the saved rate is valued at the frame-global FBR
                // water level because those bits are redistributed globally.
                case 21:{ const bool base_ok=(struct_mask<=is_qratio*qsave_mask); enable=(modes[fi]!=uint8_t(StereoMode::LR))?base_ok:(base_ok&&(rate_save>0.0)&&((exact_is_mask-exact_full_mask)<=rd_gain_global)); }break;
                // 22: channel-relative R332 distortion.  Each physical channel is
                // normalized by its source energy plus its actual full-stereo
                // quantizer floor, so a quiet channel cannot be sacrificed for free.
                case 22:{ const bool base_ok=(struct_mask<=is_qratio*qsave_mask); enable=(modes[fi]!=uint8_t(StereoMode::LR))?base_ok:(base_ok&&(rate_save>0.0)&&((rel_is-rel_full)<=rd_gain_rel)); }break;
                // 23: exact shadow quantizer indices + universal signed-integer
                // description length, avoiding the continuous Laplace-rate proxy.
                case 23:{ const bool base_ok=(struct_mask<=is_qratio*qsave_mask); enable=(modes[fi]!=uint8_t(StereoMode::LR))?base_ok:(base_ok&&(int_rate_save>0.0)&&((exact_is_mask-exact_full_mask)<=rd_gain_int)); }break;
                // 24: architecture-delta crossover. Preserve the old selector
                // unless True Independent L/R alone reverses the quantized
                // distortion ordering between full stereo and IS.
                case 24:{ const bool base_ok=(struct_mask<=is_qratio*qsave_mask); const bool lr=modes[fi]==uint8_t(StereoMode::LR); const bool flipped=(exact_oldfull_mask>=exact_is_mask)&&(exact_full_mask<exact_is_mask); enable=base_ok && !(lr&&flipped); }break;
                // 25: strict Pareto veto. Preserve baseline IS unless R332 full
                // L/R is simultaneously no worse in quantized distortion and
                // no more expensive under the integer shadow description length.
                case 25:{ const bool base_ok=(struct_mask<=is_qratio*qsave_mask); const bool lr=modes[fi]==uint8_t(StereoMode::LR); const bool dominated=(exact_full_mask<=exact_is_mask)&&(int_rate_full<=int_rate_is+side_bits); enable=base_ok && !(lr&&dominated); }break;
                // 26: conservative model consensus.  The validated old selector
                // is vetoed only when both independent R332 rate models reject IS.
                case 26:{
                    const bool base_ok=(struct_mask<=is_qratio*qsave_mask);
                    if(modes[fi]!=uint8_t(StereoMode::LR)) enable=base_ok;
                    else{
                        const bool lap_ok=(rate_save>0.0)&&((exact_is_mask-exact_full_mask)<=rd_gain);
                        const bool int_ok=(int_rate_save>0.0)&&((exact_is_mask-exact_full_mask)<=rd_gain_int);
                        enable=base_ok && (lap_ok||int_ok);
                    }
                }break;
            }
        }
        }
        if(enable) is_start[fi]=uint8_t(IS_FIRST_BAND);

        // VOXEL-IS suffix mode: CELT-style variable intensity start, but
        // with a VOXEL pooled shared-shape representation.  Evaluate each of
        // the four possible HF suffixes analytically and choose the admissible
        // suffix with maximum entropy saving.  No reference and no tuned threshold.
        if(is_cfg==4 && is_style==6){
            int best_start=ERB_BANDS; double best_rate_save=0.0; uint8_t best_q=0;
            for(int s0=IS_FIRST_BAND;s0<ERB_BANDS;++s0){
                double sel=0.0,ser=0.0,scr=0.0;
                for(int b=s0;b<ERB_BANDS;++b){sel+=statEL[fi][b];ser+=statER[fi][b];scr+=statCR[fi][b];}
                const double sgn=std::copysign(1.0,scr==0.0?1.0:scr);
                const float theta=float(std::atan2(sgn*std::sqrt(std::max(ser,0.0)),std::sqrt(std::max(sel,0.0))));
                const uint8_t qa=quantize_is_angle(theta);
                const auto axis=intensity_axis_d(qa); const double ct=axis.cosine,st=axis.sine;
                double sl2=0.0,ql2=0.0,smask=0.0,qmask=0.0,ns=0.0,nq=0.0,rf=0.0,ri=0.0;
                const bool smode=modes[fi]!=0;
                const auto& ed=edges_for(plan[fi].block);
                const float* pl=cL.data()+plan[fi].coeff_off; const float* pr=cR.data()+plan[fi].coeff_off;
                for(int b=s0;b<ERB_BANDS;++b){
                    const double el=statEL[fi][b],er=statER[fi][b],cr=statCR[fi][b],T=el+er;
                    const double sg=std::copysign(1.0,st==0.0?1.0:st);
                    const double ez=std::max(el+er+2.0*sg*cr,tiny_is);
                    const double g=std::sqrt(std::max(T,0.0)/ez),sa=g,sb=sg*g;
                    const double Ezz=sa*sa*el+sb*sb*er+2.0*sa*sb*cr;
                    const double xdot=ct*(sa*el+sb*cr)+st*(sa*cr+sb*er);
                    const double lost=std::max(0.0,T+Ezz-2.0*xdot);
                    const size_t ci=fi*ERB_BANDS+b; const int sk=int(delta_scale[ci]),oct=sk>>4,frac=sk&15;
                    const double delta=std::ldexp(qbase*double(LX_DELTA_FRAC16[frac]),oct);
                    const int lo=ed[b],hi=ed[b+1],n=hi-lo;
                    const double qsave=double(n)*delta*delta/12.0;
                    const double mask=std::max(double(statMask[fi][b]),tiny_is);
                    sl2+=lost; ql2+=qsave; smask+=lost/mask; qmask+=qsave/mask;
                    ns+=lost/std::max(T,tiny_is); nq+=qsave/std::max(T,tiny_is);
                    const double l1full=double(smode?statL1MS[fi][b]:statL1LR[fi][b]); double l1z=0.0;
                    for(int i=lo;i<hi;++i) l1z+=std::abs(sa*double(pl[i])+sb*double(pr[i]));
                    rf+=double(2*n)*qlap_entropy_pc(l1full,2*n,delta);
                    ri+=double(n)*qlap_entropy_pc(l1z,n,delta);
                }
                // Active suffix start requires two bits (1..4 bands) plus one
                // pooled projective angle code. Enable itself reuses ctrl bit 2.
                const double side=2.0+double(is_bits);
                const double rs=rf-ri-side;
                bool admissible=false;
                switch(is_metric){
                    case 1: admissible=sl2<=ql2; break;
                    case 2: admissible=smask<=qmask; break;
                    case 3: admissible=ns<=nq; break;
                    default: admissible=ns<=nq; break;
                }
                if(rs>best_rate_save && rs>0.0 && admissible){best_rate_save=rs;best_start=s0;best_q=qa;}
            }
            is_start[fi]=uint8_t(best_start);
            if(best_start<ERB_BANDS){
                const double st=intensity_axis_d(best_q).sine;
                for(int b=best_start;b<ERB_BANDS;++b){
                    const double el=statEL[fi][b],er=statER[fi][b],cr=statCR[fi][b],T=el+er;
                    const double sg=std::copysign(1.0,st==0.0?1.0:st);
                    const double ez=std::max(el+er+2.0*sg*cr,tiny_is);
                    const double g=std::sqrt(std::max(T,0.0)/ez);
                    is_angle[fi][b]=best_q; is_src_l[fi][b]=float(g); is_src_r[fi][b]=float(sg*g);
                }
            }
        }
    }

    // one-pass rate reservoir.  The global waterfall fixes the relative
    // RD geometry for the entire track.  A compact analytical entropy forecast
    // then assigns a cumulative physical-bit schedule to superframes. During
    // the single real quantizer/entropy pass, measured bit debt is converted
    // directly to a common future quantizer offset through dH/dlog2(Delta).
    // No coefficient is ever quantized twice.
    const int fbr_nc_mode=[](){const char* p=::lxvoxel_release_env("LX_FBR_NC_MODE");return p?std::atoi(p):2;}();
    std::vector<double> fbr_pred_sf_bits, fbr_pred_sf_slope, fbr_pred_prefix;
    std::vector<double> fbr_cell_slope, fbr_cell_w_rd, fbr_cell_w_mask;
    double fbr_coded_target_bits=0.0;
    double fbr_debug_pred_total=0.0, fbr_debug_slope_total=0.0, fbr_debug_fixed_bits=0.0;
    double fbr_pred_total_direct=0.0, fbr_slope_total_direct=0.0;
    if(fbr_physical_kbps>0.0){
        const size_t psf=(plan.size()+MODEL_SUPERFRAME-1)/MODEL_SUPERFRAME;
        if(fbr_nc_mode==0){
            fbr_pred_sf_bits.assign(psf,0.0); fbr_pred_sf_slope.assign(psf,0.0); fbr_pred_prefix.assign(psf+1,0.0);
            fbr_cell_w_rd.assign(cells.size(),1.0);
            fbr_cell_w_mask.assign(cells.size(),1.0);
        }
        // both streaming FBR and the global noncausal Fast-FBR solver
        // need the local entropy Jacobian.  In Fast FBR it is the exact
        // constraint vector for the whole-song minimum-modulation projection.
        if(fbr_nc_mode==0 || fbr_nc_mode==2) fbr_cell_slope.assign(cells.size(),0.0);
        const double base_delta=std::sqrt(12.0/double(K_LONG))*std::exp2(-double(bpc));
        auto h_and_slope=[](double meanabs,double delta)->std::pair<double,double>{
            if(!(meanabs>0.0) || !(delta>0.0)) return {0.0,0.0};
            const double d=delta/meanabs, a=std::exp(-0.5*d);
            if(a<1e-15) return {0.0,0.0};
            if(a>1.0-1e-12) return {std::max(0.0,std::log2(2.0*std::numbers::e*meanabs/delta)),1.0};
            const double r=a*a, om=1.0-a, omr=1.0-r;
            const double H=-om*std::log2(om)-a*std::log2(0.5*omr)-a*(1.0+r)/omr*std::log2(a);
            const double gp=(1.0+r)/omr + 4.0*r/(omr*omr);
            const double dHnat_da=(std::log(1.0-a)+1.0)+(-std::log(0.5*omr)+2.0*r/omr)+(-gp*std::log(a)-(1.0+r)/omr);
            const double dHdx=-0.5*d*a*dHnat_da; // dH / d log2(Delta), negative
            return {std::max(H,0.0),std::max(-dHdx,0.0)};
        };
        for(size_t fi=0;fi<plan.size();++fi){
            const size_t sf=fi/MODEL_SUPERFRAME; const auto& ed=edges_for(plan[fi].block); const bool smode=modes[fi]!=0;
            for(int b=0;b<ERB_BANDS;++b){
                const int n=ed[b+1]-ed[b], sk=int(delta_scale[fi*ERB_BANDS+b]); const int oct=sk>>4, frac=sk&15;
                const double delta=std::ldexp(base_delta*double(LX_DELTA_FRAC16[frac]),oct);
                const double dim=(b>=int(is_start[fi]))?double(n):2.0*double(n);
                const double l1=double(smode?statL1MS[fi][b]:statL1LR[fi][b]);
                const double meanabs=l1/std::max(2.0*double(n),1.0);
                const auto [H,S]=h_and_slope(meanabs,delta);
                const size_t ci=fi*ERB_BANDS+b;
                const double cellSlope=dim*S;
                const double cellBits=dim*H;
                fbr_pred_total_direct+=cellBits; fbr_slope_total_direct+=cellSlope;
                if(!fbr_cell_slope.empty()) fbr_cell_slope[ci]=cellSlope;
                if(fbr_nc_mode==0){
                    // Causal/streaming-only projection weights.  The global
                    // noncausal file path never consumes these arrays, so skip
                    // their extra exp/mask work entirely there.
                    const double rr=double(allocation_bpc[ci]);
                    const double Drd=std::exp(double(cells[ci].loga)-rd_beta[ci]*rr);
                    fbr_cell_w_rd[ci]=std::max(dim*Drd,1e-30);
                    const double mask=std::max(double(statMask[fi][b]),1e-30);
                    fbr_cell_w_mask[ci]=std::max(dim*delta*delta/(12.0*mask),1e-30);
                    fbr_pred_sf_bits[sf]+=cellBits; fbr_pred_sf_slope[sf]+=cellSlope;
                }
            }
        }
        if(fbr_nc_mode==0)
            for(size_t sf=0;sf<psf;++sf) fbr_pred_prefix[sf+1]=fbr_pred_prefix[sf]+fbr_pred_sf_bits[sf];
        // These two syntax terms are known before the quantizer pass and are
        // therefore reserved exactly.  SFC/rANS residual overhead is learned
        // causally from the already emitted superframes.
        uint64_t known_is_bits=0; for(size_t fi=0;fi<plan.size();++fi){const int ist=int(is_start[fi]);const bool on=ist<ERB_BANDS;const uint64_t ib=on?uint64_t((is_cfg==4?2:0)+(is_cfg==4?is_bits_cfg():is_code_count_cfg()*is_bits_cfg())):0u;known_is_bits+=1u+ib;}
        const double fixed_tail_bits=double(4*plan.size())+double(known_is_bits);
        fbr_coded_target_bits=std::max(0.0,fbr_target_bits-fixed_tail_bits);
        fbr_debug_pred_total=(fbr_nc_mode==0)?fbr_pred_prefix.back():fbr_pred_total_direct;
        if(fbr_nc_mode==0){ for(double v:fbr_pred_sf_slope) fbr_debug_slope_total+=v; }
        else fbr_debug_slope_total=fbr_slope_total_direct;
        fbr_debug_fixed_bits=fixed_tail_bits;
        if(const char* dbg=::lxvoxel_release_env("LX_DEBUG_RATE_MODEL")){if(std::atoi(dbg)!=0) std::cout<<"rm_pred_bits="<<fbr_debug_pred_total<<" rm_slope="<<fbr_debug_slope_total<<" rm_fixed_bits="<<fbr_debug_fixed_bits<<" rm_kkt_mu="<<fbr_debug_kkt_mu<<" ";}
    }

    // production Fast FBR: noncausal whole-song physical-rate calibration.
    // The complete FEAR/KKT surface and final 1/16-octave delta_scale are known
    // before packet emission.  A stored physical-rate calibration converts the
    // analytical entropy surface into a whole-song target, then two bounded
    // Newton updates solve the global quantizer offset.  No coefficient is
    // quantized here and no pilot/second encode is required.
    if(fbr_nc_mode==2 && fbr_physical_kbps>0.0 && fbr_pred_total_direct>0.0){
        auto nc_hs=[](double meanabs,double delta)->std::pair<double,double>{
            if(!(meanabs>0.0) || !(delta>0.0)) return {0.0,0.0};
            const double d=delta/meanabs, a=std::exp(-0.5*d);
            if(a<1e-15) return {0.0,0.0};
            if(a>1.0-1e-12) return {std::max(0.0,std::log2(2.0*std::numbers::e*meanabs/delta)),1.0};
            const double r=a*a, om=1.0-a, omr=1.0-r;
            const double H=-om*std::log2(om)-a*std::log2(0.5*omr)-a*(1.0+r)/omr*std::log2(a);
            const double gp=(1.0+r)/omr + 4.0*r/(omr*omr);
            const double dHnat_da=(std::log(1.0-a)+1.0)+(-std::log(0.5*omr)+2.0*r/omr)+(-gp*std::log(a)-(1.0+r)/omr);
            const double dHdx=-0.5*d*a*dHnat_da;
            return {std::max(H,0.0),std::max(-dHdx,0.0)};
        };
        const double nc_rho=[&](double target_kbps){
            if(const char* p=::lxvoxel_release_env("LX_FBR_NC_RHO")) return std::atof(p);
            // R220 measured low-rate physical anchor.  Same role as the frozen
            // R212 rho table: compensate the changed zero/SFC symbol mixture so
            // R339: one physical-rate model across the entire lossy range.
            // The former fixed <140-kbit/s RHO128_LOW caused severe 96-kbit/s
            // overshoot after the R338 sideinfo changes and prevented a smooth
            // 96/112/128 calibration.
            return double(lx::rate_model::rho(float(target_kbps)));
        }(fbr_physical_kbps);
        const double Htarget=fbr_coded_target_bits/std::max(nc_rho,1e-9);
        const double base_delta=std::sqrt(12.0/double(K_LONG))*std::exp2(-double(bpc));

        // Iteration 0 reuses the exact H/S values already accumulated by the
        // mandatory FBR predictor pass above.  This removes one full 48-band
        // whole-song entropy traversal with no change to the mathematical model.
        double Htot=fbr_pred_total_direct, Stot=fbr_slope_total_direct;
        for(int it=0;it<2;++it){
            if(it!=0){
                Htot=0.0; Stot=0.0;
                for(size_t fi=0;fi<plan.size();++fi){
                    const auto& ed=edges_for(plan[fi].block); const bool smode=modes[fi]!=0;
                    for(int b=0;b<ERB_BANDS;++b){
                        const int n=ed[b+1]-ed[b], sk=int(delta_scale[fi*ERB_BANDS+b]);
                        const double delta=std::ldexp(base_delta*double(LX_DELTA_FRAC16[sk&15]),sk>>4);
                        const double dim=(b>=int(is_start[fi]))?double(n):2.0*double(n);
                        const double l1=double(smode?statL1MS[fi][b]:statL1LR[fi][b]);
                        const double meanabs=l1/std::max(2.0*double(n),1.0);
                        const auto [H,S]=nc_hs(meanabs,delta);
                        const double cellSlope=dim*S;
                        Htot+=dim*H; Stot+=cellSlope;
                        fbr_cell_slope[fi*ERB_BANDS+b]=cellSlope;
                    }
                }
            }
            const double x=(Htot-Htarget)/std::max(Stot,1.0);
            if(std::abs(x)<(0.5/16.0)) break;

            // global noncausal minimum-modulation projection.
            // The uniform Newton step x and the minimum-L2 field below have
            // the same first-order entropy correction
            //
            //   C = sum_i S_i s_i = x * sum_i S_i .
            //
            // The minimum-norm solution is
            //
            //   s_i^MM = C S_i / sum_j S_j^2 .
            //
            // As in the validated R128 FBR70 controller, preserve 30% common
            // motion and project 70% onto the minimum-modulation direction.
            // Unlike R128 this is solved once over the complete song, so frame
            // zero has exactly the same future knowledge as every later frame.
            constexpr double NC_MM_BLEND=0.70;
            const double C=x*Stot;
            double den=0.0;
            for(double a:fbr_cell_slope) den+=a*a;
            const double lam=C/std::max(den,1e-30);
            bool changed=false;
            for(size_t ci=0;ci<delta_scale.size();++ci){
                const double smm=lam*fbr_cell_slope[ci];
                const double sh=(1.0-NC_MM_BLEND)*x + NC_MM_BLEND*smm;
                const int dsk=int(std::nearbyint(16.0*sh));
                if(dsk!=0){
                    delta_scale[ci]=uint16_t(std::clamp(int(delta_scale[ci])+dsk,0,30*16));
                    changed=true;
                }
            }
            if(!changed) break;
        }
    }

    // R220 constrained low-rate rebalance.  Preserve the calibrated full-band
    // Fast-FBR entropy point, then remove >18-kHz source entropy and spend exactly
    // that analytical budget below the cutoff while respecting tonal ridge caps.
    if(fbr_nc_mode==2 && ((hf_hybrid_profile && lowrate_cutoff) || (lowrate128 && lowrate_tone_floor))){
        auto lr_hs=[](double meanabs,double delta)->std::pair<double,double>{
            if(!(meanabs>0.0) || !(delta>0.0)) return {0.0,0.0};
            const double d=delta/meanabs,a=std::exp(-0.5*d);
            if(a<1e-15) return {0.0,0.0};
            if(a>1.0-1e-12) return {std::max(0.0,std::log2(2.0*std::numbers::e*meanabs/delta)),1.0};
            const double r=a*a,om=1.0-a,omr=1.0-r;
            const double H=-om*std::log2(om)-a*std::log2(0.5*omr)-a*(1.0+r)/omr*std::log2(a);
            const double gp=(1.0+r)/omr+4.0*r/(omr*omr);
            const double dHnat=(std::log(1.0-a)+1.0)+(-std::log(0.5*omr)+2.0*r/omr)+(-gp*std::log(a)-(1.0+r)/omr);
            return {std::max(H,0.0),std::max(0.5*d*a*dHnat,0.0)};
        };
        const double base=std::sqrt(12.0/double(K_LONG))*std::exp2(-double(bpc));
        const size_t ncells=plan.size()*ERB_BANDS;
        std::vector<double> lr_meanabs(ncells,0.0),lr_dim(ncells,0.0);

        // One coefficient pass per frame computes the active (<18-kHz) Laplace
        // statistics.  Newton iterations below touch only these cached scalars.
        for(size_t fi=0;fi<plan.size();++fi){
            const auto& fr=plan[fi]; const auto& ed=edges_for(fr.block);
            const int cb=lowrate_cut_bins(fr.k); const float* L=cL.data()+fr.coeff_off; const float* R=cR.data()+fr.coeff_off;
            const bool ms=modes[fi]!=0;
            for(int b=0;b<ERB_BANDS;++b){
                const size_t ci=fi*ERB_BANDS+size_t(b);
                const int lo=ed[b], hi=ed[b+1];
                const bool above=(lo>=cb);
                const bool keep_direct=(!above) || (hf_escape_mode>0 && lowrate_hf_direct[ci]);
                const int ahi=keep_direct?hi:std::min(hi,cb);
                const int n=std::max(0,ahi-lo);
                if(n<=0) continue;
                double l1=0.0;
                if(ms){
                    for(int i=lo;i<ahi;++i){const double m=(double(L[i])+R[i])*double(INV_SQRT2),ss=(double(L[i])-R[i])*double(INV_SQRT2);l1+=std::abs(m)+std::abs(ss);}
                }else{
                    for(int i=lo;i<ahi;++i) l1+=std::abs(double(L[i]))+std::abs(double(R[i]));
                }
                lr_meanabs[ci]=l1/std::max(2.0*double(n),1.0);
                lr_dim[ci]=(b>=int(is_start[fi]))?double(n):2.0*double(n);
            }
        }

        auto hs_cached=[&](size_t ci,int sk)->std::pair<double,double>{
            if(!(lr_dim[ci]>0.0)) return {0.0,0.0};
            const double delta=std::ldexp(base*double(LX_DELTA_FRAC16[sk&15]),sk>>4);
            const auto [H,S]=lr_hs(lr_meanabs[ci],delta);
            return {lr_dim[ci]*H,lr_dim[ci]*S};
        };

        // Reference entropy is the already calibrated full-band Fast-FBR point.
        // Reuse the frozen per-band L1 statistics; no coefficient rescan is needed.
        double Href=0.0;
        for(size_t fi=0;fi<plan.size();++fi){
            const auto& ed=edges_for(plan[fi].block); const bool ms=modes[fi]!=0;
            for(int b=0;b<ERB_BANDS;++b){
                const int n=ed[b+1]-ed[b],sk=int(delta_scale[fi*ERB_BANDS+b]);
                const double l1=double(ms?statL1MS[fi][b]:statL1LR[fi][b]);
                const double meanabs=l1/std::max(2.0*double(n),1.0);
                const double dim=(b>=int(is_start[fi]))?double(n):2.0*double(n);
                const double delta=std::ldexp(base*double(LX_DELTA_FRAC16[sk&15]),sk>>4);
                Href+=dim*lr_hs(meanabs,delta).first;
            }
        }

        // Enforce ridge floors before solving the residual budget.
        for(size_t ci=0;ci<delta_scale.size();++ci)
            delta_scale[ci]=uint16_t(std::min(int(delta_scale[ci]),int(tonal_scale_cap[ci])));

        for(int it=0;it<4;++it){
            double H=0.0,Sall=0.0;
            for(size_t ci=0;ci<ncells;++ci){
                const int sk=int(delta_scale[ci]); const auto [h,sl]=hs_cached(ci,sk); H+=h; Sall+=sl;
            }
            const double err=H-Href; if(std::abs(err)<1.0) break;

            // For a positive entropy error, tonal-floor cells are saturated and
            // cannot be coarsened; remove their slope from the active Newton set.
            double S=Sall;
            if(err>0.0){
                for(size_t ci=0;ci<ncells;++ci)
                    if(int(delta_scale[ci])>=int(tonal_scale_cap[ci])) S-=hs_cached(ci,int(delta_scale[ci])).second;
            }
            if(!(S>0.0)) break;
            const int dsk=int(std::nearbyint(16.0*err/S)); if(dsk==0) break;
            bool changed=false;
            for(size_t ci=0;ci<ncells;++ci){
                if(!(lr_dim[ci]>0.0)) continue;
                int next=std::clamp(int(delta_scale[ci])+dsk,0,30*16);
                next=std::min(next,int(tonal_scale_cap[ci]));
                if(next!=int(delta_scale[ci])){delta_scale[ci]=uint16_t(next);changed=true;}
            }
            if(!changed) break;
        }
    }

    // Dead R81 post-waterfill qabs optional path removed; it was compile-time disabled.

    // Obsolete disabled pilot-quantization rate solver removed in R108.1.


    if(vox) vox->set_base_delta(float(std::sqrt(12.0/double(K_LONG))*std::exp2(-double(bpc))));

    fear4w.stop();
    mdctR.stop();


    alignas(64) std::array<int32_t,K_LONG> qA{},qB{};

    // v0.2.7 entropy grammar:
    //   class: {zero,+,-}, conditioned by adjacent FEAR-ERB pair (24 contexts)
    //   magnitude low byte: conditioned by same 24 contexts
    //   magnitude high-present flag: one global binary model
    //   scale residual: median-2D prediction, low byte + 2-bit high symbol
    CtxQ44SoAModels<9,CLASS_TUPLE_CTX> class_model;
    CtxQ44SoAModels<MAG_ALPHA,MAG_CTX> mag_model;
    lx_rans4_fast::Model<256> scale_lo_model, lr_scale_model;
    {std::array<uint32_t,256> h{};h.fill(1);scale_lo_model.build_encoder_only(h);lr_scale_model.build_encoder_only(h);}
    std::array<uint64_t,256> scale_hist_mem{}, lr_scale_hist_mem{};

    lx_rans4_fast::Scratch rsClass,rsMag,rsScaleLo,rsLrScale;
    constexpr size_t MAX_COORD=size_t(2)*MODEL_SUPERFRAME*K_LONG;
    auto keyClass=std::make_unique_for_overwrite<uint16_t[]>(MAX_COORD);
    auto keyMag=std::make_unique_for_overwrite<uint16_t[]>(MAX_COORD);
    auto symScaleLo=std::make_unique_for_overwrite<uint8_t[]>(MODEL_SUPERFRAME*ERB_BANDS);
    auto symLrScale=std::make_unique_for_overwrite<uint8_t[]>(MODEL_SUPERFRAME*ERB_BANDS);
    std::vector<int16_t> lr_balance_q(plan.size()*ERB_BANDS,0);
    std::vector<std::array<float,ERB_BANDS>> lr_gain(plan.size());
    for(auto& g:lr_gain)g.fill(1.0f);
    std::array<std::array<int16_t,16>,4> lr_pred_prev{};
    std::array<std::array<uint8_t,16>,4> lr_pred_valid{};

    uint64_t bytes=0;double mse=0,en=0;size_t n1024=0,n512=0,n256=0,msc=0,is_frames=0,is_bands=0;uint64_t is_side_bits=0,sfc_bits=0;size_t sfc_frames=0;uint64_t prec_bins=0,prec_coords=0;uint64_t tns_side_bits=0,tns_frames=0;uint64_t sgc_side_bits=0,sgc_bands=0;
    uint64_t cpm_frames=0,cpm_anchors=0,cpm_cont=0,cpm_bits=0;
    CpmTrackState cpm_state{};
    // consecutive SFC exposure per top-8 band, in 2048-sample support units.
    // Product builds are deliberately frame-serial here because this state is a
    // decoder-synchronous temporal prior, not an encoder-only heuristic.
    std::array<double,lx_sfc::SFC_BANDS> sfc_diffuse_exposure{};
    // R204: lossless temporal magnitude-model predictor.  This state never
    // changes q; it only selects the rANS probability context.  State 0 means
    // no previous non-zero magnitude, states 1..3 are octave buckets of the
    // previous band's mean |q|: [1,2), [2,4), [4,inf).
    std::array<std::array<uint64_t,ERB_BANDS>,2> magpred_prev_q1{};
    std::array<std::array<uint32_t,ERB_BANDS>,2> magpred_prev_nz{};
    std::array<std::array<uint8_t,ERB_BANDS>,2> magpred_prev_valid{};
    // R206 research: decoder-synchronous long-term entropy memory across
    // 32-frame superframes. Integer EWMA, denominator 8; zero side information.
    std::array<std::array<uint64_t,9>,CLASS_TUPLE_CTX> class_hist_mem{};
    std::array<std::array<uint64_t,MAG_ALPHA>,MAG_CTX> mag_hist_mem{};
    double cpm_min_anchor_forecast=std::numeric_limits<double>::infinity(),cpm_min_cont_j=std::numeric_limits<double>::infinity(),cpm_min_true_cont=std::numeric_limits<double>::infinity(); uint64_t cpm_cont_eligible=0;
    double entropy_ns=0.0;uint64_t entropy_symbols=0;
    // Quantizer scale code is discrete (0..480) and bpc is constant for the
    // entire encode.  Materialize the exact float/double delta surfaces once;
    // frame hot loops then perform loads instead of repeated ldexp/multiply.
    const double quant_base_delta=std::sqrt(12.0/double(K_LONG))*std::exp2(-double(bpc));
    std::array<float,481> delta_code_f{};
    std::array<double,481> delta_code_d{};
    for(int sc=0;sc<=480;++sc){
        const int octave=sc>>4, fraction=sc&15;
        delta_code_f[sc]=std::ldexp(float(quant_base_delta)*LX_DELTA_FRAC16[fraction],octave);
        delta_code_d[sc]=std::ldexp(quant_base_delta*double(LX_DELTA_FRAC16[fraction]),octave);
    }
    const size_t nsf=(plan.size()+MODEL_SUPERFRAME-1)/MODEL_SUPERFRAME;

    // C44 duration-invariant causal physical-rate reservoir.
    // The old R81 controller divided the current cumulative bit debt by the
    // slope of *all remaining superframes* but applied the resulting correction
    // only to the current superframe. That made identical audio quantize
    // differently solely as a function of remaining file duration.
    //
    // C44 amortizes debt over a fixed H=8-superframe horizon and uses only
    // already-emitted packet bytes plus a causal multiplicative calibration rho
    // of the analytical entropy slope. No track-end quantity appears.
    //
    //   x     = debt / (H * rho * S_current)
    //   shift = A * tanh(x/A), H=8, A=1 octave
    //
    // The tanh map is linear at the origin and provides a continuous trust
    // region for large model errors; there are no signal-value branches.
    double res_pred_cum=0.0, res_actual_packet_bits_cum=0.0;
    // cumulative *assigned perceptual* payload budget.  This is distinct
    // from a uniform wall-clock schedule: bits intentionally granted to an
    // early hard superframe are not debt.
    double fbr_target_bits_cum=0.0;
    const double res_prior_bits=(!fbr_pred_sf_bits.empty())?std::max(fbr_pred_sf_bits.front(),1.0):1.0;
    constexpr double RES_H=8.0;
    constexpr double RES_A=1.0;
    // fast one-pass VBR controller.  Exact-rate mode intentionally keeps
    // the frozen R122 C44 controller above.  The fast path uses the same
    // analytical entropy forecast, but turns the already-observed model ratio
    // into a per-superframe physical-bit target and repays only half of the
    // accumulated debt in the next superframe.  This avoids the large,
    // duration-dependent overshoot of the old one-pass path without re-encoding.
    // Product default: Fast FBR.  The historical LX_R81_ONEPASS_RATE knob is
    // retained for compatibility: unset/nonzero = Fast FBR, zero = Exact FBR.
    const bool causal_vbr_fast=[](){
        const char* q=::lxvoxel_release_env("LX_R81_ONEPASS_RATE");
        return !q || std::atoi(q)!=0;
    }();
    // fast VBR: hybrid physical schedule. The perceptual FEAR/KKT
    // allocation is unchanged; only the one-pass physical-rate controller
    // blends the constant-time target with the normalized global entropy
    // complexity forecast, so hard superframes may spend more bits.
    constexpr double CAUSAL_VBR_BLEND=0.50;
    constexpr double CAUSAL_DEBT_H=2.0;
    constexpr double CAUSAL_TRUST_OCT=2.0;
    // research: project the required physical-rate correction over the
    // complete time-frequency cell set of the current superframe instead of
    // forcing a common quality jump on every cell.
    //   mode 0: R127 common shift
    //   mode 1: minimum ||s||^2
    //   mode 2: FEAR-RD curvature weighted
    //   mode 3: mask/noise-curvature weighted
    //   mode 4: FEAR-RD + temporal continuity (quadratic field)
    //   mode 5: mask curvature + temporal continuity (quadratic field)
    //   mode 6: geometric FEAR-RD x mask curvature
    //   mode 7: geometric curvature + temporal continuity
    //   mode 8/9/10: 25/50/75% minimum-modulation blended with common shift
    //   mode 11: production default, 70% minimum-modulation + 30% common shift
    // default: preserve 30% of the common-mode correction while projecting
    // 70% onto the minimum-modulation rate-slope direction. LX_FBR_PROJ remains
    // an explicit research override; unset means the production mode 11.
    const int fbr_proj_mode=[](){const char* p=::lxvoxel_release_env("LX_FBR_PROJ");return p?std::atoi(p):11;}();
    std::array<double,ERB_BANDS> fbr_prev_band_shift{};
    // research: startup physical-rate mini-model.  Mode 0 is the R129
    // reference (no SF0 correction). Modes 1..10 are deterministic analytical
    // priors evaluated before the first packet is emitted.  They use only the
    // frozen FEAR entropy forecast, known syntax budget and requested physical
    // rate; no signal-value branch is introduced in the quantizer loops.
    const int fbr_start_model=[](){const char* p=::lxvoxel_release_env("LX_FBR_START_MODEL");return p?std::atoi(p):0;}();
    // Conditional-D_n gains are needed only to synthesize optional encoder-side
    // recon PCM.  Normal .vox encoding lets the decoder derive them directly, so
    // do not allocate or rescan every quantized band in the product hot path.
    std::vector<std::array<float,ERB_BANDS>> recon_direct_gain_l,recon_direct_gain_r;
    if(recon){
        recon_direct_gain_l.resize(plan.size());recon_direct_gain_r.resize(plan.size());
        for(auto& g:recon_direct_gain_l)g.fill(1.0f);for(auto& g:recon_direct_gain_r)g.fill(1.0f);
    }
    const bool psq_enabled_hot = psq_enable_cfg()!=0;
    const int psq_detector_hot = psq_detector_cfg();
    const bool sgc_enabled_hot = sgc_enable_cfg()!=0;
    // Product configuration is immutable for one encode call.  Resolve all
    // environment-backed entropy-model knobs once, never in the superframe loop.
    const int class_mem_num=[](){const char* p=::lxvoxel_release_env("LX_CLASS_MEMORY_NUM"); if(p)return std::clamp(std::atoi(p),0,7); const char*q=::lxvoxel_release_env("LX_MODEL_MEMORY_NUM"); return q?std::clamp(std::atoi(q),0,7):5;}();
    const int mag_mem_num=[](){const char* p=::lxvoxel_release_env("LX_MAG_MEMORY_NUM"); if(p)return std::clamp(std::atoi(p),0,7); const char*q=::lxvoxel_release_env("LX_MODEL_MEMORY_NUM"); return q?std::clamp(std::atoi(q),0,7):5;}();
    const int class_backoff=[](){const char* p=::lxvoxel_release_env("LX_CLASS_BACKOFF"); if(p)return std::max(0,std::atoi(p)); const char*q=::lxvoxel_release_env("LX_PRED_BACKOFF"); return q?std::max(0,std::atoi(q)):8;}();
    const int mag_backoff=[](){const char* p=::lxvoxel_release_env("LX_MAG_BACKOFF"); if(p)return std::max(0,std::atoi(p)); const char*q=::lxvoxel_release_env("LX_PRED_BACKOFF"); return q?std::max(0,std::atoi(q)):16;}();
    const bool profile_entropy=[](){const char* p=::lxvoxel_release_env("LX_PROFILE_ENTROPY"); return p && std::atoi(p)!=0;}();
    for(size_t sf=0;sf<std::min(nsf,max_superframes);++sf){
        const size_t beg=sf*MODEL_SUPERFRAME,end=std::min(plan.size(),beg+MODEL_SUPERFRAME);
        double fbr_target_cur_assigned=0.0;
        double fbr_shift_applied=0.0;
        if(fbr_nc_mode==0 && fbr_physical_kbps>0.0 && (sf>0 || fbr_start_model>0 || startup_pilot_entropy_bits>0.0 || (startup_pilots && sf<startup_pilots->entropy_bits.size())) && !fbr_pred_sf_bits.empty()){
            double rho=(res_actual_packet_bits_cum+res_prior_bits)/(res_pred_cum+res_prior_bits);
            const double S=std::max(fbr_pred_sf_slope[sf],1.0);
            if(sf==0){
                const size_t cur0=plan[beg].sample_pos;
                const size_t cur1=(end<plan.size())?plan[end].sample_pos:ns;
                const double target_uniform=fbr_physical_kbps*1000.0*double(cur1-cur0)/double(SR);
                const double pred_cur=std::max(fbr_pred_sf_bits[0],1.0);
                const double pred_total=std::max(fbr_pred_prefix.back(),1.0);
                const double total_phys=fbr_physical_kbps*1000.0*double(ns)/double(SR);
                const double header_bits=8.0*double(sizeof(vox_beta::Header));
                const double coded_phys=std::max(total_phys-header_bits,1.0);
                const double rho_track=coded_phys/pred_total;
                const double r_local=target_uniform/pred_cur;
                const double r_geom=std::sqrt(std::max(rho_track*r_local,1e-30));
                const double r_harm=2.0*rho_track*r_local/std::max(rho_track+r_local,1e-30);
                const double complexity=pred_cur/(pred_total/std::max(double(nsf),1.0));
                switch(fbr_start_model){
                    case 1: rho=1.0; break;
                    case 2: rho=rho_track; break;
                    case 3: rho=r_local; break;
                    case 4: rho=r_geom; break;
                    case 5: rho=r_harm; break;
                    case 6: rho=0.5*(rho_track+r_local); break;
                    case 7: rho=rho_track*std::sqrt(std::max(complexity,1e-30)); break;
                    case 8: rho=rho_track/std::sqrt(std::max(complexity,1e-30)); break;
                    case 9: rho=std::exp(0.75*std::log(std::max(rho_track,1e-30))+0.25*std::log(std::max(r_local,1e-30))); break;
                    case 10: rho=std::exp(0.25*std::log(std::max(rho_track,1e-30))+0.75*std::log(std::max(r_local,1e-30))); break;
                    default: break;
                }
            }
            double shift_log2=0.0;
            if((sf==0 && startup_pilot_entropy_bits>0.0) || (startup_pilots && sf<startup_pilots->entropy_bits.size())){
                const size_t cur0=plan[beg].sample_pos;
                const size_t cur1=(end<plan.size())?plan[end].sample_pos:ns;
                const double target_uniform=fbr_physical_kbps*1000.0*double(cur1-cur0)/double(SR);
                const double pred_cur=std::max(fbr_pred_sf_bits[sf],1.0);
                const double pred_total=std::max(fbr_pred_prefix.back(),1.0);
                const double total_phys=fbr_physical_kbps*1000.0*double(ns)/double(SR);
                const double header_bits=8.0*double(sizeof(vox_beta::Header));
                const double alloc_phys=std::max(total_phys-header_bits,1.0);
                const double rho_plan=alloc_phys/pred_total;
                const double plan_cur=alloc_phys*pred_cur/pred_total;
                const double target_cur=(1.0-CAUSAL_VBR_BLEND)*target_uniform + CAUSAL_VBR_BLEND*plan_cur;
                const double debt=res_actual_packet_bits_cum-fbr_target_bits_cum;
                const double desired_cur=std::max(0.0,target_cur-debt/CAUSAL_DEBT_H);
                double pe=startup_pilot_entropy_bits;
                if(startup_pilots && sf<startup_pilots->entropy_bits.size()) pe=startup_pilots->entropy_bits[sf];
                const double rho_pilot=std::max(pe/pred_cur,1e-12);
                fbr_target_cur_assigned=target_cur;
                const double x=(rho_plan*pred_cur-desired_cur)/std::max(rho_pilot*S,1.0);
                shift_log2=CAUSAL_TRUST_OCT*std::tanh(x/CAUSAL_TRUST_OCT);
            }else if(sf==0 && fbr_start_model>0){
                // Pre-startup mini-model: estimate the first superframe's physical
                // entropy cost from the whole-track analytical forecast, then give
                // SF0 its perceptual-complexity share before any packet history exists.
                const size_t cur0=plan[beg].sample_pos;
                const size_t cur1=(end<plan.size())?plan[end].sample_pos:ns;
                const double target_uniform=fbr_physical_kbps*1000.0*double(cur1-cur0)/double(SR);
                const double pred_cur=std::max(fbr_pred_sf_bits[0],1.0);
                const double pred_total=std::max(fbr_pred_prefix.back(),1.0);
                const double total_phys=fbr_physical_kbps*1000.0*double(ns)/double(SR);
                const double header_bits=8.0*double(sizeof(vox_beta::Header));
                const double alloc_phys=std::max(total_phys-header_bits,1.0);
                const double rho_track=alloc_phys/pred_total;
                const double plan_linear=alloc_phys*pred_cur/pred_total;
                auto power_plan=[&](double p)->double{
                    double den=0.0; for(double v:fbr_pred_sf_bits) den+=std::pow(std::max(v,1.0),p);
                    return alloc_phys*std::pow(pred_cur,p)/std::max(den,1e-30);
                };
                double target_cur=target_uniform;
                double rho0=rho_track;
                switch(fbr_start_model){
                    case 1: target_cur=target_uniform; break;
                    case 2: target_cur=plan_linear; break;
                    case 3: target_cur=0.75*target_uniform+0.25*plan_linear; break;
                    case 4: target_cur=0.50*target_uniform+0.50*plan_linear; break;
                    case 5: target_cur=0.25*target_uniform+0.75*plan_linear; break;
                    case 6: target_cur=power_plan(0.50); break;
                    case 7: target_cur=power_plan(0.75); break;
                    case 8: target_cur=0.50*target_uniform+0.50*power_plan(0.50); break;
                    case 9: target_cur=0.50*target_uniform+0.50*power_plan(0.75); break;
                    case 10: target_cur=0.25*target_uniform+0.75*power_plan(0.75); break;
                    default: break;
                }
                fbr_target_cur_assigned=target_cur;
                const double x=(rho0*pred_cur-target_cur)/std::max(rho0*S,1.0);
                shift_log2=CAUSAL_TRUST_OCT*std::tanh(x/CAUSAL_TRUST_OCT);
            }else if(causal_vbr_fast){
                const size_t cur0=plan[beg].sample_pos;
                const size_t cur1=(end<plan.size())?plan[end].sample_pos:ns;
                const double target_uniform=fbr_physical_kbps*1000.0*double(cur1-cur0)/double(SR);
                const double pred_cur=(sf<fbr_pred_sf_bits.size())?fbr_pred_sf_bits[sf]:target_uniform;

                // Complexity-aware physical schedule.  The already-computed
                // global FEAR/KKT entropy forecast is used only as a normalized
                // VBR *shape*: hard superframes are explicitly allowed more
                // physical bits, easy ones fewer.  This does not re-waterfill
                // or re-encode coefficients.
                const double pred_total=std::max(fbr_pred_prefix.empty()?0.0:fbr_pred_prefix.back(),1.0);
                const double total_phys=fbr_physical_kbps*1000.0*double(ns)/double(SR);
                const double header_bits=8.0*double(sizeof(vox_beta::Header));
                const double alloc_phys=std::max(total_phys-header_bits,0.0);
                const double plan_cur=alloc_phys*pred_cur/pred_total;

                const double target_cur=(1.0-CAUSAL_VBR_BLEND)*target_uniform
                                      + CAUSAL_VBR_BLEND*plan_cur;
                // Only deviation from budgets that were actually assigned in
                // earlier superframes is feedback error.  In particular, an
                // intentional SF0 startup allocation is never reclassified as
                // debt against a different retrospective schedule.
                const double debt=res_actual_packet_bits_cum-fbr_target_bits_cum;
                const double desired_cur=std::max(0.0,target_cur-debt/CAUSAL_DEBT_H);
                fbr_target_cur_assigned=target_cur;
                const double x=(rho*pred_cur-desired_cur)/std::max(rho*S,1.0);
                shift_log2=CAUSAL_TRUST_OCT*std::tanh(x/CAUSAL_TRUST_OCT);
            }else{
                const double debt=res_actual_packet_bits_cum-fbr_target_bits_cum;
                fbr_target_cur_assigned=fbr_physical_kbps*1000.0*
                    double(((end<plan.size())?plan[end].sample_pos:ns)-plan[beg].sample_pos)/double(SR);
                // Frozen R122/C44 controller used by the exact outer-rate path.
                const double x=debt/std::max(RES_H*rho*S,1.0);
                shift_log2=RES_A*std::tanh(x/RES_A);
            }
            fbr_shift_applied=shift_log2;
            if(fbr_proj_mode==0){
                const int shift_sk=int(std::nearbyint(16.0*shift_log2));
                for(size_t fi=beg;fi<end;++fi) for(int b=0;b<ERB_BANDS;++b){
                    const size_t ci=fi*ERB_BANDS+b;
                    delta_scale[ci]=uint16_t(std::clamp(int(delta_scale[ci])+shift_sk,0,30*16));
                }
                for(int b=0;b<ERB_BANDS;++b) fbr_prev_band_shift[b]=shift_log2;
            }else{
                // Required first-order entropy reduction in bits.
                const double C=shift_log2*S;
                const size_t nfcur=end-beg;
                std::vector<double> avec(nfcur*ERB_BANDS), wvec(nfcur*ERB_BANDS), zvec(nfcur*ERB_BANDS);
                for(size_t lf=0;lf<nfcur;++lf){
                    const size_t fi=beg+lf;
                    for(int b=0;b<ERB_BANDS;++b){
                        const size_t ci=fi*ERB_BANDS+b, k=lf*ERB_BANDS+b;
                        avec[k]=fbr_cell_slope[ci];
                        if(fbr_proj_mode==2 || fbr_proj_mode==4) wvec[k]=fbr_cell_w_rd[ci];
                        else if(fbr_proj_mode==3 || fbr_proj_mode==5) wvec[k]=fbr_cell_w_mask[ci];
                        else if(fbr_proj_mode==6 || fbr_proj_mode==7) wvec[k]=std::sqrt(fbr_cell_w_rd[ci]*fbr_cell_w_mask[ci]);
                        else wvec[k]=1.0;
                    }
                }
                if(fbr_proj_mode<=3 || fbr_proj_mode==6 || (fbr_proj_mode>=8 && fbr_proj_mode<=11)){
                    // min sum_i w_i s_i^2  s.t. sum_i a_i s_i = C
                    double den=0.0;
                    for(size_t k=0;k<avec.size();++k) den += avec[k]*avec[k]/std::max(wvec[k],1e-30);
                    const double lam=C/std::max(den,1e-30);
                    for(size_t k=0;k<avec.size();++k) zvec[k]=lam*avec[k]/std::max(wvec[k],1e-30);
                    if(fbr_proj_mode>=8 && fbr_proj_mode<=11){
                        const double th=(fbr_proj_mode==11) ? 0.70 : 0.25*double(fbr_proj_mode-7);
                        for(size_t k=0;k<zvec.size();++k) zvec[k]=(1.0-th)*shift_log2 + th*zvec[k];
                    }
                }else{
                    // Whole-superframe quadratic time-frequency field. For each
                    // ERB solve the exact tridiagonal system generated by
                    //   sum w s^2 + sum sqrt(w_t w_{t-1})(s_t-s_{t-1})^2
                    // with the previous superframe band state as the boundary.
                    // One global Lagrange multiplier then enforces the bit target.
                    std::vector<double> y(avec.size(),0.0);
                    double aden=0.0, abase=0.0;
                    for(int b=0;b<ERB_BANDS;++b){
                        const size_t n=nfcur;
                        std::vector<double> lo(n),di(n),hi(n),rhsA(n),rhsP(n);
                        for(size_t t=0;t<n;++t){
                            const size_t k=t*ERB_BANDS+b;
                            const double w=std::max(wvec[k],1e-30);
                            di[t]=w; rhsA[t]=avec[k]; rhsP[t]=0.0;
                            if(t>0){const double v=std::sqrt(w*std::max(wvec[(t-1)*ERB_BANDS+b],1e-30)); lo[t]=-v; di[t]+=v;}
                            if(t+1<n){const double v=std::sqrt(w*std::max(wvec[(t+1)*ERB_BANDS+b],1e-30)); hi[t]=-v; di[t]+=v;}
                        }
                        // Previous-SF boundary is another temporal edge.
                        if(n){ const double v=std::max(wvec[b],1e-30); di[0]+=v; rhsP[0]=v*fbr_prev_band_shift[b]; }
                        auto solveTri=[&](std::vector<double> rhs){
                            std::vector<double> c=hi,d=di;
                            for(size_t i=1;i<n;++i){const double m=lo[i]/std::max(d[i-1],1e-30);d[i]-=m*c[i-1];rhs[i]-=m*rhs[i-1];}
                            if(n){rhs[n-1]/=std::max(d[n-1],1e-30);for(size_t ii=n-1;ii-->0;)rhs[ii]=(rhs[ii]-c[ii]*rhs[ii+1])/std::max(d[ii],1e-30);}
                            return rhs;
                        };
                        const auto ya=solveTri(rhsA);
                        const auto yp=solveTri(rhsP);
                        for(size_t t=0;t<n;++t){const size_t k=t*ERB_BANDS+b;y[k]=ya[t];zvec[k]=yp[t];abase+=avec[k]*yp[t];aden+=avec[k]*ya[t];}
                    }
                    const double lam=(C-abase)/std::max(aden,1e-30);
                    for(size_t k=0;k<zvec.size();++k) zvec[k]+=lam*y[k];
                }
                // Same continuous trust region as R127, now cell-local.
                std::array<double,ERB_BANDS> bsum{}; std::array<double,ERB_BANDS> bdim{};
                for(size_t lf=0;lf<nfcur;++lf){
                    const size_t fi=beg+lf;
                    for(int b=0;b<ERB_BANDS;++b){
                        const size_t k=lf*ERB_BANDS+b,ci=fi*ERB_BANDS+b;
                        const double sh=CAUSAL_TRUST_OCT*std::tanh(zvec[k]/CAUSAL_TRUST_OCT);
                        const int sk=int(std::nearbyint(16.0*sh));
                        delta_scale[ci]=uint16_t(std::clamp(int(delta_scale[ci])+sk,0,30*16));
                        const double d=double(cells[ci].dim);bsum[b]+=d*sh;bdim[b]+=d;
                    }
                }
                for(int b=0;b<ERB_BANDS;++b) fbr_prev_band_shift[b]=bsum[b]/std::max(bdim[b],1.0);
            }
        }
        vox_beta::SuperframePacket vp;
        if(vox){ vp.h.frame_count=uint16_t(end-beg); vp.side.reserve(end-beg); vp.block_hi.assign(((end-beg)+7u)>>3,0u); }
        // Fixed superframe geometry: keep decision scratch on the stack and avoid
        // six heap allocations per superframe. Only indices [0,end-beg) are used.
        std::array<SfcFrameDecision,MODEL_SUPERFRAME> sf_sfc{};
        std::array<std::array<uint8_t,lx_sfc::SFC_BANDS>,MODEL_SUPERFRAME> sf_ac_on{}, sf_ac_gain{};
        std::array<CpmFrameDecision,MODEL_SUPERFRAME> sf_cpm{};
        std::array<std::array<uint8_t,ERB_BANDS>,MODEL_SUPERFRAME> sf_sgc_on{}, sf_sgc_q{};
        std::array<std::array<uint32_t,9>,CLASS_TUPLE_CTX> hClass{};
        std::array<std::array<uint32_t,MAG_ALPHA>,MAG_CTX> hMag{};
        std::array<uint32_t,256> hScaleLo{},hLrScale{};
        size_t nClass=0,nMag=0,nScale=0,nLrScale=0,raw_hi_bytes=0,raw_scale_bytes=0,raw_lr_scale_bytes=0;

        // Decoder-known scale surface first: exact integer predictor, no reference.
        for(size_t fi=beg;fi<end;++fi){
            for(int b=0;b<ERB_BANDS;++b){
                const int pred=scale_predict(delta_scale,fi,b);
                const int32_t d=int32_t(delta_scale[fi*ERB_BANDS+b])-pred;
                const uint32_t zz=zigzag_i32(d);
                const uint8_t ssym=uint8_t(std::min(zz,255u));
                symScaleLo[nScale]=ssym;++hScaleLo[ssym];++nScale;
                const uint32_t esc=uint32_t(zz>=255u);
                raw_scale_bytes+=esc*(1u+uint32_t(zz>=128u)+uint32_t(zz>=16384u)+uint32_t(zz>=2097152u)+uint32_t(zz>=268435456u));
                if(vox && esc) vox_beta::put_uleb128(vp.scale_raw,zz);
            }
        }

        // R332 normative true-independent L/R geometry is signalled per EGV,
        // i.e. at the exact vector granularity where D_n parity is enforced.
        // q is a signed 1/16-octave axis offset:
        //   Delta_L = Delta_g * 2^(q/16), Delta_R = Delta_g * 2^(-q/16).
        // One q per EGV avoids paying 48 independent side symbols per frame while
        // keeping L/R truly independent inside every lattice vector.
        for(size_t fi=beg;fi<end;++fi){
            for(int b=0;b<ERB_BANDS;++b){lr_balance_q[fi*ERB_BANDS+size_t(b)]=0;lr_gain[fi][b]=1.0f;}
            if(modes[fi]!=0) continue;
            const int istart=int(is_start[fi]);
            const int bt=int(plan[fi].block);
            auto code_groups=[&](const auto& groups){
                int gi=0; int left_q=0;
                for(const auto g:groups){
                    const int b0=int(g.b0), b1=std::min<int>(int(g.b1),istart);
                    if(b0>=b1){++gi;continue;}
                    double el=0.0,er=0.0;
                    int qlo=-480,qhi=480;
                    for(int b=b0;b<b1;++b){
                        el+=double(statEL[fi][b]);er+=double(statER[fi][b]);
                        const int common=int(delta_scale[fi*ERB_BANDS+size_t(b)]);
                        qlo=std::max(qlo,std::max(-common,common-480));
                        qhi=std::min(qhi,std::min(480-common,common));
                    }
                    el=std::max(el,std::numeric_limits<double>::min());
                    er=std::max(er,std::numeric_limits<double>::min());
                    const int q=std::clamp(int(std::nearbyint(4.0*std::log2(el/er))),qlo,qhi);
                    const float group_gain=std::exp2(float(q)*(1.0f/16.0f));
                    for(int b=b0;b<b1;++b){
                        lr_balance_q[fi*ERB_BANDS+size_t(b)]=int16_t(q);
                        lr_gain[fi][b]=group_gain;
                    }
                    int pred=left_q;
                    if(lr_pred_valid[bt][gi]){
                        const int top=int(lr_pred_prev[bt][gi]);
                        pred=(gi==0)?top:int(std::nearbyint(0.5*double(left_q+top)));
                    }
                    const uint32_t zz=zigzag_i32(int32_t(q-pred));
                    const uint8_t ssym=uint8_t(std::min(zz,255u));
                    symLrScale[nLrScale]=ssym;++hLrScale[ssym];++nLrScale;
                    const uint32_t esc=uint32_t(zz>=255u);
                    raw_lr_scale_bytes+=esc*(1u+uint32_t(zz>=128u)+uint32_t(zz>=16384u)+uint32_t(zz>=2097152u)+uint32_t(zz>=268435456u));
                    if(vox && esc) vox_beta::put_uleb128(vp.lr_raw,zz);
                    lr_pred_prev[bt][gi]=int16_t(q);lr_pred_valid[bt][gi]=1u;
                    left_q=q;++gi;
                }
            };
            switch(plan[fi].block){
                case BlockSize::Long2048:code_groups(EGV_LONG);break;
                case BlockSize::Medium1024:code_groups(EGV_1024);break;
                case BlockSize::Short512:code_groups(EGV_SHORT);break;
                case BlockSize::Micro256:code_groups(EGV_256);break;
            }
        }

        // Diffuse Guard carries decoder-synchronous state across frames, so
        // this decision/reconstruction pass is intentionally frame-serial.
        alignas(64) std::array<float,ERB_BANDS> energy_prev_src_amp{},energy_prev_q_amp{};
        std::array<uint8_t,ERB_BANDS> energy_prev_valid{};
        alignas(64) std::array<float,ERB_BANDS> shell_prev_src_amp{},shell_prev_q_amp{};
        std::array<uint8_t,ERB_BANDS> shell_prev_valid{};
        for(long long fi_ll=(long long)beg;fi_ll<(long long)end;++fi_ll){
            const size_t fi=(size_t)fi_ll;
                    const auto&fr=plan[fi];float*l=cL.data()+fr.coeff_off,*r=cR.data()+fr.coeff_off;const bool smode=modes[fi]!=0;
            // TNS is not part of the current bitstream.  Quantization reads the
            // transform coefficients in-place; do not materialize the obsolete
            // legacy TNS scratch spectra here.
            const float*qLsrc=l; const float*qRsrc=r;
            const bool indep_lr_hot=indep_lr_cfg()!=0;
            const auto sfc_guard_exposure=sfc_diffuse_exposure;
            const double base_delta=quant_base_delta;
            alignas(64) std::array<float,ERB_BANDS> band_delta{};
            alignas(64) std::array<double,ERB_BANDS> band_delta_d{};
            for(int b=0;b<ERB_BANDS;++b){
                const int scale_code=int(delta_scale[fi*ERB_BANDS+b]);
                band_delta[b]=delta_code_f[size_t(scale_code)];
                band_delta_d[b]=delta_code_d[size_t(scale_code)];
            }

            double frame_error_energy=0.0,frame_source_energy=0.0;
            auto quantize_egv=[&](const auto& erb,int b0,int b1){
                const int p=erb.edge[b0],n=erb.edge[b1]-p;
                const int istart=int(is_start[fi]);
                const bool overlaps_is=(b1>istart);
                if(overlaps_is){
                    (void)quantize_egv_fear_is_direct(qLsrc,qRsrc,p,n,smode,erb,b0,b1,
                        cells.data()+fi*ERB_BANDS,float(base_delta),maxb,delta_scale.data()+fi*ERB_BANDS,band_delta.data(),istart,is_angle[fi].data(),
                        is_src_l[fi].data(),is_src_r[fi].data(),qA.data(),qB.data(),
                        energy_prev_src_amp.data(),energy_prev_q_amp.data(),energy_prev_valid.data());
                }else if(indep_lr_hot && !smode){
                    (void)quantize_egv_indep_lr_direct(qLsrc,qRsrc,p,n,erb,b0,b1,
                        band_delta.data(),lr_gain[fi].data(),qA.data(),qB.data());
                }else if(psq_enabled_hot){
                    (void)quantize_egv_psq_direct(qLsrc,qRsrc,p,n,smode,psq_mode[fi].data(),erb,b0,b1,
                        float(base_delta),delta_scale.data()+fi*ERB_BANDS,band_delta.data(),lr_gain[fi].data(),qA.data(),qB.data());
                }else{
                    (void)quantize_egv_fear_direct(qLsrc,qRsrc,p,n,smode,erb,b0,b1,
                        cells.data()+fi*ERB_BANDS,float(base_delta),maxb,delta_scale.data()+fi*ERB_BANDS,band_delta.data(),qA.data(),qB.data(),
                        energy_prev_src_amp.data(),energy_prev_q_amp.data(),energy_prev_valid.data());
                }
                // R238: refine the just-quantized bands immediately while source and q
                // data are cache-hot. Search and objective are unchanged.
                if(!(indep_lr_hot && !smode))
                    (void)lx_refine_dn_shell_pairs(qLsrc,qRsrc,smode,erb,int(is_start[fi]),is_angle[fi].data(),
                        is_src_l[fi].data(),is_src_r[fi].data(),band_delta.data(),qA.data(),qB.data(),
                        shell_prev_src_amp.data(),shell_prev_q_amp.data(),shell_prev_valid.data(),b0,b1);
            };

            switch(fr.block){
                case BlockSize::Long2048:   for(const auto g:EGV_LONG)  quantize_egv(eL,g.b0,g.b1); break;
                case BlockSize::Medium1024: for(const auto g:EGV_1024)  quantize_egv(e1024,g.b0,g.b1); break;
                case BlockSize::Short512:   for(const auto g:EGV_SHORT) quantize_egv(eS,g.b0,g.b1); break;
                case BlockSize::Micro256:   for(const auto g:EGV_256)   quantize_egv(e256,g.b0,g.b1); break;
            }
            // R220 direct-coding cutoff.  Do not erase the source spectrum: SFC/fSFC
            // still sees the true 16.59..20-kHz target and may synthesize it from the
            // lower decoded support.  Only direct D_n coordinates above 16.59 kHz are
            // forbidden, so their entropy is genuinely recycled below the cutoff.
            if(hf_hybrid_profile && lowrate_cutoff){
                const int cb=lowrate_cut_bins(fr.k);
                const auto& edcut=edges_for(fr.block);
                for(int b=0;b<ERB_BANDS;++b){
                    const int lo=edcut[b],hi=edcut[b+1];
                    if(lo<cb) continue;
                    const size_t ci=fi*ERB_BANDS+size_t(b);
                    if(hf_escape_mode>0 && lowrate_hf_direct[ci]) continue;
                    for(int i=lo;i<hi;++i){qA[i]=0;qB[i]=0;}
                }
                // Any coded-domain tail above the last ERB remains absent.
                for(int i=edcut[ERB_BANDS];i<fr.k;++i){qA[i]=0;qB[i]=0;}
            }

            // SFC/FEAR decision.  First materialize the decoder-faithful
            // baseline spectrum once.  SFC is then evaluated only in the top
            // eight FEAR bands.  The Lagrange price is derived from the current
            // frame's FEAR-weighted quantization error, so there is no hand-set
            // audibility threshold: a band is formed only when its FEAR error
            // reduction/rate trade beats the frame's present marginal RD cost.
            alignas(64) std::array<float,K_LONG> decL{},decR{},tmpL{},tmpR{};
            auto decode_q_baseline=[&](const auto& erb_dec){
                for(int b=0;b<ERB_BANDS;++b){
                    const float delta=band_delta[b];
                    const int lo=erb_dec.edge[b],hi=erb_dec.edge[b+1];
                    if(b>=int(is_start[fi])){
                        const auto axis=intensity_axis_f(is_angle[fi][b]);
                        const float ct=axis.cosine,st=axis.sine;
                        for(int i=lo;i<hi;++i){const float ww=lx_warp_weight(b,i-lo,hi-lo);const float z=float(qA[i])*(delta/ww);decL[i]=ct*z;decR[i]=st*z;}
                    }else{
                        const bool refine=!indep_lr_hot && psq_enabled_hot && psq_detector_hot>=10 && bool(psq_mode[fi][b]);
                        const bool bms=indep_lr_hot?smode:(psq_enabled_hot?((psq_detector_hot>=10)?(refine?true:smode):bool(psq_mode[fi][b])):smode);
                        const float pg=indep_lr_hot?(bms?1.0f:lr_gain[fi][b]):(psq_enabled_hot?((psq_detector_hot>=10)?(refine?psq_side_gain(b):1.0f):(bms?psq_side_gain(b):1.0f)):1.0f);
                        const float dA=delta*pg,dB=delta/pg;
                        for(int i=lo;i<hi;++i){
                            const float ww=lx_warp_weight(b,i-lo,hi-lo); const float a=float(qA[i])*(dA/ww), bb=float(qB[i])*(dB/ww);
                            if(!bms){decL[i]=a;decR[i]=bb;}
                            else{decL[i]=(a+bb)*INV_SQRT2;decR[i]=(a-bb)*INV_SQRT2;}
                        }
                    }
                }
            };
            switch(fr.block){case BlockSize::Long2048:decode_q_baseline(eL);break;case BlockSize::Medium1024:decode_q_baseline(e1024);break;case BlockSize::Short512:decode_q_baseline(eS);break;case BlockSize::Micro256:decode_q_baseline(e256);break;}

            // Group-SGC: adjacent ERB pairs share one energy-preserving
            // Procrustes correction angle.  This amortizes the angle over two
            // bands while retaining local stereo geometry.  Default precision
            // is 3 bits; signalling is sparse at the group level.
            if(sgc_enabled_hot && smode){
                const auto sgc_apply=[&](const auto& erb_dec){
                    constexpr int AB=3;
                    const int maxb_sgc=std::min<int>(lx_sfc::SFC_FIRST_BAND,int(is_start[fi]));
                    const int ng=maxb_sgc/2;
                    for(int g=0;g<ng;++g){
                        const int b0=2*g,b1=b0+1;
                        const int lo=erb_dec.edge[b0],hi=erb_dec.edge[b1+1],nn=hi-lo;
                        if(nn<=0) continue;
                        double A=0.0,Bx=0.0,qE=0.0,d2sum=0.0;
                        for(int b=b0;b<=b1;++b){
                            const double db=band_delta_d[b];
                            const int blo=erb_dec.edge[b],bhi=erb_dec.edge[b+1];
                            d2sum += double(bhi-blo)*db*db;
                        }
                        for(int i=lo;i<hi;++i){
                            const double x=double(l[i]),y=double(r[i]),qx=double(decL[i]),qy=double(decR[i]);
                            A += x*qx+y*qy;
                            Bx += y*qx-x*qy;
                            qE += qx*qx+qy*qy;
                        }
                        double delta=std::atan2(Bx,A);
                        const double drms=std::sqrt(d2sum/double(std::max(1,nn)));
                        const double rmsq=std::sqrt(qE/double(std::max(1,nn)));
                        const double dmax=std::atan2(drms,std::max(rmsq,1e-30));
                        delta=std::clamp(delta,-dmax,dmax);
                        const int levels=1<<AB;
                        int qi=int(std::floor((delta/dmax+1.0)*0.5*levels)); qi=std::clamp(qi,0,levels-1);
                        const double dq=dmax*(-1.0+2.0*(double(qi)+0.5)/double(levels));
                        const double c=std::cos(dq),ss=std::sin(dq);
                        double c0=0.0,c1=0.0;
                        for(int i=lo;i<hi;++i){
                            const double x=l[i],y=r[i],qx=decL[i],qy=decR[i];
                            const double rx=c*qx-ss*qy,ry=ss*qx+c*qy;
                            const double e0x=x-qx,e0y=y-qy,e1x=x-rx,e1y=y-ry;
                            c0+=e0x*e0x+e0y*e0y; c1+=e1x*e1x+e1y*e1y;
                        }
                        const double gain=c0-c1;
                        const double bit_price=std::log(2.0)*c0/double(std::max(1,nn));
                        if(gain>bit_price*double(AB)){
                            sgc_side_bits+=uint64_t(AB); sgc_bands+=2;
                            sf_sgc_on[fi-beg][g]=1; sf_sgc_q[fi-beg][g]=uint8_t(qi);
                            for(int i=lo;i<hi;++i){const float qx=decL[i],qy=decR[i];decL[i]=float(c*qx-ss*qy);decR[i]=float(ss*qx+c*qy);}
                        }
                    }
                };
                switch(fr.block){case BlockSize::Long2048:sgc_apply(eL);break;case BlockSize::Medium1024:sgc_apply(e1024);break;case BlockSize::Short512:sgc_apply(eS);break;case BlockSize::Micro256:sgc_apply(e256);break;}
            }

            double max_log_a=-std::numeric_limits<double>::infinity();
            for(int b=0;b<ERB_BANDS;++b) max_log_a=std::max(max_log_a,double(cells[fi*ERB_BANDS+b].loga));
            const double tiny_sel=std::numeric_limits<double>::min();
            alignas(64) std::array<double,ERB_BANDS> aud_w{};
            for(int b=0;b<ERB_BANDS;++b)
                aud_w[b]=std::exp(double(cells[fi*ERB_BANDS+b].loga)-max_log_a);
            auto aud_cost=[&](double err,int b)->double{
                const double mask=std::max(double(statMask[fi][b]),tiny_sel);
                const double z=err/mask;
#if LX_SFC_SEL_MODE == 1
                return z;
#elif LX_SFC_SEL_MODE == 2
                return std::log1p(z);
#elif LX_SFC_SEL_MODE == 3
                return z/(1.0+z);
#elif LX_SFC_SEL_MODE == 4
                return z*(1.0+0.5*z);
#elif LX_SFC_SEL_MODE == 5
                return z*std::sqrt(1.0+z);
#elif LX_SFC_SEL_MODE == 6
                const double e=double(statEL[fi][b])+double(statER[fi][b]);
                const double occ=e/(e+mask+tiny_sel);
                return z*(0.5+0.5*occ);
#elif LX_SFC_SEL_MODE == 7
                { const double w=aud_w[b]; return w*err*(1.0+z); }
#elif LX_SFC_SEL_MODE == 8
                { const double w=aud_w[b]; return w*err*std::sqrt(1.0+z); }
#elif LX_SFC_SEL_MODE == 9
                { const double w=aud_w[b]; return w*err*(1.0+std::log1p(z)); }
#elif LX_SFC_SEL_MODE == 10
                { const double w=aud_w[b]; return w*err*(1.0+z/(1.0+z)); }
#elif LX_SFC_SEL_MODE == 14
                { const double w=aud_w[b]; return w*err*std::sqrt(1.0+z); }
#elif LX_SFC_SEL_MODE == 17 || LX_SFC_SEL_MODE == 18
                { const double w=aud_w[b]; return w*err*(1.0+z/(1.0+z)); }
#else
                const double w=aud_w[b];
                return w*err;
#endif
            };

            double weighted_qerr=0.0; double weighted_dims=0.0;
            std::array<double,lx_sfc::SFC_BANDS> base_dq{}, base_qtw{}, base_xe{};
            auto accum_qerr=[&](const auto& erb_dec){
                for(int b=0;b<lx_sfc::SFC_FIRST_BAND;++b){
                    const int lo=erb_dec.edge[b],hi=erb_dec.edge[b+1]; double de=0.0;
                    for(int i=lo;i<hi;++i){const double dl=double(l[i])-decL[i],dr=double(r[i])-decR[i];de+=dl*dl+dr*dr;}
                    weighted_qerr+=aud_cost(de,b); weighted_dims+=2.0*double(hi-lo);
                }
                for(int b=lx_sfc::SFC_FIRST_BAND;b<ERB_BANDS;++b){
                    const int lo=erb_dec.edge[b],hi=erb_dec.edge[b+1]; double de=0.0,qtw0=0.0,xe0=0.0;
                    for(int i=lo;i<hi;++i){const double xl=double(l[i]),xr=double(r[i]);const double dl=xl-decL[i],dr=xr-decR[i];const double e2=dl*dl+dr*dr;const double x2=xl*xl+xr*xr;de+=e2;qtw0+=e2*x2;xe0+=x2;}
                    const int j=b-lx_sfc::SFC_FIRST_BAND;base_dq[j]=de;base_qtw[j]=qtw0;base_xe[j]=xe0;
                    weighted_qerr+=aud_cost(de,b); weighted_dims+=2.0*double(hi-lo);
                }
            };
            switch(fr.block){case BlockSize::Long2048:accum_qerr(eL);break;case BlockSize::Medium1024:accum_qerr(e1024);break;case BlockSize::Short512:accum_qerr(eS);break;case BlockSize::Micro256:accum_qerr(e256);break;}
            double lambda=(2.0*LN2)*weighted_qerr/std::max(weighted_dims,1.0);
#if LX_SFC_SEL_MODE == 11
            lambda*=std::exp(sfc_frame_logmask[fi]-sfc_global_logmask);
#elif LX_SFC_SEL_MODE == 12 || LX_SFC_SEL_MODE == 14
            lambda*=std::exp(0.5*(sfc_frame_logmask[fi]-sfc_global_logmask));
#elif LX_SFC_SEL_MODE == 13 || LX_SFC_SEL_MODE == 17
            lambda*=std::exp(0.25*(sfc_frame_logmask[fi]-sfc_global_logmask));
#elif LX_SFC_SEL_MODE == 15
            lambda*=std::exp((1.0/3.0)*(sfc_frame_logmask[fi]-sfc_global_logmask));
#elif LX_SFC_SEL_MODE == 16
            lambda*=std::exp(0.4*(sfc_frame_logmask[fi]-sfc_global_logmask));
#elif LX_SFC_SEL_MODE == 18
            lambda*=std::exp(0.5*(sfc_frame_logmask[fi]-sfc_global_logmask));
#endif

            std::array<double,lx_sfc::SFC_BANDS> sfc_j{},sfc_fold_j{};
            std::array<uint8_t,lx_sfc::SFC_BANDS> sfc_gain{},sfc_tilt{};
            alignas(64) std::array<float,512> sfcBandL{},sfcBandR{},foldBandL{},foldBandR{};
            auto eval_sfc=[&](const auto& erb_dec){
                for(int j=0;j<lx_sfc::SFC_BANDS;++j){
                    const int b=lx_sfc::SFC_FIRST_BAND+j;
                    const int lo=erb_dec.edge[b],hi=erb_dec.edge[b+1];
                    const double gain_rms=std::sqrt((double(statEL[fi][b])+double(statER[fi][b]))/(2.0*double(fr.k)));
                    const uint8_t gq=lx_sfc::quantize_gain32_ac(float(gain_rms)); sfc_gain[j]=gq;
                    const float target_l2=lx_sfc::decode_gain32(gq)*std::sqrt(2.0f*float(fr.k));
                    lx_sfc::continue_stereo_band_eval(decL.data(),decR.data(),lo,hi,target_l2,uint32_t(fi),
                                                      sfcBandL.data(),sfcBandR.data());
                    const uint8_t tq=lx_sfc::select_spectral_tilt3(l+lo,r+lo,sfcBandL.data(),sfcBandR.data(),hi-lo,target_l2); sfc_tilt[j]=tq;
                    lx_sfc::apply_spectral_tilt3(sfcBandL.data(),sfcBandR.data(),0,hi-lo,target_l2,tq);
                    lx_sfc::apply_sfc_diffuse_guard(sfcBandL.data(),sfcBandR.data(),0,hi-lo,target_l2,tq,sfc_guard_exposure[j]);

                    // fSFC folding candidate. Source is the immediately preceding
                    // decoded ERB band. It is evaluated from the ordinary quantized
                    // baseline here; the subset search below forbids folding when
                    // that source band is itself replaced by SFC, thereby keeping
                    // the source explicitly quantized whenever folding needs it.
                    {
                        // Folding only reads the immediately preceding reconstructed
                        // slice.  Write the candidate directly into local scratch;
                        // copying the complete spectrum for every candidate is unnecessary.
                        lx_sfc::fold_stereo_band_to(decL.data(),decR.data(),lo,hi,target_l2,
                                                    foldBandL.data(),foldBandR.data());
                    }
                    const double dq=base_dq[j]; double ds=0.0;
                    const double qtw=base_qtw[j],xe=base_xe[j]; double stw=0.0;
                    double q4=0.0,s4=0.0; const double nd=2.0*double(hi-lo);
                    for(int i=lo;i<hi;++i){const double xl=double(l[i]),xr=double(r[i]);const int bi=i-lo;const double sl=xl-sfcBandL[bi],sr=xr-sfcBandR[bi];const double se=sl*sl+sr*sr;ds+=se;const double x2=xl*xl+xr*xr;stw+=se*x2;s4+=sl*sl*sl*sl+sr*sr*sr*sr;}
                    double cq=aud_cost(dq,b), cs=aud_cost(ds,b);
#if LX_SFC_TONE_MODE == 1
                    // Target-energy-weighted error.  Mean weight is exactly one,
                    // so isolated energetic spectral lines receive proportionally
                    // more protection without an absolute threshold.
                    { const double norm=nd/std::max(xe,tiny_sel);
                      cq=aud_cost(qtw*norm,b); cs=aud_cost(stw*norm,b); }
#elif LX_SFC_TONE_MODE == 2
                    // Equal blend of ordinary FEAR error and target-energy-weighted
                    // FEAR error; both terms have identical physical units.
                    { const double norm=nd/std::max(xe,tiny_sel);
                      const double tq=aud_cost(qtw*norm,b), ts=aud_cost(stw*norm,b);
                      cq=std::sqrt(std::max(cq,tiny_sel)*std::max(tq,tiny_sel));
                      cs=std::sqrt(std::max(cs,tiny_sel)*std::max(ts,tiny_sel)); }
#elif LX_SFC_TONE_MODE == 3
                    // L4-equivalent concentrated-error energy: equals L2 for a
                    // uniform error field and rises when error collapses into peaks.
                    { const double tq=std::sqrt(std::max(q4*nd,tiny_sel));
                      const double ts=std::sqrt(std::max(s4*nd,tiny_sel));
                      cq=aud_cost(tq,b); cs=aud_cost(ts,b); }
#elif LX_SFC_TONE_MODE == 4
                    // Geometric mean of ordinary and concentrated-error FEAR costs.
                    { const double tq=aud_cost(std::sqrt(std::max(q4*nd,tiny_sel)),b);
                      const double ts=aud_cost(std::sqrt(std::max(s4*nd,tiny_sel)),b);
                      cq=std::sqrt(std::max(cq,tiny_sel)*std::max(tq,tiny_sel));
                      cs=std::sqrt(std::max(cs,tiny_sel)*std::max(ts,tiny_sel)); }
#elif LX_SFC_TONE_MODE == 5
                    // Combine target salience and concentrated-error protection.
                    { const double norm=nd/std::max(xe,tiny_sel);
                      const double tq=aud_cost(qtw*norm,b);
                      const double ts=aud_cost(stw*norm,b);
                      const double pq=aud_cost(std::sqrt(std::max(q4*nd,tiny_sel)),b);
                      const double ps=aud_cost(std::sqrt(std::max(s4*nd,tiny_sel)),b);
                      cq=std::cbrt(std::max(cq,tiny_sel)*std::max(tq,tiny_sel)*std::max(pq,tiny_sel));
                      cs=std::cbrt(std::max(cs,tiny_sel)*std::max(ts,tiny_sel)*std::max(ps,tiny_sel)); }
#elif LX_SFC_TONE_MODE >= 6 && LX_SFC_TONE_MODE <= 11
                    // Convex salience blend: protect energetic target lines while
                    // retaining the established FEAR band distortion as anchor.
                    { const double norm=nd/std::max(xe,tiny_sel);
                      const double tq=aud_cost(qtw*norm,b), ts=aud_cost(stw*norm,b);
#if LX_SFC_TONE_MODE == 6
                      constexpr double a=0.25;
#elif LX_SFC_TONE_MODE == 7
                      constexpr double a=0.50;
#elif LX_SFC_TONE_MODE == 8
                      constexpr double a=0.75;
#elif LX_SFC_TONE_MODE == 9
                      constexpr double a=0.40;
#elif LX_SFC_TONE_MODE == 10
                      constexpr double a=0.60;
#else
                      const double a=1.0/(1.0+std::exp(sfc_frame_logmask[fi]-sfc_global_logmask));
#endif
                      cq=(1.0-a)*cq+a*tq; cs=(1.0-a)*cs+a*ts; }
#endif
                    const size_t ci=fi*ERB_BANDS+b;
                    const double rfull=double(cells[ci].bpc)*double(cells[ci].dim);
                    // SFC may only claim the rate above the tonal-audibility
                    // guard as savings.  Protected shape bits are not fungible:
                    // they exist precisely because the source is both exposed
                    // and structured.  For masked/diffuse texture guard=0 and
                    // the original SFC RD decision is recovered exactly.
                    const double rguard=double(hf_guard_bpc[ci])*double(cells[ci].dim);
                    double rsavable=std::max(0.0,rfull-rguard);
                    // Same structural-deficit variable also governs whether SFC
                    // may claim true-shape rate as savings.  If the pilot RD
                    // point predicts an unresolved tonal object, continuation
                    // cannot freely consume those shape bits.
                    if(fear_deficit_sfc!=0.0){
                        const double q=std::max(0.0,double(statFearTonal[fi][b]));
                        const double h=q*std::exp(-rd_beta[ci]*double(cells[ci].bpc));
                        rsavable /= (1.0+fear_deficit_sfc*h);
                    }

                    // LX-FEAR SFC gate (frozen production formula).
                    // Structured/tonal bands are allowed to claim progressively
                    // less D_N rate as SFC savings.  The C1 smoothstep keeps the
                    // gate continuous at both endpoints and leaves diffuse texture
                    // essentially unchanged.
                    const double tau0=std::clamp(rd_tau[ci],0.0,1.0);
                    const double lp=(line_use&2)?std::clamp(line_psi[ci],0.0,1.0):0.0;
                    const double tau=1.0-(1.0-tau0)*(1.0-lp);
                    // production: SFC may claim savings only to the degree
                    // that the source is simultaneously diffuse in the established
                    // band-tonality metric and free of a persistent exposed line.
                    // D=(1-tau)(1-L) is a continuous diffuse probability; D^2 is
                    // the joint two-factor confidence that parametric continuation
                    // is appropriate rather than true shape quantization.
                    const double diffuse=(1.0-tau0)*(1.0-lp);
                    rsavable *= diffuse*diffuse;
                    // CPM has first claim on tonal bits. SFC can only claim the
                    // unprotected complement of the already-reserved CPM utility.
                    if(cpm_protect_formula>0){
                        const double pg=std::max(0.0,cpm_protect[ci]);
                        rsavable *= std::exp(-pg);
                    }

                    // Relative line-survival RD term. Compare SFC and the actual
                    // D_n reconstruction at the strongest source line in log
                    // amplitude. The term is zero when both preserve the line
                    // equally and positive only when SFC damages it more.
                    const double dfline=double(SR)/(2.0*double(fr.k));
                    const int ipline=std::clamp(int(std::floor(double(statLinePeakFreq[fi][b])/dfline)),lo,hi-1);
                    const int biline=ipline-lo;
                    const double atline=std::hypot(double(l[ipline]),double(r[ipline]));
                    const double aqline=std::hypot(double(decL[ipline]),double(decR[ipline]));
                    const double asline=std::hypot(double(sfcBandL[biline]),double(sfcBandR[biline]));
                    const double lq=std::log((aqline+tiny_sel)/(atline+tiny_sel));
                    const double ls=std::log((asline+tiny_sel)/(atline+tiny_sel));
                    const double line_cost=lp*std::max(cq,tiny_sel)*(ls*ls-lq*lq);
                    // v0.15 pSFC/fSFC syntax. Gain + the one mode bit are common.
                    // Prediction additionally needs AR(2)+tilt; folding does not.
                    const double psfc_bits=double(lx_sfc::SFC_GAIN_BITS + 1 + 2*lx_sfc::SFC_AR_BITS + lx_sfc::SFC_TILT_BITS);
                    const double fsfc_bits=double(lx_sfc::SFC_GAIN_BITS + 1);
                    sfc_j[j]=(cs-cq)+line_cost+lambda*(psfc_bits-rsavable);

                    double dfs=0.0,ftw=0.0,f4=0.0;
                    for(int i=lo;i<hi;++i){
                        const double xl=double(l[i]),xr=double(r[i]); const int bi=i-lo;
                        const double fl=xl-double(foldBandL[bi]),frx=xr-double(foldBandR[bi]);
                        const double fe=fl*fl+frx*frx, x2=xl*xl+xr*xr;
                        dfs+=fe; ftw+=fe*x2; f4+=fl*fl*fl*fl+frx*frx*frx*frx;
                    }
                    double cf=aud_cost(dfs,b);
#if LX_SFC_TONE_MODE == 1
                    { const double norm=nd/std::max(xe,tiny_sel); cf=aud_cost(ftw*norm,b); }
#elif LX_SFC_TONE_MODE == 2
                    { const double norm=nd/std::max(xe,tiny_sel); const double tf=aud_cost(ftw*norm,b); cf=std::sqrt(std::max(cf,tiny_sel)*std::max(tf,tiny_sel)); }
#elif LX_SFC_TONE_MODE == 3
                    cf=aud_cost(std::sqrt(std::max(f4*nd,tiny_sel)),b);
#elif LX_SFC_TONE_MODE == 4
                    { const double tf=aud_cost(std::sqrt(std::max(f4*nd,tiny_sel)),b); cf=std::sqrt(std::max(cf,tiny_sel)*std::max(tf,tiny_sel)); }
#elif LX_SFC_TONE_MODE == 5
                    { const double norm=nd/std::max(xe,tiny_sel); const double tf=aud_cost(ftw*norm,b); const double pf=aud_cost(std::sqrt(std::max(f4*nd,tiny_sel)),b); cf=std::cbrt(std::max(cf,tiny_sel)*std::max(tf,tiny_sel)*std::max(pf,tiny_sel)); }
#elif LX_SFC_TONE_MODE >= 6 && LX_SFC_TONE_MODE <= 11
                    { const double norm=nd/std::max(xe,tiny_sel); const double tf=aud_cost(ftw*norm,b);
#if LX_SFC_TONE_MODE == 6
                      constexpr double af=0.25;
#elif LX_SFC_TONE_MODE == 7
                      constexpr double af=0.50;
#elif LX_SFC_TONE_MODE == 8
                      constexpr double af=0.75;
#elif LX_SFC_TONE_MODE == 9
                      constexpr double af=0.40;
#elif LX_SFC_TONE_MODE == 10
                      constexpr double af=0.60;
#else
                      const double af=1.0/(1.0+std::exp(sfc_frame_logmask[fi]-sfc_global_logmask));
#endif
                      cf=(1.0-af)*cf+af*tf; }
#endif
                    const double afline=std::hypot(double(foldBandL[biline]),double(foldBandR[biline]));
                    const double lf=std::log((afline+tiny_sel)/(atline+tiny_sel));
                    const double fold_line_cost=lp*std::max(cq,tiny_sel)*(lf*lf-lq*lq);
                    sfc_fold_j[j]=(cf-cq)+fold_line_cost+lambda*(fsfc_bits-rsavable);

                    // exact line-survival feasibility constraint for SFC.
                    // A persistent exposed line is not fungible with diffuse band energy.
                    // rho(s)=exp(-alpha*s/(1-s)) is a continuous relative-amplitude
                    // error envelope: rho->infinity-equivalent/no constraint as s->0,
                    // rho->0 as s->1.  If formed SFC violates the envelope at the
                    // strongest source line, the SFC representation is infeasible.
                    constexpr bool sfc_survive=false;
                    if(sfc_survive){
                        const double sa=std::clamp(line_psi[ci],0.0,1.0);
                        constexpr double aa=0.75;
                        const double rho=std::exp(-aa*sa/(1.0-sa+std::numeric_limits<double>::epsilon()));
                        const double df=double(SR)/(2.0*double(fr.k));
                        const int ip=std::clamp(int(std::floor(double(statLinePeakFreq[fi][b])/df)),lo,hi-1);
                        const int bi=ip-lo;
                        const double at=std::hypot(double(l[ip]),double(r[ip]));
                        const double ah=std::hypot(double(sfcBandL[bi]),double(sfcBandR[bi]));
                        const double rel=std::fabs(ah-at)/std::max(at,std::numeric_limits<double>::min());
                        if(rel>rho) sfc_j[j]=std::numeric_limits<double>::infinity();
                    }
                }
            };
            switch(fr.block){case BlockSize::Long2048:eval_sfc(eL);break;case BlockSize::Medium1024:eval_sfc(e1024);break;case BlockSize::Short512:eval_sfc(eS);break;case BlockSize::Micro256:eval_sfc(e256);break;}

            // Exact subset RD search. The objective has only one-bit memory:
            // fSFC for band b is legal iff b-1 stays direct.  Dynamic programming
            // therefore solves the same k-of-8 problem in O(B*K) instead of
            // enumerating all 255 subsets.
            struct SfcSubsetState { double cost; uint8_t mask; };
            constexpr double INF=std::numeric_limits<double>::infinity();
            std::array<std::array<SfcSubsetState,2>,lx_sfc::SFC_MAX_ACTIVE+1> dp{}, next{};
            for(auto& row:dp) for(auto& v:row) v={INF,0};
            dp[0][0]={0.0,0};
            for(int band=0;band<lx_sfc::SFC_BANDS;++band){
                for(auto& row:next) for(auto& v:row) v={INF,0};
                for(int k=0;k<=lx_sfc::SFC_MAX_ACTIVE;++k){
                    for(int prev=0;prev<2;++prev){
                        const auto cur=dp[k][prev]; if(!std::isfinite(cur.cost)) continue;
                        // Keep this band direct. This also restores fold eligibility
                        // for the following band.
                        if(cur.cost<next[k][0].cost) next[k][0]=cur;
                        if(k==lx_sfc::SFC_MAX_ACTIVE) continue;
                        const bool source_direct=(band==0)||(prev==0);
                        const double selected_cost=(source_direct && sfc_fold_j[band]<sfc_j[band])?sfc_fold_j[band]:sfc_j[band];
                        const SfcSubsetState take{cur.cost+selected_cost,uint8_t(cur.mask|(1u<<band))};
                        if(take.cost<next[k+1][1].cost) next[k+1][1]=take;
                    }
                }
                dp=next;
            }
            uint8_t best_mask=0; double best_j=0.0;
            for(int k=1;k<=lx_sfc::SFC_MAX_ACTIVE;++k){
                for(int prev=0;prev<2;++prev){
                    if(!std::isfinite(dp[k][prev].cost)) continue;
                    const double total=dp[k][prev].cost+lambda*double(lx_sfc::sfc_subset_bits(k));
                    if(total<best_j){best_j=total;best_mask=dp[k][prev].mask;}
                }
            }
            std::array<uint8_t,lx_sfc::SFC_BANDS> sfc_mode{};
            for(int bb=0;bb<lx_sfc::SFC_BANDS;++bb) if(best_mask&(1u<<bb)){
                const bool source_direct=(bb==0)||((best_mask&(1u<<(bb-1)))==0);
                sfc_mode[bb]=uint8_t(source_direct && sfc_fold_j[bb]<sfc_j[bb]);
            }
            // CPM is applied after SFC/Precision formation, where it can own
            // only the five-bin tonal subspace while SFC continues to own the
            // diffuse remainder of the same ERB.
            CpmFrameDecision cd{};
            sf_cpm[fi-beg]=cd;

            SfcFrameDecision fd{}; fd.mask=best_mask; fd.gain=sfc_gain; fd.tilt=sfc_tilt; fd.mode=sfc_mode;
            if(best_mask){
                const int k=std::popcount(unsigned(best_mask));
                int payload_bits=lx_sfc::sfc_subset_bits(k);
                for(int j=0;j<lx_sfc::SFC_BANDS;++j) if(best_mask&(1u<<j))
                    payload_bits += lx_sfc::SFC_GAIN_BITS + 1 + (fd.mode[j] ? 0 : (2*lx_sfc::SFC_AR_BITS + lx_sfc::SFC_TILT_BITS));
                fd.nbits=uint16_t(payload_bits);
                auto fit_ar_side=[&](const auto& erb_ar){
                    for(int j=0;j<lx_sfc::SFC_BANDS;++j) if((best_mask&(1u<<j)) && !fd.mode[j]){
                        const int b=lx_sfc::SFC_FIRST_BAND+j;
                        const int lo=erb_ar.edge[b],hi=erb_ar.edge[b+1];
                        const auto aq=lx_sfc::fit_shared_ar2_q(l+lo,r+lo,hi-lo);
                        fd.ar_k1[j]=aq.k1; fd.ar_k2[j]=aq.k2;
                    }
                };
                switch(fr.block){case BlockSize::Long2048:fit_ar_side(eL);break;case BlockSize::Medium1024:fit_ar_side(e1024);break;case BlockSize::Short512:fit_ar_side(eS);break;case BlockSize::Micro256:fit_ar_side(e256);break;}
            }
            sf_sfc[fi-beg]=fd;
            // Advance the guard state only after this frame's SFC subset has been
            // fixed.  Inactive bands break the consecutive exposure run exactly.
            const double sfc_frame_support=double(fr.k)/double(K_LONG);
            for(int gj=0;gj<lx_sfc::SFC_BANDS;++gj)
                sfc_diffuse_exposure[gj]=(best_mask&(1u<<gj))?(sfc_guard_exposure[gj]+sfc_frame_support):0.0;


            // Final decoder-synchronous SFC spectrum: selected bands are formed
            // in ascending frequency order, so a formed lower neighbor becomes
            // valid local context for the next formed band exactly as in decode.
            tmpL=decL; tmpR=decR;
            auto apply_sfc=[&](const auto& erb_dec){
                // same S=U*C*E*K*T*P router and same transmitted IFC
                // syntax, but evaluate each candidate analytically.  The old R52
                // path rebuilt/renormalized 64 full bands and re-evaluated their
                // residuals.  Here one candidate needs a single fused dot-product
                // pass; all post-normalization distortion follows from quadratic
                // forms.  No signal-dependent branch exists in that inner pass.
                for(int j=0;j<lx_sfc::SFC_BANDS;++j){
                    if(!(best_mask&(1u<<j))) continue;
                    const int b=lx_sfc::SFC_FIRST_BAND+j,lo=erb_dec.edge[b],hi=erb_dec.edge[b+1],n=hi-lo;
                    const float target_l2=lx_sfc::decode_gain32(sfc_gain[j])*std::sqrt(2.0f*float(fr.k));
                    if(fd.mode[j]){
                        const int sb=b-1; const int slo=erb_dec.edge[sb],shi=erb_dec.edge[sb+1];
                        lx_sfc::fold_stereo_band(tmpL.data(),tmpR.data(),slo,shi,lo,hi,target_l2);
                    }else lx_sfc::continue_stereo_band_guided(tmpL.data(),tmpR.data(),lo,hi,target_l2,uint32_t(fi),fd.ar_k1[j],fd.ar_k2[j]);
                    if(!fd.mode[j]){
                        lx_sfc::apply_spectral_tilt3(tmpL.data(),tmpR.data(),lo,hi,target_l2,fd.tilt[j]);
                        lx_sfc::apply_sfc_diffuse_guard(tmpL.data(),tmpR.data(),lo,hi,target_l2,fd.tilt[j],sfc_guard_exposure[j]);
                    }

                    double ds=0.0,stw=0.0,s4=0.0,xe=0.0;
                    double eL=0.0,eR=0.0,xs=0.0,wx2=0.0,ws2=0.0,wxs=0.0;
                    for(int i=lo;i<hi;++i){
                        const double xl=double(l[i]),xr=double(r[i]);
                        const double sl=double(tmpL[i]),sr=double(tmpR[i]);
                        const double dl=xl-sl,dr=xr-sr,e2=dl*dl+dr*dr,x2=xl*xl+xr*xr;
                        ds+=e2; stw+=e2*x2; s4+=dl*dl*dl*dl+dr*dr*dr*dr; xe+=x2;
                        eL+=sl*sl; eR+=sr*sr; xs+=xl*sl+xr*sr;
                        wx2+=x2*x2; ws2+=x2*(sl*sl+sr*sr); wxs+=x2*(xl*sl+xr*sr);
                    }
                    const double nd=2.0*double(std::max(n,1));
                    auto local_cost=[&](double err,double tw)->double{
                        double c0=aud_cost(err,b);
#if LX_SFC_TONE_MODE == 2
                        const double norm=nd/std::max(xe,tiny_sel);
                        c0=std::sqrt(std::max(c0,tiny_sel)*std::max(aud_cost(tw*norm,b),tiny_sel));
#endif
                        return c0;
                    };
                    const double cbase=local_cost(ds,stw);
                    // Exact zero-risk pre-gate for the U*C*E*K*T*P router.
                    // Since gain=cbase-candidate_cost and candidate_cost>=0,
                    // best_gain <= cbase. Therefore the final score is bounded by
                    // cbase*E*K*T*P/(lambda*R). If that bound is <=1, no IFC
                    // candidate can possibly activate and the 64-state search is
                    // mathematically unnecessary.
                    const double maskE=std::max(double(statMask[fi][b]),tiny_sel);
                    const double z=ds/maskE;
                    const double E=z/(1.0+z);
                    const double denomK=std::max(nd*s4,tiny_sel);
                    const double K=std::clamp(1.0-(ds*ds)/denomK,0.0,1.0);
                    const size_t lci=fi*ERB_BANDS+b;
                    const double T0=std::clamp(rd_tau[lci],0.0,1.0), Lp=(line_use&4)?std::clamp(line_psi[lci],0.0,1.0):0.0;
                    const double T=1.0-(1.0-T0)*(1.0-Lp);
                    const auto band_energy=[&](size_t f)->double{return double(statEL[f][b])+double(statER[f][b]);};
                    const size_t fp=(fi>0)?fi-1:fi, fn=(fi+1<plan.size())?fi+1:fi;
                    const double ec=band_energy(fi),ep=band_energy(fp),enx=band_energy(fn);
                    const auto sim=[&](double a,double bb)->double{return 2.0*std::sqrt(std::max(a,0.0)*std::max(bb,0.0))/(a+bb+tiny_sel);};
                    const double P=std::sqrt(std::clamp(sim(ep,ec),0.0,1.0)*std::clamp(sim(ec,enx),0.0,1.0));
                    const double route_factor=E*K*T*P;
                    if(cbase*route_factor <= lambda*double(lx_sfc::SFC_IFC_PAYLOAD_BITS)) continue;
                    const double es=eL+eR+2.0*tiny_sel;
                    const float wLf=float(std::sqrt((eL+tiny_sel)/es)),wRf=float(std::sqrt((eR+tiny_sel)/es));
                    const double g=double(target_l2), g2=g*g;
                    alignas(64) float rvb[512],svb[512],wrvb[512],wsvb[512],x2b[512];
                    for(int ii=0;ii<n;++ii){
                        const int i=lo+ii; const float xl=l[i],xr=r[i],sl=tmpL[i],sr=tmpR[i];
                        const float x2=xl*xl+xr*xr, sv0=sl*wLf+sr*wRf, rv0=(xl-sl)*wLf+(xr-sr)*wRf;
                        rvb[ii]=rv0; svb[ii]=sv0; wrvb[ii]=x2*rv0; wsvb[ii]=x2*sv0; x2b[ii]=x2;
                    }
                    const auto& ab = fr.block==BlockSize::Long2048 ? g_tables.ifcL[j]
                                   : fr.block==BlockSize::Medium1024 ? g_tables.ifc1024[j]
                                   : fr.block==BlockSize::Short512 ? g_tables.ifcS[j]
                                                                    : g_tables.ifc256[j];
                    double best_gain=0.0; uint8_t bm=0,bp=0,bq=0;
                    for(uint8_t m=0;m<lx_sfc::SFC_IFC_FILTERS;++m){
                        for(uint8_t pp=0;pp<lx_sfc::SFC_IFC_POSITIONS;++pp){
                            const size_t off=size_t(m*lx_sfc::SFC_IFC_POSITIONS+pp)*size_t(n);
                            const float* atom=ab.a.data()+off; const float* atom2=ab.a2.data()+off;
                            float rvf=0.0f,svf=0.0f,wrvf=0.0f,wsvf=0.0f,wv2f=0.0f;
#if defined(__AVX2__)
                            __m256 vr=_mm256_setzero_ps(),vs=_mm256_setzero_ps(),vwr=_mm256_setzero_ps(),vws=_mm256_setzero_ps(),vv2=_mm256_setzero_ps();
                            int ii=0; for(;ii+8<=n;ii+=8){
                                const __m256 a=_mm256_loadu_ps(atom+ii),a2=_mm256_loadu_ps(atom2+ii);
                                vr=_mm256_fmadd_ps(a,_mm256_load_ps(rvb+ii),vr);
                                vs=_mm256_fmadd_ps(a,_mm256_load_ps(svb+ii),vs);
                                vwr=_mm256_fmadd_ps(a,_mm256_load_ps(wrvb+ii),vwr);
                                vws=_mm256_fmadd_ps(a,_mm256_load_ps(wsvb+ii),vws);
                                vv2=_mm256_fmadd_ps(a2,_mm256_load_ps(x2b+ii),vv2);
                            }
                            auto hsum8=[](__m256 v){__m128 lo=_mm256_castps256_ps128(v),hi=_mm256_extractf128_ps(v,1);__m128 x=_mm_add_ps(lo,hi);x=_mm_hadd_ps(x,x);x=_mm_hadd_ps(x,x);return _mm_cvtss_f32(x);};
                            rvf=hsum8(vr);svf=hsum8(vs);wrvf=hsum8(vwr);wsvf=hsum8(vws);wv2f=hsum8(vv2);
                            for(;ii<n;++ii){const float a=atom[ii];rvf+=a*rvb[ii];svf+=a*svb[ii];wrvf+=a*wrvb[ii];wsvf+=a*wsvb[ii];wv2f+=atom2[ii]*x2b[ii];}
#else
                            for(int ii=0;ii<n;++ii){const float a=atom[ii];rvf+=a*rvb[ii];svf+=a*svb[ii];wrvf+=a*wrvb[ii];wsvf+=a*wsvb[ii];wv2f+=atom2[ii]*x2b[ii];}
#endif
                            const double rv=double(rvf),sv=double(svf),xv=sv+rv,wsv=double(wsvf),wxv=wsv+double(wrvf),wv2=double(wv2f);
                            const uint8_t qa=lx_sfc::quantize_ifc_amp3(rv/std::max(g,tiny_sel));
                            const double A=double(lx_sfc::decode_ifc_amp3(qa))*g;
                            const double eu=eL+eR+2.0*A*sv+A*A;
                            const double sc=g/std::sqrt(std::max(eu,tiny_sel));
                            const double di=xe+g2-2.0*sc*(xs+A*xv);
                            const double yWy=sc*sc*(ws2+2.0*A*wsv+A*A*wv2);
                            const double twi=wx2+yWy-2.0*sc*(wxs+A*wxv);
                            const double gain=cbase-local_cost(std::max(di,0.0),std::max(twi,0.0));
                            if(gain>best_gain){best_gain=gain;bm=m;bp=pp;bq=qa;}
                        }
                    }
                    const double C=std::clamp(best_gain/std::max(cbase,tiny_sel),0.0,1.0);
                    const double U=best_gain/(lambda*double(lx_sfc::SFC_IFC_PAYLOAD_BITS)+tiny_sel);
                    const double score=U*C*route_factor;
                    if(score>1.0){
                        fd.ifc_on[j]=1;fd.ifc_mode[j]=bm;fd.ifc_pos[j]=bp;fd.ifc_amp[j]=bq;
                        lx_sfc::apply_ifc_band(tmpL.data(),tmpR.data(),lo,hi,target_l2,bm,bp,bq);
                        fd.nbits=uint16_t(fd.nbits+lx_sfc::SFC_IFC_PAYLOAD_BITS);
                    }
                }
            };
            switch(fr.block){case BlockSize::Long2048:apply_sfc(eL);break;case BlockSize::Medium1024:apply_sfc(e1024);break;case BlockSize::Short512:apply_sfc(eS);break;case BlockSize::Micro256:apply_sfc(e256);break;}

            // R196 research: once FEAR replaces an HF band with pSFC/fSFC,
            // the ordinary Direct-Dn shape is no longer part of the decoder
            // reconstruction.  Remove those discarded symbols before entropy
            // coding.  Precision-Dn, when enabled below, repopulates only the
            // residual coordinates whose RD utility justifies their rate.
            {
                const auto& ed=edges_for(fr.block);
                for(int j=0;j<lx_sfc::SFC_BANDS;++j){
                    if(!(best_mask&(1u<<j))) continue;
                    const int b=lx_sfc::SFC_FIRST_BAND+j;
                    for(int i=ed[b];i<ed[b+1];++i){qA[i]=0;qB[i]=0;}
                }
            }
            // IFC decisions are made during decoder-synchronous sequential SFC
            // formation; publish the completed decision only after that pass.
            sf_sfc[fi-beg]=fd;

#if LX_PREC_RES_MODE > 0
            // Precision-Dn: coefficient-domain sparse enhancement over SFC.
            // SFC remains the base at every coefficient.  Encoder-selected
            // correction coordinates from *all* SFC bands in this frame are
            // gathered into one normalized vector, quantized jointly on D_N,
            // then scattered back.  q==0 means pure SFC; q!=0 is both the
            // implicit sparse position map and the lattice correction value.
            // No band is switched from SFC to Precision and no explicit mask is
            // transmitted.
            auto precision_dn_frame=[&](const auto& erb_dec){
                struct Cand{
                    double utility;
                    double red_cost;
                    double rate;
                    float u;       // residual coordinate normalized by pdelta
                    float pdelta;
                    int i;
                    int b;
                    uint8_t axis;  // 0=A, 1=B in decoder quantizer basis
                };
                std::vector<Cand> cand;
                cand.reserve(size_t(fr.coded_bins));

                // Start with no Precision symbols in every SFC-active band.
                for(int j=0;j<lx_sfc::SFC_BANDS;++j){
                    if(!(best_mask&(1u<<j))) continue;
                    const int b=lx_sfc::SFC_FIRST_BAND+j;
                    for(int i=erb_dec.edge[b];i<erb_dec.edge[b+1];++i){qA[i]=0;qB[i]=0;}
                }

                for(int j=0;j<lx_sfc::SFC_BANDS;++j){
                    if(!(best_mask&(1u<<j))) continue;
                    // q=0 is an exact ZERO SFC state.  It intentionally carries
                    // no Precision anchors: otherwise the zero state would no
                    // longer mean zero and ultra-low HF could be re-excited.
                    if(sfc_gain[j]==0) continue;
                    const int b=lx_sfc::SFC_FIRST_BAND+j,lo=erb_dec.edge[b],hi=erb_dec.edge[b+1];
                    const int nch=(b>=int(is_start[fi]))?1:2;
                    const int sk=int(delta_scale[fi*ERB_BANDS+b]);
                    const float qdelta=delta_code_f[std::clamp(sk,0,480)];
#if LX_PREC_RES_MODE == 2 || LX_PREC_RES_MODE == 6
                    const float pdelta=0.5f*qdelta;
#elif LX_PREC_RES_MODE == 3
                    const float pdelta=0.25f*qdelta;
#elif LX_PREC_RES_MODE == 4
                    const float pdelta=2.0f*qdelta;
#else
                    const float pdelta=qdelta;
#endif
                    const auto axis=(nch==1)?intensity_axis_f(is_angle[fi][b]):IntensityAxisF{1.0f,0.0f};
                    const float ct=axis.cosine,st=axis.sine;
                    const double band_e=std::max(double(statEL[fi][b])+double(statER[fi][b]),tiny_sel);
                    const double nd=2.0*double(hi-lo);
                    const double rfull=double(cells[fi*ERB_BANDS+b].bpc)*double(cells[fi*ERB_BANDS+b].dim);
                    const double rcoord=rfull/double(std::max(1,nch*(hi-lo)));

                    for(int i=lo;i<hi;++i){
                        const float el=l[i]-tmpL[i], er=r[i]-tmpR[i];
                        const double x2=double(l[i])*l[i]+double(r[i])*r[i];
                        const double sal=std::sqrt(std::max(nd*x2/band_e,tiny_sel));
                        float aerr,berr=0.f;
                        if(nch==1) aerr=ct*el+st*er;
                        else if(!smode){aerr=el;berr=er;}
                        else{aerr=(el+er)*INV_SQRT2;berr=(el-er)*INV_SQRT2;}

                        const auto push_axis=[&](float e,uint8_t axis){
                            const int32_t qs=int32_t(std::nearbyint(e/pdelta));
                            if(qs==0) return;
                            const double before=double(e)*e;
                            const double after=double(e-float(qs)*pdelta)*double(e-float(qs)*pdelta);
                            const double red=before-after;
                            if(!(red>0.0)) return;
                            // Local FEAR derivative: aud_cost is monotone in band
                            // error; salience only distributes that band value over
                            // coefficients and protects exposed tonal peaks.
                            // FEAR-consistent Precision utility (p=1).
                            // Precision must buy error reduction in the same perceptual
                            // geometry as the SFC/FBR path, not raw PCM-MSE geometry.
                            // w_b is normalized per-frame FEAR importance in (0,1].
                            const double fear_w=aud_w[b];
                            const double red_cost=fear_w*red*sal;
                            const double utility=red_cost-lambda*double(LX_PREC_RATE_MUL)*rcoord;
                            if(utility>0.0) cand.push_back({utility,red_cost,rcoord,e/pdelta,pdelta,i,b,axis});
                        };
                        push_axis(aerr,0);
                        if(nch==2) push_axis(berr,1);
                    }
                }

                if(cand.empty()) return;
                // D_N is invariant under coordinate permutation, so a full utility
                // sort is pure overhead.  Only select the best K_LONG entries when
                // the candidate set actually exceeds the lattice scratch capacity.
                if(cand.size()>size_t(K_LONG)){
                    std::nth_element(cand.begin(),cand.begin()+K_LONG,cand.end(),
                        [](const Cand&a,const Cand&b){return a.utility>b.utility;});
                    cand.resize(K_LONG);
                }

                // Joint D_N quantization in dimensionless coordinates.  Per-band
                // quantizer steps are absorbed into u=e/pdelta, so one D_N lattice
                // spans coefficients from multiple ERBs without breaking their
                // established FEAR scales.
                const int n=int(std::min(cand.size(),size_t(K_LONG)));
                alignas(64) std::array<float,K_LONG> u{};
                alignas(64) std::array<int32_t,K_LONG> qv{};
                for(int n0=0;n0<n;++n0) u[n0]=cand[size_t(n0)].u;
                dn_nearest_i32(u.data(),n,1.0f,qv.data());

                for(int n0=0;n0<n;++n0){
                    const auto& c=cand[size_t(n0)];
                    const int32_t q=qv[n0];
                    if(q==0) continue;
                    if(c.axis==0) qA[c.i]=q; else qB[c.i]=q;
                    ++prec_coords;
                }

                // Materialize in the exact decoder order.  Precision from a
                // lower formed band becomes context for continuation of the next
                // band, so regenerate the final SFC chain and apply each band's
                // correction immediately after forming it.
                tmpL=decL; tmpR=decR;
                for(int j=0;j<lx_sfc::SFC_BANDS;++j){
                    if(!(best_mask&(1u<<j))) continue;
                    const int b=lx_sfc::SFC_FIRST_BAND+j,lo=erb_dec.edge[b],hi=erb_dec.edge[b+1];
                    const float target_l2=lx_sfc::decode_gain32(sfc_gain[j])*std::sqrt(2.0f*float(fr.k));
                    if(fd.mode[j]){
                        const int sb=b-1; const int slo=erb_dec.edge[sb],shi=erb_dec.edge[sb+1];
                        lx_sfc::fold_stereo_band(tmpL.data(),tmpR.data(),slo,shi,lo,hi,target_l2);
                    }else lx_sfc::continue_stereo_band_guided(tmpL.data(),tmpR.data(),lo,hi,target_l2,uint32_t(fi),fd.ar_k1[j],fd.ar_k2[j]);
                    if(!fd.mode[j]){
                        lx_sfc::apply_spectral_tilt3(tmpL.data(),tmpR.data(),lo,hi,target_l2,fd.tilt[j]);
                        lx_sfc::apply_sfc_diffuse_guard(tmpL.data(),tmpR.data(),lo,hi,target_l2,fd.tilt[j],sfc_guard_exposure[j]);
                    }
                    if(fd.ifc_on[j]) lx_sfc::apply_ifc_band(tmpL.data(),tmpR.data(),lo,hi,target_l2,fd.ifc_mode[j],fd.ifc_pos[j],fd.ifc_amp[j]);
                    const int nch=(b>=int(is_start[fi]))?1:2;
                    const int sk=int(delta_scale[fi*ERB_BANDS+b]);
                    const float qdelta=delta_code_f[std::clamp(sk,0,480)];
#if LX_PREC_RES_MODE == 2 || LX_PREC_RES_MODE == 6
                    const float pdelta=0.5f*qdelta;
#elif LX_PREC_RES_MODE == 3
                    const float pdelta=0.25f*qdelta;
#elif LX_PREC_RES_MODE == 4
                    const float pdelta=2.0f*qdelta;
#else
                    const float pdelta=qdelta;
#endif
                    const auto axis=(nch==1)?intensity_axis_f(is_angle[fi][b]):IntensityAxisF{1.0f,0.0f}; const float ct=axis.cosine,st=axis.sine;
                    for(int i=lo;i<hi;++i) prec_bins+=uint64_t((qA[i]!=0)||(qB[i]!=0));
                    lx_sfc::guide_stereo_band_with_precision(tmpL.data(),tmpR.data(),lo,hi,target_l2,
                        qA.data()+lo,qB.data()+lo,nch,smode,ct,st,pdelta);
                }
            };
            switch(fr.block){case BlockSize::Long2048:precision_dn_frame(eL);break;case BlockSize::Medium1024:precision_dn_frame(e1024);break;case BlockSize::Short512:precision_dn_frame(eS);break;case BlockSize::Micro256:precision_dn_frame(e256);break;}
#endif

            // CPM rank-1 tonal subspace, decoder-synchronous.  SFC remains
            // the diffuse base; CPM replaces only five bins around one persistent
            // partial.  Anchors code the absolute D_N line once. Continuations
            // predict canonical magnitude with 5->1->5 rank-1 and reuse the normal
            // q stream for zigzagged D_N magnitude innovation.
            {
                CpmFrameDecision bestcd{}; double bestcj=0.0;
                std::array<int32_t,10> bestpack{}; std::array<float,5> bestA{},bestB{}; int bestip=0,bestnch=0;
                auto eval_cpm_post=[&](const auto& erb_dec){
                    
                    for(int j=0;j<lx_sfc::SFC_BANDS;++j){
                        if(!(best_mask&(1u<<j))) continue;
                        const int b=lx_sfc::SFC_FIRST_BAND+j,lo=erb_dec.edge[b],hi=erb_dec.edge[b+1]; if(hi-lo<5)continue;
                        // CPM may replace a pure SFC band only when no sparse Precision
                        // symbol already owns another coordinate in it.
                        bool clean=true;for(int ii=lo;ii<hi;++ii)clean&=(qA[ii]==0 && qB[ii]==0);if(!clean)continue;
                        const int nch=(b>=int(is_start[fi]))?1:2; const auto axis=(nch==1)?intensity_axis_f(is_angle[fi][b]):IntensityAxisF{1.0f,0.0f}; const float ct=axis.cosine,st=axis.sine;
                        const double df=double(SR)/(2.0*double(fr.k));
                        const bool state_cont=cpm_state.valid;
                        const int pred_ip=state_cont?int(std::floor(double(cpm_state.freq_hz)/df)):-1;
                        const bool cont=state_cont && pred_ip>=lo+2 && pred_ip<=hi-3;
                        const int ip=cont?pred_ip:std::clamp(int(std::floor(double(statLinePeakFreq[fi][b])/df)),lo+2,hi-3);
                        const int sk=int(delta_scale[fi*ERB_BANDS+b]);const float delta=delta_code_f[std::clamp(sk,0,480)]; const float cpm_delta=std::ldexp(delta,-LX_CPM_PREC_SHIFT);
                        auto basis_src=[&](int ii,float&A,float&B){const float L=l[ii],R=r[ii];if(nch==1){A=ct*L+st*R;B=0;}else if(!smode){A=L;B=R;}else{A=(L+R)*INV_SQRT2;B=(L-R)*INV_SQRT2;}};
                        std::array<int32_t,10> pack{};std::array<float,5> ca{},cb{};
                        if(cont){
                            // physical CPM state.  The persistent line lives in global
                            // L/R cosine/sine coordinates, never in LR/MS/IS or block-local transform bins.
                            float tc[5],ts[5],pl[5],pr[5];
                            const bool hl=fr.sample_pos!=0;
                            const bool hr=fr.sample_pos+size_t(fr.k)+size_t(K_MIN)+size_t(OVL)<=ns;
                            switch(fr.block){
                                case BlockSize::Long2048: lx_cpm::physical_line_templates(dLong,cpm_state.freq_hz,fr.sample_pos,hl,hr,ip,tc,ts); break;
                                case BlockSize::Medium1024: lx_cpm::physical_line_templates(d1024,cpm_state.freq_hz,fr.sample_pos,hl,hr,ip,tc,ts); break;
                                case BlockSize::Short512: lx_cpm::physical_line_templates(dShort,cpm_state.freq_hz,fr.sample_pos,hl,hr,ip,tc,ts); break;
                                case BlockSize::Micro256: lx_cpm::physical_line_templates(d256,cpm_state.freq_hz,fr.sample_pos,hl,hr,ip,tc,ts); break;
                            }
                            lx_cpm::synth_complex_line5(cpm_state.al,cpm_state.bl,tc,ts,pl);
                            lx_cpm::synth_complex_line5(cpm_state.ar,cpm_state.br,tc,ts,pr);
                            std::array<float,10> pv{},iv{};
                            for(int t=0;t<5;++t){
                                float A,B;basis_src(ip+t-2,A,B);
                                if(nch==1){pv[t]=ct*pl[t]+st*pr[t];pv[5+t]=0.f;}
                                else if(!smode){pv[t]=pl[t];pv[5+t]=pr[t];}
                                else{pv[t]=(pl[t]+pr[t])*INV_SQRT2;pv[5+t]=(pl[t]-pr[t])*INV_SQRT2;}
                                iv[t]=A-pv[t];iv[5+t]=B-pv[5+t];
                            }
                            std::array<int32_t,10> qv{};dn_nearest_i32(iv.data(),nch==1?5:10,1.0f/cpm_delta,qv.data());
                            for(int t=0;t<5;++t){ca[t]=pv[t]+float(qv[t])*cpm_delta;cb[t]=(nch==1)?0.f:pv[5+t]+float(qv[5+t])*cpm_delta;pack[t]=qv[t];if(nch==2)pack[5+t]=qv[5+t];}
                        }else{
                            std::array<float,10> xv{};for(int t=0;t<5;++t){float A,B;basis_src(ip+t-2,A,B);xv[t]=A;xv[5+t]=B;}std::array<int32_t,10> qv{};dn_nearest_i32(xv.data(),nch==1?5:10,1.0f/cpm_delta,qv.data());for(int t=0;t<5;++t){ca[t]=float(qv[t])*cpm_delta;cb[t]=(nch==1)?0.f:float(qv[5+t])*cpm_delta;pack[t]=qv[t];if(nch==2)pack[5+t]=qv[5+t];}
                        }
                        double curde=0,candde=0,curtw=0,candtw=0,xe=0,rate=cont?1.0:12.0;
                        for(int ii=lo;ii<hi;++ii){const double x2=double(l[ii])*l[ii]+double(r[ii])*r[ii];const double e0=(double(l[ii])-tmpL[ii])*(double(l[ii])-tmpL[ii])+(double(r[ii])-tmpR[ii])*(double(r[ii])-tmpR[ii]);curde+=e0;curtw+=e0*x2;xe+=x2;}
                        candde=curde;candtw=curtw;
                        for(int t=0;t<5;++t){const int ii=ip+t-2;float cl,cr;if(nch==1){cl=ct*ca[t];cr=st*ca[t];}else if(!smode){cl=ca[t];cr=cb[t];}else{cl=(ca[t]+cb[t])*INV_SQRT2;cr=(ca[t]-cb[t])*INV_SQRT2;}const double x2=double(l[ii])*l[ii]+double(r[ii])*r[ii];const double e0=(double(l[ii])-tmpL[ii])*(double(l[ii])-tmpL[ii])+(double(r[ii])-tmpR[ii])*(double(r[ii])-tmpR[ii]);const double e1=(double(l[ii])-cl)*(double(l[ii])-cl)+(double(r[ii])-cr)*(double(r[ii])-cr);candde+=e1-e0;candtw+=(e1-e0)*x2;rate+=cpm_int_rate(pack[t]);if(nch==2)rate+=cpm_int_rate(pack[5+t]);}
                        const double nd=2.0*double(hi-lo);auto lc=[&](double de,double tw){const double c0=aud_cost(std::max(de,tiny_sel),b);const double norm=nd/std::max(xe,tiny_sel);return std::sqrt(std::max(c0,tiny_sel)*std::max(aud_cost(std::max(tw*norm,tiny_sel),b),tiny_sel));};
                        // Structural persistence weights only the line-specific RD gain;
                        // no threshold is introduced. C^2 P is the frozen CPM detector.
                        double zsum=0,z0=0;for(int t=0;t<5;++t){const int ii=ip+t-2;const double z=double(l[ii])*l[ii]+double(r[ii])*r[ii];zsum+=z;if(t==2)z0=z;}const double C=z0/std::max(zsum,tiny_sel);
                        auto aat=[&](size_t ff)->double{const auto& pf=plan[ff];if(pf.block!=fr.block)return std::sqrt(z0);const float* pl=cL.data()+pf.coeff_off,*pr=cR.data()+pf.coeff_off;return std::hypot(double(pl[ip]),double(pr[ip]));};const size_t fp=fi?fi-1:fi,fn=(fi+1<plan.size())?fi+1:fi;const double ac=std::sqrt(z0),ap=aat(fp),an=aat(fn);auto sm=[&](double x,double y){return 2*x*y/std::max(x*x+y*y,tiny_sel);};const double P=std::sqrt(std::clamp(sm(ac,ap),0.0,1.0)*std::clamp(sm(ac,an),0.0,1.0));const double detector=C*C*P;
                        const double dcur=lc(curde,curtw),dcand=lc(std::max(candde,tiny_sel),std::max(candtw,tiny_sel));
                        const double line_gain=(dcur-dcand)*detector;const double jc=-line_gain+lambda*rate;
                        cpm_min_cont_j=std::min(cpm_min_cont_j,jc);if(cont){++cpm_cont_eligible;cpm_min_true_cont=std::min(cpm_min_true_cont,jc);}
                        const bool take = jc<0.0 && (!bestcd.on || jc<bestcj);
                        if(take){bestcj=jc;bestcd.on=1;bestcd.anchor=cont?0:1;bestcd.band=uint8_t(j);bestcd.pos=uint8_t(ip-lo);bestcd.nbits=uint16_t(cont?1:12);bestpack=pack;bestA=ca;bestB=cb;bestip=ip;bestnch=nch;}
                    }
                    if(bestcd.on){
                        const int b=lx_sfc::SFC_FIRST_BAND+int(bestcd.band),lo=erb_dec.edge[b],hi=erb_dec.edge[b+1];
                        for(int ii=lo;ii<hi;++ii){qA[ii]=0;qB[ii]=0;}
                        for(int t=0;t<5;++t){
                            const int ii=bestip+t-2;
                            qA[ii]=bestpack[t]; qB[ii]=(bestnch==1)?0:bestpack[5+t];
                            const auto axis=(bestnch==1)?intensity_axis_f(is_angle[fi][b]):IntensityAxisF{1.0f,0.0f}; const float ct=axis.cosine,st=axis.sine;
                            if(bestnch==1){tmpL[ii]=ct*bestA[t];tmpR[ii]=st*bestA[t];}
                            else if(!smode){tmpL[ii]=bestA[t];tmpR[ii]=bestB[t];}
                            else{tmpL[ii]=(bestA[t]+bestB[t])*INV_SQRT2;tmpR[ii]=(bestA[t]-bestB[t])*INV_SQRT2;}
                        }
                        sf_cpm[fi-beg]=bestcd;

                        // CPM is part of the final decoded low-to-high spectral context.
                        // Any later SFC band must therefore be regenerated after the CPM
                        // replacement, otherwise encoder and decoder condition AR(2) on
                        // different lower-frequency samples (visible first at N=256 where
                        // one 5-bin line occupies a much larger fraction of an ERB).
                        for(int jj=int(bestcd.band)+1;jj<lx_sfc::SFC_BANDS;++jj){
                            if(!(fd.mask&(1u<<jj))) continue;
                            const int bb=lx_sfc::SFC_FIRST_BAND+jj;
                            const int l2=erb_dec.edge[bb],h2=erb_dec.edge[bb+1];
                            const float target_l2=lx_sfc::decode_gain32(fd.gain[jj])*std::sqrt(2.0f*float(fr.k));
                            if(fd.mode[jj]){
                        const int sb=bb-1; const int slo=erb_dec.edge[sb],shi=erb_dec.edge[sb+1];
                        lx_sfc::fold_stereo_band(tmpL.data(),tmpR.data(),slo,shi,l2,h2,target_l2);
                    }else lx_sfc::continue_stereo_band_guided(tmpL.data(),tmpR.data(),l2,h2,target_l2,uint32_t(fi),fd.ar_k1[jj],fd.ar_k2[jj]);
                            if(!fd.mode[jj]){
                                lx_sfc::apply_spectral_tilt3(tmpL.data(),tmpR.data(),l2,h2,target_l2,fd.tilt[jj]);
                                lx_sfc::apply_sfc_diffuse_guard(tmpL.data(),tmpR.data(),l2,h2,target_l2,fd.tilt[jj],sfc_guard_exposure[jj]);
                            }
                            if(fd.ifc_on[jj]) lx_sfc::apply_ifc_band(tmpL.data(),tmpR.data(),l2,h2,target_l2,fd.ifc_mode[jj],fd.ifc_pos[jj],fd.ifc_amp[jj]);
#if LX_PREC_RES_MODE > 0
                            const int nch2=(bb>=int(is_start[fi]))?1:2;
                            const int sk2=int(delta_scale[fi*ERB_BANDS+bb]);
                            const float qdelta2=delta_code_f[std::clamp(sk2,0,480)];
#if LX_PREC_RES_MODE == 2 || LX_PREC_RES_MODE == 6
                            const float pdelta2=0.5f*qdelta2;
#elif LX_PREC_RES_MODE == 3
                            const float pdelta2=0.25f*qdelta2;
#elif LX_PREC_RES_MODE == 4
                            const float pdelta2=2.0f*qdelta2;
#else
                            const float pdelta2=qdelta2;
#endif
                            const auto axis2=(nch2==1)?intensity_axis_f(is_angle[fi][bb]):IntensityAxisF{1.0f,0.0f}; const float ct2=axis2.cosine,st2=axis2.sine;
                            lx_sfc::guide_stereo_band_with_precision(tmpL.data(),tmpR.data(),l2,h2,target_l2,
                                qA.data()+l2,qB.data()+l2,nch2,smode,ct2,st2,pdelta2);
#endif
                        }
                    }
                };
                switch(fr.block){case BlockSize::Long2048:eval_cpm_post(eL);break;case BlockSize::Medium1024:eval_cpm_post(e1024);break;case BlockSize::Short512:eval_cpm_post(eS);break;case BlockSize::Micro256:eval_cpm_post(e256);break;}
            }

            // Publish exact decoder-reconstructed CPM state as a physical sinusoid:
            // global frequency plus L/R cosine/sine amplitudes.  No transform or stereo
            // mode appears in the persistent state.
            { const auto& cfd=sf_cpm[fi-beg];
              if(cfd.on){
                const int b=lx_sfc::SFC_FIRST_BAND+int(cfd.band);
                const auto& cedge=edges_for(fr.block);
                const int lo=cedge[b];
                const int hi=cedge[b+1];
                const int ip=std::clamp(lo+int(cfd.pos),lo+2,hi-3);
                const double df=double(SR)/(2.0*double(fr.k));
                if(cfd.anchor || !cpm_state.valid)cpm_state.freq_hz=float((double(ip)+0.5)*df);
                float tc[5],ts[5],xl[5],xr[5];
                const bool hl=fr.sample_pos!=0;
                const bool hr=fr.sample_pos+size_t(fr.k)+size_t(K_MIN)+size_t(OVL)<=ns;
                switch(fr.block){
                    case BlockSize::Long2048: lx_cpm::physical_line_templates(dLong,cpm_state.freq_hz,fr.sample_pos,hl,hr,ip,tc,ts); break;
                    case BlockSize::Medium1024: lx_cpm::physical_line_templates(d1024,cpm_state.freq_hz,fr.sample_pos,hl,hr,ip,tc,ts); break;
                    case BlockSize::Short512: lx_cpm::physical_line_templates(dShort,cpm_state.freq_hz,fr.sample_pos,hl,hr,ip,tc,ts); break;
                    case BlockSize::Micro256: lx_cpm::physical_line_templates(d256,cpm_state.freq_hz,fr.sample_pos,hl,hr,ip,tc,ts); break;
                }
                for(int t=0;t<5;++t){const int ii=ip+t-2;xl[t]=tmpL[ii];xr[t]=tmpR[ii];}
                lx_cpm::fit_complex_line5(xl,tc,ts,cpm_state.al,cpm_state.bl);
                lx_cpm::fit_complex_line5(xr,tc,ts,cpm_state.ar,cpm_state.br);
                cpm_state.valid=true;cpm_state.band=cfd.band;cpm_state.pos=cfd.pos;cpm_state.frame=fi;
                ++cpm_frames;cpm_bits+=cfd.nbits;if(cfd.anchor)++cpm_anchors;else ++cpm_cont;
              } }

            // gain-only anti-collapse.  The decoder can infer this state
            // exactly from the ordinary direct quantizer: a non-SFC high band
            // whose complete coded shape is ZERO consumes one 7-bit gain from
            // the separate AC tail.  No per-band mode flag is transmitted.
            {
                auto apply_ac=[&](const auto& erb_dec){
                    for(int j=0;j<lx_sfc::SFC_BANDS;++j){
                        if(fd.mask&(1u<<j)) continue;
                        const int b=lx_sfc::SFC_FIRST_BAND+j, lo=erb_dec.edge[b], hi=erb_dec.edge[b+1];
                        const int nch=(b>=int(is_start[fi]))?1:2;
                        uint32_t nz=0;
                        for(int ii=lo;ii<hi;++ii) nz |= uint32_t(qA[ii]!=0) | uint32_t((nch==2)&&(qB[ii]!=0));
                        if(nz) continue;
                        const double band_e=double(statEL[fi][b])+double(statER[fi][b]);
                        const double gain_rms=std::sqrt(band_e/(2.0*double(fr.k)));
                        // ungated anti-collapse.  A collapsed direct HF band carries
                        // its source-band RMS gain directly; masking no longer attenuates the
                        // anti-collapse amplitude.  Syntax remains unchanged and decoder-inferable.
                        const uint8_t gq=lx_sfc::quantize_gain32_ac(float(gain_rms));
                        sf_ac_on[fi-beg][j]=1; sf_ac_gain[fi-beg][j]=gq;
                        const float target_l2=lx_sfc::decode_gain32(gq)*std::sqrt(2.0f*float(fr.k));
                        const auto axis=(nch==1)?intensity_axis_f(is_angle[fi][b]):IntensityAxisF{1.0f,0.0f};
                        const float ct=axis.cosine,st=axis.sine;
                        lx_sfc::anti_collapse_stereo_band(tmpL.data(),tmpR.data(),lo,hi,target_l2,uint32_t(fi),nch==1,ct,st,smode);
                    }
                };
                switch(fr.block){case BlockSize::Long2048:apply_ac(eL);break;case BlockSize::Medium1024:apply_ac(e1024);break;case BlockSize::Short512:apply_ac(eS);break;case BlockSize::Micro256:apply_ac(e256);break;}
            }

            // Encoder-side gain bookkeeping exists solely for optional recon PCM.
            // The bitstream decoder derives the same gain from q, so normal encoding
            // skips this complete per-band q-vector scan.
            if(recon){
                auto& dgL=recon_direct_gain_l[fi]; auto& dgR=recon_direct_gain_r[fi];
                const auto& cfd=sf_cpm[fi-beg];
                const int cpm_band=(cfd.on ? (lx_sfc::SFC_FIRST_BAND+int(cfd.band)) : -1);
                const auto& ed=edges_for(fr.block);
                for(int b=0;b<ERB_BANDS;++b){
                    if(b>=lx_sfc::SFC_FIRST_BAND){
                        const int j=b-lx_sfc::SFC_FIRST_BAND;
                        if((fd.mask&(1u<<j)) || sf_ac_on[fi-beg][j] || b==cpm_band) continue;
                    }
                    const int lo=ed[b],hi=ed[b+1];
                    const int nch=(b>=int(is_start[fi]))?1:2;
                    const int sk=int(delta_scale[fi*ERB_BANDS+b]);
                    const float delta=delta_code_f[std::clamp(sk,0,480)];
                    if(!smode && nch==2){
                        const float g=lr_gain[fi][b];
                        dgL[b]=lx_conditional_gain::conditional_dn_gain(qA.data()+lo,nullptr,hi-lo,1,b,delta*g);
                        dgR[b]=lx_conditional_gain::conditional_dn_gain(qB.data()+lo,nullptr,hi-lo,1,b,delta/g);
                    }else{
                        const float gg=lx_conditional_gain::conditional_dn_gain(qA.data()+lo,qB.data()+lo,hi-lo,nch,b,delta);
                        dgL[b]=gg;dgR[b]=gg;
                    }
                }
            }

            // Replace quantizer bookkeeping by the actual final reconstruction
            // error, including formed bands, so reported MSE/SNR are real.
            frame_error_energy=0.0; frame_source_energy=0.0;
            for(int i=0;i<fr.coded_bins;++i){
                const double dl=double(l[i])-tmpL[i],dr=double(r[i])-tmpR[i];
                frame_error_energy+=dl*dl+dr*dr;
                frame_source_energy+=double(l[i])*l[i]+double(r[i])*r[i];
            }

            // Decoder-synchronous spectral state must not depend on whether the
            // caller requested a PCM reconstruction file.  Subsequent temporal
            // tools (SFC/CPM/etc.) consume this reconstructed spectrum, so keep
            // the encoder state identical in CLI, GUI and benchmark modes.
            std::copy(tmpL.begin(),tmpL.begin()+fr.coded_bins,l);
            std::copy(tmpR.begin(),tmpR.begin()+fr.coded_bins,r);
            std::fill(l+fr.coded_bins,l+fr.k,0.f);
            std::fill(r+fr.coded_bins,r+fr.k,0.f);
#if defined(LX_OMP_PARALLEL)
            #pragma omp ordered
#endif
            {
            auto tokenize_bands=[&](const auto& erb_dec){
                for(int b=0;b<ERB_BANDS;++b){
                    const uint8_t fctx=uint8_t(b>>1);
                    const int lo=erb_dec.edge[b],hi=erb_dec.edge[b+1];
                    const int nch=(b>=int(is_start[fi]))?1:2;
                    for(int ch=0;ch<nch;++ch){
                        const int32_t* x=ch?qB.data():qA.data();
                        uint8_t ps=0;
                        if(magpred_prev_valid[ch][b] && magpred_prev_nz[ch][b]){
                            const uint64_t q1=magpred_prev_q1[ch][b];
                            const uint64_t nz=magpred_prev_nz[ch][b];
                            ps = uint8_t(1u + uint32_t(q1 >= 2u*nz) + uint32_t(q1 >= 4u*nz));
                        }
                        const uint8_t mbase=uint8_t(unsigned(MAG_CTX_MAP[b])*MAG_PRED_STATES + ps);
                        uint64_t cur_q1=0; uint32_t cur_nz=0;
                        uint8_t prev_count=0;
                        uint8_t prev_mag_state=0;
                        for(int i=lo;i<hi;i+=2){
                            const int32_t v0=x[i];
                            const uint32_t m0=uint32_t(v0<0?-int64_t(v0):int64_t(v0));
                            const uint8_t c0=uint8_t(v0==0?0:(v0>0?1:2));
                            uint32_t m1=0; uint8_t c1=0;
                            if(i+1<hi){
                                const int32_t v1=x[i+1];
                                m1=uint32_t(v1<0?-int64_t(v1):int64_t(v1));
                                c1=uint8_t(v1==0?0:(v1>0?1:2));
                            }
                            const uint8_t tup=uint8_t(c0+3u*c1);
                            uint8_t ts=0;
                            if(magpred_prev_valid[ch][b]){
                                const uint32_t den=uint32_t(std::max(1,hi-lo));
                                ts=uint8_t(1u+std::min(2u,(3u*magpred_prev_nz[ch][b])/den));
                            }
                            const uint16_t cctx=uint16_t((unsigned(fctx)*3u+prev_count)*CLASS_TEMP_STATES+ts);
                            keyClass[nClass++]=uint16_t(unsigned(cctx)*9u+tup);
                            ++hClass[cctx][tup];
                            auto emit_mag=[&](uint32_t mag){
                                if(!mag)return;
                                cur_q1 += mag; ++cur_nz;
                                const uint32_t u=mag-1u;
                                const uint8_t msym=uint8_t(std::min(u,uint32_t(MAG_ESC)));
                                const uint16_t mctx=uint16_t(unsigned(mbase)*MAG_LOCAL_STATES + prev_mag_state);
                                keyMag[nMag++]=uint16_t(unsigned(mctx)*MAG_ALPHA+msym);++hMag[mctx][msym];
                                prev_mag_state=uint8_t(1u + uint32_t(mag>1u));
                                const uint32_t esc=uint32_t(u>=uint32_t(MAG_ESC));
                                if(vox && esc) vox_beta::put_uleb128(vp.mag_raw,u);
                                raw_hi_bytes+=esc*(1u+uint32_t(u>=128u)+uint32_t(u>=16384u)+uint32_t(u>=2097152u)+uint32_t(u>=268435456u));
                            };
                            emit_mag(m0); if(i+1<hi)emit_mag(m1);
                            prev_count=uint8_t((c0!=0u)+(c1!=0u));
                        }
                        magpred_prev_q1[ch][b]=cur_q1;
                        magpred_prev_nz[ch][b]=cur_nz;
                        magpred_prev_valid[ch][b]=1u;
                    }
                    for(int ch=nch;ch<2;++ch){magpred_prev_q1[ch][b]=0;magpred_prev_nz[ch][b]=0;magpred_prev_valid[ch][b]=0;}
                }
            };
            switch(fr.block){case BlockSize::Long2048:tokenize_bands(eL);break;case BlockSize::Medium1024:tokenize_bands(e1024);break;case BlockSize::Short512:tokenize_bands(eS);break;case BlockSize::Micro256:tokenize_bands(e256);break;}
            const size_t nib=size_t(ERB_BANDS-int(is_start[fi]));
            is_bands+=nib;is_frames+=size_t(nib!=0);
            // Syntax overhead relative to the existing LR/MS + block header:
            // one Intensity-enable flag per frame, then 2 bits for each active
            // high-band KLT angle.  With four bands this is exactly 9 bits/frame.
            is_side_bits+=(nib?uint64_t((is_cfg==4?2:0)+(is_cfg==4?is_bits_cfg():is_code_count_cfg()*is_bits_cfg())):0u);
            mse+=frame_error_energy;
            en+=frame_source_energy;
            msc+=smode;
            n1024+=fr.block==BlockSize::Medium1024;
            n512 +=fr.block==BlockSize::Short512;
            n256 +=fr.block==BlockSize::Micro256;
            } // ordered tokenization / deterministic stream assembly
        }

        std::vector<uint8_t> sfc_side_syms; unsigned sfc_side_k=0;
        if(vox){
            // R338 hybrid: Gain + AR1 use bounded adaptive Rice residual coding.
            // AR2 + Tilt share a geometric rANS prefix with the existing scale stream.
            std::array<std::vector<std::pair<int,int>>,3> rice_residuals; // SFC gain, AC gain, AR1
            sfc_side_syms.clear(); sfc_side_syms.reserve(512);
            lx_gainpred::State sgp, a1p, a2p, tp, acp;
            auto emit_side=[&](int q,int pr){if(pr>=0)sfc_side_syms.push_back(lx_sideinfo_rans::symbol(q,pr));};
            for(size_t fi=beg;fi<end;++fi){
                const int lf=int(fi-beg);sgp.begin_frame(lf);a1p.begin_frame(lf);a2p.begin_frame(lf);tp.begin_frame(lf);
                const auto& gd=sf_sfc[fi-beg];
                for(int j=0;j<lx_sfc::SFC_BANDS;++j) if(gd.mask&(1u<<j)){
                    int q=int(gd.gain[j]),pr=sgp.predict(j);rice_residuals[0].emplace_back(q,pr);sgp.observe(j,q);
                    if(!gd.mode[j]){
                        q=int(gd.ar_k1[j]);pr=a1p.predict(j);rice_residuals[2].emplace_back(q,pr);a1p.observe(j,q);
                        q=int(gd.ar_k2[j]);pr=a2p.predict(j);emit_side(q,pr);a2p.observe(j,q);
                        q=int(gd.tilt[j]);pr=tp.predict(j);emit_side(q,pr);tp.observe(j,q);
                    }
                }
            }
            for(size_t fi=beg;fi<end;++fi){
                const int lf=int(fi-beg);acp.begin_frame(lf);
                for(int j=0;j<lx_sfc::SFC_BANDS;++j) if(sf_ac_on[fi-beg][j]){
                    const int q=int(sf_ac_gain[fi-beg][j]),pr=acp.predict(j);rice_residuals[1].emplace_back(q,pr);acp.observe(j,q);
                }
            }
            auto best_rice_parameter=[](const auto& pairs)->unsigned{
                unsigned best=0;uint64_t best_bits=~uint64_t(0);
                for(unsigned k=0;k<=lx_gainpred::MAX_RICE_K;++k){
                    uint64_t bits=0;for(const auto& qp:pairs)bits+=lx_gainpred::bit_cost(qp.first,qp.second,k);
                    if(bits<best_bits){best_bits=bits;best=k;}
                }
                return best;
            };
            const unsigned gain_k=best_rice_parameter(rice_residuals[0]);
            const unsigned ac_gain_k=best_rice_parameter(rice_residuals[1]);
            const unsigned ar1_k=best_rice_parameter(rice_residuals[2]);

            // R348: SGC uses 3-bit angle indices but entropy-codes the exact
            // index residual against a deterministic temporal/spatial predictor.
            // A two-bit Rice parameter is selected once per superframe.
            constexpr int SGC_NG=lx_sfc::SFC_FIRST_BAND/2;
            struct SgcPredState {
                std::array<int,SGC_NG> last{}, prior{}, current{};
                SgcPredState(){last.fill(-1);prior.fill(-1);current.fill(-1);}
                void begin_frame(){prior=last;current.fill(-1);}
                int predict(int g) const {
                    const int t=prior[g];
                    const int a=g?current[g-1]:-1;
                    const int c=g?prior[g-1]:-1;
                    if(t>=0 && a>=0 && c>=0){const int z=t+a-c;return std::clamp(z,std::min(t,a),std::max(t,a));}
                    if(t>=0)return t;if(a>=0)return a;return -1;
                }
                void observe(int g,int q){current[g]=q;last[g]=q;}
            };
            auto sgc_zz=[](int v)->unsigned{return v>=0?unsigned(2*v):unsigned(-2*v-1);};
            auto sgc_cost=[&](int q,int pr,unsigned k)->unsigned{
                if(pr<0)return 3u;
                const unsigned quot=sgc_zz(q-pr)>>k;
                return quot<3u ? quot+1u+k : 6u;
            };
            unsigned sgc_k=0; uint64_t sgc_best_bits=~uint64_t(0);
            for(unsigned k=0;k<=2;++k){
                uint64_t bits=0;SgcPredState st;
                for(size_t fi=beg;fi<end;++fi){st.begin_frame();for(int g=0;g<SGC_NG;++g)if(sf_sgc_on[fi-beg][g]){const int q=int(sf_sgc_q[fi-beg][g]);const int pr=st.predict(g);bits+=sgc_cost(q,pr,k);st.observe(g,q);}}
                if(bits<sgc_best_bits){sgc_best_bits=bits;sgc_k=k;}
            }
            if(!sfc_side_syms.empty()){
                size_t best_bytes=~size_t(0);lx_rans4_fast::Scratch tmp;
                for(unsigned k=0;k<=7;++k){auto m=lx_sideinfo_rans::model(k);const auto e=lx_rans4_fast::encode4(sfc_side_syms.data(),sfc_side_syms.size(),m,tmp);if(e.bytes<best_bytes){best_bytes=e.bytes;sfc_side_k=k;}}
            }

            lx_gainpred::State sfc_writer_state,ar1_writer_state,ar2_writer_state,tilt_writer_state,ac_writer_state;
            SgcPredState sgc_writer_state;
            std::vector<uint8_t> sfc_main,ac_tail;vox_beta::SfcBitWriter sfcw(sfc_main);vox_beta::SfcBitWriter acw(ac_tail);
            sfcw.put(gain_k,3);sfcw.put(ar1_k,3);sfcw.put(sgc_k,2);sgc_side_bits+=2;if(!rice_residuals[1].empty())acw.put(ac_gain_k,3);
            auto put_sgc_rice=[&](int q,int pr)->unsigned{
                if(pr<0){sfcw.put(unsigned(q),3);return 3u;}
                const unsigned z=sgc_zz(q-pr),quot=z>>sgc_k;
                if(quot>=3u){sfcw.put(7u,3);sfcw.put(unsigned(q),3);return 6u;}
                if(quot)sfcw.put((1u<<quot)-1u,quot);sfcw.put(0u,1);if(sgc_k)sfcw.put(z&((1u<<sgc_k)-1u),sgc_k);return 1u+quot+sgc_k;
            };
            for(size_t fi=beg;fi<end;++fi){
                sfc_writer_state.begin_frame(int(fi-beg));ar1_writer_state.begin_frame(int(fi-beg));ar2_writer_state.begin_frame(int(fi-beg));tilt_writer_state.begin_frame(int(fi-beg));ac_writer_state.begin_frame(int(fi-beg));sgc_writer_state.begin_frame();
                if(modes[fi]){
                    constexpr int NG=lx_sfc::SFC_FIRST_BAND/2;uint64_t mask=0;int ksgc=0;
                    for(int g=0;g<NG;++g)if(sf_sgc_on[fi-beg][g]){mask|=uint64_t(1)<<g;++ksgc;}
                    sfcw.put(ksgc?1u:0u,1);sgc_side_bits+=1;
                    if(ksgc){for(int j=0;j<ksgc;++j)sfcw.put(1u,1);sfcw.put(0u,1);sgc_side_bits+=uint64_t(ksgc+1);const unsigned rb=sgc_rank_bits(NG,ksgc);if(rb){sfcw.put(sgc_rank_mask_n(mask,NG,ksgc),rb);sgc_side_bits+=rb;}for(int g=0;g<NG;++g)if(mask&(uint64_t(1)<<g)){const int q=int(sf_sgc_q[fi-beg][g]),pr=sgc_writer_state.predict(g);sgc_side_bits+=put_sgc_rice(q,pr);sgc_writer_state.observe(g,q);}}
                }
                const auto& fd=sf_sfc[fi-beg];const int istart=int(is_start[fi]);const bool is_enabled=istart<ERB_BANDS;
                if(is_cfg!=4&&istart!=ERB_BANDS&&istart!=IS_FIRST_BAND)throw std::runtime_error("non-suffix IS syntax mismatch");
                const bool frame_ifc_any=std::any_of(fd.ifc_on.begin(),fd.ifc_on.end(),[](uint8_t v){return v!=0;});const auto& cfd=sf_cpm[fi-beg];const BlockSize bs=plan[fi].block;
                const uint8_t ctrl=uint8_t(block_low_bit(bs)|(modes[fi]?2u:0u)|(is_enabled?4u:0u)|(uint8_t(std::popcount(unsigned(fd.mask))&7u)<<3)|(frame_ifc_any?0x40u:0u)|(cfd.on?0x80u:0u));
                const size_t lf=fi-beg;vp.block_hi[lf>>3]|=uint8_t(block_high_bit(bs)<<(lf&7u));const uint32_t ang=pack_is_angle_byte(is_angle[fi]);vp.side.push_back(ctrl);
                if(is_enabled){if(is_cfg==4){const unsigned nib=unsigned(ERB_BANDS-istart);sfcw.put(nib-1u,2);sfcw.put(is_angle[fi][istart],unsigned(is_bits_cfg()));}else{const unsigned ib=unsigned(is_code_count_cfg()*is_bits_cfg());sfcw.put(ang,ib);}}
                const int k=std::popcount(unsigned(fd.mask));unsigned sfc_gain_bits_frame=0;
                if(k){sfcw.put(sfc_rank_mask(fd.mask),unsigned(lx_sfc::sfc_subset_bits(k)));for(int j=0;j<lx_sfc::SFC_BANDS;++j)if(fd.mask&(1u<<j)){
                    sfc_gain_bits_frame+=lx_gainpred::put(sfcw,fd.gain[j],sfc_writer_state.predict(j),gain_k);sfc_writer_state.observe(j,fd.gain[j]);sfcw.put(fd.mode[j],1);
                    if(!fd.mode[j]){
                        sfc_gain_bits_frame+=lx_gainpred::put(sfcw,fd.ar_k1[j],ar1_writer_state.predict(j),ar1_k);ar1_writer_state.observe(j,fd.ar_k1[j]);
                        int pr=ar2_writer_state.predict(j);sfc_gain_bits_frame+=lx_sideinfo_rans::put_raw_if_needed(sfcw,fd.ar_k2[j],pr,lx_sfc::SFC_AR_BITS);ar2_writer_state.observe(j,fd.ar_k2[j]);
                        pr=tilt_writer_state.predict(j);sfc_gain_bits_frame+=lx_sideinfo_rans::put_raw_if_needed(sfcw,fd.tilt[j],pr,lx_sfc::SFC_TILT_BITS);tilt_writer_state.observe(j,fd.tilt[j]);
                    }
                    if(frame_ifc_any)sfcw.put(fd.ifc_on[j],1);if(frame_ifc_any&&fd.ifc_on[j]){sfcw.put(fd.ifc_mode[j],lx_sfc::SFC_IFC_FILTER_BITS);sfcw.put(fd.ifc_pos[j],lx_sfc::SFC_IFC_POS_BITS);sfcw.put(fd.ifc_amp[j],lx_sfc::SFC_IFC_AMP_BITS);}
                }}
                if(cfd.on){sfcw.put(cfd.anchor?1u:0u,1);if(cfd.anchor){sfcw.put(cfd.band,3);sfcw.put(cfd.pos,8);}}
                unsigned ac_bits_frame=0;for(int j=0;j<lx_sfc::SFC_BANDS;++j)if(sf_ac_on[fi-beg][j]){ac_bits_frame+=lx_gainpred::put(acw,sf_ac_gain[fi-beg][j],ac_writer_state.predict(j),ac_gain_k);ac_writer_state.observe(j,sf_ac_gain[fi-beg][j]);}
                sfc_bits+=fd.nbits-unsigned(k)*lx_sfc::SFC_GAIN_BITS+sfc_gain_bits_frame+(frame_ifc_any?uint16_t(k):uint16_t(0))+cfd.nbits+ac_bits_frame;sfc_frames+=size_t(fd.nbits!=0);
            }
            sfc_bits+=6u+(rice_residuals[1].empty()?0u:3u);
            sfcw.finish();acw.finish();if(sfc_main.size()>65535u||ac_tail.size()>65535u||sfc_side_syms.size()>65535u)throw std::runtime_error("SFC hybrid payload too large");
            vp.sfc.clear();vp.sfc.reserve(7+sfc_main.size()+ac_tail.size());auto p16=[&](uint16_t v){vp.sfc.push_back(uint8_t(v));vp.sfc.push_back(uint8_t(v>>8));};
            p16(uint16_t(sfc_main.size()));p16(uint16_t(ac_tail.size()));vp.sfc.push_back(uint8_t(sfc_side_k));p16(uint16_t(sfc_side_syms.size()));vp.sfc.insert(vp.sfc.end(),sfc_main.begin(),sfc_main.end());vp.sfc.insert(vp.sfc.end(),ac_tail.begin(),ac_tail.end());
            sfc_bits+=7u*8u;if(vp.sfc.size()>65535u)throw std::runtime_error("SFC payload too large");vp.h.reserved=uint16_t(vp.sfc.size());
        }

        const auto e0=profile_entropy?Clock::now():Clock::time_point{};
        const auto side_model=lx_sideinfo_rans::model(sfc_side_k);
        const auto esl=lx_sideinfo_rans::encode_prefix_then_main<256>(sfc_side_syms.data(),sfc_side_syms.size(),side_model,symScaleLo.get(),nScale,scale_lo_model,rsScaleLo);
        const auto elr=lx_rans4_fast::encode4(symLrScale.get(),nLrScale,lr_scale_model,rsLrScale);
        size_t class_bytes=0,mag_bytes=0;
        uint64_t coded_symbols=esl.symbols+elr.symbols;
        if(vox){
            // Production VOX path: encode the canonical four-state stream once.
            // The previous path first produced an ISA-local 16-state stream and
            // then encoded the same symbols a second time for VOX serialization.
            // Four-state SIMD preserves the canonical lane assignment exactly.
            const auto ep=encode4_q44_key_pair(keyClass.get(),nClass,class_model,rsClass,keyMag.get(),nMag,mag_model,rsMag);
            const auto ec=ep.first; const auto em=ep.second;
            class_bytes=ec.bytes; mag_bytes=em.bytes; coded_symbols+=ec.symbols+em.symbols;
            vp.h.scale_symbols=uint32_t(nScale); vp.h.lr_symbols=uint32_t(nLrScale); vp.h.class_symbols=uint32_t(nClass); vp.h.mag_symbols=uint32_t(nMag);
            vp.scale_rans.assign(esl.data,esl.data+esl.bytes); vp.lr_rans.assign(elr.data,elr.data+elr.bytes); vp.class_rans.assign(ec.data,ec.data+ec.bytes); vp.mag_rans.assign(em.data,em.data+em.bytes);
            vp.h.scale_rans_bytes=uint32_t(vp.scale_rans.size()); vp.h.lr_rans_bytes=uint32_t(vp.lr_rans.size()); vp.h.class_rans_bytes=uint32_t(vp.class_rans.size()); vp.h.mag_rans_bytes=uint32_t(vp.mag_rans.size());
            vp.h.scale_raw_bytes=uint32_t(vp.scale_raw.size()); vp.h.lr_raw_bytes=uint32_t(vp.lr_raw.size()); vp.h.mag_raw_bytes=uint32_t(vp.mag_raw.size());
            const uint64_t res_packet_bytes_current=uint64_t(sizeof(vox_beta::SuperframeHeader))
                +uint64_t(vp.side.size())+uint64_t(vp.block_hi.size())+uint64_t(vp.sfc.size())
                +uint64_t(vp.scale_rans.size())+uint64_t(vp.lr_rans.size())+uint64_t(vp.class_rans.size())+uint64_t(vp.mag_rans.size())
                +uint64_t(vp.scale_raw.size())+uint64_t(vp.lr_raw.size())+uint64_t(vp.mag_raw.size());
            if(pilot_stats){
                const uint64_t entropy_bytes=uint64_t(vp.scale_rans.size())+uint64_t(vp.lr_rans.size())+uint64_t(vp.class_rans.size())+uint64_t(vp.mag_rans.size())
                    +uint64_t(vp.scale_raw.size())+uint64_t(vp.lr_raw.size())+uint64_t(vp.mag_raw.size());
                pilot_stats->packet_bits.push_back(8.0*double(res_packet_bytes_current));
                pilot_stats->entropy_bits.push_back(8.0*double(entropy_bytes));
                pilot_stats->overhead_bits.push_back(8.0*double(res_packet_bytes_current-entropy_bytes));
            }
            res_actual_packet_bits_cum+=8.0*double(res_packet_bytes_current);
            if(sf<fbr_pred_sf_bits.size()){
                // Calibrate rho against the entropy forecast *after* the
                // controller's own first-order quantizer displacement.  Using
                // the unshifted forecast here makes a good startup correction
                // look like a model error in the next superframe.
                const double Heff=std::max(1.0, fbr_pred_sf_bits[sf]
                    - fbr_pred_sf_slope[sf]*fbr_shift_applied);
                res_pred_cum+=Heff;
            }
            fbr_target_bits_cum+=fbr_target_cur_assigned;
            vox->add(std::move(vp));
        }else{
            const auto ep=encode4_q44_key_pair(keyClass.get(),nClass,class_model,rsClass,keyMag.get(),nMag,mag_model,rsMag);
            const auto ec=ep.first; const auto em=ep.second;
            class_bytes=ec.bytes; mag_bytes=em.bytes; coded_symbols+=ec.symbols+em.symbols;
        }
        if(profile_entropy){
            const auto e1=Clock::now();
            entropy_ns+=std::chrono::duration<double,std::nano>(e1-e0).count();
        }
        entropy_symbols+=coded_symbols;

        bytes+=class_bytes+mag_bytes+esl.bytes+elr.bytes+raw_hi_bytes+raw_scale_bytes+raw_lr_scale_bytes+12+vp.sfc.size();

        // Causal model adaptation. R206 adds a decoder-synchronous slow
        // probability memory across superframes. This attacks model reset /
        // sparse-child variance without adding contexts or side information.
        auto hClassEff=hClass;
        auto hMagEff=hMag;
        if(class_mem_num>0){
            for(int c=0;c<CLASS_TUPLE_CTX;++c)for(int a=0;a<9;++a){
                class_hist_mem[c][a]=(class_hist_mem[c][a]*uint64_t(class_mem_num)+4u)>>3;
                class_hist_mem[c][a]+=hClass[c][a];
                hClassEff[c][a]=uint32_t(std::min<uint64_t>(class_hist_mem[c][a],0xffffffffu));
            }
        }
        if(mag_mem_num>0){
            for(int c=0;c<MAG_CTX;++c)for(int a=0;a<MAG_ALPHA;++a){
                mag_hist_mem[c][a]=(mag_hist_mem[c][a]*uint64_t(mag_mem_num)+4u)>>3;
                mag_hist_mem[c][a]+=hMag[c][a];
                hMagEff[c][a]=uint32_t(std::min<uint64_t>(mag_hist_mem[c][a],0xffffffffu));
            }
        }
        // Sign-symmetric class regularization: significance is strongly
        // structured; coefficient sign is near-bypass. Pool sign-equivalent
        // tuple counts without changing the 9-symbol bitstream grammar.
        for(int c=0;c<CLASS_TUPLE_CTX;++c){
            auto &h=hClassEff[c];
            const uint32_t s10=h[1]+h[2]; const uint32_t a10=(s10+1u)>>1; h[1]=a10; h[2]=s10-a10;
            const uint32_t s01=h[3]+h[6]; const uint32_t a01=(s01+1u)>>1; h[3]=a01; h[6]=s01-a01;
            const uint32_t s11=h[4]+h[5]+h[7]+h[8]; const uint32_t q=s11>>2, r=s11&3u;
            h[4]=q+uint32_t(r>0); h[5]=q+uint32_t(r>1); h[7]=q+uint32_t(r>2); h[8]=q;
        }
        // Empirical-Bayes backoff for temporal child contexts.
        if(class_backoff>0 || mag_backoff>0){
            auto hc=hClassEff;
            for(int parent=0; parent<CLASS_FREQ_CTX*3; ++parent){
                std::array<uint32_t,9> pooled{};
                for(int t=0;t<CLASS_TEMP_STATES;++t)for(int a=0;a<9;++a)pooled[a]+=hClassEff[parent*CLASS_TEMP_STATES+t][a];
                for(int t=0;t<CLASS_TEMP_STATES;++t){
                    const int c=parent*CLASS_TEMP_STATES+t;
                    for(int a=0;a<9;++a)hc[c][a]=uint32_t(uint64_t(class_backoff)*hClassEff[c][a]+pooled[a]);
                }
            }
            auto hm=hMagEff;
            for(int parent=0; parent<MAG_BASE_CTX; ++parent){
                std::array<uint32_t,MAG_ALPHA> pooled{};
                constexpr int NST=MAG_PRED_STATES*MAG_LOCAL_STATES;
                for(int t=0;t<NST;++t)for(int a=0;a<MAG_ALPHA;++a)pooled[a]+=hMagEff[parent*NST+t][a];
                for(int t=0;t<NST;++t){
                    const int c=parent*NST+t;
                    for(int a=0;a<MAG_ALPHA;++a)hm[c][a]=uint32_t(uint64_t(mag_backoff)*hMagEff[c][a]+pooled[a]);
                }
            }
            class_model.rebuild_encoder_only(class_backoff>0?hc:hClassEff);
            mag_model.rebuild_encoder_only(mag_backoff>0?hm:hMagEff);
        }else{
            class_model.rebuild_encoder_only(hClassEff);
            mag_model.rebuild_encoder_only(hMagEff);
        }
        {
            std::array<uint32_t,256> hs=hScaleLo;
            constexpr uint32_t SCALE_MEM_NUM=5;
            for(int a=0;a<256;++a){
                scale_hist_mem[a]=(scale_hist_mem[a]*SCALE_MEM_NUM+4u)>>3;
                scale_hist_mem[a]+=hScaleLo[a];
                hs[a]=uint32_t(std::min<uint64_t>(scale_hist_mem[a],0xffffffffu));
            }
            scale_lo_model.build_encoder_only(hs);
        }
        {
            std::array<uint32_t,256> hs=hLrScale;
            constexpr uint32_t LR_SCALE_MEM_NUM=5;
            for(int a=0;a<256;++a){
                lr_scale_hist_mem[a]=(lr_scale_hist_mem[a]*LR_SCALE_MEM_NUM+4u)>>3;
                lr_scale_hist_mem[a]+=hLrScale[a];
                hs[a]=uint32_t(std::min<uint64_t>(lr_scale_hist_mem[a],0xffffffffu));
            }
            lr_scale_model.build_encoder_only(hs);
        }
    }
    bytes+=(4*plan.size()+7)/8;
    bytes+=(is_side_bits+7)/8;
    if(psq_enabled_hot) bytes+=(uint64_t(ERB_BANDS)*uint64_t(plan.size())+7u)/8u; // one PSQ basis bit/ERB
    // Scale indices are already carried by the predictive rANS scale streams.
    if(recon){
        // v0.28: decoder-synchronous Conditional-D_n correction on all Direct-LVQ bands.
        for(size_t fi=0;fi<plan.size();++fi){
            const auto& fr=plan[fi];
            const auto& edge=edges_for(fr.block);
            const size_t off=fr.coeff_off;
            for(int b=0;b<ERB_BANDS;++b){
                const int lo=edge[b],hi=edge[b+1];
                const float gL=recon_direct_gain_l[fi][b],gR=recon_direct_gain_r[fi][b];
                for(int i=lo;i<hi;++i){cL[off+size_t(i)]*=gL;cR[off+size_t(i)]*=gR;}
            }
        }
        std::vector<lx_nsmdct::Frame> fs;fs.reserve(plan.size());
        for(const auto&fr:plan){lx_nsmdct::Frame f;f.pos=fr.sample_pos;f.n=fr.k;f.off=fr.coeff_off;fs.push_back(f);}
        VariableLowOverlapMDCT invL,invR;std::vector<float>L,R;
#if defined(LXVOXEL_SINGLE_THREAD)
        invL.synthesis(cL.data(),ns,fs,L);
        invR.synthesis(cR.data(),ns,fs,R);
#else
        std::thread tr([&]{invR.synthesis(cR.data(),ns,fs,R);});
        invL.synthesis(cL.data(),ns,fs,L);
        tr.join();
#endif
        recon->resize(2*ns);
        for(size_t i=0;i<ns;++i){(*recon)[2*i]=L[i];(*recon)[2*i+1]=R[i];}
    }
    std::cerr<<"CPM frames="<<cpm_frames<<" anchors="<<cpm_anchors<<" cont="<<cpm_cont<<" bits="<<cpm_bits<<" min_anchor_forecast="<<cpm_min_anchor_forecast<<" min_cont_j="<<cpm_min_cont_j<<" cont_eligible="<<cpm_cont_eligible<<" min_true_cont="<<cpm_min_true_cont<<"\n";
    return {bytes,mse,en,plan.size(),n1024,n512,n256,msc,is_frames,is_bands,is_side_bits,sfc_bits,sfc_frames,prec_bins,prec_coords,entropy_ns,entropy_symbols,bpc,tns_side_bits,tns_frames,rate_saturated,fbr_physical_kbps,fbr_debug_kkt_mu,fbr_debug_kkt_C};
}
