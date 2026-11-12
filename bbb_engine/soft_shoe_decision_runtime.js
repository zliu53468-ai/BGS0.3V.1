/**
 * BBB soft-shoe B/P/T-only inference. No cards/ranks are ever claimed observed.
 * Deterministic per-history seeded Monte Carlo; all returned probabilities/EVs
 * are MODEL ESTIMATES, not demonstrated profitable betting edges.
 */
(function(root,factory){
 "use strict";
 const api=factory();
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.BBBSoftShoe=api;
})(typeof self!=="undefined"?self:typeof window!=="undefined"?window:null,function(){
 "use strict";
 const VERSION="BBB_SOFT_SHOE_BPT_EV_V5_1";
 const DEFAULT=Object.freeze({
  PARTICLE_COUNT:448,LIKELIHOOD_DRAWS:80,NEXT_HAND_DRAWS:128,
  MIN_EV_THRESHOLD:.0030,MIN_CONFIDENCE:.75,FRACTIONAL_KELLY:.225,
  MAX_BET_FRACTION:.0175,AVERAGE_CARDS_PER_HAND:4.95,DEPTH_RELIABILITY:.5,
  MID_SHOE_MIN_EV_THRESHOLD:.00235,LATE_SHOE_MIN_EV_THRESHOLD:.00190,
  EARLY_SHOE_REMAINING_RATIO:.60,LATE_SHOE_REMAINING_RATIO:.30,
  LATE_CONFIDENCE_REMAINING_RATIO:.35,LATE_MIN_CONFIDENCE:.80,
  MIN_BET_EV_GAP:.0015,MIN_BET_ESS_RATIO:.45,MIN_BET_QUALITY:.65,
  MAX_BET_EV_SE:.060,MIN_REMAINING_CARDS_FOR_BET:25,
  END_SHOE_OVERRIDE_MIN_EV:.025,END_SHOE_OVERRIDE_CONFIDENCE:.90,
  END_SHOE_OVERRIDE_MAX_EV_SE:.030,
  RESAMPLE_ESS_RATIO:.5,MAX_HANDS:72,LOWER_BOUND_Z:1.645,SEQUENCE_WEIGHT:0,
  MIN_DIRECTION_ROUNDS:3,MIN_DIRECTION_EV_GAP:.010,
  MIN_DIRECTION_GAP_SNR:1.05,MIN_DIRECTION_ESS_RATIO:.40,
  MIN_DIRECTION_DIVERSITY:.25,MAX_DIRECTION_GAP_SE:.070,
  RESEARCH_DIRECTION_RELAXED:true,
  ADAPTIVE_EXTRA_DRAWS:32,ADAPTIVE_EXTRA_DRAWS_MAX:48,
  ADAPTIVE_MULTI_HORIZON_ENABLED:true,ADAPTIVE_NEXT_HAND_SHARE:.60,
  ADAPTIVE_HORIZON_HANDS:2,ADAPTIVE_HORIZON_WEIGHT:.15,
  ADAPTIVE_MIN_ESS_RATIO:.42,ADAPTIVE_MIN_DIVERSITY:.28,
  ADAPTIVE_MAX_ESS_RATIO:.75,ADAPTIVE_ROLLBACK_TOLERANCE:.08,
  ADAPTIVE_IMPORTANCE_MIX:.10,
  ADAPTIVE_GAP_MARGIN:.008,ADAPTIVE_SNR_MARGIN:.25,
  ROAD_AUX_ENABLED:true,ROAD_AUX_MIN_HISTORY:12,
  ROAD_AUX_AGREEMENT_MIN:.67,ROAD_AUX_CONFIDENCE_MIN:.45,
  ROAD_AUX_MAX_BOOST:.08,ROAD_AUX_CONFLICT_PENALTY:.06,
  ROAD_AUX_REQUIRE_SOFT_ELIGIBLE:true
 });
 // Research-display thresholds before V5.1's display-only relaxation.
 // Adaptive next-hand refinement keeps these references unchanged.
 const STRICT_DIRECTION_GATE=Object.freeze({MIN_DIRECTION_EV_GAP:.015,
  MIN_DIRECTION_GAP_SNR:1.15,MIN_DIRECTION_ESS_RATIO:.45,
  MIN_DIRECTION_DIVERSITY:.30,MAX_DIRECTION_GAP_SE:.060});
 const BASE=[128,32,32,32,32,32,32,32,32,32],TOTAL=416;
 const BASE_PROB={B:.458597,P:.446247,T:.095156};
 const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
 const sum=a=>a.reduce((x,y)=>x+y,0);
 const isResult=r=>r==="B"||r==="P"||r==="T";
 function config(c={}){
  const v={...DEFAULT,...c};
  for(const k of ["PARTICLE_COUNT","LIKELIHOOD_DRAWS","NEXT_HAND_DRAWS","MAX_HANDS",
   "MIN_DIRECTION_ROUNDS","ADAPTIVE_EXTRA_DRAWS","ADAPTIVE_EXTRA_DRAWS_MAX",
   "ADAPTIVE_HORIZON_HANDS"])
   if(!Number.isInteger(v[k])||v[k]<1||
      v[k]>(k==="MAX_HANDS"?75:k==="ADAPTIVE_EXTRA_DRAWS_MAX"?48:512))
    throw Error("invalid_"+k);
  if(v.PARTICLE_COUNT<8||v.LIKELIHOOD_DRAWS<2||v.NEXT_HAND_DRAWS<2)
   throw Error("sample_budget_too_small");
  if(typeof v.ADAPTIVE_MULTI_HORIZON_ENABLED!=="boolean")
   throw Error("invalid_ADAPTIVE_MULTI_HORIZON_ENABLED");
  if(typeof v.RESEARCH_DIRECTION_RELAXED!=="boolean")
   throw Error("invalid_RESEARCH_DIRECTION_RELAXED");
  for(const k of ["ROAD_AUX_ENABLED","ROAD_AUX_REQUIRE_SOFT_ELIGIBLE"])
   if(typeof v[k]!=="boolean")throw Error("invalid_"+k);
  if(!Number.isInteger(v.ROAD_AUX_MIN_HISTORY)||
     v.ROAD_AUX_MIN_HISTORY<1||v.ROAD_AUX_MIN_HISTORY>72)
   throw Error("invalid_ROAD_AUX_MIN_HISTORY");
  if(v.ADAPTIVE_EXTRA_DRAWS>v.ADAPTIVE_EXTRA_DRAWS_MAX||
     v.NEXT_HAND_DRAWS+v.ADAPTIVE_EXTRA_DRAWS_MAX>512)
   throw Error("adaptive_sample_budget_exceeded");
  if(v.ADAPTIVE_HORIZON_HANDS<2||v.ADAPTIVE_HORIZON_HANDS>3)
   throw Error("invalid_ADAPTIVE_HORIZON_HANDS");
  for(const [k,lo,hi] of [["MIN_EV_THRESHOLD",0,.1],["MIN_CONFIDENCE",0,1],
   ["MID_SHOE_MIN_EV_THRESHOLD",0,.1],["LATE_SHOE_MIN_EV_THRESHOLD",0,.1],
   ["EARLY_SHOE_REMAINING_RATIO",0,1],["LATE_SHOE_REMAINING_RATIO",0,1],
   ["LATE_CONFIDENCE_REMAINING_RATIO",0,1],["LATE_MIN_CONFIDENCE",0,1],
   ["MIN_BET_EV_GAP",0,.1],["MIN_BET_ESS_RATIO",0,1],["MIN_BET_QUALITY",0,1],
   ["MAX_BET_EV_SE",.001,.5],["MIN_REMAINING_CARDS_FOR_BET",0,60],
   ["END_SHOE_OVERRIDE_MIN_EV",0,.2],["END_SHOE_OVERRIDE_CONFIDENCE",0,1],
   ["END_SHOE_OVERRIDE_MAX_EV_SE",.001,.5],
   ["FRACTIONAL_KELLY",0,1],["MAX_BET_FRACTION",0,.1],
   ["AVERAGE_CARDS_PER_HAND",4,6],["DEPTH_RELIABILITY",0,1],
   ["RESAMPLE_ESS_RATIO",0,1],["LOWER_BOUND_Z",0,4],["SEQUENCE_WEIGHT",0,.05],
   ["MIN_DIRECTION_EV_GAP",0,.2],["MIN_DIRECTION_GAP_SNR",0,4],
   ["MIN_DIRECTION_ESS_RATIO",0,1],["MIN_DIRECTION_DIVERSITY",0,1],
   ["MAX_DIRECTION_GAP_SE",.001,.5],
    ["ADAPTIVE_GAP_MARGIN",0,.04],["ADAPTIVE_SNR_MARGIN",0,1],
   ["ADAPTIVE_NEXT_HAND_SHARE",0,1],["ADAPTIVE_HORIZON_WEIGHT",0,.25],
   ["ADAPTIVE_MIN_ESS_RATIO",0,1],["ADAPTIVE_MIN_DIVERSITY",0,1],
   ["ADAPTIVE_MAX_ESS_RATIO",0,1],["ADAPTIVE_ROLLBACK_TOLERANCE",0,.08],
   ["ADAPTIVE_IMPORTANCE_MIX",.01,.25],
   ["ROAD_AUX_AGREEMENT_MIN",0,1],["ROAD_AUX_CONFIDENCE_MIN",0,1],
   ["ROAD_AUX_MAX_BOOST",0,.10],["ROAD_AUX_CONFLICT_PENALTY",0,1]])
   if(!Number.isFinite(v[k])||v[k]<lo||v[k]>hi)throw Error("invalid_"+k);
  // Enforce meaningful tier ordering even when callers override defaults.
  if(!(v.ADAPTIVE_MIN_ESS_RATIO<v.ADAPTIVE_MAX_ESS_RATIO &&
       v.EARLY_SHOE_REMAINING_RATIO>v.LATE_SHOE_REMAINING_RATIO &&
       v.MIN_EV_THRESHOLD>=v.MID_SHOE_MIN_EV_THRESHOLD &&
       v.MID_SHOE_MIN_EV_THRESHOLD>=v.LATE_SHOE_MIN_EV_THRESHOLD &&
       v.LATE_MIN_CONFIDENCE>=v.MIN_CONFIDENCE &&
       v.END_SHOE_OVERRIDE_MIN_EV>=v.MIN_EV_THRESHOLD &&
       v.END_SHOE_OVERRIDE_CONFIDENCE>=v.LATE_MIN_CONFIDENCE &&
       v.END_SHOE_OVERRIDE_MAX_EV_SE<=v.MAX_BET_EV_SE))
   throw Error("invalid_wager_gate_tier_order");
  return Object.freeze(v);
 }
 function seedFor(history){
  let h=2166136261;
  for(const c of history){h^=c.charCodeAt(0);h=Math.imul(h,16777619);}
  return h>>>0;
 }
 function rngFrom(seed){
  let x=seed>>>0||0x9e3779b9;
  return()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return(x>>>0)/4294967296;};
 }
 function deal(counts,rng){
  if(!Array.isArray(counts)||counts.length!==10||sum(counts)<6)return null;
  const left=counts.slice();let available=sum(left),draws=0;
  function draw(){
   let x=Math.floor(rng()*available);
   for(let k=0;k<10;k++){
    x-=left[k];
    if(x<0){left[k]--;available--;draws++;return k;}
   }
   throw Error("virtual_deal_exhausted");
  }
  const p1=draw(),b1=draw(),p2=draw(),b2=draw();
  let p=(p1+p2)%10,b=(b1+b2)%10;
  if(p<8&&b<8){
   let p3=-1;
   if(p<=5){p3=draw();p=(p+p3)%10;}
   const bankDraw=p3<0?b<=5:
    b<=2?true:b===3?p3!==8:b===4?p3>=2&&p3<=7:
    b===5?p3>=4&&p3<=7:b===6?p3===6||p3===7:false;
   if(bankDraw)b=(b+draw())%10;
  }
  return {counts:left,draws,player:p,banker:b,result:b>p?"B":p>b?"P":"T"};
 }
 function normalize(a){const n=sum(a);return n>0?a.map(x=>x/n):null;}
 function resampleSystematic(weights,rng){
  const n=weights.length,selected=[],step=1/n,start=rng()*step;
  let i=0,c=weights[0];
  for(let j=0;j<n;j++){
   const u=start+j*step;
   while(u>c&&i<n-1)c+=weights[++i];
   selected.push(i);
  }
  return selected;
 }
 function posterior(history,opts={},seed=seedFor(history)){
  const cfg=config(opts),n=cfg.PARTICLE_COUNT,rng=rngFrom(seed);
  let particles=Array.from({length:n},()=>BASE.slice());
  let weights=Array(n).fill(1/n),ess=n,diversity=1,resampleCount=0;
  let depthExpected=TOTAL;
  if(history.length>cfg.MAX_HANDS)return{available:false,reason:"shoe_depth_limit",config:cfg,seed};
  for(let hand=0;hand<history.length;hand++){
   const observed=history[hand];
   if(!isResult(observed))throw Error("invalid_bpt_outcome");
   const proposals=new Array(n),evidence=new Array(n).fill(0);
   const target=TOTAL-(hand+1)*cfg.AVERAGE_CARDS_PER_HAND;
   const sigma=2.5+1.0*Math.sqrt(hand+1);
   for(let i=0;i<n;i++){
    const original=particles[i];
    let hits=0,chosen=null;
    for(let j=0;j<cfg.LIKELIHOOD_DRAWS;j++){
     const trial=deal(original,rng);
     if(trial&&trial.result===observed){
      hits++;
      if(rng()*hits<1)chosen=trial;
     }
    }
    if(chosen){
     const remaining=sum(chosen.counts),z=(remaining-target)/sigma;
     const depthLikelihood=Math.exp(-.5*cfg.DEPTH_RELIABILITY*z*z);
     evidence[i]=(hits/cfg.LIKELIHOOD_DRAWS)*depthLikelihood;
     proposals[i]=chosen.counts;
    }
   }
   // Sequential importance weights accumulate observed-outcome evidence;
   // resetting to uniform is valid only after an actual resampling step.
   const w=normalize(evidence.map((likelihood,i)=>likelihood*weights[i]));
   if(!w)return{available:false,reason:"particle_collapse",config:cfg,seed};
   ess=1/sum(w.map(x=>x*x));
   const next=new Array(n);
   if(ess/n<cfg.RESAMPLE_ESS_RATIO){
    const parents=resampleSystematic(w,rng),used=new Map();
    for(let k=0;k<n;k++){
     const parent=parents[k],times=used.get(parent)||0;
     used.set(parent,times+1);
     // Duplicated ancestors get distinct legal conditionally drawn hands,
     // with a bounded fallback to the accepted proposal (no random mutation).
     let proposal=proposals[parent];
     if(times>0){
      for(let j=0;j<64;j++){
       const trial=deal(particles[parent],rng);
       if(trial&&trial.result===observed){proposal=trial.counts;break;}
      }
     }
     next[k]=proposal.slice();
    }
    weights=Array(n).fill(1/n);
    diversity=used.size/n;
    resampleCount++;
   }else{
    for(let i=0;i<n;i++)next[i]=proposals[i]||particles[i];
    weights=w;
    diversity=w.filter(value=>value>0).length/n;
   }
   particles=next;depthExpected=target;
   if(particles.every(p=>sum(p)<6))
    return{available:false,reason:"shoe_exhausted",config:cfg,seed};
  }
  return{available:true,particles,weights,ess,diversity,resampleCount,depthExpected,
   config:cfg,seed,observed_actual_cards:false,soft_depth_constraint:true};
 }

 function frozenCore(history,sequenceRuntime=null,sequenceModel=null){
  // Prefer the actual 8x8 LSTM inference ONLY with verified frozen real-shoe
  // weights. The production bundle contains none. Untrained diagnostics do
  // not gain voting weight and are explicitly labelled as such.
  if(sequenceRuntime&&typeof sequenceRuntime.predict==="function"){
   const inferred=sequenceRuntime.predict(history,sequenceModel);
   if(inferred?.available&&inferred.trained&&inferred.validated)
    return inferred;
  }
  // A bounded, weak, untrained sequence diagnostic fallback. NOT an LSTM,
  // NOT evidence of a predictive edge and NEVER eligible for direct staking.
  const a=history.slice(-24).filter(isResult),last=a[a.length-1];
  let weights=[0,0,0],norm=0;
  for(let i=0;i<a.length;i++){
   const recency=Math.exp((i-a.length+1)/12);
   const attention=recency*Math.exp(a[i]===last?.25:0);
   weights[["B","P","T"].indexOf(a[i])]+=attention;norm+=attention;
  }
  const prior=[BASE_PROB.B,BASE_PROB.P,BASE_PROB.T];
  const prob=prior.map((p,i)=>.98*p+.02*(norm?weights[i]/norm:p));
  return{architecture:"untrained_weak_sequence_diagnostic",trained:false,
   validated:false,probabilities:{B:prob[0],P:prob[1],T:prob[2]},
   history_support:a.length,reason:"validated_frozen_lstm_unavailable"};
 }
 function handProbabilities(particles,cfg,seed,weights=null,drawCount=cfg.NEXT_HAND_DRAWS){
  if(!Number.isInteger(drawCount)||drawCount<1)throw Error("invalid_draw_count");
  const rng=rngFrom(seed^0x52ec17e),per=[],counts={B:0,P:0,T:0};
  const n=particles.length,w=normalize(weights||Array(n).fill(1/n));
  if(!w||w.length!==n)return null;
  let samples=0;
  for(let i=0;i<n;i++){
   const particle=particles[i],local={B:0,P:0,T:0};
   for(let j=0;j<drawCount;j++){
    const x=deal(particle,rng);if(!x)continue;
    counts[x.result]+=w[i];local[x.result]++;samples++;
   }
   per.push(local);
  }
  if(!samples)return null;
  const total=sum(Object.values(counts));
  const p={B:counts.B/total,P:counts.P/total,T:counts.T/total};
  const weightSquares=sum(w.map(x=>x*x)),sd={};
  for(const side of ["B","P","T"]){
   const av=per.map(x=>x[side]/drawCount);
   const mean=sum(av.map((x,i)=>x*w[i]));
   const spread=sum(av.map((x,i)=>w[i]*(x-mean)**2))/
    Math.max(1e-12,1-weightSquares);
   const within=p[side]*(1-p[side])/drawCount;
   // Preserve between-shoe latent dispersion; only MC noise is reduced by
   // the effective importance-weighted number of posterior particles.
   sd[side]=Math.sqrt(Math.max(0,spread-within)+within*weightSquares);
  }
  return{p,sd,samples,per,weights:w,drawsPerParticle:drawCount};
 }
 // Pool independent draws per particle without changing posterior weights.
 function poolPhysical(base,extra){
  const n=base.per.length,per=[],w=base.weights;
  const draws=(base.samples+extra.samples)/n;
  if(!Number.isInteger(draws)||draws<1)return base;
  const counts={B:0,P:0,T:0};
  for(let i=0;i<n;i++){
   const local={};
   for(const side of ["B","P","T"]){
    local[side]=base.per[i][side]+extra.per[i][side];
    counts[side]+=local[side]*w[i];
   }
   per.push(local);
  }
  const total=sum(Object.values(counts));
  if(!(total>0))return base;
  const p={B:counts.B/total,P:counts.P/total,T:counts.T/total};
  const w2=sum(w.map(x=>x*x)),sd={};
  for(const side of ["B","P","T"]){
   const av=per.map(x=>x[side]/draws);
   const mean=sum(av.map((x,i)=>x*w[i]));
   const spread=sum(av.map((x,i)=>w[i]*(x-mean)**2))/Math.max(1e-12,1-w2);
   const within=p[side]*(1-p[side])/draws;
   sd[side]=Math.sqrt(Math.max(0,spread-within)+within*w2);
  }
  return{p,sd,samples:base.samples+extra.samples,per,weights:w};
 }
 function shouldRefine(v,rounds,essRatio,diversity,cfg){
  if(!cfg.ADAPTIVE_MULTI_HORIZON_ENABLED||
     rounds<cfg.MIN_DIRECTION_ROUNDS||
     essRatio<cfg.ADAPTIVE_MIN_ESS_RATIO||
     diversity<cfg.ADAPTIVE_MIN_DIVERSITY||
     essRatio>=cfg.ADAPTIVE_MAX_ESS_RATIO)return false;
  const nearGap=Math.abs(v.abs_ev_gap-STRICT_DIRECTION_GATE.MIN_DIRECTION_EV_GAP)<=
   cfg.ADAPTIVE_GAP_MARGIN;
  const nearSnr=Math.abs(v.ev_gap_snr-STRICT_DIRECTION_GATE.MIN_DIRECTION_GAP_SNR)<=
   cfg.ADAPTIVE_SNR_MARGIN;
  return nearGap||nearSnr;
 }
 function conditionalPosteriorResample(post,seed,cfg){
  const n=post.particles.length;
  const target=normalize(post.weights||Array(n).fill(1/n));
  if(!target||target.length!==n)return null;
  const mix=cfg.ADAPTIVE_IMPORTANCE_MIX;
  const proposal=normalize(target.map(w=>(1-mix)*w+mix/n));
  if(!proposal)return null;
  const cdf=[];let cumulative=0;
  for(const q of proposal){cumulative+=q;cdf.push(cumulative);}
  cdf[n-1]=1;
  const rng=rngFrom(seed^0x6d2b79f5),indices=[];
  // One draw in each equal-width CDF stratum; no future outcome is used.
  for(let j=0;j<n;j++){
   const u=(j+rng())/n;let i=0;
   while(i<n-1&&u>=cdf[i])i++;
   indices.push(i);
  }
  // Target/proposal correction preserves the current posterior distribution.
  const weights=normalize(indices.map(i=>target[i]/proposal[i]));
  if(!weights)return null;
  const ess=1/sum(weights.map(w=>w*w));
  return{particles:indices.map(i=>post.particles[i]),weights,indices,
   ess,ess_ratio:ess/n,diversity:new Set(indices).size/n,
   future_outcome_conditioned:false};
 }
 function shouldRollbackAdaptive(before,after,cfg){
  const threshold=1-cfg.ADAPTIVE_ROLLBACK_TOLERANCE;
  return after.ess_ratio<before.ess_ratio*threshold||
   after.diversity<before.diversity*threshold;
 }
 function directionConsistency(outcomes,direction){
  const decided=outcomes.filter(x=>x==="B"||x==="P");
  return decided.length?decided.filter(x=>x===direction).length/decided.length:.5;
 }
 function futureEdgeScore(nextGapSnr,pathConsistency,pathUncertaintyNorm){
  if(![nextGapSnr,pathConsistency,pathUncertaintyNorm].every(Number.isFinite))
   return null;
  return clamp(nextGapSnr*.55+pathConsistency*.30+
   (1-pathUncertaintyNorm)*.15,0,1);
 }
 function simulateFuturePaths(particles,weights,direction,cfg,seed,
                              drawsPerParticle,nextHandEv,nextGapSnr){
  if(!particles?.length||!["B","P"].includes(direction)||
     !Number.isInteger(drawsPerParticle)||drawsPerParticle<1)
   return{available:false,reason:"path_budget_unavailable",
    estimated_only:true,validated_betting_edge:false};
  const n=particles.length,w=normalize(weights||Array(n).fill(1/n));
  if(!w)return{available:false,reason:"invalid_weights",
   estimated_only:true,validated_betting_edge:false};
  const selected=Array.from({length:n},(_,i)=>i)
   .sort((a,b)=>w[b]-w[a]).slice(0,Math.max(1,Math.ceil(n*.2)));
  const selectedMass=sum(selected.map(i=>w[i]));
  if(!(selectedMass>0))return{available:false,reason:"empty_posterior_mass",
   estimated_only:true,validated_betting_edge:false};
  const rng=rngFrom(seed^0x3c6ef372);
  const payoff=result=>result==="T"?0:
   direction==="B"?(result==="B"?.95:-1):(result==="P"?1:-1);
  const values={1:[],2:[],3:[]},stats=[];
  let simulatedHands=0;
  for(const index of selected){
   const byH={1:[],2:[],3:[]};let matched=0,decided=0;
   for(let r=0;r<drawsPerParticle;r++){
    let counts=particles[index].slice(),cumulativePayoff=0;
    for(let h=1;h<=3;h++){
     const hand=deal(counts,rng);if(!hand)break;
     counts=hand.counts;cumulativePayoff+=payoff(hand.result);simulatedHands++;
     if(h<=cfg.ADAPTIVE_HORIZON_HANDS&&
        (hand.result==="B"||hand.result==="P")){
      decided++;if(hand.result===direction)matched++;
     }
     byH[h].push(cumulativePayoff/h);
    }
   }
   const means={};
   for(const h of [1,2,3]){
    means[h]=byH[h].length?sum(byH[h])/byH[h].length:null;
    if(Number.isFinite(means[h]))
     values[h].push({mean:means[h],weight:w[index]});
   }
   stats.push({weight:w[index],means,
    consistency:decided?matched/decided:.5});
  }
  const pathEv={};
  for(const h of [1,2,3]){
   const rows=values[h],mass=sum(rows.map(x=>x.weight));
   pathEv[h]=mass>0?sum(rows.map(x=>x.mean*x.weight))/mass:null;
  }
  const horizon=cfg.ADAPTIVE_HORIZON_HANDS;
  const hRows=stats.filter(x=>Number.isFinite(x.means[horizon]));
  const hMass=sum(hRows.map(x=>x.weight));
  let uncertainty=null,consistency=null;
  if(hMass>0){
   const mean=sum(hRows.map(x=>x.weight*x.means[horizon]))/hMass;
   uncertainty=Math.sqrt(sum(hRows.map(x=>
    x.weight*(x.means[horizon]-mean)**2))/hMass);
   consistency=sum(hRows.map(x=>x.weight*x.consistency))/hMass;
  }
  const uncertaintyNorm=Number.isFinite(uncertainty)?
   clamp(uncertainty/1.5,0,1):null;
  const adjacent=horizon===2?3:2;
  const blended=Number.isFinite(pathEv[horizon])&&Number.isFinite(pathEv[adjacent])?
   (1-cfg.ADAPTIVE_HORIZON_WEIGHT)*pathEv[horizon]+
   cfg.ADAPTIVE_HORIZON_WEIGHT*pathEv[adjacent]:pathEv[horizon];
  return{
   available:Number.isFinite(pathEv[2])&&Number.isFinite(pathEv[3]),
   direction,next_hand_ev:nextHandEv,path_ev_1:pathEv[1],
   path_ev_2:pathEv[2],path_ev_3:pathEv[3],
   path_consistency:consistency,path_uncertainty:uncertainty,
   path_uncertainty_norm:uncertaintyNorm,
   primary_horizon_hands:horizon,horizon_weighted_ev:blended,
   next_ev_gap_snr:nextGapSnr,
   future_edge_score:Number.isFinite(pathEv[2])&&Number.isFinite(pathEv[3])?
    futureEdgeScore(nextGapSnr,consistency,uncertaintyNorm):null,
   selected_particle_count:selected.length,posterior_mass_covered:selectedMass,
   draws_per_selected_particle:drawsPerParticle,simulated_hands:simulatedHands,
   estimated_only:true,validated_betting_edge:false
  };
 }
 function evs(p){return{B:.95*p.B-p.P,P:p.P-p.B,T:8*p.T-(1-p.T)};}
 function eor(particles,cfg,seed,weights=null){
  // Paired common-random-number simulations for 0..9 hypothetical removal;
  // diagnostic-only sensitivity, not another independent observation.
  const effects=Array(10).fill(0),support=Array(10).fill(0);
  const stride=Math.max(1,Math.floor(particles.length/32));
  const selected=weights?resampleSystematic(weights,rngFrom(seed^0x9a3f0b1e)):null;
  for(let i=0;i<particles.length;i+=stride){
   const counts=particles[selected?selected[i]:i];
   for(let rank=0;rank<10;rank++){
    if(counts[rank]<1)continue;
    const removed=counts.slice();removed[rank]--;
    let difference=0;
    for(let k=0;k<cfg.NEXT_HAND_DRAWS;k++){
     const seed0=(seed^Math.imul(i+1,0x9e3779b1)^Math.imul(rank+1,0x85ebca6b)^Math.imul(k+1,0xc2b2ae35))>>>0;
     const first=deal(counts,rngFrom(seed0)),second=deal(removed,rngFrom(seed0));
     if(!first||!second)continue;
     const payoff=s=>s==="B"?1.95:s==="P"?-2:0;
     difference+=payoff(second.result)-payoff(first.result);support[rank]++;
    }
    effects[rank]+=difference;
   }
  }
  return effects.map((x,i)=>support[i]?x/support[i]:0);
 }
 function features48(posteriorState,physical,history,effects,frozen){
  const parts=posteriorState.particles;
  const weights=posteriorState.weights||Array(parts.length).fill(1/parts.length);
  const avg=Array.from({length:10},(_,rank)=>sum(parts.map((p,i)=>p[rank]*weights[i])));
  const stdev=avg.map((v,rank)=>Math.sqrt(sum(parts.map((p,i)=>weights[i]*(p[rank]-v)**2))));
  const prob=physical.p,ev=evs(prob);
  // v2 feature schema: 30 latent shoe features + 6 probabilities/EV +
  // 4 depth/ESS/quality + 5 weak recent-road features + 3 Frozen Core
  // soft probabilities. Do not apply any old v1 48D classifier to this layout.
  const seq=[];
  for(const len of [4,16]){
   const a=history.slice(-len),n=a.length;
   seq.push(n?a.filter(x=>x==="B").length/n:BASE_PROB.B);
   seq.push(n?a.filter(x=>x==="T").length/n:BASE_PROB.T);
  }
  let streak=0;
  if(history.length){
   const side=history[history.length-1];
   for(let i=history.length-1;i>=0&&history[i]===side;i--)streak++;
  }
  seq.push(Math.min(1,streak/6));
  seq.push(frozen.probabilities.B,frozen.probabilities.P,frozen.probabilities.T);
  const progress=1-sum(avg)/TOTAL;
  const quality=[progress,posteriorState.ess/parts.length,posteriorState.diversity,
   physical.samples/(parts.length*posteriorState.config.NEXT_HAND_DRAWS)];
  const f=[...avg.map(x=>x/TOTAL),...stdev.map(x=>x/TOTAL),...effects,
   prob.B,prob.P,prob.T,ev.B,ev.P,ev.T,...quality,...seq];
  if(f.length!==48||f.some(x=>!Number.isFinite(x)))throw Error("invalid_48d_features");
  return f;
 }
 function evaluateXgbTree(tree,features){
  let node=typeof tree==="string"?JSON.parse(tree):tree;
  for(let d=0;d<64;d++){
   if(Number.isFinite(node?.leaf))return node.leaf;
   const index=Number(String(node?.split??"").replace(/^f/,""));
   if(!Number.isInteger(index)||index<0||index>=48||
      !Number.isFinite(node?.split_condition)||!Array.isArray(node.children))
    throw Error("invalid_xgb_tree");
   const childId=features[index]<node.split_condition?node.yes:node.no;
   node=node.children.find(c=>c.nodeid===childId);
   if(!node)throw Error("xgb_child_missing");
  }
  throw Error("xgb_tree_depth_exceeded");
 }
 function predictXgbJson(model,features){
  if(!Array.isArray(model.trees)||!model.trees.length||
     model.trees.length>2000||!Array.isArray(model.base_logits)||
     model.base_logits.length!==3||model.base_logits.some(v=>!Number.isFinite(v)))
   throw Error("invalid_xgb_bundle");
  const scores=model.base_logits.slice();
  const eta=Number.isFinite(model.learning_rate)?model.learning_rate:1;
  if(!(eta>0&&eta<=1))throw Error("invalid_xgb_learning_rate");
  for(let i=0;i<model.trees.length;i++){
   const cls=model.tree_info?.[i]??i%3;
   if(!Number.isInteger(cls)||cls<0||cls>2)throw Error("invalid_xgb_class");
   scores[cls]+=eta*evaluateXgbTree(model.trees[i],features);
  }
  const temperature=model.temperature??1;
  if(!Number.isFinite(temperature)||temperature<=0)throw Error("invalid_xgb_temperature");
  const m=Math.max(...scores),exps=scores.map(x=>Math.exp((x-m)/temperature)),z=sum(exps);
  return{B:exps[0]/z,P:exps[1]/z,T:exps[2]/z};
 }
 function modelDecision(p,features,frozen,model){
  // Incompatible 57D historical XGB / synthetic-only fits never qualify.
  const verified=Boolean(model?.trained===true&&model.calibrated===true&&
    model.schema==="soft_shoe_bpt_48d_v2"&&model.feature_dimension===48&&
    model.validation?.out_of_shoe_holdout===true&&
    model.validation?.real_shoe_data===true);
  if(verified){
   let candidate=null;
   try{candidate=typeof model.predict==="function"?
     model.predict(features,frozen.probabilities):predictXgbJson(model,features);}
   catch(_){candidate=null;}
   if(candidate&&["B","P","T"].every(x=>Number.isFinite(candidate[x])&&candidate[x]>=0)){
    const z=candidate.B+candidate.P+candidate.T;
    if(Math.abs(z-1)<1e-6)return{p:candidate,model:"validated_external_xgb"};
   }
  }
  return{p,model:"physics_only_untrained_decision_fallback"};
 }

 // Moderate selective direction DISPLAY gate. This is not a validated betting
 // edge or a calibrated confidence interval; it only chooses whether the
 // model's B/P EV ordering is well-separated relative to its MC noise.
 function moderateDirectionGate({candidate,ev,physical,observations,essRatio,diversity,config:cfg}){
  const gate=cfg.RESEARCH_DIRECTION_RELAXED?cfg:{...cfg,...STRICT_DIRECTION_GATE};
  const gap=ev.B-ev.P,absGap=Math.abs(gap),n=physical.samples;
  // Gap payoff is +1.95 on Banker, -2 on Player, 0 on Tie.
  const gapSecondMoment=1.95*1.95*physical.p.B+4*physical.p.P;
  const oneHandVariance=Math.max(0,gapSecondMoment-gap*gap);
  const weights=physical.weights;
  const weightSquares=weights?sum(weights.map(x=>x*x)):0;
  const drawsPerParticle=physical.per?.length?physical.samples/physical.per.length:
   cfg.NEXT_HAND_DRAWS;
  const mcVariance=weights?oneHandVariance*weightSquares/drawsPerParticle:
   oneHandVariance/Math.max(1,n);
  const each=(physical.per||[]).map(h=>(1.95*h.B-2*h.P)/drawsPerParticle);
  const m=each.length?(weights?sum(each.map((x,i)=>x*weights[i])):
   sum(each)/each.length):gap;
  const spread=each.length>1?(weights?
   sum(each.map((x,i)=>weights[i]*(x-m)**2))/Math.max(1e-12,1-weightSquares):
   sum(each.map(x=>(x-m)**2))/(each.length-1)):0;
  // Monte Carlo noise in a per-particle gap has variance variance/NEXT_HAND_DRAWS.
  // Debias between-particle variation, then scale as an estimator-mean error.
  const latentMeanVariance=Math.max(0,spread-oneHandVariance/drawsPerParticle)/
   (weights?Math.max(1,1/weightSquares):Math.max(1,each.length));
  const gapSe=Math.sqrt(mcVariance+latentMeanVariance);
  const snr=gapSe>0?absGap/gapSe:0;
  const reasons=[];
  if(observations<gate.MIN_DIRECTION_ROUNDS)reasons.push("insufficient_rounds");
  if(absGap<gate.MIN_DIRECTION_EV_GAP)reasons.push("weak_ev_separation");
  if(!(gapSe>0&&Number.isFinite(gapSe))||gapSe>gate.MAX_DIRECTION_GAP_SE)
   reasons.push("high_estimator_uncertainty");
  if(snr<gate.MIN_DIRECTION_GAP_SNR)reasons.push("low_gap_to_noise");
  if(essRatio<gate.MIN_DIRECTION_ESS_RATIO)reasons.push("low_particle_ess");
  if(diversity<gate.MIN_DIRECTION_DIVERSITY)reasons.push("low_particle_diversity");
  return Object.freeze({
   direction:reasons.length?"SKIP":candidate,
   candidate,eligible:reasons.length===0,reason:reasons[0]||"moderate_gate_pass",
   reasons,abs_ev_gap:absGap,ev_gap:gap,ev_gap_se:gapSe,
   ev_gap_snr:snr,ess_ratio:essRatio,ancestor_diversity:diversity,
   estimated_only:true,validated_betting_edge:false
  });
 }

 // Decision-only wager selection; does not mutate PF / Physics / Frozen Core /
 // XGBoost forecasts or the separate moderate B/P research-display gate.
 function wagerPolicy(remainingCards,cfg=DEFAULT){
  const ratio=clamp(remainingCards/TOTAL,0,1);
  const minEV=ratio>cfg.EARLY_SHOE_REMAINING_RATIO?cfg.MIN_EV_THRESHOLD:
   ratio>=cfg.LATE_SHOE_REMAINING_RATIO?cfg.MID_SHOE_MIN_EV_THRESHOLD:
   cfg.LATE_SHOE_MIN_EV_THRESHOLD;
  const minConfidence=ratio<cfg.LATE_CONFIDENCE_REMAINING_RATIO?
   cfg.LATE_MIN_CONFIDENCE:cfg.MIN_CONFIDENCE;
  return Object.freeze({
   remaining_ratio:ratio,min_ev_threshold:minEV,min_confidence:minConfidence
  });
 }
 function wagerGate({cfg,remainingCards,ess,quality,ev,candidate,confidence,se,
                     lower,enoughDepth,observations,model}){
  const policy=wagerPolicy(remainingCards,cfg),bestEV=ev[candidate],
   evGap=Math.abs(ev.B-ev.P);
  const reasons=[];
  if(model!=="validated_external_xgb")reasons.push("trained_decision_model_unavailable");
  if(observations<1||!enoughDepth)reasons.push("insufficient_shoe_depth");
  if(!(Number.isFinite(ess)&&ess>=cfg.MIN_BET_ESS_RATIO*cfg.PARTICLE_COUNT))
   reasons.push("low_particle_ess");
  if(!(Number.isFinite(quality)&&quality>=cfg.MIN_BET_QUALITY))
   reasons.push("low_particle_quality");
  if(!(Number.isFinite(evGap)&&evGap>=cfg.MIN_BET_EV_GAP))
   reasons.push("ev_gap_too_small");
  if(!(Number.isFinite(bestEV)&&bestEV>=policy.min_ev_threshold))
   reasons.push("ev_below_dynamic_threshold");
  if(!(Number.isFinite(confidence)&&confidence>=policy.min_confidence))
   reasons.push("confidence_below_dynamic_gate");
  if(!(Number.isFinite(se)&&se<=cfg.MAX_BET_EV_SE))
   reasons.push("ev_uncertainty_too_high");
  if(!(Number.isFinite(lower)&&lower>policy.min_ev_threshold))
   reasons.push("ev_lower_bound_too_low");
  if(remainingCards<cfg.MIN_REMAINING_CARDS_FOR_BET &&
     !(bestEV>=cfg.END_SHOE_OVERRIDE_MIN_EV &&
       confidence>=cfg.END_SHOE_OVERRIDE_CONFIDENCE &&
       Number.isFinite(se)&&se<=cfg.END_SHOE_OVERRIDE_MAX_EV_SE))
   reasons.push("shoe_end_no_bet");
  return Object.freeze({
   eligible:reasons.length===0,reason:reasons[0]||"positive_ev_with_validated_model",
   reasons,min_ev_threshold:policy.min_ev_threshold,
   min_confidence:policy.min_confidence,
   estimated_remaining_ratio:policy.remaining_ratio,
   estimated_remaining_cards:remainingCards,
   ev_gap:evGap,ess_ratio:ess/cfg.PARTICLE_COUNT,
   max_ev_se:cfg.MAX_BET_EV_SE,
   model_verified:model==="validated_external_xgb"
  });
 }
 // Road operations are disconnected from the posterior, EV and wager gate.
 // This module never feeds ranks, outcomes or pattern estimates to particles.
 function fuseRoadResearch(research,futurePath,road,cfg,candidate){
  if(!cfg.ROAD_AUX_ENABLED)
   return {research_view:research,future_path:futurePath};
  const snr=Math.max(0,Number(research?.ev_gap_snr)||0);
  const base=Number.isFinite(futurePath?.future_edge_score)?
   clamp(futurePath.future_edge_score,0,1):clamp(snr/(1+snr),0,1);
  let edge=base,confirmed=false,nearConfirmed=false,conflict=false;
  if(road?.available&&road.estimated_only===true&&
     road.validated_betting_edge===false){
   const agreementOk=road.road_agreement>=cfg.ROAD_AUX_AGREEMENT_MIN;
   const confidenceOk=road.road_confidence>=cfg.ROAD_AUX_CONFIDENCE_MIN;
   const eligible=research?.eligible===true;
   // If explicitly relaxed, allow a *near-eligible* diagnostic score boost
   // only with adequate ESS, diversity and SE. Never promote eligibility.
   const nearEligible=!eligible&&!cfg.ROAD_AUX_REQUIRE_SOFT_ELIGIBLE&&
    research?.candidate===candidate&&
    research.abs_ev_gap>=.9*cfg.MIN_DIRECTION_EV_GAP&&
    research.ev_gap_snr>=.9*cfg.MIN_DIRECTION_GAP_SNR&&
    research.ess_ratio>=cfg.MIN_DIRECTION_ESS_RATIO&&
    research.ancestor_diversity>=cfg.MIN_DIRECTION_DIVERSITY&&
    research.ev_gap_se>0&&research.ev_gap_se<=cfg.MAX_DIRECTION_GAP_SE;
   if((eligible||nearEligible)&&agreementOk&&confidenceOk&&
      road.road_side===candidate){
    edge=clamp(edge+Math.min(cfg.ROAD_AUX_MAX_BOOST,.10)*
     road.road_agreement,0,1);
    confirmed=eligible;
    nearConfirmed=nearEligible;
   }else if(agreementOk&&road.road_side!=="NEUTRAL"&&
             road.road_side!==candidate){
    edge=clamp(edge-cfg.ROAD_AUX_CONFLICT_PENALTY,0,1);
    conflict=true;
   }
  }
  const label=confirmed?"雙訊號同向":nearConfirmed?
   "接近門檻但仍觀望（牌路同向）":conflict?
   "物理與牌路分歧（觀望偏安全）":null;
  const adjustedPath={...futurePath,
   ...(Number.isFinite(futurePath?.future_edge_score)?
    {future_edge_score:edge,
     unadjusted_future_edge_score:futurePath.future_edge_score}:{}),
   research_aux_edge_score:edge,road_aux_adjustment:edge-base,
   estimated_only:true,validated_betting_edge:false};
  return {
   research_view:{...research,road_confirmed:confirmed,
    road_near_confirmed:nearConfirmed,road_conflict:conflict,
    road_confidence_label:label,research_aux_edge_score:edge,
    research_aux_base_edge_score:base,estimated_only:true,
    validated_betting_edge:false},
   future_path:adjustedPath
  };
 }
 function derivedRoadAux(history,cfg){
  if(!cfg.ROAD_AUX_ENABLED)return {
   available:false,reason:"disabled",big_road_columns:0,
   big_road_depth:history.filter(x=>x!=="T").length,
   probes:{big_eye:"N",small:"N",cockroach:"N"},
   road_side:"NEUTRAL",road_agreement:0,road_confidence:0,
   estimated_only:true,validated_betting_edge:false};
  const dep=(typeof module==="object"&&module.exports)?
   require("./derived_road_aux.js"):
   (typeof self!=="undefined"?self:typeof window!=="undefined"?window:null)?.BBBDerivedRoadAux;
  if(!dep||typeof dep.analyze!=="function")
   throw Error("road_aux_dependency_missing");
  return dep.analyze(history,{minHistory:cfg.ROAD_AUX_MIN_HISTORY});
 }
 function run(history,opts={},externalModel=null,sequenceRuntime=null,sequenceModel=null){
  if(!Array.isArray(history)||history.some(x=>!isResult(x)))throw Error("invalid_bpt_history");
  const cfg=config(opts),seed=seedFor(history),post=posterior(history,cfg,seed);
  const road_aux=derivedRoadAux(history,cfg);
  const skip=(reason,extra={})=>({
   version:VERSION,observations:history.length,action:"Skip",direction:"SKIP",
   reason,bankroll_fraction:0,confidence:0,validated_edge:false,observed_actual_cards:false,
   adaptive_sampling:{version:"v5_1",enabled:cfg.ADAPTIVE_MULTI_HORIZON_ENABLED,
    attempted:false,accepted:false,refined:false,baseline_draws:cfg.NEXT_HAND_DRAWS,
    extra_draws:0,extra_next_hand_draws:0,extra_horizon_draws:0,rollback_reason:null},
   future_path:{available:false,reason:"forecast_unavailable",
    estimated_only:true,validated_betting_edge:false},
   config:cfg,seed,road_aux,...extra
  });
  if(!post.available)return skip(post.reason);
  let physical=handProbabilities(post.particles,cfg,seed,post.weights);
  if(!physical)return skip("physical_simulation_unavailable");
  const preliminaryEV=evs(physical.p);
  const preliminary=moderateDirectionGate({
   candidate:preliminaryEV.B>=preliminaryEV.P?"B":"P",
   ev:preliminaryEV,physical,observations:history.length,
   essRatio:post.ess/cfg.PARTICLE_COUNT,diversity:post.diversity,config:cfg
  });
  let posteriorForPath=post;
  const nextExtra=cfg.ADAPTIVE_EXTRA_DRAWS>=2?
   Math.min(cfg.ADAPTIVE_EXTRA_DRAWS-1,
    Math.max(1,Math.round(cfg.ADAPTIVE_EXTRA_DRAWS*cfg.ADAPTIVE_NEXT_HAND_SHARE))):
   cfg.ADAPTIVE_EXTRA_DRAWS;
  const horizonExtra=cfg.ADAPTIVE_EXTRA_DRAWS-nextExtra;
  const baselinePathDraws=Math.max(0,cfg.ADAPTIVE_EXTRA_DRAWS-
   Math.round(cfg.ADAPTIVE_EXTRA_DRAWS*cfg.ADAPTIVE_NEXT_HAND_SHARE));
  let pathDrawsPerParticle=cfg.ADAPTIVE_MULTI_HORIZON_ENABLED?
   baselinePathDraws:0;
  const adaptiveSampling={version:"v5_1",
   enabled:cfg.ADAPTIVE_MULTI_HORIZON_ENABLED,attempted:false,accepted:false,
   refined:false,baseline_draws:cfg.NEXT_HAND_DRAWS,extra_draws:0,
   extra_next_hand_draws:0,extra_horizon_draws:0,rollback_reason:null};
  if(cfg.ADAPTIVE_MULTI_HORIZON_ENABLED&&shouldRefine(preliminary,history.length,
      post.ess/cfg.PARTICLE_COUNT,post.diversity,cfg)){
   adaptiveSampling.attempted=true;
   const refined=conditionalPosteriorResample(post,seed^0x27d4eb2f,cfg);
   if(!refined){
    adaptiveSampling.rollback_reason="posterior_resample_unavailable";
   }else{
    const before={ess_ratio:post.ess/cfg.PARTICLE_COUNT,
     diversity:post.diversity};
    const after={ess_ratio:refined.ess_ratio,diversity:refined.diversity};
    adaptiveSampling.before=before;adaptiveSampling.after=after;
    if(shouldRollbackAdaptive(before,after,cfg)){
     adaptiveSampling.rollback_reason="ess_or_diversity_drop_over_8pct";
    }else{
     const refinedPhysical=handProbabilities(refined.particles,cfg,
      seed^0x13579bdf,refined.weights,cfg.NEXT_HAND_DRAWS+nextExtra);
     if(!refinedPhysical){
      adaptiveSampling.rollback_reason="refined_physical_simulation_unavailable";
     }else{
      physical=refinedPhysical;
      posteriorForPath={...post,particles:refined.particles,
       weights:refined.weights,ess:refined.ess,diversity:refined.diversity};
      pathDrawsPerParticle=horizonExtra;
      adaptiveSampling.accepted=true;adaptiveSampling.refined=true;
      adaptiveSampling.extra_draws=cfg.ADAPTIVE_EXTRA_DRAWS;
      adaptiveSampling.extra_next_hand_draws=nextExtra;
      adaptiveSampling.extra_horizon_draws=horizonExtra;
     }
    }
   }
  }
  const effects=eor(post.particles,cfg,seed,post.weights),
   frozen=frozenCore(history,sequenceRuntime,sequenceModel);
  const vector=features48(post,physical,history,effects,frozen);
  // Weak sequence can influence physical probability only after a verified
  // frozen model exists. The default is zero, so live results remain purely
  // physical until independent real-shoe validation.
  const sequenceWeightApplied=frozen.validated?cfg.SEQUENCE_WEIGHT:0;
  const weighted=Object.fromEntries(["B","P","T"].map(side=>
   [side,(1-sequenceWeightApplied)*physical.p[side]+
      sequenceWeightApplied*frozen.probabilities[side]]));
  const applied=modelDecision(weighted,vector,frozen,externalModel);
  const p=applied.p,ev=evs(p);
  const roiSe={};
  // Multinomial ROI variance and latent posterior SD are both relevant.
  for(const side of ["B","P"]){
   const win=side==="B"?p.B:p.P,loss=side==="B"?p.P:p.B,odds=side==="B"?.95:1;
   const secondMoment=odds*odds*win+loss;
   const mc=Math.sqrt(Math.max(0,secondMoment-ev[side]*ev[side])/physical.samples);
   const latent=Math.sqrt((odds*physical.sd[side])**2+
    physical.sd[side==="B"?"P":"B"]**2);
   roiSe[side]=Math.hypot(mc,latent);
  }
  const candidate=ev.B>=ev.P?"B":"P";
  const bestEV=ev[candidate],se=roiSe[candidate];
  const confidence=clamp(.5+.5*Math.tanh(bestEV/(Math.max(se,1e-9)*Math.sqrt(2))),0,1);
  const lower=bestEV-cfg.LOWER_BOUND_Z*se;
  const enoughDepth=history.length<=cfg.MAX_HANDS&&post.particles.every(x=>sum(x)>=6);
  const quality=clamp((post.ess/cfg.PARTICLE_COUNT)*post.diversity,0,1);
  const research_view=moderateDirectionGate({
   candidate,ev,physical,observations:history.length,
   essRatio:post.ess/cfg.PARTICLE_COUNT,diversity:post.diversity,config:cfg
  });
  const remainingCards=sum(post.particles.map(x=>sum(x)))/post.particles.length;
  const wager_gate=wagerGate({
   cfg,remainingCards,ess:post.ess,quality,ev,candidate,confidence,
   se,lower,enoughDepth,observations:history.length,model:applied.model
  });
  const qualifies=wager_gate.eligible;
  const futurePath=cfg.ADAPTIVE_MULTI_HORIZON_ENABLED?
   simulateFuturePaths(posteriorForPath.particles,posteriorForPath.weights,
    candidate,cfg,seed^0x4cf5ad43,pathDrawsPerParticle,ev[candidate],
    research_view.ev_gap_snr):
   {available:false,reason:"feature_disabled",estimated_only:true,
    validated_betting_edge:false};
  const fusion=fuseRoadResearch(research_view,futurePath,road_aux,cfg,candidate);
  const researchConfidenceTag=fusion.research_view.road_confidence_label||
   (futurePath.available&&
    futurePath.future_edge_score>=.72&&research_view.eligible===true?
    "高信心短中期優勢":null);
  // Until matched, independent holdout-calibrated XGB is supplied, EV is
  // research-only and the safe live decision is always SKIP.
  const win=p[candidate],loss=p[candidate==="B"?"P":"B"],odds=candidate==="B"?.95:1;
  const fullKelly=Math.max(0,(odds*win-loss)/(odds*(win+loss)||1));
  const fraction=qualifies?Math.min(cfg.MAX_BET_FRACTION,cfg.FRACTIONAL_KELLY*fullKelly):0;
  const reason=wager_gate.reason;
  return{
   version:VERSION,observations:history.length,action:qualifies?(candidate==="B"?"Bet Banker":"Bet Player"):"Skip",
   direction:qualifies?candidate:"SKIP",candidate,reason,probabilities:p,
   physical_probabilities:physical.p,ev_banker:ev.B,ev_player:ev.P,ev_tie:ev.T,
   se_banker:roiSe.B,se_player:roiSe.P,conservative_lower_ev:lower,
   confidence,quality,particle_ess:post.ess,ancestor_diversity:post.diversity,
   estimated_remaining_cards:remainingCards,estimated_penetration:
    1-remainingCards/TOTAL,
   eor:effects,features48:vector,feature_dimension:48,
   feature_schema:"soft_shoe_bpt_48d_v2",
   frozen_core:frozen,sequence_weight_applied:sequenceWeightApplied,
   model:applied.model,simulation_samples:physical.samples,bankroll_fraction:fraction,
   adaptive_sampling:adaptiveSampling,
   future_path:{...fusion.future_path,research_label:researchConfidenceTag},
   research_view:fusion.research_view,road_aux,wager_gate,
   validated_edge:qualifies,observed_actual_cards:false,calibrated_confidence:false,
   seed,config:cfg
  };
 }
 return Object.freeze({VERSION,DEFAULT,config,seedFor,deal,posterior,frozenCore,
  handProbabilities,poolPhysical,shouldRefine,conditionalPosteriorResample,
  shouldRollbackAdaptive,directionConsistency,futureEdgeScore,simulateFuturePaths,
  evs,eor,features48,evaluateXgbTree,predictXgbJson,modelDecision,
  moderateDirectionGate,wagerPolicy,wagerGate,fuseRoadResearch,derivedRoadAux,run});
});
