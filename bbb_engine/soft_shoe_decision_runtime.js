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
 const VERSION="BBB_SOFT_SHOE_BPT_EV_V2";
 const DEFAULT=Object.freeze({
  PARTICLE_COUNT:128,LIKELIHOOD_DRAWS:14,NEXT_HAND_DRAWS:14,
  MIN_EV_THRESHOLD:.0015,MIN_CONFIDENCE:.65,FRACTIONAL_KELLY:.30,
  MAX_BET_FRACTION:.025,AVERAGE_CARDS_PER_HAND:4.95,DEPTH_RELIABILITY:.5,
  RESAMPLE_ESS_RATIO:.5,MAX_HANDS:72,LOWER_BOUND_Z:1.645,SEQUENCE_WEIGHT:0,
  MIN_DIRECTION_ROUNDS:3,MIN_DIRECTION_EV_GAP:.020,
  MIN_DIRECTION_GAP_SNR:.70,MIN_DIRECTION_ESS_RATIO:.35,
  MIN_DIRECTION_DIVERSITY:.20,MAX_DIRECTION_GAP_SE:.080
 });
 const BASE=[128,32,32,32,32,32,32,32,32,32],TOTAL=416;
 const BASE_PROB={B:.458597,P:.446247,T:.095156};
 const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
 const sum=a=>a.reduce((x,y)=>x+y,0);
 const isResult=r=>r==="B"||r==="P"||r==="T";
 function config(c={}){
  const v={...DEFAULT,...c};
  for(const k of ["PARTICLE_COUNT","LIKELIHOOD_DRAWS","NEXT_HAND_DRAWS","MAX_HANDS","MIN_DIRECTION_ROUNDS"])
   if(!Number.isInteger(v[k])||v[k]<1||v[k]>(k==="MAX_HANDS"?75:512))throw Error("invalid_"+k);
  if(v.PARTICLE_COUNT<8||v.LIKELIHOOD_DRAWS<2||v.NEXT_HAND_DRAWS<2)throw Error("sample_budget_too_small");
  for(const [k,lo,hi] of [["MIN_EV_THRESHOLD",0,.1],["MIN_CONFIDENCE",0,1],
   ["FRACTIONAL_KELLY",0,1],["MAX_BET_FRACTION",0,.1],
   ["AVERAGE_CARDS_PER_HAND",4,6],["DEPTH_RELIABILITY",0,1],
   ["RESAMPLE_ESS_RATIO",0,1],["LOWER_BOUND_Z",0,4],["SEQUENCE_WEIGHT",0,.05],
   ["MIN_DIRECTION_EV_GAP",0,.2],["MIN_DIRECTION_GAP_SNR",0,4],
   ["MIN_DIRECTION_ESS_RATIO",0,1],["MIN_DIRECTION_DIVERSITY",0,1],
   ["MAX_DIRECTION_GAP_SE",.001,.5]])
   if(!Number.isFinite(v[k])||v[k]<lo||v[k]>hi)throw Error("invalid_"+k);
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
  let particles=Array.from({length:n},()=>BASE.slice()),ess=n,diversity=1;
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
   const w=normalize(evidence);
   if(!w)return{available:false,reason:"particle_collapse",config:cfg,seed};
   ess=1/sum(w.map(x=>x*x));
   const parents=resampleSystematic(w,rng),used=new Map();
   const next=new Array(n);
   for(let k=0;k<n;k++){
    const parent=parents[k],times=used.get(parent)||0;
    used.set(parent,times+1);
    // Repeated ancestors receive new legal conditional latent hands, not
    // arbitrary mutations of counts. A bounded fallback clones the proposal.
    let proposal=proposals[parent];
    if(times>0){
     for(let j=0;j<64;j++){
      const trial=deal(particles[parent],rng);
      if(trial&&trial.result===observed){proposal=trial.counts;break;}
     }
    }
    next[k]=proposal.slice();
   }
   particles=next;diversity=used.size/n;
   depthExpected=target;
   if(particles.every(p=>sum(p)<6))
    return{available:false,reason:"shoe_exhausted",config:cfg,seed};
  }
  return{available:true,particles,ess,diversity,depthExpected,config:cfg,seed,
   observed_actual_cards:false,soft_depth_constraint:true};
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
 function handProbabilities(particles,cfg,seed){
  const rng=rngFrom(seed^0x52ec17e),per=[],counts={B:0,P:0,T:0};
  let samples=0;
  for(const particle of particles){
   const local={B:0,P:0,T:0};
   for(let j=0;j<cfg.NEXT_HAND_DRAWS;j++){
    const x=deal(particle,rng);if(!x)continue;
    counts[x.result]++;local[x.result]++;samples++;
   }
   per.push(local);
  }
  if(!samples)return null;
  const p={B:counts.B/samples,P:counts.P/samples,T:counts.T/samples};
  const sd={};
  for(const side of ["B","P","T"]){
   const av=per.map(x=>x[side]/cfg.NEXT_HAND_DRAWS);
   const mean=sum(av)/av.length;
   const spread=sum(av.map(x=>(x-mean)**2))/Math.max(1,av.length-1);
   const within=p[side]*(1-p[side])/cfg.NEXT_HAND_DRAWS;
   // Between-shoe latent dispersion is NOT divided by N as if observed data.
   sd[side]=Math.sqrt(Math.max(0,spread-within)+within/particles.length);
  }
  return{p,sd,samples,per};
 }
 function evs(p){return{B:.95*p.B-p.P,P:p.P-p.B,T:8*p.T-(1-p.T)};}
 function eor(particles,cfg,seed){
  // Paired common-random-number simulations for 0..9 hypothetical removal;
  // diagnostic-only sensitivity, not another independent observation.
  const effects=Array(10).fill(0),support=Array(10).fill(0);
  const stride=Math.max(1,Math.floor(particles.length/32));
  for(let i=0;i<particles.length;i+=stride){
   const counts=particles[i];
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
  const avg=Array.from({length:10},(_,rank)=>sum(parts.map(p=>p[rank]))/parts.length);
  const stdev=avg.map((v,rank)=>Math.sqrt(sum(parts.map(p=>(p[rank]-v)**2))/parts.length));
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
  const gap=ev.B-ev.P,absGap=Math.abs(gap),n=physical.samples;
  // Gap payoff is +1.95 on Banker, -2 on Player, 0 on Tie.
  const gapSecondMoment=1.95*1.95*physical.p.B+4*physical.p.P;
  const oneHandVariance=Math.max(0,gapSecondMoment-gap*gap);
  const mcVariance=oneHandVariance/Math.max(1,n);
  const each=(physical.per||[]).map(h=>(1.95*h.B-2*h.P)/cfg.NEXT_HAND_DRAWS);
  const m=each.length?sum(each)/each.length:gap;
  const spread=each.length>1?sum(each.map(x=>(x-m)**2))/(each.length-1):0;
  // Monte Carlo noise in a per-particle gap has variance variance/NEXT_HAND_DRAWS.
  // Debias between-particle variation, then scale as an estimator-mean error.
  const latentMeanVariance=Math.max(0,spread-oneHandVariance/cfg.NEXT_HAND_DRAWS)/
   Math.max(1,each.length);
  const gapSe=Math.sqrt(mcVariance+latentMeanVariance);
  const snr=gapSe>0?absGap/gapSe:0;
  const reasons=[];
  if(observations<cfg.MIN_DIRECTION_ROUNDS)reasons.push("insufficient_rounds");
  if(absGap<cfg.MIN_DIRECTION_EV_GAP)reasons.push("weak_ev_separation");
  if(!(gapSe>0&&Number.isFinite(gapSe))||gapSe>cfg.MAX_DIRECTION_GAP_SE)
   reasons.push("high_estimator_uncertainty");
  if(snr<cfg.MIN_DIRECTION_GAP_SNR)reasons.push("low_gap_to_noise");
  if(essRatio<cfg.MIN_DIRECTION_ESS_RATIO)reasons.push("low_particle_ess");
  if(diversity<cfg.MIN_DIRECTION_DIVERSITY)reasons.push("low_particle_diversity");
  return Object.freeze({
   direction:reasons.length?"SKIP":candidate,
   candidate,eligible:reasons.length===0,reason:reasons[0]||"moderate_gate_pass",
   reasons,abs_ev_gap:absGap,ev_gap:gap,ev_gap_se:gapSe,
   ev_gap_snr:snr,ess_ratio:essRatio,ancestor_diversity:diversity,
   estimated_only:true,validated_betting_edge:false
  });
 }
 function run(history,opts={},externalModel=null,sequenceRuntime=null,sequenceModel=null){
  if(!Array.isArray(history)||history.some(x=>!isResult(x)))throw Error("invalid_bpt_history");
  const cfg=config(opts),seed=seedFor(history),post=posterior(history,cfg,seed);
  const skip=(reason,extra={})=>({
   version:VERSION,observations:history.length,action:"Skip",direction:"SKIP",
   reason,bankroll_fraction:0,confidence:0,validated_edge:false,observed_actual_cards:false,
   config:cfg,seed,...extra
  });
  if(!post.available)return skip(post.reason);
  const physical=handProbabilities(post.particles,cfg,seed);
  if(!physical)return skip("physical_simulation_unavailable");
  const effects=eor(post.particles,cfg,seed),
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
  // Calibrated confidence is not available; fail closed on weak ESS/diversity.
  const qualifies=history.length>0&&enoughDepth&&quality>=.65&&
   bestEV>=cfg.MIN_EV_THRESHOLD&&confidence>=cfg.MIN_CONFIDENCE&&lower>cfg.MIN_EV_THRESHOLD &&
   applied.model==="validated_external_xgb";
  // Until matched, independent holdout-calibrated XGB is supplied, EV is
  // research-only and the safe live decision is always SKIP.
  const win=p[candidate],loss=p[candidate==="B"?"P":"B"],odds=candidate==="B"?.95:1;
  const fullKelly=Math.max(0,(odds*win-loss)/(odds*(win+loss)||1));
  const fraction=qualifies?Math.min(cfg.MAX_BET_FRACTION,cfg.FRACTIONAL_KELLY*fullKelly):0;
  const reason=qualifies?"positive_ev_with_validated_model":
   applied.model!=="validated_external_xgb"?"trained_decision_model_unavailable":
   quality<.65?"low_particle_quality":lower<=cfg.MIN_EV_THRESHOLD?"ev_uncertainty_too_high":
   bestEV<cfg.MIN_EV_THRESHOLD?"nonpositive_ev":"confidence_below_gate";
  return{
   version:VERSION,observations:history.length,action:qualifies?(candidate==="B"?"Bet Banker":"Bet Player"):"Skip",
   direction:qualifies?candidate:"SKIP",candidate,reason,probabilities:p,
   physical_probabilities:physical.p,ev_banker:ev.B,ev_player:ev.P,ev_tie:ev.T,
   se_banker:roiSe.B,se_player:roiSe.P,conservative_lower_ev:lower,
   confidence,quality,particle_ess:post.ess,ancestor_diversity:post.diversity,
   estimated_remaining_cards:sum(post.particles[0]),estimated_penetration:
    1-sum(post.particles.map(x=>sum(x)))/post.particles.length/TOTAL,
   eor:effects,features48:vector,feature_dimension:48,
   feature_schema:"soft_shoe_bpt_48d_v2",
   frozen_core:frozen,sequence_weight_applied:sequenceWeightApplied,
   model:applied.model,simulation_samples:physical.samples,bankroll_fraction:fraction,
   research_view,
   validated_edge:qualifies,observed_actual_cards:false,calibrated_confidence:false,
   seed,config:cfg
  };
 }
 return Object.freeze({VERSION,DEFAULT,config,seedFor,deal,posterior,frozenCore,
  handProbabilities,evs,eor,features48,evaluateXgbTree,predictXgbJson,modelDecision,moderateDirectionGate,run});
});
