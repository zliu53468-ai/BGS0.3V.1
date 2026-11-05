/**
 * BBB V4 正式推論核心：僅輸入當局閒／莊 0–9 最終總點。
 * 方向來自條件式 PF 平滑後 EV 相對基準的偏移；原始值另行保留；
 * Deplete MC 使用相同資料獨立重新抽樣以交叉檢查，不當成額外證據。
 * 不建立 LinUCB、不計算 256D 上下文、不使用牌路特徵修改方向。
 */
(function(root,factory){
 "use strict";
 const quant=root?.BBBQuant256Online||
  (typeof module==="object"&&module.exports?require("./quant256_online_runtime.js"):null);
 const physics=root?.BBBPhysicsForecastFirst||
  (typeof module==="object"&&module.exports?require("./physics_forecast_first_runtime.js"):null);
 const deplete=root?.BBBDepleteMC||
  (typeof module==="object"&&module.exports?require("./deplete_mc_runtime.js"):null);
 const api=factory(quant,physics,deplete);
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.BBBPhysicalV4=api;
})(typeof window!=="undefined"?window:null,function(quant,physics,deplete){
 "use strict";
 if(!quant?.BlindPhysicsEstimator||!physics?.summarizeHypotheses||!physics?.adaptiveScoreSmoother||
   !physics?.BASE||!deplete?.attachShadow||!deplete?.dealWithUniforms)
   throw Error("missing_physical_v4_dependencies");
 const VERSION="BBB_PF_EOR_DEPLETE_V4_1_NEUTRAL_SMOOTHED_RANDOM";
 const CONFIG=Object.freeze({
  particles:96,simulationsPerParticle:256,
  // 使用 24 個分層抽取的假想粒子，每個點數影響各做 96 次配對模擬。
  eorParticleCount:24,eorSamples:96,
  // Floor and sampling-uncertainty multiplier for an adaptive EV-gap gate.
  // 0.65 is a moderate signal-noise heuristic, NOT a statistical z-score or
  // an observed win-rate confidence interval.
  minEvGap:.0040,noiseGuardMultiplier:.65,minPosteriorQuality:.50,
  // EOR can veto a fragile signal when a one-card hypothetical perturbation
  // is as large as the indicated EV-gap; it never chooses the direction.
  eorFragilityFloor:.025,
  crossCheckZ:2.5,crossCheckFloor:.008,crossCheckDirectionFloor:.005
 });
 // 參數集中在 CONFIG；0.80 僅留在對照工具，不由前端環境變數偷偷覆寫。
 function validateConfig(c){
  if(!c||typeof c!=="object")throw Error("invalid_v41_config");
  const integer=(name,lo,hi)=>{if(!Number.isInteger(c[name])||c[name]<lo||c[name]>hi)throw Error("invalid_"+name);};
  integer("particles",8,96);integer("simulationsPerParticle",8,1024);
  integer("eorParticleCount",1,c.particles);integer("eorSamples",16,1024);
  for(const [name,lo,hi] of [["minEvGap",0,.5],["noiseGuardMultiplier",0,5],
   ["minPosteriorQuality",0,1],["eorFragilityFloor",0,1],["crossCheckZ",0,5],["crossCheckFloor",0,1],["crossCheckDirectionFloor",0,.5]]){
   if(!Number.isFinite(c[name])||c[name]<lo||c[name]>hi)throw Error("invalid_"+name);
  }
  return Object.freeze({...c});
 }
 function valid(x){return Number.isInteger(x)&&x>=0&&x<=9;}
 function outcome(p,b){return b>p?"B":p>b?"P":"T";}
 const abs=Math.abs,finite=Number.isFinite;
 const clamp=(x,a,b)=>Math.min(b,Math.max(a,x));
 function seeded(seed){
  let s=seed>>>0||0x9e3779b9;
  return()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return(s>>>0)/4294967296;};
 }
 const gapOf=prob=>1.95*prob.B-2*prob.P;
 // 「50/50 中立」是移除基準本身的 EV 不對稱，不修改真實機率或賠率。
 // 兩邊各自相對基準的 EV 改變相等時，方向偏移為零；門檻仍會觀望。
 const BASE_EV=Object.freeze({B:.95*physics.BASE.B-physics.BASE.P,
  P:physics.BASE.P-physics.BASE.B});
 const BASE_GAP=BASE_EV.B-BASE_EV.P;
 const gapScore=x=>x==="B"?1.95:x==="P"?-2:0;

 function computeEOR(estimator,pair,{particles=CONFIG.eorParticleCount,
  samples=CONFIG.eorSamples,seedSalt=0}={}){
  const rows=estimator.particles;
  if(!Array.isArray(rows)||rows.length<1||
   !Number.isInteger(particles)||particles<1||particles>rows.length||
   !Number.isInteger(samples)||samples<16||samples>1024)
    throw Error("invalid_eor_samples");
  const delta=Array(10).fill(0),supported=Array(10).fill(0),pointMass=Array(10).fill(0);
  let scenarios=0,validSampledParticles=0;
  // 分層選取後驗粒子；每次在同一假想組成中扣掉一張 0–9 點數牌，
  // 與未再扣牌版本使用相同亂數，降低 EOR 差分的蒙地卡羅雜訊。
  for(let i=0;i<particles;i++){
   const idx=Math.min(rows.length-1,Math.floor((i+.5)*rows.length/particles));
   const counts=rows[idx],total=counts.reduce((a,b)=>a+b,0);
   if(total<7)continue;
   validSampledParticles++;
   for(let rank=0;rank<10;rank++)pointMass[rank]+=counts[rank]/total;
   const rng=seeded(0x45d3e77^seedSalt^Math.imul(pair.player+1,0x9e3779b1)^
    Math.imul(pair.banker+1,0x85ebca6b)^Math.imul(idx+1,0x27d4eb2f));
   // 同一個假想剩餘組成計算 10 種點數移除的配對差值；
   // 這些是敏感度導數的近似，不是額外觀察到的真實牌。
   for(let j=0;j<samples;j++){
    const u=Array.from({length:6},()=>rng());
    const control=gapScore(deplete.dealWithUniforms(counts,u));
    for(let rank=0;rank<10;rank++){
     if(counts[rank]<1)continue;
     const removed=counts.slice();removed[rank]--;
     delta[rank]+=gapScore(deplete.dealWithUniforms(removed,u))-control;
     supported[rank]++;scenarios++;
    }
   }
  }
  const effects=delta.map((v,i)=>supported[i]?v/supported[i]:null);
  if(effects.some(v=>v===null))throw Error("eor_insufficient_support");
  const maxAbs=Math.max(...effects.map(abs));
  const weightedMean=effects.reduce((v,e,i)=>v+e*pointMass[i],0)/
    Math.max(1,validSampledParticles);
  return{effects_by_point:effects,max_abs_ev_gap_delta:maxAbs,
   weighted_ev_gap_delta:weightedMean,simulated_paired_scenarios:scenarios,
   hypothetically_removed_cards:1,diagnostic_only:true,
   actual_remaining_counts_known:false,independent_evidence:false};
 }

function rawQualityDiagnostics(raw,pf){
  const n=Math.max(1,pf.size||raw.particle_support);
  const ess=clamp((raw.posterior_update_ess??n)/n,0,1);
  const activeESS=clamp((raw.particle_ess??n)/n,0,1);
  const ancestorDiversity=clamp(Array.isArray(pf.ancestorIds)?
   new Set(pf.ancestorIds).size/n:1,0,1);
  const likelihoodNoise=clamp(pf.lastLikelihoodRelativeSE??0,0,1);
  const quality=clamp((.5*ess+.25*activeESS+.25*ancestorDiversity)/
   (1+likelihoodNoise),0,1);
  // Use the debiased posterior MEAN standard error, never the raw spread
  // across 160-draw per-particle MC estimates as an estimator SE.
  return{
   mode:"adaptive_confidence_only",probability_shrinkage:0,ev_modified:false,
   quality,ess_ratio:ess,active_ess_ratio:activeESS,
   ancestor_diversity:ancestorDiversity,likelihood_relative_se:likelihoodNoise,
   mc_gap_se:raw.mc_se_gap,
   posterior_gap_spread:raw.posterior_sd_gap_debiased,
   posterior_raw_gap_spread:raw.posterior_sd_gap,
   posterior_mean_se_gap:raw.posterior_mean_se_gap,
   estimator_gap_se:raw.pf_estimator_se_gap,
   history_progress_used:false,validated_hit_rate:false
  };
 }

 class QualityFilterV4{
  constructor(config=CONFIG){this.config=validateConfig(config);}
  decide(report){
   const c=this.config,d=report?.quality,eor=report?.eor,shadow=report?.deplete_mc;
   const evB=report?.ev_banker,evP=report?.ev_player;
   // Compare changes versus the baseline; retain raw commission-adjusted EV separately.
   const rawSigned=finite(evB)&&finite(evP)?evB-evP:NaN;
   const signed=rawSigned-BASE_GAP;
   const candidate=finite(signed)?(signed>=0?"B":"P"):null;
   const mandatory=[
    evB,evP,d?.quality,d?.ess_ratio,d?.ancestor_diversity,
    d?.mc_gap_se,d?.posterior_gap_spread,d?.posterior_mean_se_gap,
    d?.estimator_gap_se,eor?.max_abs_ev_gap_delta,
    shadow?.ev_banker,shadow?.ev_player,shadow?.paired_mc_gap_se
   ];
   if(!candidate||mandatory.some(v=>!finite(v))||
    report?.physical_ev?.simulation_exhausted||
    shadow?.status!=="complete"||
    !report?.point_observation_independent)
    return{direction:"SKIP",candidate,reason:"diagnostics_unavailable",
     validated_hit_rate:false,verified_edge:false};
   const quality=clamp(d.quality,0,1);
   const mcSe=Math.max(0,d.mc_gap_se),posteriorMeanSe=Math.max(0,d.posterior_mean_se_gap);
   const estimatedSe=Math.hypot(mcSe,posteriorMeanSe);
   // Fail closed if the supplied total does not match independently combined
   // MC + latent-posterior estimator uncertainty.
   if(Math.abs(estimatedSe-d.estimator_gap_se)>1e-8||
    !finite(estimatedSe)||estimatedSe>.5)
    return{direction:"SKIP",candidate,reason:"diagnostics_unavailable",
     validated_hit_rate:false,verified_edge:false};
   const threshold=Math.max(c.minEvGap,c.noiseGuardMultiplier*estimatedSe);
   const gap=abs(signed),sensitivity=Math.max(0,eor.max_abs_ev_gap_delta);
   const fragile=sensitivity>Math.max(c.eorFragilityFloor,gap);
   const mcSigned=(shadow.comparison_ev_banker??shadow.ev_banker)-(shadow.comparison_ev_player??shadow.ev_player)-BASE_GAP;
   const tolerance=Math.max(c.crossCheckFloor,c.crossCheckZ*
    Math.hypot(estimatedSe,Math.max(0,shadow.paired_mc_gap_se)));
   const deviation=abs(signed-mcSigned);
   const stable=deviation<=tolerance;
   const opposite= signed*mcSigned<0 &&
    abs(mcSigned)>Math.max(c.crossCheckDirectionFloor,c.noiseGuardMultiplier*shadow.paired_mc_gap_se);
   const goodQuality=quality>=c.minPosteriorQuality;
   const pass=goodQuality&&gap>threshold&&!fragile&&stable&&!opposite;
   const reason=!goodQuality?"poor_posterior_quality":
    gap<=threshold?"weak_ev_gap":fragile?"eor_fragile":
    !stable?"deplete_disagreement":
    opposite?"opposite_deplete_direction":"quality_pass";
   return{direction:pass?candidate:"SKIP",candidate,reason,
    gap,threshold,quality,ev_difference:rawSigned,decision_ev_gap:signed,baseline_ev_gap:BASE_GAP,
    ev_banker:evB,ev_player:evP,decision_metric:"baseline_centered_ev_gap",
    chosen_ev:candidate==="B"?evB:evP,
    mc_gap_se:mcSe,posterior_mean_se:posteriorMeanSe,
    estimator_gap_se:estimatedSe,noise_guard_multiplier:c.noiseGuardMultiplier,
    eor_sensitivity:sensitivity,eor_fragile:fragile,
    deplete_gap_deviation:deviation,deplete_agreement_limit:tolerance,
    deplete_consistent:stable&&!opposite,
    validated_hit_rate:false,verified_edge:false};
  }
 }


 // 僅改變抽樣流，不隨機選方向；沒有熵來源就失敗關閉。
 function checkedSeed(value){
  if(!Number.isInteger(value)||value<0||value>0xffffffff)throw Error("invalid_analysis_seed");
  return value>>>0;
 }
 function freshSeed(){
  const cryptoAPI=globalThis.crypto;
  if(!cryptoAPI?.getRandomValues)throw Error("secure_random_unavailable");
  return cryptoAPI.getRandomValues(new Uint32Array(1))[0];
 }
 class PhysicalPointV4{
  constructor(config=CONFIG){this.config=validateConfig(config);this.filter=new QualityFilterV4(this.config);}
  analyze(player,banker,{seed}={}){
   if(!valid(player)||!valid(banker))throw Error("invalid_total_points");
   const replay=seed!==undefined;
   const baseSeed=replay?checkedSeed(seed):freshSeed();
   // 不同salt分離PF、EOR、Deplete的偽亂數流；它們仍共享同一觀察。
   const eorSeed=Math.imul(baseSeed,0x85ebca6b)>>>0;
   const depleteSeed=Math.imul(baseSeed,0xc2b2ae35)>>>0;
   const pair={player,banker},result=outcome(player,banker);
   // 每次重新建立與八副牌先驗一致的粒子後驗；上一局不寫入本次組成。
   const pf=new quant.BlindPhysicsEstimator({particles:this.config.particles,
    exactSingleScore:true,seed:(0xBBB256^baseSeed)>>>0});
   pf.observe(result,pair);
   const raw=physics.summarizeHypotheses(pf,[result],
    {samplesPerParticle:this.config.simulationsPerParticle});
   if(raw.simulation_exhausted||raw.particle_support!==this.config.particles)
    throw Error("pf_posterior_unavailable");
   // 恢復既有自適應機率平滑。原始報告另外保留，不重複扣佣金。
   const smoothed=physics.adaptiveScoreSmoother(raw,pf);
   const physical=smoothed.physical,rawQuality=rawQualityDiagnostics(raw,pf);
   // 不把資料相依平滑誤當新增證據：保留原始估計誤差，採保守品質門檻。
   const quality={...rawQuality,mode:"adaptive_confidence_with_score_smoothing",
    probability_shrinkage:1-smoothed.diagnostics.blend,ev_modified:true};
   const core={version:VERSION,point_observation_independent:true,
    history_length:1,point_score_observation_count:1,
    observed_card_evidence:false,physical_ev:physical,
    ev_banker:physical.ev_banker,ev_player:physical.ev_player,
    quality,adaptive_confidence_gate:quality,
    effective_research_gate:null,research_signal:{direction:null,available:false}};
   const eor=computeEOR(pf,pair,{particles:this.config.eorParticleCount,
    samples:this.config.eorSamples,seedSalt:eorSeed});
   const checked=deplete.attachShadow({...core,physical_ev:raw,
    ev_banker:raw.ev_banker,ev_player:raw.ev_player,quality:rawQuality},pair,{seedSalt:depleteSeed});
   const shadow=checked.deplete_mc,blend=smoothed.diagnostics.blend;
   // 以相同平滑係數與同一基準核對，原始Deplete資料不覆寫。
   const comparison=shadow.status==="complete"?{...shadow,
    comparison_ev_banker:BASE_EV.B+blend*(shadow.ev_banker-BASE_EV.B),
    comparison_ev_player:BASE_EV.P+blend*(shadow.ev_player-BASE_EV.P),
    comparison_smoothing_blend:blend}:shadow;
   const report={...core,eor,deplete_mc:comparison,deplete_raw:shadow,raw_quality:rawQuality,
    raw_simulated_probability:raw.simulated_probability,adaptive_score_smoothing:smoothed.diagnostics,
    randomization:{mode:replay?"explicit_seed_replay":"fresh_secure_seed",
     seed:baseSeed,pf_seed:(0xBBB256^baseSeed)>>>0,eor_salt:eorSeed,deplete_salt:depleteSeed},
    model_posterior:{sampled_compositions:pf.size,model_based:true,
     actual_card_ranks_known:false},
    simulated_probability:physical.simulated_probability,
    baseline_probability:physics.BASE,
    original_pf_ev:{B:raw.ev_banker,P:raw.ev_player},
    markov_used_for_direction:false,linucb_used_for_direction:false,
    context256_used_for_direction:false,road_used_for_direction:false,
    deplete_blended_into_primary:false};
   return{...report,v4:this.filter.decide(report)};
  }
 }
 return Object.freeze({VERSION,CONFIG,BASE_EV,BASE_GAP,validateConfig,computeEOR,rawQualityDiagnostics,
  QualityFilterV4,PhysicalPointV4});
});
