/** 多批數值檢查器；原PF/EOR/Deplete單批計算維持不變，不增加獨立觀測。 */
(function(root,factory){const node=typeof module==='object'&&module.exports;
 const api=factory(node?require('./pf_eor_v4_runtime.js'):root.BBBPhysicalV4,
 node?require('./physics_forecast_first_runtime.js'):root.BBBPhysicsForecastFirst,
 node?require('./stability_reference.js'):root.BBBExactPointReference);
 if(node)module.exports=api;else root.BBBStability=api;
})(typeof self!=='undefined'?self:this,function(base,physics,reference){
 'use strict';
 const VERSION='BBB_V41_ADAPTIVE_BATCH_STABILITY_V1';

 // Explicit user policy: SE is uncertainty, never a win probability.
 const USER_ENTRY=Object.freeze({minEstimatorSE:.0055,minAbsoluteEVGap:0,minBatchAgreement:.55});
 function userEntryDecision(report){
  const se=report?.quality?.estimator_gap_se;
  const b=report?.ev_banker,p=report?.ev_player;
  const rb=report?.original_pf_ev?.B,rp=report?.original_pf_ev?.P;
  const valid=[se,b,p,rb,rp].every(Number.isFinite)&&se>=0&&
   report?.physical_ev?.simulation_exhausted!==true;
  const gap=valid?Math.abs(b-p):null,rawGap=valid?Math.abs(rb-rp):null;
  const candidate=valid&&b!==p?(b>p?'B':'P'):null;
  const reasons=[];
  if(!valid)reasons.push('user_diagnostics_unavailable');
  else{
   if(!(se>USER_ENTRY.minEstimatorSE))reasons.push('user_se_not_above_055_percent');
   if(!candidate||rb===rp||(b>p)!==(rb>rp))reasons.push('user_ev_direction_conflict');
  }
  const agreement=report?.sampling_stability?.agreement;
  if(!Number.isFinite(agreement)||agreement<USER_ENTRY.minBatchAgreement||agreement>1)
   reasons.push('user_batch_agreement_below_55_percent');
  return {direction:reasons.length?'SKIP':candidate,candidate,
   reason:reasons[0]||'user_entry_pass',reasons,
   estimator_gap_se:se,user_smoothed_ev_gap:gap,user_raw_ev_gap:rawGap,
   user_batch_agreement:agreement,
   user_entry_policy:USER_ENTRY,validated_hit_rate:false,verified_edge:false};
 }

 const SETTINGS=Object.freeze({minBatches:5,maxBatches:5,minAgreement:.55,nearThresholdFraction:.5,
  referenceZ:2.5,referenceFloor:.005});
 if(reference?.rows?.length!==100)throw Error('exact_reference_missing');
 let evidence=0;reference.rows.forEach((r,i)=>{if(r[0]*10+r[1]!==i||!r.every(Number.isFinite)||r[2]<=0||
  r.slice(3).some(v=>v<0||v>1)||Math.abs(r[3]+r[4]+r[5]-1)>1e-10)throw Error('invalid_exact_reference');evidence+=r[2];});
 if(Math.abs(evidence-1)>1e-10)throw Error('invalid_reference_evidence');
 const mean=xs=>xs.reduce((a,b)=>a+b,0)/xs.length;
 const ev=q=>({B:.95*q.B-q.P,P:q.P-q.B});
 function fresh(){if(!globalThis.crypto?.getRandomValues)throw Error('secure_random_unavailable');return globalThis.crypto.getRandomValues(new Uint32Array(1))[0];}
 function seedAt(seed,index){if(index===0)return seed;let x=(seed+Math.imul(index,0x9e3779b9))>>>0;
  x=Math.imul(x^(x>>>16),0x85ebca6b);x=Math.imul(x^(x>>>13),0xc2b2ae35);return(x^(x>>>16))>>>0;}
 function aggregate(batches,player,banker,config){
  if(!Array.isArray(batches)||batches.length<3||batches.length>5||batches.some(r=>
   !['B','P'].includes(r?.v4?.candidate)||!r.raw_simulated_probability||
   ['B','P','T'].some(s=>!Number.isFinite(r.raw_simulated_probability[s])||r.raw_simulated_probability[s]<0)||
   Math.abs(Object.values(r.raw_simulated_probability).reduce((s,x)=>s+x,0)-1)>1e-10||
   ![r.raw_quality?.mc_gap_se,r.raw_quality?.posterior_mean_se_gap,r.raw_quality?.quality].every(Number.isFinite)))
   throw Error('invalid_batch_diagnostics');
  const k=batches.length,first=batches[0],q=Object.fromEntries(['B','P','T'].map(s=>[s,mean(batches.map(r=>r.raw_simulated_probability[s]))]));
  const rawEV=ev(q),gaps=batches.map(r=>r.original_pf_ev.B-r.original_pf_ev.P-base.BASE_GAP),g=rawEV.B-rawEV.P-base.BASE_GAP;
  const mcVar=batches.reduce((s,r)=>s+r.raw_quality.mc_gap_se**2,0)/(k*k);
  const latentVar=batches.reduce((s,r)=>s+r.raw_quality.posterior_mean_se_gap**2,0)/(k*k);
  const measuredVar=k>1?gaps.reduce((s,x)=>s+(x-g)**2,0)/(k*(k-1)):0;
  // 批次平均的實測分散已含批次內噪音：取保守最大值，不再次相加。
  const totalVar=Math.max(mcVar+latentVar,measuredVar),posteriorVar=Math.max(0,totalVar-mcVar);
  const totalParticles=k*config.particles;
  const raw={...first.physical_ev,simulated_probability:q,ev_banker:rawEV.B,ev_player:rawEV.P,
   remaining:mean(batches.map(r=>r.physical_ev.remaining)),draws_mean:mean(batches.map(r=>r.physical_ev.draws_mean)),
   ev_gap:rawEV.B-rawEV.P,ev_gap_deviation:g,samples:k*config.particles*config.simulationsPerParticle,
   particle_support:totalParticles,mc_se_gap:Math.sqrt(mcVar),posterior_mean_se_gap:Math.sqrt(posteriorVar),
   pf_estimator_se_gap:Math.sqrt(totalVar),se_gap:Math.sqrt(totalVar),
   particle_ess:batches.reduce((s,r)=>s+r.physical_ev.particle_ess,0),
   posterior_update_ess:batches.reduce((s,r)=>s+r.physical_ev.posterior_update_ess,0),
   posterior_sd_gap:Math.sqrt(mean(batches.map(r=>r.raw_quality.posterior_raw_gap_spread**2))),
   posterior_sd_gap_debiased:Math.sqrt(mean(batches.map(r=>r.raw_quality.posterior_gap_spread**2))),
   draw_length_diagnostics:null,ev_uncertainty_diagnostics:{combined_estimator_se:Math.sqrt(totalVar),
    batch_mean_variance:measuredVar,analytic_mean_variance:mcVar+latentVar,variance_combination:'maximum_not_sum',
    calibrated_confidence_interval:false,estimated_uncertainty_only:true}};
  const descriptor={size:totalParticles,lastESS:raw.posterior_update_ess,
   ancestorIds:Array.from({length:totalParticles},(_,i)=>i),
   lastLikelihoodRelativeSE:Math.max(...batches.map(r=>r.raw_quality.likelihood_relative_se))};
  const smooth=physics.adaptiveScoreSmoother(raw,descriptor),physical=smooth.physical,blend=smooth.diagnostics.blend;
  const rawQuality=base.rawQualityDiagnostics(raw,descriptor);
  // 品質不高於任何批次，避免合併掩蓋低品質批次。
  rawQuality.quality=Math.min(...batches.map(r=>r.raw_quality.quality));
  const quality={...rawQuality,mode:'adaptive_confidence_with_score_smoothing',probability_shrinkage:1-blend,ev_modified:true};
  const eor={...first.eor,effects_by_point:Array.from({length:10},(_,i)=>mean(batches.map(r=>r.eor.effects_by_point[i]))),
   simulated_paired_scenarios:batches.reduce((s,r)=>s+r.eor.simulated_paired_scenarios,0)};
  eor.max_abs_ev_gap_delta=Math.max(...eor.effects_by_point.map(Math.abs));
  eor.weighted_ev_gap_delta=mean(batches.map(r=>r.eor.weighted_ev_gap_delta));
  if(batches.some(r=>r.deplete_raw.status!=='complete'))throw Error('batch_deplete_unavailable');
  const dp=Object.fromEntries(['B','P','T'].map(s=>[s,mean(batches.map(r=>r.deplete_raw.probability[s]))])),de=ev(dp);
  const shadow={...first.deplete_raw,probability:dp,ev_banker:de.B,ev_player:de.P,
   aggregation:'equal_weight_independent_batches',candidate_direction:null,primary_direction:null,
   candidate_ev_banker:null,candidate_ev_player:null,
   conditioning_proposals:batches.reduce((s,r)=>s+r.deplete_raw.conditioning_proposals,0),
   paired_disagreements:batches.reduce((s,r)=>s+r.deplete_raw.paired_disagreements,0),
   samples:batches.reduce((s,r)=>s+r.deplete_raw.samples,0),completed_shoes:batches.reduce((s,r)=>s+r.deplete_raw.completed_shoes,0),
   paired_mc_gap_se:Math.sqrt(batches.reduce((s,r)=>s+r.deplete_raw.paired_mc_gap_se**2,0))/k};
  const comparison={...shadow,comparison_ev_banker:base.BASE_EV.B+blend*(de.B-base.BASE_EV.B),
   comparison_ev_player:base.BASE_EV.P+blend*(de.P-base.BASE_EV.P),comparison_smoothing_blend:blend};
  const report={...first,version:VERSION,physical_ev:physical,raw_simulated_probability:q,raw_quality:rawQuality,
   simulated_probability:physical.simulated_probability,original_pf_ev:rawEV,
   ev_banker:physical.ev_banker,ev_player:physical.ev_player,quality,adaptive_confidence_gate:quality,
   adaptive_score_smoothing:smooth.diagnostics,eor,deplete_raw:shadow,deplete_mc:comparison,
   model_posterior:{...first.model_posterior,sampled_compositions:totalParticles}};
  const decision=new base.QualityFilterV4(config).decide(report),candidate=decision.candidate;
  const votes={B:0,P:0};for(const r of batches)votes[r.v4.candidate]++;
  const agreement=votes[candidate]/k;
  const row=reference.rows[player*10+banker],refProb={B:row[3],P:row[4],T:row[5]},refEV=ev(refProb);
  const refGap=refEV.B-refEV.P-base.BASE_GAP,delta=Math.abs(g-refGap);
  const tolerance=Math.max(SETTINGS.referenceFloor,SETTINGS.referenceZ*quality.estimator_gap_se);
  const refOK=delta<=tolerance,stable=agreement>=SETTINGS.minAgreement;
  const reasons=[];if(decision.direction==='SKIP')reasons.push(decision.reason);
  if(!stable)reasons.push('batch_direction_unstable');if(!refOK)reasons.push('exact_reference_disagreement');
  report.v4={...decision,direction:reasons.length?'SKIP':candidate,reason:reasons[0]||'quality_pass',reasons};
  report.legacy_v4_decision=report.v4;
  report.sampling_stability={batches:k,min_batches:5,max_batches:5,votes,agreement,required_agreement:.55,
   stable,per_batch_samples:config.particles*config.simulationsPerParticle,
   seeds:batches.map(r=>r.randomization.seed),batch_mean_variance:measuredVar,analytic_mean_variance:mcVar+latentVar};
  report.exact_reference={probability:refProb,neutral_ev_gap:refGap,pf_raw_gap:g,deviation:delta,tolerance,
   consistent:refOK,independent_evidence:false,model_assumption:reference.assumption,used_for_direction:false};
  report.v4={...decision,...userEntryDecision(report)};
  return report;
 }
 class StablePhysicalPointV4{
  constructor(config=base.CONFIG){this.config=base.validateConfig(config);this.engine=new base.PhysicalPointV4(this.config);}
  analyze(player,banker,{seed}={}){
   if(!Number.isInteger(player)||player<0||player>9||!Number.isInteger(banker)||banker<0||banker>9)throw Error('invalid_total_points');
   const replay=seed!==undefined,rootSeed=replay?seed:fresh();
   if(!Number.isInteger(rootSeed)||rootSeed<0||rootSeed>0xffffffff)throw Error('invalid_analysis_seed');
   const batches=[];for(let i=0;i<SETTINGS.minBatches;i++)batches.push(this.engine.analyze(player,banker,{seed:seedAt(rootSeed,i)}));
   let r=aggregate(batches,player,banker,this.config);
   r.randomization={...r.randomization,mode:replay?'explicit_seed_replay':'fresh_secure_seed',seed:rootSeed};
   r.sampling_stability.expanded=false;
   r.sampling_stability.expansion_reason={required:'fixed five independent batches'};
   return r;
  }
 }
 return Object.freeze({VERSION,SETTINGS,USER_ENTRY,userEntryDecision,StablePhysicalPointV4,aggregate,seedAt});
});
