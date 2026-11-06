/** Independent conditional-depletion Monte Carlo with a paired fresh-shoe control.
 * Shares the one-hand observation and prior with PF: this is NOT new evidence.
 */
(function(root,factory){
 const quant=root?.BBBQuant256Online||
  (typeof module==='object'&&module.exports?require('./quant256_online_runtime.js'):null);
 const api=factory(quant);
 if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.BBBDepleteMC=api;
})(typeof window!=='undefined'?window:null,function(quant){
 'use strict';
 if(!quant?.conditionalHandProposal)throw Error('missing_deplete_conditional_sampler');
 const VERSION='BBB_DEPLETE_PAIRED_MC_V1';
 const BASE=Object.freeze({B:.458597,P:.446247,T:.095156});
 const INITIAL=Object.freeze([128,32,32,32,32,32,32,32,32,32]);
 const CONFIG=Object.freeze({batches:8,shoesPerBatch:32,samplesPerShoe:128,
  proposalBudget:4096,candidateWeight:.20,defaultMode:'candidate'});
 const valid=v=>Number.isInteger(v)&&v>=0&&v<=9;
 const rngFor=seed=>{let s=seed>>>0||0xa341316c;return()=>{
  s^=s<<13;s^=s>>>17;s^=s<<5;return(s>>>0)/4294967296;
 };};
 const index=o=>o==='B'?0:o==='P'?1:2;
 function dealWithUniforms(counts,u){
  const c=counts.slice();let total=c.reduce((a,b)=>a+b,0),at=0;
  const draw=()=>{
   let x=Math.floor(u[at++]*total),rank=0;
   while(rank<9&&x>=c[rank])x-=c[rank++];
   if(!c[rank])throw Error('deplete_invalid_draw');
   c[rank]--;total--;return rank;
  };
  const p1=draw(),b1=draw(),p2=draw(),b2=draw();
  let p=(p1+p2)%10,b=(b1+b2)%10;
  if(p<8&&b<8){
   if(p<=5){const x=draw();p=(p+x)%10;
    if(b<=2||(b===3&&x!==8)||(b===4&&x>=2&&x<=7)||
      (b===5&&x>=4&&x<=7)||(b===6&&(x===6||x===7)))b=(b+draw())%10;
   }else if(b<=5)b=(b+draw())%10;
  }
  return b>p?'B':p>b?'P':'T';
 }
 function estimate(player,banker,{seedSalt=0}={}){
  if(!Number.isInteger(seedSalt)||seedSalt<0||seedSalt>0xffffffff)throw Error("invalid_deplete_seed");
  if(!valid(player)||!valid(banker))throw Error('invalid_point_observation');
  const outcome=banker>player?'B':player>banker?'P':'T';
  // Distinct streams for conditioning and prediction, and a new stream per request.
  const condition=rngFor(0xd3e1e7e1^seedSalt^Math.imul(player+1,0x9e3779b1)^Math.imul(banker+1,0x85ebca6b));
  const forecast=rngFor(0x51ad0b17^Math.imul(seedSalt,0x9e3779b1)^Math.imul(player+1,0xc2b2ae35)^Math.imul(banker+1,0x27d4eb2f));
  const batchDeltas=[];let proposals=0,pairedDisagreements=0;
  for(let batch=0;batch<CONFIG.batches;batch++){
   const delta=[0,0,0];
   for(let i=0;i<CONFIG.shoesPerBatch;i++){
    let accepted=null;
    for(let j=0;j<CONFIG.proposalBudget;j++){
     proposals++;
     const trial=quant.conditionalHandProposal(INITIAL,condition,outcome,{player,banker});
     if(trial.weight>0&&condition()<trial.weight){accepted=trial.hand;break;}
    }
    if(!accepted)throw Error('deplete_conditioning_budget_exhausted');
    for(let sample=0;sample<CONFIG.samplesPerShoe;sample++){
     const u=Array.from({length:6},()=>forecast());
     const depleted=dealWithUniforms(accepted.counts,u);
     const control=dealWithUniforms(INITIAL,u);
     delta[index(depleted)]++;delta[index(control)]--;
     if(depleted!==control)pairedDisagreements++;
    }
   }
   batchDeltas.push(delta.map(v=>v/(CONFIG.shoesPerBatch*CONFIG.samplesPerShoe)));
  }
  const keys=['B','P','T'];
  const probability=Object.fromEntries(keys.map((key,i)=>[key,BASE[key]+
   batchDeltas.reduce((a,b)=>a+b[i],0)/CONFIG.batches]));
  if(keys.some(k=>!Number.isFinite(probability[k])||probability[k]<0||probability[k]>1)||
    Math.abs(keys.reduce((a,k)=>a+probability[k],0)-1)>1e-10)
   throw Error('deplete_invalid_probability');
  const gaps=batchDeltas.map(d=>1.95*d[0]-2*d[1]);
  const mean=gaps.reduce((a,b)=>a+b,0)/gaps.length;
  const se=Math.sqrt(gaps.reduce((a,b)=>a+(b-mean)**2,0)/(gaps.length*(gaps.length-1)));
  return {version:VERSION,probability,ev_banker:.95*probability.B-probability.P,
   ev_player:probability.P-probability.B,paired_mc_gap_se:se,
   completed_shoes:CONFIG.batches*CONFIG.shoesPerBatch,
   samples:CONFIG.batches*CONFIG.shoesPerBatch*CONFIG.samplesPerShoe,
   conditioning_proposals:proposals,paired_disagreements:pairedDisagreements,
   prior:'fresh_eight_deck_single_final_score',independent_observation:false,
   observed_card_evidence:false,validated_edge:false};
 }
 function candidate(report,diagnostic){
  if(report.history_length!==1||report.point_score_observation_count!==1||
    report.observed_card_evidence!==false||report.physical_ev?.simulation_exhausted)
   throw Error('deplete_requires_unobserved_independent_score');
  const old=report.physical_ev.simulated_probability,w=CONFIG.candidateWeight;
  const p=Object.fromEntries(['B','P','T'].map(k=>[k,(1-w)*old[k]+w*diagnostic.probability[k]]));
  const evB=.95*p.B-p.P,evP=p.P-p.B,baseGap=1.95*BASE.B-2*BASE.P;
  const direction=evB>=evP?'B':'P',deviation=evB-evP-baseGap;
  // These branches share the SAME observation; do not multiply likelihoods,
  // shrink uncertainty as independent evidence, or alter the V4 thresholds.
  const d=report.quality||report.adaptive_score_smoothing||report.early_shoe_guard;
  const quality={...d,mc_gap_se:Math.max(d.mc_gap_se,diagnostic.paired_mc_gap_se)};
  const gate=report.effective_research_gate;
  const available=Number.isFinite(gate)&&Math.abs(deviation)>=gate;
  return {...report,ev_banker:evB,ev_player:evP,ev_gap_deviation:deviation,
   display_direction:direction,hypothetical_candidate:direction,early_shoe_guard:quality,
   physical_ev:{...report.physical_ev,simulated_probability:p,ev_banker:evB,
    ev_player:evP,ev_gap:evB-evP,ev_gap_deviation:deviation,
    mc_se_gap:quality.mc_gap_se,early_shoe_guard:quality,
    draw_length_diagnostics:report.physical_ev.draw_length_diagnostics?
     {...report.physical_ev.draw_length_diagnostics,source:'primary_pf_only'}:null},
   research_signal:{...report.research_signal,direction,score:Math.abs(deviation),available},
   action:'SKIP',bet_allowed:false,validated_edge:false};
 }
 function attach(report,pair,{mode=CONFIG.defaultMode,seedSalt=0}={}){
  if(mode!=='shadow'&&mode!=='candidate')throw Error('invalid_deplete_mode');
  const diagnostic=estimate(pair.player,pair.banker,{seedSalt});
  const fused=candidate(report,diagnostic);
  const detail={...diagnostic,status:'complete',mode,candidate_weight:CONFIG.candidateWeight,
   candidate_ev_banker:fused.ev_banker,candidate_ev_player:fused.ev_player,
   primary_direction:report.ev_banker>=report.ev_player?'B':'P',
   candidate_direction:fused.display_direction,
   affects_display:mode==='candidate',validated_hit_rate_gain:false};
  return {...(mode==='candidate'?fused:report),deplete_mc:detail};
 }
 function attachShadow(report,pair,{seedSalt=0}={}){
  try{return attach(report,pair,{mode:'shadow',seedSalt});}
  catch(_){return {...report,deplete_mc:{version:VERSION,status:'unavailable',
   mode:'shadow',affects_display:false,validated_hit_rate_gain:false,
   reason:'auxiliary_calculation_unavailable'}};}
 }
 function attachPrediction(report,pair){
  try{return attach(report,pair,{mode:'candidate'});}
  catch(_){return {...report,deplete_mc:{version:VERSION,status:'unavailable',
   mode:'candidate',affects_display:false,validated_hit_rate_gain:false,
   reason:'auxiliary_calculation_unavailable'}};}
 }
 return Object.freeze({VERSION,CONFIG,estimate,attach,attachShadow,attachPrediction,dealWithUniforms});
});
