/**
 * BBB V2.2: strictly causal first-order Markov research diagnostics.
 * No cross-shoe training, no observed-card EV override, no particle
 * mutation or duplicate use of the same B/P/T sequence as independent
 * likelihood evidence. Shadow calibration MUST NOT drive BET/gates by default.
 */
(function(root,factory){
  "use strict";
  const api=factory();
  if(typeof module==="object"&&module.exports)module.exports=api;
  if(root)root.BBBMarkovResidual=api;
})(typeof window!=="undefined"?window:null,function(){
"use strict";
const VERSION="BBB_V22_MARKOV_RESIDUAL_SHADOW_V1";
const BASE_BANKER=.458597/(.458597+.446247);
const SETTINGS=Object.freeze({
  priorStrength:64,minSupport:12,maxShadowWeight:.10,
  maxConditionalShift:.003,appliedWeight:0
});
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const finite=x=>typeof x==="number"&&Number.isFinite(x);
/**
 * Ignore ties for directional transitions, but preserve tie counts.
 * Each observation is used once; no future outcomes influence the prefix.
 */
function transitionStats(history,{priorStrength=SETTINGS.priorStrength,
  minSupport=SETTINGS.minSupport,baselineBanker=BASE_BANKER}={}){
  if(!Array.isArray(history)||history.some(x=>x!=="B"&&x!=="P"&&x!=="T"))
    throw Error("invalid_markov_history");
  if(!finite(priorStrength)||priorStrength<=0||priorStrength>10000||
    !Number.isInteger(minSupport)||minSupport<1||
    !finite(baselineBanker)||baselineBanker<=0||baselineBanker>=1)
    throw Error("invalid_markov_options");
  const rows={B:{B:0,P:0},P:{B:0,P:0}};
  let previous=null,nonTie=0,ties=0;
  for(const result of history){
    if(result==="T"){ties++;continue;}
    if(previous!==null)rows[previous][result]++;
    previous=result;nonTie++;
  }
  const current=previous;
  const count=current?rows[current]:{B:0,P:0};
  const support=count.B+count.P;
  const posteriorBanker=(count.B+priorStrength*baselineBanker)/
    (support+priorStrength);
  const eligible=Boolean(current)&&support>=minSupport;
  return{
    version:VERSION,model:"first_order_dirichlet_beta",
    current_side:current,from:current,transition_counts:rows,
    last_transition_count:Math.max(0,nonTie-1),non_tie_rounds:nonTie,
    ties_ignored_for_transitions:ties,
    selected_support:support,min_support:minSupport,
    prior_strength:priorStrength,prior_banker:baselineBanker,
    posterior_conditional:{B:posteriorBanker,P:1-posteriorBanker},
    raw_conditional:support?{B:count.B/support,P:count.P/support}:null,
    reliability:support/(support+priorStrength),eligible,
    estimated_edge_validated:false,independent_evidence:false
  };
}
/**
 * A diagnostic WHAT-IF only: max blend 10% and max B/P conditional move
 * +/-0.003. Never edits physics or passes a modified probability to BET.
 * Treat Markov and PF as using overlapping history, not independent samples.
 */
function shadowCalibrate(markov,physical,{observed=false,
  maxShadowWeight=SETTINGS.maxShadowWeight,
  maxConditionalShift=SETTINGS.maxConditionalShift}={}){
  if(!markov||!markov.posterior_conditional)throw Error("missing_markov_stats");
  if(!finite(maxShadowWeight)||maxShadowWeight<0||maxShadowWeight>.10||
    !finite(maxConditionalShift)||maxConditionalShift<0||
    maxConditionalShift>.003)throw Error("invalid_markov_calibration");
  const prob=physical?.simulated_probability||physical?.win_probabilities;
  const mass=prob&&finite(prob.B)&&finite(prob.P)&&finite(prob.T)?
    prob.B+prob.P:0;
  const usable=mass>.05&&prob.B>=0&&prob.P>=0&&prob.T>=0&&
    Math.abs(mass+prob.T-1)<1e-4&&
    !physical?.simulation_exhausted&&!observed&&markov.eligible;
  const pConditional=mass>0?prob.B/mass:null;
  const rawDifference=usable?
    markov.posterior_conditional.B-markov.prior_banker:0;
  const proposedShift=usable?clamp(
    maxShadowWeight*markov.reliability*rawDifference,
    -maxConditionalShift,maxConditionalShift):0;
  const candidateBanker=usable?
    clamp(pConditional+proposedShift,0,1):pConditional;
  const candidateProbability=mass>0?{
    B:mass*candidateBanker,P:mass*(1-candidateBanker),T:prob.T
  }:null;
  const candidateEV=candidateProbability?{
    B:.95*candidateProbability.B-candidateProbability.P,
    P:candidateProbability.P-candidateProbability.B
  }:null;
  return{
    version:VERSION,shadow_only:true,eligible:usable,
    candidate_probability:candidateProbability,candidate_ev:candidateEV,
    conditional_shift:proposedShift,
    theoretical_shadow_weight:maxShadowWeight,
    max_conditional_shift:maxConditionalShift,
    applied_weight:SETTINGS.appliedWeight,
    modifies_physical_ev:false,modifies_particle_weights:false,
    modifies_bet_gate:false,independent_evidence:false,
    validated_edge:false,validated_calibration:false
  };
}
return{VERSION,SETTINGS,BASE_BANKER,transitionStats,shadowCalibrate};
});
