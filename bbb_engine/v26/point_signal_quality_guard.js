/**
 * BBB V2.6 / BGS LINE shared moderate DISPLAY-ONLY quality guard.
 *
 * Single-final-score PF96 is evidence-limited: a small conditional EV gap is
 * not a verified edge or actual hit probability. This heuristic abstains only
 * on weak *relative* EV signals. No EV, PF, Early-Shoe, gate or BET mutation.
 */
(function(root,factory){
  "use strict";
  const api=factory();
  if(typeof module==="object"&&module.exports)module.exports=api;
  if(root)root.BBBPointSignalQualityGuard=api;
})(typeof window!=="undefined"?window:null,function(){
"use strict";
const VERSION="BBB_V26_SELECTIVE_ENTRY_V3";
// Rollback V3 -> previous moderate V2: set ACTIVE_PROFILE="moderate" in
// both repositories. Legacy V1: set MODERATE_ENTRY.enabled=false.
// Neither setting changes the physical EV, PF, early-shoe or raw B/P choice.
const ENTRY_PROFILES=Object.freeze({
  moderate:Object.freeze({
    additional_ev_gap:.00105,
    pf_quality_threshold:.62,
    low_quality_extra:.00035,
    uncertainty_proxy_threshold:.52,
    high_uncertainty_extra:.00025
  }),
  moderate_high:Object.freeze({
    additional_ev_gap:.00200,
    pf_quality_threshold:.68,
    low_quality_extra:.00055,
    uncertainty_proxy_threshold:.48,
    high_uncertainty_extra:.00035
  })
});
const ACTIVE_PROFILE="moderate_high";
const MODERATE_ENTRY=Object.freeze({
  enabled:true,
  ...ENTRY_PROFILES[ACTIVE_PROFILE]
});
const finite=Number.isFinite;
const clamp=(v,low,high)=>Math.max(low,Math.min(high,v));
function decide(report,{enabled=MODERATE_ENTRY.enabled,profile=ACTIVE_PROFILE}={}){
  const tuning=ENTRY_PROFILES[profile];
  if(!tuning)throw Error("invalid_entry_profile");
  const evB=report?.ev_banker,evP=report?.ev_player;
  const d=report?.early_shoe_guard;
  const invalid=!finite(evB)||!finite(evP)||
    report?.physical_ev?.simulation_exhausted===true||
    d?.mode!=="independent_final_score_quality"||
    ![d.quality,d.ess_ratio,d.ancestor_diversity,
       d.mc_gap_se,d.posterior_gap_spread].every(finite);
  const rawDirection=finite(evB)&&finite(evP)?
    (evB>=evP?"B":"P"):null;
  if(invalid)return{version:VERSION,direction:"SKIP",raw_direction:rawDirection,
    reason:"insufficient_quality_diagnostics",threshold:null,gap:null,
    verified_edge:false,validated_hit_rate:false};
  const quality=clamp(d.quality,0,1);
  const ess=clamp(d.ess_ratio,0,1);
  const diversity=clamp(d.ancestor_diversity,0,1);
  const mc=clamp(d.mc_gap_se/.015,0,1);
  const spread=clamp(d.posterior_gap_spread/.20,0,1);
  // EV-unit threshold: baseline 0.085 percentage points, rising *modestly*
  // with posterior/MC uncertainty and diminished particle support.
  // It is NOT a calibrated real-world confidence bound.
  const legacyThreshold=.00085+.00055*(1-quality)+.00040*(1-ess)+
    .00030*(1-diversity)+.00060*mc+.00035*spread;
  // Do not confuse the MC/posterior uncertainty proxy with 57D physics noise.
  // All additions ONLY narrow display signals. No alternate direction or
  // fabricated positive-EV qualification is introduced.
  const uncertaintyProxy=Math.max(mc,spread);
  const addition=enabled?
    tuning.additional_ev_gap+
    (quality<tuning.pf_quality_threshold?
      tuning.low_quality_extra:0)+
    (uncertaintyProxy>tuning.uncertainty_proxy_threshold?
      tuning.high_uncertainty_extra:0):0;
  const threshold=legacyThreshold+addition;
  const gap=Math.abs(evB-evP);
  const skip=gap<threshold;
  return{version:VERSION,direction:skip?"SKIP":rawDirection,
    raw_direction:rawDirection,reason:skip?"small_ev_gap_or_uncertain":"relative_ev_gap_pass",
    threshold,gap,legacy_threshold:legacyThreshold,threshold_addition:addition,
    moderate_entry_enabled:Boolean(enabled),entry_profile:profile,
    uncertainty_proxy:uncertaintyProxy,
    quality,ess_ratio:ess,ancestor_diversity:diversity,
    mc_gap_se:d.mc_gap_se,posterior_gap_spread:d.posterior_gap_spread,
    verified_edge:false,validated_hit_rate:false};
}
return{VERSION,ENTRY_PROFILES,ACTIVE_PROFILE,MODERATE_ENTRY,decide};
});
