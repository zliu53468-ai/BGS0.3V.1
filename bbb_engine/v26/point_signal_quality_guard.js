/**
 * BBB V2.6 / BGS LINE shared mild DISPLAY-ONLY quality guard.
 *
 * Single-final-score PF256 is evidence-limited: a small conditional EV gap is
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
const VERSION="BBB_V26_MILD_DISPLAY_SKIP_V1";
const finite=Number.isFinite;
const clamp=(v,low,high)=>Math.max(low,Math.min(high,v));
function decide(report){
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
  const threshold=.00085+.00055*(1-quality)+.00040*(1-ess)+
    .00030*(1-diversity)+.00060*mc+.00035*spread;
  const gap=Math.abs(evB-evP);
  const skip=gap<threshold;
  return{version:VERSION,direction:skip?"SKIP":rawDirection,
    raw_direction:rawDirection,reason:skip?"small_ev_gap_or_uncertain":"relative_ev_gap_pass",
    threshold,gap,quality,ess_ratio:ess,ancestor_diversity:diversity,
    mc_gap_se:d.mc_gap_se,posterior_gap_spread:d.posterior_gap_spread,
    verified_edge:false,validated_hit_rate:false};
}
return{VERSION,decide};
});
