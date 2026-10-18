/* B/P/T-only policy adapter. No trained model features or weights are changed. */
(() => {
"use strict";
const VERSION="BLIND_HISTORY_V1";
const clip=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
// Modes: guarded (rules-conditioned), baseline (exact legacy), requested_prior
// (the user's uncalibrated matrix, for offline comparison only).
const mode=["baseline","guarded","requested_prior"].includes(window.BBB_BLIND_UPGRADE_MODE)?window.BBB_BLIND_UPGRADE_MODE:"baseline";
const enabled=mode!=="baseline";
const requestedMatrix={B:[.45,.40,.15],P:[.45,.40,.15],T:[.10,.40,.50]};
function structure(history){
  const bp=history.filter(x=>x==="B"||x==="P");
  const windows=[5,10].map(n=>{
    const xs=history.slice(-n).filter(x=>x==="B"||x==="P");let changes=0;
    for(let i=1;i<xs.length;i++)changes+=Number(xs[i]!==xs[i-1]);
    const transitions=xs.length-1,rate=transitions>0?changes/transitions:null;
    const entropy=rate===null?null:rate===0||rate===1?0:-(rate*Math.log2(rate)+(1-rate)*Math.log2(1-rate));
    return {window:n,samples:xs.length,transitionRate:rate,entropy};
  });
  const supported=bp.length>=6&&windows.every(w=>w.samples>=4),entropy=supported?(windows[0].entropy+windows[1].entropy)/2:null;
  return {windows,supported,entropy,coreWeight:!enabled||!supported?.15:clip(.30-.25*entropy,.05,.30)};
}
function consensus(d){
  const finite=x=>typeof x==="number"&&Number.isFinite(x);
  return !!d&&d.particle_count===500&&finite(d.probability_std)&&d.probability_std>=0&&d.probability_std<=.08&&
    finite(d.probability_mc_se)&&d.probability_mc_se>=0&&d.probability_mc_se<=.015&&
    finite(d.pre_resample_ess_ratio)&&d.pre_resample_ess_ratio>=.50&&d.pre_resample_ess_ratio<=1&&
    finite(d.ancestor_ratio)&&d.ancestor_ratio>=.25&&d.ancestor_ratio<=1&&d.conditional_failures===0&&d.forecast_unsupported_particles===0;
}
function thresholds(depth,ev,band,diagnostics){
  if(!enabled)return {ev,band,stage:"baseline",consensus:false,relief:0};
  const stable=consensus(diagnostics),early=depth>=1&&depth<=15,late=depth>50;
  // Completed history length, never next-hand index. No zero/negative EV bypass.
  const relief=early?.30*Math.max(0,ev-.002):late&&stable?.002:0;
  const penalty=late?.002:0;
  return {ev:Math.max(.002,ev+penalty-relief),band:Math.max(.003,band*(early?.85:1)+(late&&!stable?.001:0)),
    stage:early?"early":late?"late":"middle",consensus:stable,relief};
}
function blendWeight(base,history,diagnostics){
  if(!enabled)return base;
  const s=structure(history),entropy=s.entropy??.5;
  const ess=diagnostics.pre_resample_ess_ratio??0;
  const qualityCap=history.length>50&&ess<.5?.10:.16;
  return clip(base*(.85+.30*entropy),.06,qualityCap);
}
window.__BBB_BLIND_UPGRADE__={version:VERSION,mode,enabled,
  requestedMatrix,structure,consensus,thresholds,blendWeight};
})();
