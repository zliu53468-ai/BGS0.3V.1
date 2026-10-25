/* Observed, completed hands only. No ranks or points are inferred from B/P/T. */
(function(root,factory){
  const api=factory();
  if(typeof module==="object"&&module.exports)module.exports=api;
  if(root)root.__BGS_MACRO_EMA__=api;
})(typeof window!=="undefined"?window:globalThis,function(){
  "use strict";
  const NAMES=["Six_Card","Low_Score","Point_Diff"];
  const clamp=(x,lo=0,hi=1)=>Math.max(lo,Math.min(hi,x));
  function integer(value,name,lo,hi){
    if(value===null||value===undefined||value==="")return null;
    if(typeof value==="boolean"||typeof value==="object"||!/^\d+$/.test(String(value).trim()))throw new Error(name+" 必須為整數");
    const n=Number(value);
    if(!Number.isInteger(n)||n<lo||n>hi)throw new Error(name+" 超出範圍");
    return n;
  }
  function normalizeHand(hand){
    const outcome=String(hand?.outcome||"").toUpperCase();
    if(!["B","P","T"].includes(outcome))throw new Error("outcome 必須為 B/P/T");
    const card_count=integer(hand.card_count,"開獎張數",4,6);
    const player_score=integer(hand.player_score,"閒點數",0,9);
    const banker_score=integer(hand.banker_score,"莊點數",0,9);
    if(player_score!==null&&banker_score!==null){
      const actual=banker_score>player_score?"B":banker_score<player_score?"P":"T";
      if(actual!==outcome)throw new Error("點數與莊／閒／和結果不一致");
    }
    return {outcome,card_count,player_score,banker_score};
  }
  function createState(shoeId=""){
    return {shoe_id:String(shoeId),rounds:0,ema:Object.fromEntries(NAMES.map(k=>[k,{short:null,long:null,count:0,last_round:0}]))};
  }
  function values(hand){
    const h=normalizeHand(hand),winner=h.outcome==="B"?h.banker_score:h.player_score;
    return {
      Six_Card:h.card_count===null?null:Number(h.card_count===6),
      Low_Score:h.outcome==="T"?0:winner===null?null:Number(winner<=3),
      Point_Diff:h.player_score===null||h.banker_score===null?null:Math.abs(h.player_score-h.banker_score)
    };
  }
  function snapshot(state){
    const out={shoe_id:state.shoe_id,rounds:state.rounds,counts:{},ages:{}};
    for(const key of NAMES){
      const e=state.ema[key];
      out[key+"_Short"]=e.short;out[key+"_Long"]=e.long;
      out[key+"_Spread"]=e.count?e.short-e.long:null;
      out.counts[key]=e.count;out.ages[key]=e.count?state.rounds-e.last_round:null;
    }
    return out;
  }
  function update(state,hand,config={}){
    const raw=values(hand); // Validate everything before mutating any EMA.
    state.rounds++;
    for(const key of NAMES){
      const x=raw[key],e=state.ema[key];if(x===null)continue;
      const short=clamp(config.short_alpha??.20,.05,.50),long=clamp(config.long_alpha??.05,.01,.20);
      e.short=e.count?short*x+(1-short)*e.short:x;
      e.long=e.count?long*x+(1-long)*e.long:x;
      e.count++;e.last_round=state.rounds;
    }
    return snapshot(state);
  }
  function noiseGate(noise,low=.60,high=.75){
    const n=typeof noise==="number"&&Number.isFinite(noise)?clamp(noise):1;
    return clamp((high-n)/Math.max(1e-9,high-low));
  }
  function adjustment(probabilityB,features,progress,noise,config={}){
    const p=clamp(probabilityB),result={input:p,value:p,delta:0,signal:0,ready:false,reason:"disabled"};
    if(config.enabled!==true)return result;
    const number=(key,fallback)=>typeof config[key]==="number"&&Number.isFinite(config[key])?config[key]:fallback;
    const minimum=Math.max(1,Math.floor(number("min_observations",12))),maxAge=Math.max(0,number("max_age",3));
    // Each observed feature is independently eligible. A missing card_count
    // MUST NOT suppress Low_Score / Point_Diff when final scores were supplied.
    // Missing/stale readings do not become zero-valued observations.
    const weights=config.coefficients||{},active=[],available=[],ignored=[],readiness={};
    const coefficientNames={Six_Card:"six_card",Low_Score:"low_score",Point_Diff:"point_diff"};
    const beta=k=>typeof weights[k]==="number"&&Number.isFinite(weights[k])?clamp(weights[k],-10,10):0;
    let signal=0,configured=0;
    for(const name of NAMES){
      const key=coefficientNames[name],weight=beta(key);
      const value=features?.[name+"_Spread"];
      const count=features?.counts?.[name],age=features?.ages?.[name];
      const valid=Number.isFinite(value)&&Number.isInteger(count)&&count>=minimum&&
        Number.isInteger(age)&&age>=0&&age<=maxAge;
      readiness[key]=valid;
      if(valid)available.push(key);
      if(weight===0)continue;
      configured++;
      if(!valid){ignored.push(key);continue;}
      active.push(key);
      signal+=weight*value/(name==="Point_Diff"?9:1);
    }
    // Readiness is diagnostic only. The checked-in production bundle keeps all
    // three macro coefficients at zero, so this module must return delta=0.
    const availability={available_features:available,active_features:active,
      ignored_configured_features:ignored,feature_ready:readiness,
      configured_coefficient_count:configured,all_coefficients_zero:configured===0};
    if(!active.length)return {...result,...availability,
      reason:configured?"insufficient_observations":"zero_coefficients_or_signal"};
    const start=clamp(number("progress_start",.30),0,.99),power=clamp(number("progress_power",2),1,4);
    const weight=clamp((progress-start)/(1-start))**power,cap=clamp(number("max_delta",.03),0,.05);
    // v2 coefficients are literal probability-point sensitivities per normalized spread.
    // Legacy bundles retain their prior cap*tanh(signal) interpretation.
    const gate=noiseGate(noise,number("noise_low",.60),number("noise_high",.75)),probabilityUnits=config.coefficient_units==="probability_delta";
    const boundedSignal=clamp(signal,-cap,cap);
    const delta=(probabilityUnits?boundedSignal:cap*Math.tanh(signal))*weight*gate;
    // B/P EV preference boundary: .95*p-(1-p) = (1-p)-p.
    const boundary=2/3.95,raw=clamp(p+delta);
    const value=p>=boundary?Math.max(boundary,raw):Math.min(boundary,raw);
    return {...result,...availability,value,delta:value-p,signal,ready:true,progress_weight:weight,noise_gate:gate,max_delta:cap,
      reason:signal===0?"zero_coefficients_or_signal":gate===0?"high_noise":"applied"};
  }
  return {NAMES,createState,normalizeHand,values,update,snapshot,noiseGate,adjustment};
});
