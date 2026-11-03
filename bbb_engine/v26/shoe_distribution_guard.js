/**
 * BBB V2 Dynamic Shoe Distribution Guard.
 * Causal, B/P/T-only, recent-window change-point *diagnostic*.
 * Detects descriptive shifts; does not extrapolate streaks or flip direction.
 * The gate applies only to unobserved-card research indications.
 */
(function(root,factory){
 "use strict";
 const api=factory();
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.BBBShoeDistributionGuard=api;
})(typeof window!=="undefined"?window:null,function(){
 "use strict";
 const VERSION="BBB_V2_DYNAMIC_SHOE_DISTRIBUTION_GUARD_V1";
 const WINDOWS=Object.freeze([8,16,24]);
 const HISTORY_LIMIT=500;
 const PRIOR_ODDS=.10/.90;
 const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
 const factorialLogs=[0];
 function logFactorial(n){
   for(let i=factorialLogs.length;i<=n;i++)
     factorialLogs.push(factorialLogs[i-1]+Math.log(i));
   return factorialLogs[n];
 }
 // Uniform Beta(1,1) prior integrated over the ordered Bernoulli sequence.
 // Binomial coefficients are intentionally omitted because outcomes are
 // ordered and identical in the pooled/split hypotheses.
 function logMarginal(b,p){
   return logFactorial(b)+logFactorial(p)-logFactorial(b+p+1);
 }
 function counts(a){
   let b=0,p=0;
   for(const s of a){if(s==="B")b++;else if(s==="P")p++;}
   return{B:b,P:p,n:b+p,banker_share:b/(b+p||1)};
 }
 function evaluate(history,{observed=false}={}){
   if(!Array.isArray(history)||history.length>HISTORY_LIMIT||
      history.some(v=>v!=="B"&&v!=="P"&&v!=="T"))
     throw Error("invalid_distribution_history");
   const bp=history.filter(x=>x!=="T");
   const windows=WINDOWS.map(w=>{
     if(bp.length<2*w)return{window:w,eligible:false,previous:null,recent:null,
       bayes_factor:null,change_probability:null,significant:false};
     const recent=counts(bp.slice(-w)),previous=counts(bp.slice(-2*w,-w));
     const split=logMarginal(previous.B,previous.P)+
       logMarginal(recent.B,recent.P);
     const pooled=logMarginal(previous.B+recent.B,previous.P+recent.P);
     // Numerically stable posterior from a 10% prior shift probability.
     const logBF=split-pooled;
     const logOdds=logBF+Math.log(PRIOR_ODDS);
     const changeProb=logOdds>=0?1/(1+Math.exp(-logOdds)):
       Math.exp(logOdds)/(1+Math.exp(logOdds));
     const shift=Math.abs(recent.banker_share-previous.banker_share);
     const significant=changeProb>=.80&&shift>=.20;
     return{window:w,eligible:true,previous,recent,
       bayes_factor:Math.exp(Math.min(40,logBF)),
       change_probability:changeProb,
       difference:recent.banker_share-previous.banker_share,
       significant};
   });
   const strongest=windows.filter(w=>w.significant)
     .sort((a,b)=>b.change_probability-a.change_probability)[0]||null;
   const recent=counts(bp.slice(-8));
   // Never override or reweight physical particles: this is purely a
   // conservative extra research-signal filter during strong shifts.
   const additionalGate=observed||!strongest?0:
     .004*clamp((strongest.change_probability-.80)/.20,0,1);
   return{
     version:VERSION,active:!observed,observed_card_bypass:observed,
     total_rounds:history.length,non_tie_rounds:bp.length,
     current_stage:bp.length<16?"insufficient_data":
       strongest?"distribution_shift":"no_strong_change_evidence",
     recent_banker_share:recent.n?recent.banker_share:null,
     strongest_window:strongest?.window??null,
     change_probability:strongest?.change_probability??null,
     additional_gate:additionalGate,
     predicted_next_direction:null,
     calibrated_win_rate:false,
     windows
   };
 }
 return{VERSION,WINDOWS,evaluate};
});
