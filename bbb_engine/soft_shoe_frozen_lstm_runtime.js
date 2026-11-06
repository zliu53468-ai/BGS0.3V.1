/**
 * Tiny frozen LSTM inference adapter for B/P/T outcomes + weak road features.
 *
 * No training occurs in production. No weights are bundled or invented.
 * A real-shoe trained, independently held-out, frozen & calibrated model may
 * supply validated weights through soft_shoe_frozen_model_bundle.js.
 * Invalid/untrained bundles are rejected; callers use a labelled fallback.
 */
(function(root,factory){
 "use strict";
 const api=factory();
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.BBBFrozenSequenceLSTM=api;
})(typeof self!=="undefined"?self:typeof window!=="undefined"?window:null,function(){
 "use strict";
 const SCHEMA="frozen_lstm_bpt_8x8_v1",INPUT=8,HIDDEN=8,WINDOW=24;
 const keys=["i","f","g","o"];
 const clamp=(x,min,max)=>Math.min(max,Math.max(min,x));
 const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
 const finite=a=>Array.isArray(a)&&a.every(x=>Number.isFinite(x)&&Math.abs(x)<=16);
 const sum=a=>a.reduce((s,x)=>s+x,0);
 function isValidated(model){
  return model?.schema===SCHEMA&&model?.input_dim===INPUT&&
   model?.hidden_dim===HIDDEN&&model?.window===WINDOW&&
   model?.trained===true&&model?.frozen===true&&model?.calibrated===true&&
   model.validation?.real_shoe_data===true&&
   model.validation?.out_of_shoe_holdout===true;
 }
 function features(history){
  if(!Array.isArray(history)||history.some(x=>!["B","P","T"].includes(x)))
   throw Error("invalid_lstm_bpt_history");
  const seq=history.slice(-WINDOW),rows=[];
  for(let i=0;i<seq.length;i++){
   const r=seq[i],recent=seq.slice(Math.max(0,i-7),i+1);
   let run=1;
   for(let j=i-1;j>=0&&seq[j]===r;j--)run++;
   const samePrev=i>0&&seq[i-1]===r?1:0;
   // 8-dimensional input: B/P/T one hot, streak, continuation,
   // and trailing 8-hand Banker/Player/Tie frequencies.
   rows.push([
    +(r==="B"),+(r==="P"),+(r==="T"),
    clamp(run/6,0,1),samePrev,
    recent.filter(v=>v==="B").length/recent.length,
    recent.filter(v=>v==="P").length/recent.length,
    recent.filter(v=>v==="T").length/recent.length
   ]);
  }
  return rows;
 }
 function validWeights(model){
  if(!isValidated(model))return false;
  const weights=model.weights;
  if(!weights||typeof weights!=="object")return false;
  for(const k of keys){
   if(!finite(weights["W"+k])||weights["W"+k].length!==HIDDEN*INPUT||
      !finite(weights["U"+k])||weights["U"+k].length!==HIDDEN*HIDDEN||
      !finite(weights["b"+k])||weights["b"+k].length!==HIDDEN)return false;
  }
  if(!finite(weights.Wy)||weights.Wy.length!==3*HIDDEN||
     !finite(weights.by)||weights.by.length!==3)return false;
  return true;
 }
 function predict(history,model){
  if(!Array.isArray(history)||history.some(x=>!["B","P","T"].includes(x)))
   throw Error("invalid_lstm_bpt_history");
  if(!validWeights(model))return{available:false,reason:
   isValidated(model)?"invalid_lstm_weights":"no_validated_frozen_lstm"};
  const w=model.weights,rows=features(history);
  let hidden=Array(HIDDEN).fill(0),cell=Array(HIDDEN).fill(0);
  for(const row of rows){
   const nextH=Array(HIDDEN),nextC=Array(HIDDEN);
   for(let h=0;h<HIDDEN;h++){
    const gate={};
    for(const k of keys){
     let x=w["b"+k][h];
     for(let j=0;j<INPUT;j++)x+=w["W"+k][h*INPUT+j]*row[j];
     for(let j=0;j<HIDDEN;j++)x+=w["U"+k][h*HIDDEN+j]*hidden[j];
     gate[k]=k==="g"?Math.tanh(x):sigmoid(x);
    }
    nextC[h]=gate.f*cell[h]+gate.i*gate.g;
    nextH[h]=gate.o*Math.tanh(nextC[h]);
   }
   hidden=nextH;cell=nextC;
  }
  const logits=Array.from({length:3},(_,k)=>{
   let x=w.by[k];
   for(let j=0;j<HIDDEN;j++)x+=w.Wy[k*HIDDEN+j]*hidden[j];
   return x;
  });
  const temp=model.temperature??1;
  if(!Number.isFinite(temp)||temp<.05||temp>10)
   return{available:false,reason:"invalid_lstm_temperature"};
  const max=Math.max(...logits),exp=logits.map(x=>Math.exp((x-max)/temp)),z=sum(exp);
  if(!(z>0&&Number.isFinite(z)))return{available:false,reason:"invalid_lstm_probability"};
  const p=exp.map(x=>x/z);
  return{
   available:true,architecture:"frozen_lstm_bpt_8x8_v1",
   trained:true,validated:true,history_support:rows.length,
   probabilities:{B:p[0],P:p[1],T:p[2]},
   frozen:true,features_per_round:INPUT
  };
 }
 return Object.freeze({SCHEMA,INPUT,HIDDEN,WINDOW,features,isValidated,validWeights,predict});
});
