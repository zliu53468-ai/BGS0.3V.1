/* Device/browser-local prequential probability calibration. No model training,
 * network requests, route-pattern rules or writes to formal wager snapshots. */
(function(root,factory){
  const api=factory();
  if(typeof module==="object"&&module.exports)module.exports=api;
  if(root)root.__BBB_BROWSER_CALIBRATION__=api;
})(typeof window!=="undefined"?window:globalThis,function(){
  "use strict";
  // bgs* is managed by the existing verified-license IndexedDB isolation layer.
  const KEY="bgs_browser_calibration_v1",VERSION=1;
  const defaults={enabled:true,min_samples:24,min_validation:16,prior_strength:100,
    max_delta:.005,residual_decay:1,validation_window:32,min_brier_gain:.00005,noise_start:.55,noise_stop:.85};
  const finite=x=>typeof x==="number"&&Number.isFinite(x);
  const clamp=(x,lo=0,hi=1)=>Math.max(lo,Math.min(hi,x));
  function settings(input={}){
    const c={...defaults};
    for(const key of Object.keys(c))if(finite(input[key]))c[key]=input[key];
    c.enabled=input.enabled!==false;
    c.min_samples=Math.round(clamp(c.min_samples,12,200));
    c.min_validation=Math.round(clamp(c.min_validation,8,100));
    c.prior_strength=clamp(c.prior_strength,50,1000);
    c.max_delta=clamp(c.max_delta,0,.005);
    c.validation_window=Math.round(clamp(c.validation_window,c.min_validation,200));
    c.min_brier_gain=clamp(c.min_brier_gain,0,.01);
    c.residual_decay=clamp(c.residual_decay,.95,1);
    c.noise_start=clamp(c.noise_start,0,.8);c.noise_stop=clamp(c.noise_stop,c.noise_start+.01,1);
    c.stage_anchors={};
    for(const key of ["prior_strength","residual_decay","validation_window"]){
      const a=input.stage_anchors?.[key];
      if(Array.isArray(a)&&a.length>=2&&a.every((v,i)=>Array.isArray(v)&&v.length===2&&v.every(finite)&&(!i||v[0]>a[i-1][0])))c.stage_anchors[key]=a;
    }
    return c;
  }
  function atRound(config,round){
    const c={...config};
    for(const [key,a] of Object.entries(config.stage_anchors||{})){
      const r=clamp(round,a[0][0],a.at(-1)[0]);let value=a.at(-1)[1];
      for(let i=1;i<a.length;i++)if(r<=a[i][0]){const [x,v]=a[i-1],[y,w]=a[i];value=v+(w-v)*(r-x)/(y-x);break;}
      c[key]=value;
    }
    c.prior_strength=clamp(c.prior_strength,50,1000);c.residual_decay=clamp(c.residual_decay,.95,1);
    c.validation_window=Math.round(clamp(c.validation_window,c.min_validation,200));
    return c;
  }
  function candidate(p,noise,state,c){
    if(!c.enabled||state.n<c.min_samples||p===.5||!finite(noise))return p;
    const gate=clamp((c.noise_stop-noise)/(c.noise_stop-c.noise_start));
    const delta=clamp(state.residual/(c.prior_strength+(state.effective_n??state.n)),-c.max_delta,c.max_delta)*gate;
    if(gate===0||Math.abs(delta)<=1e-12)return p;
    const adjusted=clamp(p+delta);
    // Local correction never reverses the underlying Physics/PF side.
    return p>.5?Math.max(.500001,adjusted):Math.min(.499999,adjusted);
  }
  function replay(events,c,nextRound){
    const state={n:0,effective_n:0,residual:0,trials:[],seen:events.length,ties:0};
    for(const event of events){
      if(event.actual==="T"){state.ties++;continue;}
      if(!finite(event.p)||event.p<0||event.p>1)continue;
      const active=atRound(c,event.index+1);
      const proposed=candidate(event.p,event.noise,state,active),y=event.actual==="B"?1:0;
      // Score the proposal from PRIOR results, then learn this result. No
      // in-sample validation or use of the just-observed outcome for its forecast.
      if(state.n>=c.min_samples&&Math.abs(proposed-event.p)>1e-12){
        state.trials.push({base:(event.p-y)**2,local:(proposed-y)**2});
        if(state.trials.length>active.validation_window)state.trials.shift();
      }
      state.residual=active.residual_decay*state.residual+(y-event.p);
      state.effective_n=active.residual_decay*state.effective_n+1;state.n++;
    }
    const active=atRound(c,nextRound??(events.at(-1)?.index+2||1));
    state.trials=state.trials.slice(-active.validation_window);
    state.validation_count=state.trials.length;
    state.brier_gain=state.trials.length?state.trials.reduce((sum,t)=>sum+t.base-t.local,0)/state.trials.length:0;
    state.ready=c.enabled&&state.n>=c.min_samples&&state.validation_count>=c.min_validation&&state.brier_gain>=c.min_brier_gain;
    return state;
  }
  function fingerprint(...values){
    let hash=2166136261;
    for(const value of values){const text=JSON.stringify(value);for(let i=0;i<text.length;i++){hash^=text.charCodeAt(i);hash=Math.imul(hash,16777619);}}
    return (hash>>>0).toString(16); // Versioning only; not an authentication hash.
  }
  function createClient(storage){
    let storageError="";
    function read(history,shoe,model){
      let saved=null;try{saved=JSON.parse(storage.getItem(KEY)||"null");}catch(_){storageError="read_failed";}
      if(saved?.version!==VERSION||saved.shoe!==shoe||saved.model!==model||!Array.isArray(saved.events))return {version:VERSION,shoe,model,events:[]};
      const events=saved.events.slice(0,500).filter(e=>Number.isInteger(e?.index)&&e.index>=0&&e.index<history.length&&e.prefix===history.slice(0,e.index).join("")&&e.actual===history[e.index]&&["B","P","T"].includes(e.actual)&&(e.p===null||finite(e.p)&&e.p>=0&&e.p<=1)).sort((a,b)=>a.index-b.index);
      // A corrupted/duplicated journal must not learn an outcome twice.
      const unique=events.filter((e,i)=>!i||events[i-1].index!==e.index);
      return {version:VERSION,shoe,model,events:unique};
    }
    function save(journal){
      try{storage.setItem(KEY,JSON.stringify(journal));storageError="";return true;}
      catch(_){storageError="write_failed";return false;}
    }
    function inspect(history,shoe,model,config){
      const c=settings(config),state=replay(read(history,shoe,model).events,c,history.length+1);
      return {enabled:c.enabled,seen:state.seen,directional_samples:state.n,ties:state.ties,
        validation_count:state.validation_count,brier_gain:state.brier_gain,ready:state.ready,
        max_delta:c.max_delta,storage_error:storageError};
    }
    function prepare(history,forecast,shoe,model){
      return {version:VERSION,shoe,model,index:history.length,prefix:history.join(""),
        p:forecast?.available&&finite(forecast.directional?.B)?forecast.directional.B:null,
        noise:finite(forecast?.uncertainty)?forecast.uncertainty:null};
    }
    function commit(ticket,actual,history,config){
      if(!ticket||!Array.isArray(history)||history.length!==ticket.index+1||history.at(-1)!==actual||history.slice(0,-1).join("")!==ticket.prefix||!["B","P","T"].includes(actual))return false;
      const journal=read(history,ticket.shoe,ticket.model);
      if(journal.events.some(e=>e.index===ticket.index))return false;
      journal.events.push({index:ticket.index,prefix:ticket.prefix,actual,p:ticket.p,noise:ticket.noise});
      return save(journal);
    }
    function apply(forecast,history,shoe,model,config){
      const c=settings(config),journal=read(history,shoe,model),state=replay(journal.events,c,history.length+1);
      const status=inspect(history,shoe,model,config);
      if(!forecast?.available)return {forecast,status:{...status,applied:false,delta:0,reason:"model_unavailable"}};
      const p=forecast.directional.B,proposed=candidate(p,forecast.uncertainty,state,atRound(c,history.length+1)),adjusted=state.ready&&!storageError?proposed:p;
      const delta=adjusted-p,mass=forecast.probabilities.B+forecast.probabilities.P;
      return {forecast:{...forecast,directional:{B:adjusted,P:1-adjusted},
        probabilities:{B:mass*adjusted,P:mass*(1-adjusted),T:forecast.probabilities.T},
        weak_signal:Math.abs(adjusted-.5)<.005},status:{...status,applied:Math.abs(delta)>1e-12,delta,
        reason:!c.enabled?"disabled":storageError?"storage_error":!state.ready?"warming_or_not_better":Math.abs(delta)>1e-12?"bounded_local_calibration":"noise_or_neutral"}};
    }
    function reconcile(history,shoe,model){return save(read(history,shoe,model));}
    function clear(){try{storage.removeItem(KEY);storageError="";return true;}catch(_){storageError="clear_failed";return false;}}
    return {inspect,prepare,commit,apply,reconcile,clear};
  }
  return {KEY,VERSION,defaults,settings,atRound,fingerprint,createClient};
});
