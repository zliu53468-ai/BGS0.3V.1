"use strict";
// Data-only bridge into the unchanged, pinned BBB browser prediction scripts.
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const root=path.resolve(__dirname,".."),manifest=require("./manifest.json");
async function run(input){
  const history=input.history||[];
  if(!Array.isArray(history)||history.length>2000||history.some(x=>!["B","P","T"].includes(x)))throw Error("invalid history");
  const values=new Map(Object.entries(input.storage||{}));
  values.set("bgs256d_frozen_6x15_forward_v18",JSON.stringify({history}));
  if(!values.has("bgs_xgb_final_shoe_id_v5"))values.set("bgs_xgb_final_shoe_id_v5","bgs-isolated-shoe");
  const ctx={console:{log(){},warn(){},error(){}},BBB_BLIND_UPGRADE_MODE:"baseline",
    document:{getElementById:()=>null},
    localStorage:{getItem:k=>values.get(k)??null,setItem:(k,v)=>values.set(k,String(v)),removeItem:k=>values.delete(k)},
    fetch:async url=>{
      const name=String(url).replace(/^\.\//,"");
      if(!["physics_multitask_model.json","final_probability_model.json","cardcount_profile.json"].includes(name))throw Error("model URL not allowed");
      const payload=JSON.parse(fs.readFileSync(path.join(root,name),"utf8"));
      return {ok:true,status:200,json:async()=>payload};
    }};
  ctx.window=ctx;vm.createContext(ctx);
  for(const file of manifest.scripts){
    let source=fs.readFileSync(path.join(__dirname,file),"utf8");
    if(file==="final_probability_runtime.js"){
      // Read-only compatibility exports; no arithmetic or policy is rewritten.
      const anchor='if(typeof window!=="undefined")window.__BGS_FINAL56__=';
      if(!source.includes(anchor))throw Error("BBB export anchor mismatch");
      source=source.replace(anchor,'window.__BBB_HELPERS__={buildExtended,predictFinalProbability,physicsNoiseScore,prepareCalibration:(seq,f)=>LOCAL_CALIBRATION?.prepare(seq,f,getShoeId(),browserCalibrationTag),commitCalibration:(t,o,h)=>LOCAL_CALIBRATION?.commit(t,o,h,localCalibrationConfig())};\n'+anchor);
    }
    vm.runInContext(source,ctx,{filename:file});
  }
  const api=ctx.__BGS_FINAL56__;await api.loadModels();
  if(api.getCardcountProfile()?.display_gap!==.02)throw Error("Fixed 2pp profile unavailable");
  const status=api.getModelStatus();if(!status.physics||!status.final56)throw Error("BBB model bundles unavailable");
  const op=input.operation||"predict",helpers=ctx.__BBB_HELPERS__;
  if(op==="status")return {status,version:api.version};
  if(op==="display_direction")return {value:api.displayDirection(input.forecast),gap:.02};
  if(op==="history_vector")return {value:Array.from(api.historyVector(history))};
  if(op==="physics")return {value:api.predictPhysics(history)};
  if(op==="original7")return {value:api.buildOriginal7(history,input.core||ctx.__BGS256_CONTINUATION_TEST__.hazardChoose(history))};
  if(op==="features")return {value:helpers.buildExtended(input.core_pb,input.original7,input.physics)};
  if(op==="noise")return {value:helpers.physicsNoiseScore(input.physics,input.round_index)};
  if(op==="raw_probability")return {value:helpers.predictFinalProbability(input.bundle,input.features,input.bundle.feature_names)};
  if(op!=="predict")throw Error("unsupported operation");
  const previous=input.previous_history||[];
  const extendsPrevious=previous.length<history.length&&previous.every((x,i)=>history[i]===x);
  if(extendsPrevious&&previous.length){
    for(let i=previous.length;i<history.length;i++){
      const prefix=history.slice(0,i),observed={outcome:history[i]};
      const raw=api.applyFinalPrediction(prefix).raw_next_hand_forecast;
      const ticket=helpers.prepareCalibration(prefix,raw);
      api.settlePending(history[i],observed,prefix);
      helpers.commitCalibration(ticket,history[i],history.slice(0,i+1));
      api.recordCompletedHand(history.slice(0,i+1),observed);
    }
  }
  const isSame=previous.length===history.length&&previous.every((x,i)=>history[i]===x);
  if(!isSame&&!extendsPrevious)api.undoRuntimeHistory();
  // OCR imports are bulk history; do not invent prior predictions for missing rounds.
  api.getMacroFeatures(history);
  const prediction=api.applyFinalPrediction(history);
  if(prediction.finalProbability.error)throw Error("BBB prediction failed: "+prediction.finalProbability.error);
  api.registerPrediction(history,prediction);
  return {prediction,display_direction:api.displayDirection(prediction.next_hand_forecast),fixed_display_gap:.02,storage:Object.fromEntries(values),history,source_commit:manifest.commit};
}
let raw="";process.stdin.setEncoding("utf8");
process.stdin.on("data",chunk=>{raw+=chunk;if(raw.length>16*1024*1024){process.stderr.write("input too large");process.exit(2);}});
process.stdin.on("end",()=>run(JSON.parse(raw)).then(out=>process.stdout.write(JSON.stringify(out))).catch(e=>{process.stderr.write(String(e.message||e));process.exitCode=1;}));
