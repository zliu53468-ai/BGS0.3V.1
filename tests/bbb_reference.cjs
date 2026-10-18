"use strict";
// Independent browser-style reference: load the unchanged BBB files without
// the Python worker's compatibility exports or result mapping.
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const root=path.resolve(__dirname,".."),m=require("../bbb_engine/manifest.json");
async function create(){
  const store=new Map([["bgs_xgb_final_shoe_id_v5","bgs-isolated-shoe"]]);
  const ctx={console,document:{getElementById:()=>null},
    localStorage:{getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)},
    fetch:async n=>({ok:true,json:async()=>JSON.parse(fs.readFileSync(path.join(root,n),"utf8"))})};
  ctx.window=ctx;vm.createContext(ctx);
  for(const file of m.scripts)vm.runInContext(fs.readFileSync(path.join(root,"bbb_engine",file),"utf8"),ctx);
  const api=ctx.__BGS_FINAL56__;await api.loadModels();return {api,store};
}
function fields(p){const r=p.finalProbability;return {
  direction:p.direction,confidence:p.confidence,final_p_b:r.finalPB,raw_p_b:r.rawPB,
  ev_banker:p.ev_banker,ev_player:p.ev_player,physics_48d:r.physics,features_57d:r.extended,
  smoothing_alpha:r.smoothingAlpha,execution_order:r.execution_order,
  particle_count:r.particleDiagnostics.particle_count,source_commit:m.commit,
};}
(async()=>{
  const standalone=[];
  for(const h of ["B","T","BBBBBBBBBBBB","BPBPBPBPBPBP","BPPBTBBPPTBBPPBTBP","BBPPBBPPBBPPBBPPBBPP","BPT".repeat(17),"BPPBBPBPPBBPBT".repeat(5)]){
    const {api}=await create();standalone.push({history:h,expected:fields(api.applyFinalPrediction(h.split("")))});
  }
  const {api,store}=await create(),sequence=[];let previous=[];
  for(const h of ["BPPBTBBPPTBBPPBTBP","BPPBTBBPPTBBPPBTBPB","BPPBTBBPPTBBPPBTBPBT","BPPBTBBPPTBBPPBTBPBTP"]){
    const history=h.split("");store.set("bgs256d_frozen_6x15_forward_v18",JSON.stringify({history}));
    if(previous.length)api.settlePending(history[previous.length],{outcome:history[previous.length]},previous);
    api.getMacroFeatures(history);const p=api.applyFinalPrediction(history);api.registerPrediction(history,p);
    sequence.push({history:h,expected:fields(p)});previous=history;
  }
  process.stdout.write(JSON.stringify({source_commit:m.commit,standalone,sequence}));
})().catch(e=>{process.stderr.write(String(e.stack||e));process.exitCode=1;});
