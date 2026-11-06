"use strict";
/* Transport only. Core + untrained model are byte-identical BBB main blobs. */
const readline=require("node:readline");
global.self=global;
const core=require("./soft_shoe_decision_runtime.js");
require("./soft_shoe_model_bundle.js");
const SOURCE_COMMIT="ededd462ac68932c8a44d91f2e3fc28280c9aa88";
const CORE_BLOB="7d299113cdd9cade5739c9a3e71be3ee2d80b1cd";
function evaluate(input){
 if(!input||!Array.isArray(input.history)||
    input.history.some(x=>!["B","P","T"].includes(x)))
  throw Error("invalid_bpt_history");
 // Do not use LINE UID as a model seed: identical BPT input matches BBB web.
 const report=core.run(input.history,{},global.BBB_SOFT_SHOE_XGB_MODEL);
 if(report.version!==core.VERSION)throw Error("unexpected_bbb_core_version");
 if(report.action!=="Skip"||report.validated_edge!==false||
    report.bankroll_fraction!==0)throw Error("unverified_betting_gate_changed");
 return {source_commit:SOURCE_COMMIT,source_blob:CORE_BLOB,report};
}
const rl=readline.createInterface({input:process.stdin,crlfDelay:Infinity});
rl.on("line",line=>{
 let response;
 try{response={ok:true,result:evaluate(JSON.parse(line))};}
 catch(e){response={ok:false,error:String(e?.message||e).slice(0,250)};}
 process.stdout.write(JSON.stringify(response)+"\n");
});
module.exports={evaluate,SOURCE_COMMIT,CORE_BLOB};
