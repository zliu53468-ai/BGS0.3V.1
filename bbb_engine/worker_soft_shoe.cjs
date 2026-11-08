"use strict";
/* Transport only. Core + untrained model are byte-identical BBB main blobs. */
const readline=require("node:readline");
global.self=global;
const core=require("./soft_shoe_decision_runtime.js");
require("./soft_shoe_model_bundle.js");
const frozen=require("./soft_shoe_frozen_lstm_runtime.js");
require("./soft_shoe_frozen_model_bundle.js");
const SOURCE_COMMIT="7b33490f6a3f69cf077b2954952e6ff55af3904e";
const CORE_BLOB="b56a3bd5516434bcf47da1df78d9d6468d793cd1";
function evaluate(input){
 if(!input||!Array.isArray(input.history)||
    input.history.some(x=>!["B","P","T"].includes(x)))
  throw Error("invalid_bpt_history");
 // Do not use LINE UID as a model seed: identical BPT input matches BBB web.
 const report=core.run(input.history,{},global.BBB_SOFT_SHOE_XGB_MODEL,
  frozen,global.BBB_SOFT_SHOE_LSTM_MODEL);
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
