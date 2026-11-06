"use strict";
// Transport adapter only: the vendored BBB V4 modules are unchanged.
const readline=require("node:readline");
const {StablePhysicalPointV4,VERSION,USER_ENTRY}=require("./v4/stability_runtime.js");
const {CONFIG}=require("./v4/pf_eor_v4_runtime.js");
const SOURCE_COMMIT="aec8124dd01d98b2dc2a4823a5304823a6f9d06b";
function run(payload){
 if(!payload||!Number.isInteger(payload.player)||payload.player<0||payload.player>9||
    !Number.isInteger(payload.banker)||payload.banker<0||payload.banker>9)
  throw Error("invalid_final_score_pair");
 const options=payload.seed===undefined?{}:{seed:payload.seed};
 const report=new StablePhysicalPointV4().analyze(payload.player,payload.banker,options);
 const direction=report.v4.direction;
 return {...report,source_commit:SOURCE_COMMIT,version:VERSION,
  player:payload.player,banker:payload.banker,
  outcome:payload.banker>payload.player?"B":payload.player>payload.banker?"P":"T",
  direction,display_direction:direction,
  raw_direction:report.ev_banker>=report.ev_player?"B":"P",
  display_signal:{...report.v4,threshold:USER_ENTRY.minAbsoluteEVGap,
   gap:Math.min(report.v4.user_smoothed_ev_gap,report.v4.user_raw_ev_gap)},
  config:CONFIG,early_shoe_guard:report.quality,
  action:"SKIP",bet_allowed:false,proxy_only:true,validated_edge:false,
  final_scores_are_observed_ranks:false};
}
module.exports={run,SOURCE_COMMIT};
if(require.main===module){
 const rl=readline.createInterface({input:process.stdin,crlfDelay:Infinity});
 rl.on("line",line=>{
  let response;
  try{response={ok:true,result:run(JSON.parse(line))};}
  catch(err){response={ok:false,error:String(err?.message||err).slice(0,200)};}
  process.stdout.write(JSON.stringify(response)+"\n");
 });
}
