"use strict";
/**
 * BGS LINE final-score transport adapter to byte-for-byte BBB V2.6 source.
 * The same fresh PhysicsForecastFirstEngine(CONFIG) and syncHistory as
 * BBB's quant256_online_panel.js are used for every input pair.
 * No session, UID, LINE token, old shoe or OCR data enters the model.
 */
const readline=require("node:readline");
const {PhysicsForecastFirstEngine,VERSION}=require("./v26/physics_forecast_first_runtime.js");
const SOURCE_COMMIT="a9e25b42d003f1a721007d6ae68127082ab86a29";
const CONFIG=Object.freeze({
  threshold:.005,lossGate:.035,simulationsPerParticle:160,
  observedSamples:12000,quantOptions:{lambda:.82,alpha:.15,particles:256},
  independentFinalScoreMode:true
});
function validScore(v){return Number.isInteger(v)&&v>=0&&v<=9;}
function run(payload){
  if(!payload||!validScore(payload.player)||!validScore(payload.banker))
    throw Error("invalid_final_score_pair");
  const {player,banker}=payload;
  const outcome=banker>player?"B":player>banker?"P":"T";
  // No long-lived model cache: true BBB browser-independent single-score mode.
  const fresh=new PhysicsForecastFirstEngine(CONFIG);
  const report=fresh.syncHistory([outcome],[{player,banker}]);
  if(report.history_length!==1||report.point_score_observation_count!==1||
     fresh.quant.physics.size!==256||report.physical_ev.samples!==40960)
    throw Error("BBB_V26_independent_score_contract_mismatch");
  if(report.action!=="SKIP"||report.bet_allowed!==false)
    throw Error("unverified_final_score_must_not_be_BET");
  if(report.early_shoe_guard?.mode!=="independent_final_score_quality")
    throw Error("missing_independent_early_shoe_guard");
  const evB=report.ev_banker,evP=report.ev_player;
  if(!Number.isFinite(evB)||!Number.isFinite(evP)||
     report.physical_ev.simulation_exhausted)
    throw Error("physical_ev_unavailable");
  const direction=evB>=evP?"B":"P";
  return {
    source_commit:SOURCE_COMMIT,version:VERSION,
    outcome,player,banker,direction,display_direction:direction,
    history_length:report.history_length,
    point_score_observation_count:report.point_score_observation_count,
    config:{N:fresh.quant.physics.size,L:384,M:fresh.simulationsPerParticle,
      ess_resampling_ratio:.5,physics_weight:5,road_weight:.1,
      independentFinalScoreMode:true},
    ev_banker:evB,ev_player:evP,ev_gap:report.ev_gap,
    ev_gap_deviation:report.ev_gap_deviation,
    early_shoe_guard:report.early_shoe_guard,
    research_signal:report.research_signal,
    dynamic_gate:report.dynamic_gate,
    physical_ev:{
      simulated_probability:report.physical_ev.simulated_probability,
      samples:report.physical_ev.samples,
      posterior_update_ess:report.physical_ev.posterior_update_ess,
      mc_se_gap:report.physical_ev.mc_se_gap,
      posterior_sd_gap:report.physical_ev.posterior_sd_gap
    },
    action:report.action,bet_allowed:false,proxy_only:true,validated_edge:false,
    final_scores_are_observed_ranks:false
  };
}
const rl=readline.createInterface({input:process.stdin,crlfDelay:Infinity});
rl.on("line",line=>{
  let response;
  try{response={ok:true,result:run(JSON.parse(line))};}
  catch(err){response={ok:false,error:String(err?.message||err).slice(0,200)};}
  process.stdout.write(JSON.stringify(response)+"\n");
});
