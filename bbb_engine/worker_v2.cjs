"use strict";
/*
 * Prediction-only, long-lived Node bridge. The three imported BBB JS files
 * are byte-for-byte sourced from BBB main 7f49802f and NEVER modified here.
 * No access to LINE tokens, sessions, trial storage, OCR or environment secrets.
 */
const readline=require("node:readline");
const {PhysicsForecastFirstEngine,VERSION}=require("./physics_forecast_first_runtime.js");
const SOURCE_COMMIT="7f49802f6f514eb7bf44aa3a61bdf4b1ffcc9920";
const CONFIG=Object.freeze({threshold:.005,lossGate:.035,simulationsPerParticle:160,
  observedSamples:12000,quantOptions:{lambda:.82,alpha:.15,particles:256}});
const scopes=new Map(),MAX_SCOPES=24;
function run(input){
  const seq=input.history;
  if(!Array.isArray(seq)||seq.length>500||seq.some(v=>!["B","P","T"].includes(v)))
    throw Error("invalid_bpt_history");
  const scope=String(input.scope||"anonymous");
  if(scope.length>256)throw Error("invalid_scope");
  let model=scopes.get(scope);
  if(!model)model=new PhysicsForecastFirstEngine(CONFIG);
  else scopes.delete(scope);
  scopes.set(scope,model);
  while(scopes.size>MAX_SCOPES)scopes.delete(scopes.keys().next().value);
  const report=model.syncHistory(seq);
  const physical=report.physical_ev;
  return{
    source_commit:SOURCE_COMMIT,version:VERSION,history_length:report.history_length,
    config:{N:model.quant.physics.size,L:384,M:model.simulationsPerParticle,
      ess_resampling_ratio:.5,physics_weight:5,road_weight:.1},
    action:report.action,bet_allowed:report.bet_allowed,
    display_direction:report.display_direction,
    research_signal:report.research_signal,
    gate_phase:report.gate_phase,win_streak:report.win_streak,
    last_settlement:report.last_settlement,hit_rate:report.hit_rate,
    signal_rate:report.signal_rate,dynamic_gate:report.dynamic_gate,
    effective_research_gate:report.effective_research_gate,
    ev_banker:report.ev_banker,ev_player:report.ev_player,
    ev_gap_deviation:report.ev_gap_deviation,
    physical_ev:{
      source:physical.source,simulated_probability:physical.simulated_probability,
      remaining:physical.remaining,samples:physical.samples,
      particle_support:physical.particle_support,
      se_gap:physical.se_gap,mc_se_gap:physical.mc_se_gap,
      posterior_sd_gap:physical.posterior_sd_gap,
      posterior_update_ess:physical.posterior_update_ess,
      simulation_exhausted:physical.simulation_exhausted
    },
    proxy_only:report.proxy_only,validated_edge:report.validated_edge
  };
}
const rl=readline.createInterface({input:process.stdin,crlfDelay:Infinity});
rl.on("line",line=>{
  let payload;
  try{payload={ok:true,result:run(JSON.parse(line));}}
  catch(error){payload={ok:false,error:String(error?.message||error).slice(0,200)};}
  process.stdout.write(JSON.stringify(payload)+"\n");
});
