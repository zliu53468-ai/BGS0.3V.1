"use strict";
const assert=require("node:assert/strict");
const {run}=require("../bbb_engine/worker_v26.cjs");
const {StablePhysicalPointV4,USER_ENTRY}=require("../bbb_engine/v4/stability_runtime.js");
const base=require("../bbb_engine/v4/pf_eor_v4_runtime.js");
assert.deepEqual(USER_ENTRY,{minEstimatorSE:.0055,minAbsoluteEVGap:0,minBatchAgreement:.55});
for(const [player,banker,seed] of [[0,0,0],[5,3,0],[8,4,42],[9,9,123],[1,8,77],[6,7,1234]]){
 const before=new StablePhysicalPointV4().analyze(player,banker,{seed});
 const result=run({player,banker,seed,history:["B","P"],user_id:"ignored"});
 assert.deepEqual(result.v4,before.v4);
 assert.deepEqual(result.simulated_probability,before.simulated_probability);
 assert.deepEqual(result.quality,before.quality);
 assert.deepEqual(result.sampling_stability,before.sampling_stability);
 assert.deepEqual(result.original_pf_ev,before.original_pf_ev);
 assert.deepEqual(result.config,base.CONFIG);
 assert.equal(result.direction,before.v4.direction);
 assert.equal(result.physical_ev.samples,491520);
 assert.equal(result.sampling_stability.batches,5);
 assert.equal(result.deplete_mc.samples,327680);
 assert.equal(result.model_posterior.sampled_compositions,960);
 assert.equal(result.source_commit,"1399cd948449fa4b8d2ddf74a140d539dc2be3cc");
 assert.equal(result.action,"SKIP");
 assert.equal(result.bet_allowed,false);
}
const first=run({player:5,banker:3,seed:42});
run({player:1,banker:7,seed:12});
assert.deepEqual(run({player:5,banker:3,seed:42}),first);
assert.throws(()=>run({player:10,banker:0}));
console.log("Exact BBB V4 parity passed: 6 seeded score pairs, replay isolation and invalid input");
