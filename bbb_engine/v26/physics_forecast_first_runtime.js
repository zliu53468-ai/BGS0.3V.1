/**
 * BBB PHYSICS_FORECAST_FIRST V2 — dynamic research signal gate
 *
 * 物理主導決策，路單僅 0.1× 診斷，物理特徵 5×。
 * B/P/T 只可反推「可能」的 8 副牌路徑，無法知道實際扣了哪些牌。
 * 因此 B/P/T-only 模式可輸出模擬 EV，但永遠不能把它標為真實正 EV 或 BET。
 * 選用真實逐局完整牌值時才可經 EV + 不確定度閘門產生 BET 研究訊號。
 */
(function(root,factory){
  "use strict";
  const quant=(root&&root.BBBQuant256Online)||
    (typeof module==="object"&&module.exports&&typeof require==="function"?
      require("./quant256_online_runtime.js"):null);
  const distribution=(root&&root.BBBShoeDistributionGuard)||
    (typeof module==="object"&&module.exports&&typeof require==="function"?
      require("./shoe_distribution_guard.js"):null);
  const markov=(root&&root.BBBMarkovResidual)||
    (typeof module==="object"&&module.exports&&typeof require==="function"?
      require("./markov_residual_runtime.js"):null);
  const api=factory(quant,distribution,markov);
  if(typeof module==="object"&&module.exports)module.exports=api;
  if(root)root.BBBPhysicsForecastFirst=api;
})(typeof window!=="undefined"?window:null,function(quant,distribution,markov){
"use strict";
if(!quant||typeof quant.OnlineLinUCB256!=="function")throw Error("missing_quant256_online_runtime");
if(!distribution||typeof distribution.evaluate!=="function")throw Error("missing_shoe_distribution_guard");
if(!markov||typeof markov.transitionStats!=="function"||
  typeof markov.shadowCalibrate!=="function")throw Error("missing_markov_residual_runtime");
const VERSION="BBB_PHYSICS_FORECAST_FIRST_V2_DYNAMIC_GATE",DECKS=8;
const BASE_EV_GATE=.005,OVERDRIVE_GATE=0,ONE_HAND_CUSHION=.035;
const INITIAL=Object.freeze([128,32,32,32,32,32,32,32,32,32]);
const BASE=Object.freeze({B:.458597,P:.446247,T:.095156});
const BASE_EV=Object.freeze({B:.95*BASE.B-BASE.P,P:BASE.P-BASE.B});
const BASE_GAP=BASE_EV.B-BASE_EV.P;
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const finite=x=>typeof x==="number"&&Number.isFinite(x);
const valid=s=>["B","P","T"].includes(s);
const sum=a=>a.reduce((n,v)=>n+v,0);
function seeded(seed){
  let x=seed>>>0||0x9E3779B9;
  return()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return(x>>>0)/4294967296;};
}
/**
 * MC 隨機種子由當前虛擬殘牌的 0–9 點數組成決定，不再額外混入
 * 歷史 B/P/T 字元。合法的結果條件式粒子更新流程保持 V2 原樣。
 */
function hashComposition(shoes){
  const rows=Array.isArray(shoes[0])?shoes:[shoes];
  let x=2166136261;
  for(const row of rows){
    for(const count of row){x^=count;x=Math.imul(x,16777619);}
    x^=0x9e3779b9;x=Math.imul(x,16777619);
  }
  return x>>>0;
}
function validateCounts(counts,{observed=false}={}){
  if(!Array.isArray(counts)||counts.length!==10||counts.some((v,i)=>
    !Number.isInteger(v)||v<0||v>INITIAL[i]))throw Error("invalid_point_counts");
  const remaining=sum(counts);
  if(remaining>416||remaining<6)throw Error("remaining_shoe_not_playable");
  if(observed&&remaining===416)throw Error("no_observed_depletion");
  return remaining;
}
/** 抽一手合法的標準百家樂第三張牌（P1,B1,P2,B2,P3?,B3?）。 */
function deal(counts,rng){
  const available=counts.slice(),total=validateCounts(counts);
  let seen=0;
  function draw(){
    let pick=Math.floor(rng()*(total-seen));
    for(let i=0;i<10;i++){
      pick-=available[i];
      if(pick<0){available[i]--;seen++;return i;}
    }
    throw Error("virtual_deal_exhausted");
  }
  const cards=[draw(),draw(),draw(),draw()];
  const initialP=(cards[0]+cards[2])%10,initialB=(cards[1]+cards[3])%10;
  let p=initialP,b=initialB,p3=null;
  if(p<8&&b<8){
    if(p<=5){p3=draw();cards.push(p3);p=(p+p3)%10;}
    const bankerDraw=p3===null?b<=5:
      b<=2?true:b===3?p3!==8:b===4?p3>=2&&p3<=7:
      b===5?p3>=4&&p3<=7:b===6?p3>=6&&p3<=7:false;
    if(bankerDraw){const b3=draw();cards.push(b3);b=(b+b3)%10;}
  }
  return{cards,outcome:b>p?"B":p>b?"P":"T",player:p,banker:b};
}
/** 按出牌順序驗證完整觀測：不能用僅 B/P/T 偽造補牌數。 */
function observedHand(cards){
  if(!Array.isArray(cards)||cards.length<4||cards.length>6||
    cards.some(x=>!Number.isInteger(x)||x<0||x>9))
    throw Error("invalid_observed_hand");
  const a=(cards[0]+cards[2])%10,b=(cards[1]+cards[3])%10;
  const natural=a>=8||b>=8;
  const drawP=!natural&&a<=5;
  const p3=drawP?cards[4]:null;
  const drawB=!natural&&(p3===null?b<=5:
    b<=2?true:b===3?p3!==8:b===4?p3>=2&&p3<=7:
    b===5?p3>=4&&p3<=7:b===6?p3>=6&&p3<=7:false);
  const expectedLength=4+Number(drawP)+Number(drawB);
  if(cards.length!==expectedLength)throw Error("observed_hand_draw_rule_mismatch");
  const p=(a+(drawP?p3:0))%10;
  const bankerThird=drawB?cards[drawP?5:4]:0;
  const banker=(b+bankerThird)%10;
  return banker>p?"B":p>banker?"P":"T";
}
function observedCounts(rows,history){
  if(!Array.isArray(rows)||!Array.isArray(history)||rows.length!==history.length)
    throw Error("observed_rows_must_cover_entire_shoe");
  const counts=INITIAL.slice();
  for(let i=0;i<rows.length;i++){
    const row=rows[i],outcome=history[i],cards=row?.observation?.cards;
    if(!row||row.outcome!==outcome||!valid(outcome)||!Array.isArray(cards))
      throw Error("observed_rows_outcome_mismatch");
    if(observedHand(cards)!==outcome)throw Error("observed_result_mismatch");
    for(const card of cards){
      counts[card]--;
      if(counts[card]<0)throw Error("observed_card_exceeds_shoe");
    }
  }
  validateCounts(counts,{observed:true});
  return counts;
}
/** Monte Carlo 只從剩餘牌組成計算 B/P/T，沒有引入牌路權重。 */
function simulate(counts,{samples=2048,seed=0x13877}={}){
  validateCounts(counts);
  if(!Number.isInteger(samples)||samples<64||samples>50000)throw Error("invalid_mc_sample_count");
  const rng=seeded(seed),wins={B:0,P:0,T:0};
  let expectedCards=0;
  for(let i=0;i<samples;i++){
    const hand=deal(counts,rng);
    wins[hand.outcome]++;expectedCards+=hand.cards.length;
  }
  const p={B:wins.B/samples,P:wins.P/samples,T:wins.T/samples};
  const evB=.95*p.B-p.P,evP=p.P-p.B;
  // 莊/閒下注逢和退回原注。SE 用每局盈虧平方的樣本估計。
  const seB=Math.sqrt(Math.max(0,(.95*.95*p.B+p.P-evB*evB)/samples));
  const seP=Math.sqrt(Math.max(0,(p.B+p.P-evP*evP)/samples));
  const gap=evB-evP;
  const seGap=Math.sqrt(Math.max(0,(1.95*1.95*p.B+4*p.P-gap*gap)/samples));
  return{samples,counts:counts.slice(),remaining:sum(counts),
    win_probabilities:p,ev_banker:evB,ev_player:evP,ev_gap:gap,
    ev_gap_deviation:gap-BASE_GAP,ev_banker_delta:evB-BASE_EV.B,
    ev_player_delta:evP-BASE_EV.P,se_banker:seB,se_player:seP,
    se_gap:seGap,draws_mean:expectedCards/samples,
    payout:{banker_win:.95,player_win:1,tie_push:true}};
}
function summarizeHypotheses(physics,_history,{samplesPerParticle=96}={}){
  if(!physics||!Array.isArray(physics.particles))throw Error("invalid_physics_particles");
  if(!Number.isInteger(samplesPerParticle)||samplesPerParticle<8||samplesPerParticle>1024)
    throw Error("invalid_shadow_sample_count");
  // Weights are the posterior mass of *hypothesized* shoes, not observed
  // ranks. Old callers without weights still get the legacy equal mixture.
  const validParticles=[],rawWeights=[];
  const provided=Array.isArray(physics.weights)&&
    physics.weights.length===physics.particles.length?physics.weights:null;
  for(let i=0;i<physics.particles.length;i++){
    const counts=physics.particles[i];
    if(!Array.isArray(counts)||sum(counts)<6)continue;
    validParticles.push(counts);
    rawWeights.push(provided?Math.max(0,physics.weights[i]):1);
  }
  const mass=sum(rawWeights),n=validParticles.length*samplesPerParticle;
  if(!n||mass<=0||physics.posteriorUnavailable)
    return{...baselineResult(),particle_support:0,simulation_exhausted:true};
  const weights=rawWeights.map(w=>w/mass);
  const stableSeed=hashComposition(validParticles);
  const estimates=[];
  for(let i=0;i<validParticles.length;i++){
    const counts=validParticles[i],rng=seeded(stableSeed^Math.imul(i+1,2654435761));
    const tally={B:0,P:0,T:0};let draws=0;
    for(let j=0;j<samplesPerParticle;j++){
      const result=deal(counts,rng);
      tally[result.outcome]++;draws+=result.cards.length;
    }
    const b=tally.B/samplesPerParticle,p=tally.P/samplesPerParticle;
    const gap=1.95*b-2*p;
    estimates.push({...tally,draws,gap,
      gap_variance:Math.max(0,1.95**2*b+4*p-gap*gap)});
  }
  const prob={
    B:estimates.reduce((v,e,i)=>v+weights[i]*e.B/samplesPerParticle,0),
    P:estimates.reduce((v,e,i)=>v+weights[i]*e.P/samplesPerParticle,0),
    T:estimates.reduce((v,e,i)=>v+weights[i]*e.T/samplesPerParticle,0)
  };
  const evB=.95*prob.B-prob.P,evP=prob.P-prob.B,gap=evB-evP;
  // Two distinct uncertainties: within-shoe MC draw noise and dispersion of
  // plausible *latent shoes*. Neither is proof of a predictive edge.
  const mcVariance=estimates.reduce((v,e,i)=>
    v+weights[i]**2*e.gap_variance/samplesPerParticle,0);
  const posteriorVariance=estimates.reduce((v,e,i)=>
    v+weights[i]*(e.gap-gap)**2,0);
  const mcStd=Math.sqrt(mcVariance),hypothesisSpread=Math.sqrt(posteriorVariance);
  return{remaining:validParticles.reduce((v,c,i)=>v+weights[i]*sum(c),0),
    samples:n,particle_support:validParticles.length,
    simulated_probability:prob,ev_banker:evB,ev_player:evP,
    ev_gap:gap,ev_gap_deviation:gap-BASE_GAP,
    se_gap:Math.sqrt(mcVariance+posteriorVariance),
    mc_se_gap:mcStd,
    posterior_sd_gap:hypothesisSpread,
    // Diagnostics only. MC interval conditions on the sampled hypothesis
    // mixture; it is NOT a calibrated confidence interval for real outcomes.
    ev_uncertainty_diagnostics:{
      likelihood_method:"rao_blackwell_third_card_given_first_four",
      mc_gap_interval_95_conditional:[gap-1.96*mcStd,gap+1.96*mcStd],
      posterior_gap_dispersion_sd:hypothesisSpread,
      likelihood_relative_se:physics.lastLikelihoodRelativeSE??null,
      rejuvenated_transitions:physics.lastRejuvenated??0,
      rejuvenation_fallbacks:physics.lastRejuvenationFallbacks??0,
      ancestor_diversity:Array.isArray(physics.ancestorIds)?
        new Set(physics.ancestorIds).size/physics.ancestorIds.length:null,
      calibrated_hit_rate:false,validated_edge:false
    },
    particle_ess:1/weights.reduce((v,w)=>v+w*w,0),
    posterior_update_ess:physics.lastESS??null,
    particle_resampled:physics.lastResampled===true,
    draws_mean:estimates.reduce((v,e,i)=>
      v+weights[i]*e.draws/samplesPerParticle,0),
    source:"hypothetical_particle_posterior",simulation_exhausted:false};
}
/**
 * Early-shoe Bayesian-style shrinkage of unobserved B/P/T particle EV.
 * This reliability heuristic is not a calibrated hit rate or observed count.
 * At 20 rounds, and for real observed cards, it is an exact V2 no-op.
 */
function earlyShoeGuard(physical,particles,rounds,{observed=false,
  independentFinalScore=false}={}){
  if(observed||rounds<1||rounds>=20||physical.simulation_exhausted||
    !physical.simulated_probability)
    return{physical,gateAddition:0,diagnostics:null};
  const p=physical.simulated_probability;
  if(["B","P","T"].some(k=>!finite(p[k])))
    return{physical,gateAddition:0,diagnostics:null};
  const n=Math.max(1,particles.size||physical.particle_support);
  const ess=clamp((physical.posterior_update_ess??n)/n,0,1);
  const activeESS=clamp((physical.particle_ess??n)/n,0,1);
  const diversity=clamp(Array.isArray(particles.ancestorIds)?
    new Set(particles.ancestorIds).size/n:1,0,1);
  const likelihoodNoise=clamp(particles.lastLikelihoodRelativeSE??0,0,1);
  const quality=clamp((.5*ess+.25*activeESS+.25*diversity)/
    (1+likelihoodNoise),0,1);
  const progress=rounds/20;
  // IMPORTANT: the web-only single-score mode ALWAYS has rounds===1.
  // Treating that as an early shoe forever shrinks 86-89% of the actual
  // score-conditioned particle deviation toward a fresh eight-deck prior.
  // In this explicit mode, use posterior quality and uncertainty, NOT
  // artificial chronological progress, to determine a bounded blend.
  // Other modes retain the exact original 1-19 / >=20 logic below.
  const mcNoise=finite(physical.mc_se_gap)?
    Math.max(0,physical.mc_se_gap):.05;
  const posteriorSpread=finite(physical.posterior_sd_gap)?
    Math.max(0,physical.posterior_sd_gap):.10;
  const mcReliability=1/(1+mcNoise/.04);
  const posteriorReliability=1/(1+posteriorSpread/.12);
  const reliability=.65*mcReliability+.35*posteriorReliability;
  const baseBlend=.12+.88*progress**1.25;
  const legacyBlend=baseBlend*(1-.25*(1-progress)*(1-quality));
  const blend=independentFinalScore?
    clamp(.28+.60*quality*reliability,.28,.82):legacyBlend;
  const prob={
    B:BASE.B+blend*(p.B-BASE.B),
    P:BASE.P+blend*(p.P-BASE.P),
    T:BASE.T+blend*(p.T-BASE.T)
  };
  const evB=.95*prob.B-prob.P,evP=prob.P-prob.B;
  const se=finite(physical.mc_se_gap)?physical.mc_se_gap:0;
  const gateAddition=(independentFinalScore?1-blend:1-progress)*
    Math.min(.01,.20*Math.max(0,se));
  const diagnostics={
    active:true,rounds,progress,blend,prior_weight:1-blend,
    quality,likelihood_relative_se:likelihoodNoise,
    ess_ratio:ess,active_ess_ratio:activeESS,ancestor_diversity:diversity,
    supplemental_gate:gateAddition,validated_hit_rate:false,
    ...(independentFinalScore?{
      mode:"independent_final_score_quality",mc_gap_se:mcNoise,
      posterior_gap_spread:posteriorSpread,
      mc_reliability:mcReliability,posterior_reliability:posteriorReliability,
      evidence_reliability:reliability,validated_accuracy_gain:false
    }:{})
  };
  return{physical:{...physical,simulated_probability:prob,
    ev_banker:evB,ev_player:evP,ev_gap:evB-evP,
    ev_gap_deviation:evB-evP-BASE_GAP,early_shoe_guard:diagnostics},
    gateAddition,diagnostics};
}

function baselineResult(){
  return{samples:0,remaining:416,particle_support:0,simulated_probability:{...BASE},
    ev_banker:BASE_EV.B,ev_player:BASE_EV.P,ev_gap:BASE_GAP,
    ev_gap_deviation:0,se_gap:null,draws_mean:null,
    source:"unobserved_baseline",simulation_exhausted:false};
}
function decisionFromEV(s,{observed=false,threshold=.015,z=1.96}={}){
  if(!finite(threshold)||threshold<0)throw Error("invalid_ev_gate");
  const direction=s.ev_banker>=s.ev_player?"B":"P";
  const chosenEV=direction==="B"?s.ev_banker:s.ev_player;
  const se=direction==="B"?s.se_banker:s.se_player;
  const enough=observed&&finite(se)&&finite(s.se_gap);
  const significant=enough&&Math.abs(s.ev_gap_deviation)>threshold&&
    Math.abs(s.ev_gap_deviation)>z*s.se_gap&&
    chosenEV-z*se>0;
  return{action:significant?"BET":"SKIP",direction:significant?direction:"Skip",
    candidate:direction,bet_allowed:significant,
    reason:!observed?"unobserved_card_composition":
      !significant?"physical_ev_gate_not_confirmed":"observed_physical_ev_confirmed",
    ev_threshold:threshold,ev_gap_deviation:s.ev_gap_deviation,
    selected_ev:chosenEV,ev_lower_bound:enough?chosenEV-z*se:null,
    evidence_source:observed?"supplied_observed_cards":"bpt_virtual_only"};
}
function weightedContext(model,physical){
  const raw=model.context;
  if(!Array.isArray(raw)||raw.length!==256)throw Error("invalid_quant256_context");
  const shoe=Array(128).fill(0),road=raw.slice(128).map(x=>clamp(x,-1,1)*.1);
  const remaining=physical.remaining;
  const phase=clamp((416-remaining)/416,0,1),p=physical.simulated_probability||physical.win_probabilities;
  // 5× 物理主體：進程、三次方、點數虛擬殘量的中心差、模型 EV 與基準偏移。
  shoe[0]=5*phase;shoe[1]=5*phase**2;shoe[2]=5*phase**3;
  shoe[3]=5*(1-phase);shoe[4]=5*(1-phase)**2;shoe[5]=5*(1-phase)**3;
  shoe[6]=5*clamp(p.B-BASE.B,-1,1);
  shoe[7]=5*clamp(p.P-BASE.P,-1,1);
  shoe[8]=5*clamp(p.T-BASE.T,-1,1);
  shoe[9]=5*clamp(physical.ev_gap_deviation,-1,1);
  shoe[10]=5*clamp(physical.ev_banker-BASE_EV.B,-1,1);
  shoe[11]=5*clamp(physical.ev_player-BASE_EV.P,-1,1);
  if(Array.isArray(physical.counts)){
    for(let i=0;i<10;i++)
      shoe[16+i]=5*(physical.counts[i]/Math.max(6,remaining)-INITIAL[i]/416);
  }
  return{vector:shoe.concat(road),shoe,road,physics_weight:5,
    road_weight:.1,road_has_direction_authority:false};
}
/**
 * 研究訊號狀態機：
 * BASE(0.005) → 上一局實際提示正確：OVERDRIVE(0)
 * BASE/OVERDRIVE → 上一局實際提示錯誤：CUSHION(0.035)
 * CUSHION → 下一個非和局若未通過門檻：BASE(0.005)
 *
 * 狀態只在 observe()/syncHistory() 的逐局結算更新；重複 predict() 不會
 * 虛增連勝或重複結算。預測錯誤是從上一局「已提示的研究方向」計算，
 * 不能偷看當前輸入開獎結果。B/P/T 盲估的高密度方向不等於 BET。
 */
class PhysicsForecastFirstEngine{
  constructor({simulationsPerParticle=96,observedSamples=12000,
    threshold=BASE_EV_GATE,lossGate=ONE_HAND_CUSHION,quantOptions={},
    independentFinalScoreMode=false}={}){
    if(!finite(threshold)||threshold<0||threshold>.5)throw Error("invalid_ev_gate");
    if(!finite(lossGate)||lossGate<0||lossGate>.5)throw Error("invalid_loss_gate");
    this.simulationsPerParticle=simulationsPerParticle;
    this.observedSamples=observedSamples;
    this.threshold=threshold;this.lossGate=lossGate;
    if(typeof independentFinalScoreMode!=="boolean")
      throw Error("invalid_independent_final_score_mode");
    this.independentFinalScoreMode=independentFinalScoreMode;
    this.quant=new quant.OnlineLinUCB256(quantOptions);
    this.reset();
  }
  get history(){return this.quant.history;}
  get trainedRounds(){return this.quant.trainedRounds;}
  reset(){
    this.quant.reset();this.observedRows=null;this.lastReport=null;
    this.scoreRows=[];
    this.phase="base";this.winStreak=0;this.lastSettlement="none";
    this.settlements={signals:0,correct:0,wrong:0,ties:0,filtered:0,turns:0};
    return this;
  }
  _gate(){return this.phase==="overdrive"?OVERDRIVE_GATE:
    this.phase==="cushion"?this.lossGate:this.threshold;}
  setObservedRounds(rows){
    if(rows===null){this.observedRows=null;this.lastReport=null;return this;}
    if(!Array.isArray(rows)||rows.some(r=>!r||!valid(r.outcome)||
      !Array.isArray(r.observation?.cards)))throw Error("invalid_observed_rows");
    // 匯入前先確認整靴完整，避免先顯示真實數據後才發現驗證失敗。
    if(rows.length!==this.history.length)throw Error("observed_rows_must_cover_entire_shoe");
    observedCounts(rows,this.history);
    this.observedRows=JSON.parse(JSON.stringify(rows));
    this.lastReport=null;return this;
  }
  /**
   * 每次真正開獎只結算一次研究方向；T 不宣稱方向輸贏。
   * 上一局若是 Cushion 且 research signal 不達 .035，只消耗下一個 B/P
   * 觀測事件就恢復 BASE，不會因和局卡住。若下一局仍有訊號，
   * 正確→OVERDRIVE，錯誤→再次一局 Cushion。
   */
  _settle(prior,outcome){
    this.lastReport=null; // 結算狀態改變立即使預測快取失效
    if(outcome==="T"){this.settlements.ties++;this.lastSettlement="tie";return;}
    this.settlements.turns++;
    const signal=prior.research_signal;
    if(!signal?.available){
      this.settlements.filtered++;this.phase="base";
      this.winStreak=0;this.lastSettlement="filtered";
      return;
    }
    this.settlements.signals++;
    if(signal.direction===outcome){
      this.settlements.correct++;
      this.winStreak++;this.phase="overdrive";this.lastSettlement="win";
    }else{
      this.settlements.wrong++;
      this.winStreak=0;this.phase="cushion";this.lastSettlement="loss";
    }
  }
  syncHistory(seq,pointScores=[]){
    if(!Array.isArray(seq)||seq.some(v=>!valid(v)))throw Error("invalid_history");
    if(!Array.isArray(pointScores)||pointScores.length>seq.length)
      throw Error("invalid_point_score_history");
    const scores=seq.map((outcome,i)=>{
      const pair=pointScores[i];
      if(pair===undefined||pair===null)return null;
      if(!pair||!Number.isInteger(pair.player)||!Number.isInteger(pair.banker)||
        pair.player<0||pair.player>9||pair.banker<0||pair.banker>9||
        (pair.banker>pair.player?"B":pair.player>pair.banker?"P":"T")!==outcome)
        throw Error("invalid_point_observation");
      return{player:pair.player,banker:pair.banker};
    });
    if(this.observedRows!==null&&this.observedRows.length!==seq.length)
      throw Error("observed_rows_must_cover_entire_shoe");
    if(this.observedRows!==null&&scores.some(Boolean))
      throw Error("point_scores_conflict_observed_rows");
    let prefix=this.history.length<=seq.length;
    if(prefix)for(let i=0;i<this.history.length;i++){
      const old=this.scoreRows[i],fresh=scores[i];
      if(this.history[i]!==seq[i]||Boolean(old)!==Boolean(fresh)||
        (old&&(old.player!==fresh.player||old.banker!==fresh.banker))){
        prefix=false;break;
      }
    }
    if(!prefix){
      // Undone or corrected score observations require causal full replay.
      // Ordinary append only updates the newest particle transition.
      const remembered=this.observedRows;
      this.reset();this.observedRows=remembered;
    }
    for(let i=this.history.length;i<seq.length;i++){
      const prior=this.predict();
      this._settle(prior,seq[i]);
      this.quant.observe(seq[i],scores[i]);
      this.scoreRows.push(scores[i]);
      this.lastReport=null;
    }
    return this.predict();
  }
  predict(){
    if(this.lastReport&&this.lastReport.history_length===this.history.length)
      return this.lastReport;
    const quantPrediction=this.quant.predict();
    const observed=this.observedRows!==null;
    const counts=observed?observedCounts(this.observedRows,this.history):null;
    const physics=observed?simulate(counts,{
      samples:this.observedSamples,seed:hashComposition(counts)^0xb15f13}):
      (this.history.length?summarizeHypotheses(this.quant.physics,this.history,{
        samplesPerParticle:this.simulationsPerParticle}):baselineResult());
    const candidate=observed?
      {...physics,simulated_probability:physics.win_probabilities,source:"observed_card_count"}:
      physics;
    const independentFinalScore=this.independentFinalScoreMode&&!observed&&
      this.history.length===1&&this.scoreRows.length===1&&Boolean(this.scoreRows[0]);
    const early=earlyShoeGuard(candidate,this.quant.physics,this.history.length,{
      observed,independentFinalScore});
    const normalized=early.physical;
    // A rolling change-point monitor does not flip or blend physical EV.
    // It only adds a small research-signal penalty for a strong observed
    // distribution shift; the original dynamic V2 phase gate is unchanged.
    const distributionGuard=distribution.evaluate(this.history,{observed});
    const dynamicGate=this._gate();
    const markovStats=markov.transitionStats(this.history);
    const markovShadow=markov.shadowCalibrate(markovStats,normalized,{observed});
    const score=Math.abs(normalized.ev_gap_deviation);
    const direction=normalized.ev_banker>=normalized.ev_player?"B":"P";
    // 與真實投注分離的「研究方向」：不把 MC 抽樣不確定性當成勝率。
    // 若物理粒子全耗盡或訊號不可計算，即使 overdrive 也不假裝可用。
    const usable=!normalized.simulation_exhausted&&finite(score)&&
      finite(normalized.ev_banker)&&finite(normalized.ev_player);
    const effectiveResearchGate=dynamicGate+early.gateAddition+
      distributionGuard.additional_gate;
    const available=usable&&score>=effectiveResearchGate;
    const researchSignal={
      action:available?"SIGNAL":"SKIP",
      direction,available,score,dynamic_gate:dynamicGate,
      effective_research_gate:effectiveResearchGate,
      distribution_guard:distributionGuard,
      markov_residual:{stats:markovStats,shadow:markovShadow},
      phase:this.phase,win_streak:this.winStreak,
      last_settlement:this.lastSettlement,
      early_shoe_guard:early.diagnostics,
      source:observed?"observed_cards":"simulated_bpt_only",
      evidence_qualified:observed,
      reason:!usable?"simulation_unavailable":
        !available?"dynamic_signal_filter":"model_direction_not_validated"
    };
    // 正式 BET 仍要求觀測到每局真實點數、正 EV 和抽樣誤差下界。
    // 動態門檻會影響真實 EV 閘門，但任何連勝都無法捏造實際牌值。
    const gate=decisionFromEV(normalized,{observed,threshold:dynamicGate});
    const action=available?gate.action:"SKIP";
    const reason=!available?"dynamic_signal_filter":gate.reason;
    const features=weightedContext(quantPrediction,normalized);
    const scoreCount=this.scoreRows.filter(Boolean).length;
    const report={version:VERSION,history_length:this.history.length,
      point_score_observation_count:scoreCount,
      score_conditioned_particles:scoreCount>0,
      score_observations_are_actual_card_ranks:false,
      action,direction:action==="BET"?direction:"Skip",bet_allowed:action==="BET",
      reason,card_count_source:normalized.source,
      observed_card_evidence:observed,card_count_gate:gate,
      research_signal:researchSignal,display_direction:direction,
      early_shoe_guard:early.diagnostics,
      shoe_distribution_guard:distributionGuard,
      markov_residual:{stats:markovStats,shadow:markovShadow},
      effective_research_gate:effectiveResearchGate,
      dynamic_gate:dynamicGate,base_ev_gate:this.threshold,
      cushion_gate:this.lossGate,gate_phase:this.phase,
      win_streak:this.winStreak,last_settlement:this.lastSettlement,
      prequential:{...this.settlements},
      signal_rate:this.settlements.turns?
        this.settlements.signals/this.settlements.turns:null,
      hit_rate:this.settlements.signals?
        this.settlements.correct/this.settlements.signals:null,
      context:features.vector,shoe_context:features.shoe,road_context:features.road,
      feature_weights:{physics:5,road:.1},
      road_has_direction_authority:false,model_road_debug:quantPrediction,
      physical_ev:normalized,ev_banker:normalized.ev_banker,ev_player:normalized.ev_player,
      ev_gap_deviation:normalized.ev_gap_deviation,ev_threshold:dynamicGate,
      baseline_ev:{...BASE_EV},hypothetical_candidate:direction,
      proxy_only:!observed,validated_edge:false,
      notice:observed?
        "已輸入完整實際點數；仍需確認 MC 抽樣誤差與真實正 EV，連勝不保證下局贏。":
        scoreCount>0?
          "當局最終點數已用於條件式粒子更新，仍無法唯一辨識實際牌張，不能當作已驗證正 EV。":
          "高頻莊/閒僅為虛擬殘牌的研究方向，不是已驗證的正 EV 或投注指令。"};
    this.lastReport=report;return report;
  }
}
return{PhysicsForecastFirstEngine,observedCounts,observedHand,
  deal,simulate,summarizeHypotheses,earlyShoeGuard,
  decisionFromEV,weightedContext,
  BASE,BASE_EV,INITIAL,DECKS,VERSION,BASE_EV_GATE,OVERDRIVE_GATE,ONE_HAND_CUSHION};
});