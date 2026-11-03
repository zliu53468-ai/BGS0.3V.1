/**
 * BBB 256D Online LinUCB V3 — hyper-sensitive high-frequency experimental B/P/T engine.
 * 本模組不讀取 A~K，也不把盲估牌張當真實算牌資料。
 * 128D 牌靴 + 128D 牌路固定介面，內部只訓練經挑選的 36 個非零特徵，
 * 其餘維度為 0；活躍維度使用完整正定 Gram 矩陣，而非凍結係數。
 */
(function(root,factory){
  "use strict";
  const api=factory();
  if(typeof module==="object"&&module.exports)module.exports=api;
  if(root)root.BBBQuant256Online=api;
})(typeof window!=="undefined"?window:null,function(){
"use strict";
const DIM=256,SHOE=128,DECK=416,BASE_POINTS=[128,32,32,32,32,32,32,32,32,32];
const ACTIVE=[...Array.from({length:12},(_,i)=>i),...Array.from({length:24},(_,i)=>128+i)];
const K=ACTIVE.length,VERSION="BBB_QUANT256_ONLINE_V3_HYPER_MICRO";
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const finite=x=>typeof x==="number"&&Number.isFinite(x);
const valid=s=>s==="B"||s==="P"||s==="T";
const bp=h=>h.filter(v=>v!=="T");
const safeRatio=(n,d,f=0)=>d?n/d:f;
function windowStats(history,n){
  const a=bp(history).slice(-n),count=a.length;
  if(!count)return{count:0,banker:.5,turn:.5,regularity:.5,entropy:1,run:0};
  let turns=0;
  for(let i=1;i<count;i++)if(a[i]!==a[i-1])turns++;
  const turn=count>=2?turns/(count-1):.5;
  const entropy=turn===0||turn===1?0:-turn*Math.log2(turn)-(1-turn)*Math.log2(1-turn);
  let run=1;for(let i=count-1;i>0&&a[i]===a[i-1];i--)run++;
  return{count,banker:a.filter(x=>x==="B").length/count,turn,regularity:Math.abs(2*turn-1),entropy,run};
}
function seededRng(seed){
  let x=seed>>>0||0x6d2b79f5;
  return function(){x^=x<<13;x^=x>>>17;x^=x<<5;return(x>>>0)/4294967296;};
}
/** 正確的百家樂第三張牌規則；每組虛擬粒子保存 0~9 點的假設殘牌。 */
function sampleHand(counts,rng){
  const c=counts.slice(),total=c.reduce((a,b)=>a+b,0);
  if(total<6)return null;
  let cards=0;
  const draw=()=>{
    let r=Math.floor(rng()*(total-cards)),idx=9;
    for(let i=0;i<10;i++){r-=c[i];if(r<0){idx=i;break;}}
    if(c[idx]<=0)throw Error("virtual_deck_empty");
    c[idx]--;cards++;return idx;
  };
  const p1=draw(),b1=draw(),p2=draw(),b2=draw();
  let p=(p1+p2)%10,b=(b1+b2)%10;
  if(p<8&&b<8){
    let p3=null;
    if(p<=5){p3=draw();p=(p+p3)%10;}
    let bankerDraw=false;
    if(p3===null)bankerDraw=b<=5;
    else if(b<=2)bankerDraw=true;
    else if(b===3)bankerDraw=p3!==8;
    else if(b===4)bankerDraw=p3>=2&&p3<=7;
    else if(b===5)bankerDraw=p3>=4&&p3<=7;
    else if(b===6)bankerDraw=p3>=6&&p3<=7;
    if(bankerDraw)b=(b+draw())%10;
  }
  return{counts:c,draws:cards,result:b>p?"B":p>b?"P":"T"};
}
/**
 * Rao-Blackwellized B/P/T likelihood proposal.
 *
 * Sample the first four cards without replacement, then enumerate all legal
 * third-card branches under the real baccarat table rules. The returned
 * `weight` is P(observed B/P/T | sampled first four), NOT a fabricated card
 * observation. `hand` is one exact conditional draw among those branches.
 *
 * This integrates third-card noise analytically while preserving the fixed
 * 256 independent four-card proposals per particle and correct likelihood
 * weighting. It is not an exact full-shoe outcome likelihood.
 */
function validPointObservation(outcome,points){
  if(points===null||points===undefined)return true;
  return Boolean(points&&Number.isInteger(points.player)&&Number.isInteger(points.banker)&&
    points.player>=0&&points.player<=9&&points.banker>=0&&points.banker<=9&&
    (points.banker>points.player?"B":points.player>points.banker?"P":"T")===outcome);
}
function conditionalHandProposal(counts,rng,outcome,points=null){
  if(!valid(outcome))throw Error("invalid_outcome");
  if(!validPointObservation(outcome,points))throw Error("invalid_point_observation");
  const c=counts.slice(),total=c.reduce((a,b)=>a+b,0);
  if(total<6)return{weight:0,hand:null};
  let drawn=0;
  function draw(){
    let r=Math.floor(rng()*(total-drawn)),idx=9;
    for(let j=0;j<10;j++){r-=c[j];if(r<0){idx=j;break;}}
    if(c[idx]<=0)throw Error("virtual_deck_empty");
    c[idx]--;drawn++;return idx;
  }
  const p1=draw(),b1=draw(),p2=draw(),b2=draw();
  const p=(p1+p2)%10,b=(b1+b2)%10;
  let mass=0,selectedP3=-1,selectedB3=-1;
  const offer=(prob,finalP,finalB,p3=-1,b3=-1)=>{
    if(!(prob>0)||((finalB>finalP?"B":finalP>finalB?"P":"T")!==outcome)||
      (points!==null&&(finalP!==points.player||finalB!==points.banker)))return;
    mass+=prob;
    // Streaming weighted reservoir is exact conditional on these first four.
    if(rng()*mass<prob){selectedP3=p3;selectedB3=b3;}
  };
  const mustBankerDraw=(banker,p3)=>p3<0?banker<=5:
    banker<=2?true:banker===3?p3!==8:banker===4?p3>=2&&p3<=7:
    banker===5?p3>=4&&p3<=7:banker===6?p3>=6&&p3<=7:false;
  if(p>=8||b>=8)offer(1,p,b);
  else if(p>5){
    if(!mustBankerDraw(b,-1))offer(1,p,b);
    else for(let k=0;k<10;k++){
      if(c[k]>0)offer(c[k]/(total-4),p,(b+k)%10,-1,k);
    }
  }else{
    const remaining=total-4;
    for(let p3=0;p3<10;p3++){
      if(c[p3]<=0)continue;
      const p3mass=c[p3]/remaining,finalP=(p+p3)%10;
      if(!mustBankerDraw(b,p3)){offer(p3mass,finalP,b,p3);continue;}
      for(let b3=0;b3<10;b3++){
        const available=c[b3]-(b3===p3?1:0);
        if(available>0)
          offer(p3mass*available/(remaining-1),finalP,(b+b3)%10,p3,b3);
      }
    }
  }
  if(!(mass>0))return{weight:0,hand:null};
  if(selectedP3>=0)c[selectedP3]--;
  if(selectedB3>=0)c[selectedB3]--;
  return{weight:mass,hand:{counts:c,draws:4+
    Number(selectedP3>=0)+Number(selectedB3>=0),result:outcome}};
}
/**
 * After ESS resampling, repeated descendants of the same parent can receive
 * fresh *conditional transitions* from that parent's PRE-OBSERVATION deck.
 * Rejection with acceptance probability P(outcome | first four) is necessary:
 * accepting every nonzero proposal would bias the posterior.
 * Never mutate a residual shoe's point counts directly.
 */
function resampleConditionalTransition(parent,rng,outcome,points=null){
  // Exact-score events are rarer than outcome-only events. More independent
  // proposals reduce fallback without altering any acceptance probability.
  for(let attempt=0;attempt<(points===null?32:192);attempt++){
    const trial=conditionalHandProposal(parent,rng,outcome,points);
    if(trial.weight>0&&rng()<trial.weight)return trial.hand;
  }
  return null;
}
/**
 * 不可辨識性聲明：B/P/T 不提供實際 4/5/6 張或點數。
 * 這裡僅從合法發牌模擬抽取「符合已知結果的可能路徑」做 latent ensemble；
 * 它不是對真實牌靴的測量，亦不代表已知任何 A~K 殘量。
 */
class BlindPhysicsEstimator{
  constructor({particles=96,seed=0xBBB256}={}){
    this.size=clamp(Math.floor(particles)||96,8,96);
    this.seed=seed>>>0;this.reset();
  }
  reset(){
    this.rng=seededRng(this.seed);
    this.particles=Array.from({length:this.size},()=>BASE_POINTS.slice());
    this.weights=Array(this.size).fill(1/this.size);
    this.ancestorIds=Array.from({length:this.size},(_,i)=>i);
    this.lastLikelihoodRelativeSE=0;
    this.lastESS=this.size;this.lastResampled=false;
    this.lastRejuvenated=0;this.lastRejuvenationFallbacks=0;
    this.posteriorUnavailable=false;
    this.lastPointConditioned=false;
    this.pointConditionedRounds=0;
    this.drawHistory=[];this.densityHistory=[];this.lastAcceptance=0;return this;
  }
  observe(outcome,points=null){
    if(!valid(outcome))throw Error("invalid_outcome");
    if(!validPointObservation(outcome,points))throw Error("invalid_point_observation");
    const pointEvidence=points===null?null:{player:points.player,banker:points.banker};
    const attempts=256,next=new Array(this.size),updated=new Array(this.size);
    const priorParticles=this.particles;
    let accepted=0,sum=0,likelihood=0,likelihoodVariance=0;
    for(let i=0;i<this.size;i++){
      const original=priorParticles[i];let chosen=null,likelihoodMass=0,squaredMass=0;
      for(let tries=0;tries<attempts;tries++){
        const trial=conditionalHandProposal(original,this.rng,outcome,pointEvidence);
        const mass=trial.weight;
        likelihoodMass+=mass;squaredMass+=mass*mass;
        // Pick a legal conditional hand with probability proportional to its
        // Rao-Blackwellized first-four evidence contribution.
        if(mass>0&&this.rng()*likelihoodMass<mass)chosen=trial.hand;
      }
      next[i]=chosen?chosen.counts:original.slice();
      const pHat=likelihoodMass/attempts,priorWeight=this.weights[i];
      likelihood+=priorWeight*pHat;
      // Variance of the RB likelihood mean (not Bernoulli variance).
      const varMean=Math.max(0,(squaredMass-attempts*pHat*pHat)/
        (attempts*(attempts-1)));
      likelihoodVariance+=priorWeight*priorWeight*varMean;
      updated[i]=chosen?priorWeight*pHat:0;
      if(chosen){sum+=chosen.draws;accepted++;}
    }
    const evidence=updated.reduce((a,b)=>a+b,0);
    this.lastLikelihoodRelativeSE=likelihood>0?
      Math.sqrt(likelihoodVariance)/likelihood:1;
    this.lastAcceptance=accepted/this.size;
    this.lastResampled=false;
    this.lastRejuvenated=0;this.lastRejuvenationFallbacks=0;
    if(evidence>0){
      this.particles=next;
      this.weights=updated.map(w=>w/evidence);
      this.lastESS=1/this.weights.reduce((acc,w)=>acc+w*w,0);
      this.posteriorUnavailable=false;
      if(this.lastESS<this.size*.5){
        const step=1/this.size,start=this.rng()*step;
        const resampled=[],ancestors=[],selectedParents=new Set();
        let index=0,cumulative=this.weights[0];
        for(let j=0;j<this.size;j++){
          const u=start+j*step;
          while(u>cumulative&&index<this.size-1)
            cumulative+=this.weights[++index];
          // Regenerate only DUPLICATE transitions, never arbitrary card-rank
          // mutations. The accepted transition is conditional on the same
          // observed outcome and the same pre-observation parent.
          let particle=null;
          if(selectedParents.has(index)){
            particle=resampleConditionalTransition(priorParticles[index],this.rng,outcome,pointEvidence);
            if(particle)this.lastRejuvenated++;
            else this.lastRejuvenationFallbacks++;
          }
          resampled.push(particle?particle.counts:this.particles[index].slice());
          ancestors.push(this.ancestorIds[index]);
          selectedParents.add(index);
        }
        this.particles=resampled;
        this.ancestorIds=ancestors;
        this.weights.fill(step);
        this.lastResampled=true;
      }
    }else{
      this.lastESS=0;
      this.posteriorUnavailable=true;
    }
    this.lastPointConditioned=pointEvidence!==null;
    if(pointEvidence!==null)this.pointConditionedRounds++;
    this.drawHistory.push(accepted?sum/accepted:0);
    const snapshot=this.snapshot();
    this.densityHistory.push({high:snapshot.virtual_high_ratio,low:snapshot.virtual_low_ratio});
    return snapshot;
  }
  snapshot(){
    const mean=BASE_POINTS.map((_,k)=>this.particles.reduce(
      (s,p,i)=>s+this.weights[i]*p[k],0));
    const remain=mean.reduce((s,v)=>s+v,0);
    const high=mean[7]+mean[8]+mean[9],low=mean[1]+mean[2]+mean[3]+mean[4];
    const highBase=96/416,lowBase=128/416;
    const highRatio=remain?high/remain:highBase,lowRatio=remain?low/remain:lowBase;
    const highs=this.particles.map(p=>(p[7]+p[8]+p[9])/Math.max(1,p.reduce((s,v)=>s+v,0)));
    const variance=highs.reduce((s,v,i)=>s+this.weights[i]*(v-highRatio)**2,0);
    const recent=this.drawHistory.slice(-8).filter(x=>x>0);
    const avg=(values,n,fallback)=>{
      const tail=values.slice(-n);return tail.length?
        tail.reduce((sum,v)=>sum+v,0)/tail.length:fallback;
    };
    // 僅使用近 3 vs 8 次虛擬模擬的相對差分，不輸出累積殘牌量給模型。
    const densityHigh=this.densityHistory.map(x=>x.high),
      lows=this.densityHistory.map(x=>x.low),draws=this.drawHistory.filter(v=>v>0);
    const highMomentum=avg(densityHigh,3,highRatio)-avg(densityHigh,8,highRatio);
    const lowMomentum=avg(lows,3,lowRatio)-avg(lows,8,lowRatio);
    const drawMomentum=(avg(draws,3,4.94)-avg(draws,8,4.94))/6;
    return{
      source:"simulated_unobserved",estimated_only:true,observed_cards:0,
      virtual_remaining:remain,virtual_high_ratio:highRatio,virtual_low_ratio:lowRatio,
      virtual_high_delta:highRatio-highBase,virtual_low_delta:lowRatio-lowBase,
      virtual_high_momentum:highMomentum,virtual_low_momentum:lowMomentum,
      virtual_draw_momentum:drawMomentum,
      virtual_high_std:Math.sqrt(variance),conditional_acceptance:this.lastAcceptance,
      particle_ess:this.lastESS,
      particle_active_ess:1/this.weights.reduce((v,w)=>v+w*w,0),
      particle_count:this.size,
      particle_ancestor_diversity:new Set(this.ancestorIds).size/this.size,
      likelihood_relative_se:this.lastLikelihoodRelativeSE,
      posterior_weights_active:!this.lastResampled,
      point_conditioned_last_round:this.lastPointConditioned,
      point_conditioned_rounds:this.pointConditionedRounds,
      point_scores_are_not_observed_card_ranks:true,
      resampled:this.lastResampled,posterior_unavailable:this.posteriorUnavailable,
      likelihood_method:"rao_blackwell_third_card_given_first_four",
      likelihood_proposals_per_particle:256,
      conditional_rejuvenation_moves:this.lastRejuvenated,
      conditional_rejuvenation_fallbacks:this.lastRejuvenationFallbacks,
      expected_draws_recent:recent.length?recent.reduce((s,v)=>s+v,0)/recent.length:4.94,
      actual_draw_count_known:false,actual_remaining_known:false
    };
  }
}
function context256(history,physics){
  if(!Array.isArray(history)||history.some(v=>!valid(v)))throw Error("invalid_history");
  const shoe=Array(SHOE).fill(0),road=Array(SHOE).fill(0);
  const s2=windowStats(history,2),s5=windowStats(history,5),
    s8=windowStats(history,8),s16=windowStats(history,16);
  const nonTie=bp(history),tieRatio=n=>{
    const tail=history.slice(-n);return tail.length?
      tail.filter(x=>x==="T").length/tail.length:0;
  };
  const sim=physics.snapshot(),last=nonTie.at(-1)||"";
  const turn25=s2.turn-s5.turn,regular25=s2.regularity-s5.regularity;
  const entropy25=s2.entropy-s5.entropy;
  // 固定 128D 牌靴/進程槽：只用可得的短窗相對差與虛擬模擬動量。
  // 不將盲估的高低張殘牌當成實測事實，也不加入單調累積局數。
  shoe[0]=clamp(s2.banker-s5.banker,-1,1);
  shoe[1]=clamp(s5.banker-s8.banker,-1,1);
  shoe[2]=clamp(tieRatio(2)-tieRatio(5),-1,1);
  shoe[3]=clamp(tieRatio(5)-tieRatio(8),-1,1);
  shoe[4]=clamp(sim.virtual_high_momentum*2,-.2,.2);
  shoe[5]=clamp(sim.virtual_low_momentum*2,-.2,.2);
  shoe[6]=clamp(sim.virtual_high_momentum-sim.virtual_low_momentum,-.2,.2);
  shoe[7]=clamp(sim.virtual_draw_momentum,-.2,.2);
  shoe[8]=0; // 盲抽不提供真實剩餘牌或預測的確定性
  shoe[9]=clamp(turn25,-1,1);
  shoe[10]=clamp(s5.entropy-s8.entropy,-1,1);
  shoe[11]=clamp(entropy25,-1,1);

  // 後 128D：核心改為 2 vs 5 局。所有有效視窗 <=16 局（無 32+ 長窗）。
  // Entropy 在只有兩筆非和局時常為 0，它仍是變化檢測而非預知未來。
  road[0]=last==="B"?.5:last==="P"?-.5:0;
  road[1]=clamp(s2.run/Math.max(1,s2.count)-s5.run/Math.max(1,s5.count),-1,1);
  road[2]=clamp(s2.turn-.5,-.5,.5);
  road[3]=clamp(s5.turn-.5,-.5,.5);
  road[4]=clamp(turn25,-1,1);
  road[5]=clamp((s2.banker-.5)*(s2.regularity-.5),-1,1);
  road[6]=clamp((s5.banker-.5)*(s5.regularity-.5),-1,1);
  road[7]=clamp(regular25,-1,1);
  road[8]=clamp((s2.banker-.5)*(s2.entropy-.5),-1,1);
  road[9]=clamp(s5.entropy-s8.entropy,-1,1);
  road[10]=clamp(entropy25,-1,1);
  road[11]=s2.banker-.5;
  road[12]=s5.banker-.5;
  road[13]=clamp(s2.banker-s5.banker,-1,1);
  road[14]=clamp(s8.banker-.5,-.5,.5);
  road[15]=clamp(s5.turn-s8.turn,-1,1);
  road[16]=clamp(s5.run/Math.max(1,s5.count)-s8.run/Math.max(1,s8.count),-1,1);
  road[17]=clamp((s8.banker-.5)*(s8.regularity-.5),-1,1);
  road[18]=clamp((s5.banker-s8.banker)*(s5.entropy-s8.entropy),-1,1);
  road[19]=clamp(tieRatio(2)-tieRatio(5),-1,1);
  road[20]=nonTie.length<2?0:nonTie.at(-1)===nonTie.at(-2)?.4:-.4;
  road[21]=clamp(s2.banker-s8.banker,-1,1);
  road[22]=clamp(s5.regularity-s8.regularity,-1,1);
  road[23]=clamp(s2.turn-s8.turn,-1,1);
  const vector=shoe.concat(road);
  if(vector.length!==DIM||vector.some(x=>!finite(x)))throw Error("invalid_256D_context");
  return{vector,shoe,road,physics:sim,features:{
    regularity2:s2.regularity,regularity5:s5.regularity,regularity_delta:regular25,
    turn2:s2.turn,turn5:s5.turn,turn_rate_delta:turn25,
    entropy2:s2.entropy,entropy5:s5.entropy,entropy_delta:entropy25,
    entropy_volatility:Math.abs(entropy25),
    non_tie_sample:nonTie.length,maximum_window:16
  }};
}
function makeArm(){
  const A=Array.from({length:K},(_,i)=>Float64Array.from({length:K},(_,j)=>i===j?1:0));
  return{A,b:new Float64Array(K)};
}
/** 精確的正定 Cholesky 解方；求解 A^(-1)b 與 A^(-1)x。 */
function solveSPD(A,v){
  const L=Array.from({length:K},()=>new Float64Array(K));
  for(let i=0;i<K;i++)for(let j=0;j<=i;j++){
    let t=A[i][j];
    for(let k=0;k<j;k++)t-=L[i][k]*L[j][k];
    if(i===j){
      if(!(t>1e-9))throw Error("linucb_matrix_not_positive_definite");
      L[i][j]=Math.sqrt(t);
    }else L[i][j]=t/L[j][j];
  }
  const y=new Float64Array(K),x=new Float64Array(K);
  for(let i=0;i<K;i++){let v0=v[i];for(let j=0;j<i;j++)v0-=L[i][j]*y[j];y[i]=v0/L[i][i];}
  for(let i=K-1;i>=0;i--){let v0=y[i];for(let j=i+1;j<K;j++)v0-=L[j][i]*x[j];x[i]=v0/L[i][i];}
  return x;
}
function dot(a,b){let s=0;for(let i=0;i<K;i++)s+=a[i]*b[i];return s;}
class OnlineLinUCB256{
  constructor({lambda=.82,alpha=.15,minSamples=0,confidenceThreshold=0,
    lossGate=.04,particles=96}={}){
    if(!finite(lambda)||lambda<.75||lambda>.96)throw Error("lambda_must_be_0.75_to_0.96");
    if(!finite(alpha)||alpha<0||alpha>2)throw Error("invalid_alpha");
    if(!Number.isInteger(minSamples)||minSamples<0)throw Error("invalid_min_samples");
    if(!finite(confidenceThreshold)||confidenceThreshold<0||confidenceThreshold>.5)
      throw Error("invalid_confidence_threshold");
    if(!finite(lossGate)||lossGate<0||lossGate>.5)throw Error("invalid_loss_gate");
    this.lambda=lambda;this.alpha=alpha;this.minSamples=minSamples;
    this.confidenceThreshold=confidenceThreshold;this.lossGate=lossGate;
    this.particleCount=particles;this.reset();
  }
  reset(){
    this.history=[];this.arms={B:makeArm(),P:makeArm()};
    this.physics=new BlindPhysicsEstimator({particles:this.particleCount});
    this.trainedRounds=0;
    this.prequential={eligible:0,correct:0,wrong:0,ties:0,skipped:0};
    this.last_signal=null;this.last_settlement="none";
    this.gateCooldown=0;this.recommendationArm=null;this.recommendationStreak=0;
    return this;
  }
  _x(vector){return Float64Array.from(ACTIVE,i=>vector[i]);}
  _armScore(arm,x){
    const m=this.arms[arm],theta=solveSPD(m.A,m.b),projection=solveSPD(m.A,x);
    const mean=dot(x,theta),uncertainty=Math.sqrt(Math.max(0,dot(x,projection)));
    return{mean,uncertainty,ucb:mean+this.alpha*uncertainty};
  }
  predict(){
    const c=context256(this.history,this.physics),x=this._x(c.vector);
    const banker=this._armScore("B",x),player=this._armScore("P",x);
    const delta=Math.abs(banker.ucb-player.ucb);
    // 高密度訊號政策：平常 dynamic_gate=0；預測失誤只提高下一局至 .04。
    // 這是研究用信號強度門檻，不是下注具有正 EV 的保證。
    let dynamic_gate=Math.max(this.confidenceThreshold,this.gateCooldown);
    const enough=this.trainedRounds>=this.minSamples;
    const allowed=enough&&delta>=dynamic_gate;
    // 分數同為零時先展示 B 為固定決策，不得標記為高信心或高勝率。
    const rawDirection=banker.ucb>=player.ucb?"B":"P";
    const reason=!enough?"warmup":!allowed?"one_round_loss_gate":
      delta===0?"neutral_default_signal":"hyper_micro_signal";
    return{
      version:VERSION,direction:allowed?rawDirection:"Skip",raw_direction:rawDirection,
      bet_allowed:allowed,reason,dynamic_gate,
      ucb_banker:banker.ucb,ucb_player:player.ucb,
      delta,confidence_threshold:dynamic_gate,
      arms:{B:banker,P:player},history_length:this.history.length,
      trained_rounds:this.trainedRounds,lambda:this.lambda,alpha:this.alpha,
      context:c.vector,shoe_context:c.shoe,road_context:c.road,
      active_dimensions:K,features:c.features,blind_physics:c.physics,
      gate_cooldown_rounds:this.gateCooldown>0?1:0,
      last_settlement:this.last_settlement,
      recommendation_streak:this.recommendationStreak,
      calibrated_probability:false,claimed_edge:false,
      prequential:{...this.prequential},
      evaluated_hands:this.prequential.eligible,
      hit_rate:this.prequential.eligible-this.prequential.ties>0?
        this.prequential.correct/(this.prequential.eligible-this.prequential.ties):null,
      signal_rate:this.history.length?
        this.prequential.eligible/this.history.length:null,
      notice:"高密度推測方向不是保證命中；B/P/T 無法提供實際牌張與可驗證正EV。"
    };
  }
  _decay(m){
    // 每局 A = lambda*A+(1-lambda)*I；b = lambda*b
    const lam=this.lambda;
    for(let i=0;i<K;i++){
      m.b[i]*=lam;
      for(let j=0;j<K;j++)m.A[i][j]*=lam;
      m.A[i][i]+=1-lam;
    }
  }
  /** Prequential：先取得上一局預測，等使用者按 B/P/T 後才結算、再訓練。 */
  observe(outcome,points=null){
    if(!valid(outcome))throw Error("invalid_outcome");
    if(!validPointObservation(outcome,points))throw Error("invalid_point_observation");
    const prior=this.predict(),x=this._x(prior.context);
    this.last_signal=prior.direction;
    if(prior.bet_allowed){
      this.recommendationStreak=prior.direction===this.recommendationArm?
        this.recommendationStreak+1:1;
      this.recommendationArm=prior.direction;
      this.prequential.eligible++;
      if(outcome==="T")this.prequential.ties++;
      else if(prior.direction===outcome)this.prequential.correct++;
      else this.prequential.wrong++;
    }else{
      this.recommendationArm=null;this.recommendationStreak=0;
      this.prequential.skipped++;
    }
    // T 無法判定 B/P 預測對錯。保留剩餘的一局震盪過濾資格直到 B/P 開獎。
    if(outcome!=="T"){
      const failed=prior.bet_allowed&&prior.direction!==outcome;
      this.gateCooldown=failed?this.lossGate:0; // 成功立即歸零；Skip 亦只維持一局
      this.last_settlement=prior.bet_allowed?(failed?"loss":"win"):"skipped";
    }else{
      this.last_settlement="tie";
    }
    for(const arm of ["B","P"])this._decay(this.arms[arm]);
    if(outcome!=="T"){
      // 同一筆已知結果揭露兩個 arm 的 full-information 標籤。
      // 非數學反對稱：獲勝 +1、失敗 -2，A 雙邊相同但 b 不互為負數。
      for(const arm of ["B","P"]){
        const m=this.arms[arm],reward=arm===outcome?1:-2;
        for(let i=0;i<K;i++){
          m.b[i]+=reward*x[i];
          for(let j=0;j<K;j++)m.A[i][j]+=x[i]*x[j];
        }
      }
      this.trainedRounds++;
    }
    this.history.push(outcome);
    this.physics.observe(outcome,points);
    return this.predict();
  }
  /** 歷史/返回上一局採可重現回放；不會因點擊分析而重複訓練。 */
  syncHistory(seq){
    if(!Array.isArray(seq)||seq.some(v=>!valid(v)))throw Error("invalid_history");
    let same=this.history.length<=seq.length;
    if(same)for(let i=0;i<this.history.length;i++)
      if(this.history[i]!==seq[i]){same=false;break;}
    if(!same)this.reset();
    for(let i=this.history.length;i<seq.length;i++)this.observe(seq[i]);
    return this.predict();
  }
  static fromHistory(seq,options={}){return new OnlineLinUCB256(options).syncHistory(seq);}
}
return{OnlineLinUCB256,BlindPhysicsEstimator,conditionalHandProposal,
  context256,windowStats,
  DIM,SHOE_DIM:SHOE,ACTIVE_INDICES:ACTIVE.slice(),VERSION};
});