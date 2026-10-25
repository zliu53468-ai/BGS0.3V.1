(() => {
"use strict";

/*
Fixed-500 stateless baccarat particle posterior for the browser runtime.

Important semantics:
- Every prediction call starts from 500 fresh eight-deck particles.
- No particle array, weights, ESS, or resampling state is carried from the
  previous prediction. The current B/P/T history is replayed from scratch.
- B/P/T is the only observation. Hidden ranks/points are conditional
  hypotheses, never claimed as reconstructed casino cards.
- The particle layer is a bounded correction to Direct Physics and is not
  allowed to flip the Direct Physics B/P side by itself.
*/

const VERSION="PARTICLE500_STATELESS_V1";
const PARTICLE_COUNT=500;
const LIKELIHOOD_DRAWS=2;
const FALLBACK_DRAWS=2;
const FORECAST_DRAWS=1;
const ESS_RESAMPLE_THRESHOLD=.50;
const TIE_LIKELIHOOD_INFLUENCE=.30;
const MIN_BLEND=.06;
const MAX_BLEND=.16;
const RANKS=13;
const INITIAL_PER_RANK=32;
const OUTCOMES=new Set(["B","P","T"]);
const BLEND_MODES=new Set(["quality","off","fixed_0.06","fixed_0.10","fixed_0.16","uncertainty_only"]);
const clip=(v,lo=0,hi=1)=>Math.max(lo,Math.min(hi,Number.isFinite(+v)?+v:lo));

function normalizeBlendMode(mode){return BLEND_MODES.has(String(mode||"quality"))?String(mode||"quality"):"quality";}
function fixedBlendWeight(mode){
  if(mode==="fixed_0.06")return .06;
  if(mode==="fixed_0.10")return .10;
  if(mode==="fixed_0.16")return .16;
  return null;
}

function seedFromHistory(history){
  let h=2166136261>>>0;
  for(const token of history){
    h^=token==="B"?66:token==="P"?80:84;
    h=Math.imul(h,16777619)>>>0;
  }
  h^=history.length+0x9e3779b9;
  return h>>>0;
}
function rng32(seed){
  let x=(seed||0x6d2b79f5)>>>0;
  return ()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return (x>>>0)/4294967296;};
}
function rankValue(index){const rank=index+1;return rank<=9?rank:0;}
function bankerDraws(total,playerThird){
  if(playerThird===null)return total<=5;
  if(total<=2)return true;
  if(total===3)return playerThird!==8;
  if(total===4)return playerThird>=2&&playerThird<=7;
  if(total===5)return playerThird>=4&&playerThird<=7;
  if(total===6)return playerThird>=6&&playerThird<=7;
  return false;
}
function drawRank(counts,rng){
  let total=0;for(let i=0;i<RANKS;i++)total+=counts[i];
  if(total<=0)throw new Error("particle shoe exhausted");
  let pick=Math.floor(rng()*total),sum=0;
  for(let i=0;i<RANKS;i++){sum+=counts[i];if(pick<sum){counts[i]--;return i;}}
  return RANKS-1;
}
function deal(base,rng){
  const counts=base.slice();
  const p1=drawRank(counts,rng),b1=drawRank(counts,rng),p2=drawRank(counts,rng),b2=drawRank(counts,rng);
  const ranks=[p1,b1,p2,b2];
  let pt=(rankValue(p1)+rankValue(p2))%10,bt=(rankValue(b1)+rankValue(b2))%10;
  let p3=null;
  const natural=pt>=8||bt>=8;
  if(!natural){
    if(pt<=5){const r=drawRank(counts,rng);ranks.push(r);p3=rankValue(r);pt=(pt+p3)%10;}
    if(bankerDraws(bt,p3)){const r=drawRank(counts,rng);ranks.push(r);bt=(bt+rankValue(r))%10;}
  }
  return {outcome:bt>pt?"B":pt>bt?"P":"T",playerPoint:pt,bankerPoint:bt,ranks,counts,cardCount:ranks.length};
}
function normaliseWeights(values){
  let sum=0;for(const v of values)sum+=Math.max(0,+v||0);
  if(sum<=1e-300)return Array(values.length).fill(1/Math.max(1,values.length));
  return values.map(v=>Math.max(0,+v||0)/sum);
}
function systematicResample(particles,consumed,weights,rng){
  const n=particles.length,cdf=Array(n);let s=0;
  for(let i=0;i<n;i++){s+=weights[i];cdf[i]=s;}
  const start=rng()/n,next=Array(n),nextConsumed=Array(n);let j=0;
  for(let i=0;i<n;i++){
    const pos=start+i/n;while(j<n-1&&pos>cdf[j])j++;
    next[i]=particles[j].slice();nextConsumed[i]=consumed[j];
  }
  return {particles:next,consumed:nextConsumed,weights:Array(n).fill(1/n)};
}
function entropy3(block){
  let h=0;
  for(const p of block){if(p>1e-12)h-=p*Math.log(p);}
  return clip(h/Math.log(3));
}
function normalizeBlock(a){
  const x=a.map(v=>Math.max(0,+v||0)),sum=x.reduce((p,c)=>p+c,0);
  return sum>1e-12?x.map(v=>v/sum):Array(a.length).fill(1/a.length);
}
function blendBlock(direct,posterior,start,n,w){
  const d=normalizeBlock(direct.slice(start,start+n)),p=normalizeBlock(posterior.slice(start,start+n));
  const b=d.map((v,i)=>(1-w)*v+w*p[i]);
  const z=normalizeBlock(b);for(let i=0;i<n;i++)direct[start+i]=z[i];
}
function preserveWinnerDirection(corrected,direct){
  const dB=clip(direct[23]),dP=clip(direct[24]),dT=clip(direct[25]),dMass=Math.max(1e-9,dB+dP),dDir=dB/dMass;
  let b=clip(corrected[23]),p=clip(corrected[24]),t=clip(corrected[25]),mass=Math.max(1e-9,b+p),dir=b/mass;
  let blocked=false;
  if(dDir>.5&&dir<.5){dir=.5;blocked=true;}
  if(dDir<.5&&dir>.5){dir=.5;blocked=true;}
  const directionalMass=clip(1-t);
  corrected[23]=directionalMass*dir;corrected[24]=directionalMass*(1-dir);corrected[25]=t;
  return blocked;
}
function posteriorForecast(particles,consumed,weights,rng,draws=FORECAST_DRAWS){
  const out=Array(48).fill(0);let mass=0;
  let consumedMean=0,consumedSq=0;
  for(let i=0;i<particles.length;i++){
    const w=weights[i];consumedMean+=w*consumed[i];consumedSq+=w*consumed[i]*consumed[i];
    for(let d=0;d<draws;d++){
      const hand=deal(particles[i],rng),m=w/draws;mass+=m;
      out[{4:0,5:1,6:2}[hand.cardCount]]+=m;
      out[3+hand.playerPoint]+=m;out[13+hand.bankerPoint]+=m;
      out[23+(hand.outcome==="B"?0:hand.outcome==="P"?1:2)]+=m;
      for(const rank of hand.ranks)out[26+rank]+=m;
      const diff=hand.bankerPoint-hand.playerPoint;
      out[46]+=m*diff/9;out[47]+=m*Math.abs(diff)/9;
    }
  }
  if(mass<=1e-12)throw new Error("particle forecast empty");
  for(const [start,n] of [[0,3],[3,10],[13,10],[23,3]])for(let i=0;i<n;i++)out[start+i]/=mass;
  for(let i=26;i<39;i++)out[i]/=mass;
  out[43]=consumedMean;
  let low=0,high=0,total=0;
  for(let i=0;i<particles.length;i++){
    const w=weights[i],counts=particles[i];let localTotal=0,localLow=0,localHigh=0;
    for(let r=0;r<RANKS;r++){localTotal+=counts[r];if(r<5)localLow+=counts[r];if(r>=8)localHigh+=counts[r];}
    total+=w*localTotal;low+=w*localLow;high+=w*localHigh;
  }
  out[44]=total>0?low/total:0;out[45]=total>0?high/total:0;
  out[46]/=mass;out[47]/=mass;
  const variance=Math.max(0,consumedSq-consumedMean*consumedMean);
  return {physics:out,consumedStd:Math.sqrt(variance)};
}
function estimate(history,directPhysics,mode="quality",config={}){
  const seq=(Array.isArray(history)?history:[]).map(x=>String(x||"").toUpperCase()).filter(x=>OUTCOMES.has(x));
  if(!Array.isArray(directPhysics)||directPhysics.length!==48)throw new Error("particle500 requires Direct Physics 48D");
  const blendMode=normalizeBlendMode(mode);
  if(blendMode==="off")return {
    physics:directPhysics.map(v=>+v||0),posterior:null,
    diagnostics:{version:VERSION,enabled:false,particle_filter_enabled:false,particle_count:0,
      policy:"direct_physics_only",persistent_state:false,rebuild_from_scratch:false,
      history_rounds:seq.length,history_fingerprint:seq.join(""),particle_blend_mode:blendMode,blend_weight:0}
  };
  const rng=rng32(seedFromHistory(seq));
  let particles=Array.from({length:PARTICLE_COUNT},()=>Array(RANKS).fill(INITIAL_PER_RANK));
  let consumed=Array(PARTICLE_COUNT).fill(0),weights=Array(PARTICLE_COUNT).fill(1/PARTICLE_COUNT);
  let resamplingCount=0,lastEssRatio=1,tieUpdates=0;

  for(const actual of seq){
    const next=Array(PARTICLE_COUNT),nextConsumed=Array(PARTICLE_COUNT),rawWeights=Array(PARTICLE_COUNT);
    for(let i=0;i<PARTICLE_COUNT;i++){
      const base=particles[i],proposals=[],matches=[];
      for(let d=0;d<LIKELIHOOD_DRAWS;d++){const hand=deal(base,rng);proposals.push(hand);if(hand.outcome===actual)matches.push(hand);}
      const matchCount=matches.length;
      let likelihood=(matchCount+.20)/(LIKELIHOOD_DRAWS+.60);
      if(!matches.length){
        for(let d=0;d<FALLBACK_DRAWS;d++){const hand=deal(base,rng);if(hand.outcome===actual){matches.push(hand);break;}}
      }
      const chosen=matches.length?matches[Math.floor(rng()*matches.length)]:proposals[Math.floor(rng()*proposals.length)];
      if(!matches.length)likelihood=Math.max(likelihood,.015);
      if(actual==="T"){likelihood=(1-TIE_LIKELIHOOD_INFLUENCE)+TIE_LIKELIHOOD_INFLUENCE*likelihood;tieUpdates++;}
      next[i]=chosen.counts;nextConsumed[i]=consumed[i]+chosen.cardCount;rawWeights[i]=weights[i]*Math.max(1e-8,likelihood);
    }
    weights=normaliseWeights(rawWeights);particles=next;consumed=nextConsumed;
    let sq=0;for(const w of weights)sq+=w*w;lastEssRatio=clip((1/Math.max(1e-12,sq))/PARTICLE_COUNT);
    if(lastEssRatio<ESS_RESAMPLE_THRESHOLD){
      const r=systematicResample(particles,consumed,weights,rng);particles=r.particles;consumed=r.consumed;weights=r.weights;resamplingCount++;lastEssRatio=1;
    }
  }

  const forecastDraws=Math.round(clip(config.forecast_draws??FORECAST_DRAWS,1,5));
  const forecast=posteriorForecast(particles,consumed,weights,rng,forecastDraws),posterior=forecast.physics;
  const winner=normalizeBlock(posterior.slice(23,26));
  const spread=clip(forecast.consumedStd/18);
  const posteriorUncertainty=clip(.55*entropy3(winner)+.30*(1-lastEssRatio)+.15*spread);
  const quality=clip(.45*lastEssRatio+.35*(1-posteriorUncertainty)+.20*(1-spread));
  const minimum=clip(config.min_blend??MIN_BLEND,0,MAX_BLEND),maximum=clip(config.max_blend??MAX_BLEND,minimum,MAX_BLEND);
  const sameSide=(directPhysics[23]-directPhysics[24])*(posterior[23]-posterior[24])>0;
  const signalGate=config.signal_gate===true?(sameSide?clip(lastEssRatio)*clip(1-posteriorUncertainty):0):1;
  const qualityBlendWeight=clip(minimum+(maximum-minimum)*quality*signalGate,minimum,maximum);
  const forcedBlendWeight=fixedBlendWeight(blendMode);
  const blendWeight=forcedBlendWeight===null?qualityBlendWeight:forcedBlendWeight;

  const corrected=directPhysics.map(v=>+v||0);
  let directionFlipBlocked=false;
  if(blendMode==="uncertainty_only"){
    corrected[42]=clip(.65*clip(directPhysics[42])+.35*posteriorUncertainty);
    corrected[44]=(1-blendWeight)*corrected[44]+blendWeight*posterior[44];
    corrected[45]=(1-blendWeight)*corrected[45]+blendWeight*posterior[45];
  }else{
    blendBlock(corrected,posterior,0,3,blendWeight);
    blendBlock(corrected,posterior,3,10,blendWeight);
    blendBlock(corrected,posterior,13,10,blendWeight);
    blendBlock(corrected,posterior,23,3,blendWeight);
    directionFlipBlocked=preserveWinnerDirection(corrected,directPhysics);
    for(let i=26;i<39;i++)corrected[i]=(1-blendWeight)*corrected[i]+blendWeight*posterior[i];
    corrected[43]=(1-blendWeight)*corrected[43]+blendWeight*posterior[43];
    corrected[44]=(1-blendWeight)*corrected[44]+blendWeight*posterior[44];
    corrected[45]=(1-blendWeight)*corrected[45]+blendWeight*posterior[45];
    corrected[46]=(1-blendWeight)*corrected[46]+blendWeight*posterior[46];
    corrected[47]=(1-blendWeight)*corrected[47]+blendWeight*posterior[47];
    corrected[42]=clip(.65*clip(directPhysics[42])+.35*posteriorUncertainty);
    const pB=clip(corrected[23]),pP=clip(corrected[24]);
    corrected[39]=pB*.95-pP;corrected[40]=pP-pB;corrected[41]=corrected[39]-corrected[40];
  }

  return {
    physics:corrected,
    posterior,
    diagnostics:{
      version:VERSION,enabled:true,particle_filter_enabled:true,particle_count:PARTICLE_COUNT,
      policy:"stateless_rebuild_each_prediction",persistent_state:false,rebuild_from_scratch:true,
      history_rounds:seq.length,history_fingerprint:seq.join(""),likelihood_draws:LIKELIHOOD_DRAWS,
      fallback_draws:FALLBACK_DRAWS,forecast_draws:forecastDraws,ess_resample_threshold:ESS_RESAMPLE_THRESHOLD,
      resampling_count:resamplingCount,recent_ess_ratio:lastEssRatio,posterior_uncertainty:posteriorUncertainty,
      particle_quality:quality,signal_gate:signalGate,same_direction:sameSide,blend_min:minimum,blend_max:maximum,blend_weight:blendWeight,quality_blend_weight:qualityBlendWeight,particle_blend_mode:blendMode,
      tie_soft_update:true,tie_likelihood_influence:TIE_LIKELIHOOD_INFLUENCE,
      tie_observation_count:seq.filter(x=>x==="T").length,direction_flip_blocked:directionFlipBlocked,
      direct_directional_pb:clip(directPhysics[23]/Math.max(1e-9,directPhysics[23]+directPhysics[24])),
      corrected_directional_pb:clip(corrected[23]/Math.max(1e-9,corrected[23]+corrected[24])),
      physics_uncertainty:corrected[42],expected_consumed_cards:corrected[43],
      physical_ev_banker:corrected[39],physical_ev_player:corrected[40],physical_ev_gap:corrected[41],
    }
  };
}

if(typeof window!=="undefined")window.__BGS_PARTICLE500__={version:VERSION,particleCount:PARTICLE_COUNT,estimate};
})();
