(() => {
"use strict";

const CORE=(typeof window!=="undefined")?window.__BGS256_CONTINUATION_TEST__:null;
const PARTICLE500=(typeof window!=="undefined")?window.__BGS_PARTICLE500__:null;
const LOCAL_CALIBRATION_API=(typeof window!=="undefined")?window.__BBB_BROWSER_CALIBRATION__:null;
const LOCAL_CALIBRATION=LOCAL_CALIBRATION_API?.createClient(typeof localStorage!=="undefined"?localStorage:null)||null;
let browserCalibrationTag="unready";
const localCalibrationConfig=()=>cardcountProfile?.local_calibration||{};
const MACRO=(typeof window!=="undefined")?window.__BGS_MACRO_EMA__:null;
if(!CORE||typeof CORE.hazardChoose!=="function")return;

const EV_CONSISTENCY_V1=false; // Shadow gate failed; set true only after holdout acceptance.
const EARLY_TIE_PENALTY=.15;
const VERSION=EV_CONSISTENCY_V1
  ?"PHYSICS_57D_V14_8_FORECAST_FIRST_EVC1"
  :"PHYSICS_57D_V14_8_FORECAST_FIRST";
const PHYSICS_URL="physics_multitask_model.json";
let cardcountProfile=null,displayGapOverride=null;
function profileValue(key,roundIndex){
  if(!cardcountProfile?.enabled)return null;
  return interpAnchors(clip(roundIndex,1,70),cardcountProfile.anchors[key]);
}
function displayDirection(f){
  if(!f?.available)return null;
  const gap=displayGapOverride??(cardcountProfile?.enabled?cardcountProfile.display_gap:0);
  if(Math.abs(f.directional.B-f.directional.P)+1e-12<gap)return "Skip";
  return f.direction;
}
function macroUpdate(state,hand){
  const r=state.rounds+1;
  return MACRO.update(state,hand,cardcountProfile?.enabled?{short_alpha:profileValue("macro_short",r),long_alpha:profileValue("macro_long",r)}:{});
}
const FINAL56_URL="final_probability_model.json";
const ORIGINAL7_NAMES=["core_p_b","round_index","estimated_total_hands","remaining_ratio","sx_markov_p_same","stage","depth"];
// Keep the trained 48D/57D ABI: legacy particle_uncertainty at slot 42
// contains Physics uncertainty; feature order stays unchanged.
const PHYSICS_NAMES=[
"cards_p4","cards_p5","cards_p6",
...Array.from({length:10},(_,i)=>"player_point_p"+i),
...Array.from({length:10},(_,i)=>"banker_point_p"+i),
"winner_p_b","winner_p_p","winner_p_t",
..."A,2,3,4,5,6,7,8,9,10,J,Q,K".split(",").map(x=>"next_rank_expected_"+x),
"physical_ev_banker","physical_ev_player","physical_ev_gap","particle_uncertainty",
"shoe_consumed_cards","remaining_low_rank_density","remaining_high_rank_density",
"expected_point_diff_norm","expected_abs_point_diff_norm"];
const LEGACY_PHYSICS_NAMES=[
...PHYSICS_NAMES.slice(0,39),
"next_suit_ratio_spades","next_suit_ratio_hearts","next_suit_ratio_diamonds","next_suit_ratio_clubs",
...PHYSICS_NAMES.slice(43)];
const PHYSICS_INDEX=Object.fromEntries(PHYSICS_NAMES.map((name,index)=>[name,index]));
const LEGACY_PHYSICS_INDEX=Object.fromEntries(LEGACY_PHYSICS_NAMES.map((name,index)=>[name,index]));
const EXTENDED_NAMES=["core_p_b_external","shoe_progress_weight",...ORIGINAL7_NAMES.slice(1).map(x=>"original7_"+x),...PHYSICS_NAMES,"physics_noise_score"];
const HISTORY_WINDOW=64,HISTORY_INPUT_DIM=213,PHYSICS_DIM=48,FEATURE_DIM=57;
const ESTIMATED_TOTAL_HANDS_MIN=50,ESTIMATED_TOTAL_HANDS_MAX=70,ESTIMATED_PLAYABLE_CARDS=416-60;
const DEFAULT_BOUNDS=[.40,.60],EARLY_BOUNDS=[.42,.58],LATE_CLEAN_BOUNDS=[.35,.65],PHYSICS_NOISE_LOW_THRESHOLD=.78;
const SHOE_ERROR_CORRECTION_VERSION=0,SHOE_ERROR_CORRECTION_MEMORY=10;
const PHYSICS_DIRECT_VERSION=1;
const LOW_SKIP_CONFIDENCE_SCALE=.75;
const LOW_SKIP_EV_RELIEF=.006;
const LOW_SKIP_FINAL_BAND_RELIEF=.006;
const LOW_SKIP_CONSISTENCY_BAND_GAIN=.0015;
const LOW_SKIP_UNCERTAINTY_BAND_GAIN=.001;
const LOW_SKIP_PRECISION_SCALE=.40;
const BANKER_EV_FLOOR=-0.0025; // Candidate-only: permit a tiny banker EV floor.
const PLAYER_EV_FLOOR=0.0;
const SNAPSHOT_SCHEMA_VERSION=7;
const TRAINING_KEY="bgs_xgb_final_training_v5",PENDING_KEY="bgs_xgb_final_pending_v5";
const SHOE_KEY="bgs_xgb_final_shoe_id_v5",CUT_KEY="bgs_xgb_estimated_total_hands_v1";
const STORAGE_KEY="bgs256d_short_x_dynamic_v23",MAX_TRAINING_ROWS=15000;
const EARLY35_VERSION=1,EARLY35_BRIDGE_START=35,EARLY35_BRIDGE_END=40;
const EARLY35_EVIDENCE_ANCHORS=[[1,.15],[5,.22],[10,.32],[15,.45],[20,.55],[25,.64],[30,.72],[35,.78]];
const EARLY35_EV_RELIABILITY_ANCHORS=[[1,.20],[5,.28],[10,.38],[15,.48],[20,.58],[25,.66],[30,.73],[35,.79]];
const clip=(v,lo=0,hi=1)=>Math.max(lo,Math.min(hi,Number.isFinite(+v)?+v:lo));
const bp=seq=>seq.filter(x=>x==="B"||x==="P");
const directionalRoundCount=seq=>bp(seq).length;
const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));

let physicsBundle=null,final56Bundle=null;
let status={physics:false,final56:false,errors:[]};

function transitionSequence(seq){const a=bp(seq),out=[];for(let i=1;i<a.length;i++)out.push(a[i]===a[i-1]?"S":"X");return out;}
function sxMarkovPSame(seq,window=24,prior=1){
  const t=transitionSequence(seq);if(!t.length)return .5;
  const current=t.at(-1),start=Math.max(0,t.length-1-Math.max(2,window));let same=0,sw=0;
  for(let i=start;i<t.length-1;i++){if(t[i]!==current)continue;if(t[i+1]==="S")same++;else if(t[i+1]==="X")sw++;}
  return clip((same+prior)/(same+sw+2*prior));
}
function currentStage(seq){const a=bp(seq);if(!a.length)return 0;const side=a.at(-1);let n=1;for(let i=a.length-2;i>=0&&a[i]===side;i--)n++;return n;}
function currentDepth(seq){const t=transitionSequence(seq);if(!t.length)return 0;const token=t.at(-1);let n=1;for(let i=t.length-2;i>=0&&t[i]===token;i--)n++;return n;}
function getEstimatedTotalHands(){
  try{const c=+(window.__BGS_FINAL_CONFIG__?.estimatedTotalHands||0);if(c>=ESTIMATED_TOTAL_HANDS_MIN&&c<=ESTIMATED_TOTAL_HANDS_MAX)return c;const s=+localStorage.getItem(CUT_KEY);if(s>=ESTIMATED_TOTAL_HANDS_MIN&&s<=ESTIMATED_TOTAL_HANDS_MAX)return s;}catch(_){}
  return 60;
}
function setEstimatedTotalHands(v){const n=Math.round(+v||0);if(n<ESTIMATED_TOTAL_HANDS_MIN||n>ESTIMATED_TOTAL_HANDS_MAX)throw new Error("estimatedTotalHands must be 50..70");try{localStorage.setItem(CUT_KEY,String(n));}catch(_){}return n;}
function buildOriginal7(seq,corePrediction){
  const corePB=clip(+corePrediction?.probabilities?.B||.5),roundIndex=Math.max(1,Math.min(70,seq.length+1));
  const estimatedTotalHands=getEstimatedTotalHands(),remainingRatio=clip((estimatedTotalHands-(roundIndex-1))/Math.max(1,estimatedTotalHands));
  const signal=corePrediction?.singleHazard||null;
  return {core_p_b:corePB,round_index:roundIndex,estimated_total_hands:estimatedTotalHands,remaining_ratio:remainingRatio,
    sx_markov_p_same:sxMarkovPSame(seq),stage:Number.isFinite(+signal?.state?.length)?+signal.state.length:currentStage(seq),
    depth:Number.isFinite(+signal?.depth?.depth)?+signal.depth.depth:currentDepth(seq)};
}
function original7Vector(f){return ORIGINAL7_NAMES.map(n=>Number.isFinite(+f[n])?+f[n]:0);}

function entropy3(p){let s=0;for(const v of p)if(v>0)s-=v*Math.log(v);return s/Math.log(3);}
function ratios(seq,n){const b=n?seq.slice(-n):seq;if(!b.length)return[0,0,0];return[b.filter(x=>x==="B").length/b.length,b.filter(x=>x==="P").length/b.length,b.filter(x=>x==="T").length/b.length];}
function historyVector(seq){
  const one=Array(HISTORY_WINDOW*3).fill(0),tail=seq.slice(-HISTORY_WINDOW),offset=HISTORY_WINDOW-tail.length,map={B:0,P:1,T:2};
  tail.forEach((token,j)=>{one[(offset+j)*3+map[token]]=1;});
  const bps=bp(seq);let turns=0;for(let i=1;i<bps.length;i++)if(bps[i]!==bps[i-1])turns++;
  const r8=ratios(seq,8),r16=ratios(seq,16),r32=ratios(seq,32),rall=ratios(seq,Math.max(1,seq.length));
  const summary=[Math.min(seq.length,90)/90,Math.min(bps.length,90)/90,Math.min(currentStage(seq),12)/12,turns/Math.max(1,bps.length-1),
    ...r8,...r16,...r32,...rall,entropy3(r8),entropy3(r16),entropy3(r32),bps.length&&bps.at(-1)==="B"?1:0,bps.length&&bps.at(-1)==="P"?1:0];
  const out=[...one,...summary];if(out.length!==HISTORY_INPUT_DIM)throw new Error("history vector "+out.length);return out;
}
function dense(input,w,b,relu){
  const out=Array(b.length).fill(0);
  for(let j=0;j<b.length;j++){let s=+b[j]||0;for(let i=0;i<input.length;i++)s+=(+input[i]||0)*(+w[i][j]||0);out[j]=relu?Math.max(0,s):s;}
  return out;
}
function normalise(block,fallback){const a=block.map(v=>Math.max(0,Number.isFinite(+v)?+v:0)),s=a.reduce((x,y)=>x+y,0);return s>1e-12?a.map(v=>v/s):fallback.slice();}
function temperatureNorm(block,fallback,t=1){const p=normalise(block,fallback).map(v=>Math.max(1e-8,v)),z=p.map(v=>Math.log(v)/Math.max(.25,+t||1)),m=Math.max(...z),e=z.map(v=>Math.exp(v-m)),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s);}
function physicsBundleUsesPhysicalEv(){return Array.isArray(physicsBundle?.feature_names)&&physicsBundle.feature_names.includes("physical_ev_banker");}
function directPhysicsVersion(){return Math.max(0,Math.floor(+final56Bundle?.training?.physics_direct_version||+final56Bundle?.physics_direct_version||0));}
function directPhysicsEnabled(){return directPhysicsVersion()>=PHYSICS_DIRECT_VERSION;}
function finalModelUsesPhysicalEv(){return directPhysicsEnabled();}
function adaptiveStageEnabled(){return (+final56Bundle?.training?.stage_progress_version||0)>=2;}
function stageProgressV3Enabled(){return (+final56Bundle?.training?.stage_progress_version||0)>=3;}
function stageProgressV4Enabled(){return (+final56Bundle?.training?.stage_progress_version||0)>=4;}
function early35Version(){return Math.max(0,Math.floor(+final56Bundle?.training?.early35_version||+final56Bundle?.early35_version||0));}
function early35Enabled(){return early35Version()>=EARLY35_VERSION;}
function shoeErrorCorrectionVersion(){return Math.max(0,Math.floor(+final56Bundle?.training?.shoe_error_correction_version||+final56Bundle?.shoe_error_correction_version||0));}
function shoeErrorCorrectionEnabled(){return shoeErrorCorrectionVersion()>=SHOE_ERROR_CORRECTION_VERSION;}
function modelSemanticsCompatible(){return directPhysicsEnabled()&&physicsBundleUsesPhysicalEv();}
function sanitizePhysics(raw,temperatures={},physicalEvSemantics=physicsBundleUsesPhysicalEv()){
  if(raw.length!==PHYSICS_DIM)throw new Error("physics dim mismatch");
  const out=Array(PHYSICS_DIM).fill(0);let src=0,dst=0;
  const copyNorm=(n,fb,key)=>{const a=temperatureNorm(raw.slice(src,src+n),fb,temperatures[key]);src+=n;for(const v of a)out[dst++]=v;};
  copyNorm(3,[.58,.34,.08],"card_count");copyNorm(10,Array(10).fill(.1),"player_points");copyNorm(10,Array(10).fill(.1),"banker_points");copyNorm(3,[.4586,.4462,.0952],"winner");
  for(let i=0;i<13;i++)out[dst++]=clip(raw[src++],0,6);
  if(physicalEvSemantics){
    out[dst++]=clip(raw[src++],-1,.95);out[dst++]=clip(raw[src++],-1,1);out[dst++]=clip(raw[src++],-2,2);out[dst++]=clip(raw[src++],0,1);
  }else copyNorm(4,Array(4).fill(.25),"suit");
  out[dst++]=clip(raw[src++],0,416);out[dst++]=clip(raw[src++],0,1);out[dst++]=clip(raw[src++],0,1);
  out[dst++]=clip(raw[src++],-1,1);out[dst++]=clip(raw[src++],0,1);
  return out;
}
let lastPhysicsDiagnostics=null;
function interpAnchors(value,anchors){if(value<=anchors[0][0])return anchors[0][1];for(let i=1;i<anchors.length;i++){if(value<=anchors[i][0]){const [x0,y0]=anchors[i-1],[x1,y1]=anchors[i],t=(value-x0)/Math.max(1e-12,x1-x0);return y0+t*(y1-y0);}}return anchors.at(-1)[1];}
function predictPhysics(seq){
  if(!physicsBundle?.trained)return null;
  let x=historyVector(seq),mean=physicsBundle.scaler?.mean||[],scale=physicsBundle.scaler?.scale||[];
  x=x.map((v,i)=>(v-(+mean[i]||0))/Math.max(1e-12,+scale[i]||1));
  const coefs=physicsBundle.coefs||[],intercepts=physicsBundle.intercepts||[];
  if(!coefs.length||coefs.length!==intercepts.length)throw new Error("invalid physics bundle");
  let h=x;for(let layer=0;layer<coefs.length;layer++)h=dense(h,coefs[layer],intercepts[layer],layer<coefs.length-1);
  const lossScale=physicsBundle.loss_scale||[];h=h.map((v,i)=>(+v||0)/Math.max(1e-12,+lossScale[i]||1));
  const tMean=physicsBundle.target_scaler?.mean||[],tScale=physicsBundle.target_scaler?.scale||[];
  h=h.map((v,i)=>(+v||0)*Math.max(1e-12,+tScale[i]||1)+(+tMean[i]||0));
  const outputSlope=physicsBundle.output_calibration?.slope||[],outputIntercept=physicsBundle.output_calibration?.intercept||[];
  h=h.map((v,i)=>v*(Number.isFinite(+outputSlope[i])?+outputSlope[i]:1)+(Number.isFinite(+outputIntercept[i])?+outputIntercept[i]:0));
  const physicalSemantics=physicsBundleUsesPhysicalEv();
  const direct=sanitizePhysics(h,physicsBundle.calibration_temperatures||{},physicalSemantics);
  if(physicalSemantics){
    const roundIndex=Math.max(1,seq.length+1);
    direct[42]=clip(physicsNoiseScore(direct,roundIndex,null));
  }
  lastPhysicsDiagnostics={
    enabled:false,
    physics_direct_version:PHYSICS_DIRECT_VERSION,
    physics_source:"direct_multitask_mlp",
    physics_uncertainty:clip(direct[42]),
    expected_consumed_cards:+direct[43]||0,
    physical_ev_banker:+direct[39]||0,
    physical_ev_player:+direct[40]||0,
    physical_ev_gap:+direct[41]||0,
  };
  return sanitizePhysics(direct,{},physicalSemantics);
}

function findChild(node,id){return(node?.children||[]).find(c=>+c.nodeid===+id)||null;}
function splitIndex(split,names){const t=String(split??"");if(/^f\d+$/.test(t))return+t.slice(1);return names.indexOf(t);}
function evaluateTree(tree,vector,names){
  let node=tree,guard=0;while(node&&guard++<256){if(Object.prototype.hasOwnProperty.call(node,"leaf"))return+node.leaf||0;
    const idx=splitIndex(node.split,names),value=idx>=0?Math.fround(vector[idx]):NaN,threshold=Math.fround(+node.split_condition);
    node=findChild(node,!Number.isFinite(value)?node.missing:(value<threshold?node.yes:node.no));}
  return 0;
}
function predictFinalProbability(bundle,vector,names){
  if(!bundle?.trained||!Array.isArray(bundle.trees))return .5;
  let margin=+bundle.base_margin||0;
  for(const tree of bundle.trees)margin+=evaluateTree(tree,vector,names);
  let probability=clip(sigmoid(margin),1e-7,1-1e-7),calibration=bundle.calibration||{};
  if(calibration.method==="platt"){
    const logit=Math.log(probability/(1-probability));
    const slope=Number.isFinite(+calibration.slope)?+calibration.slope:1,intercept=Number.isFinite(+calibration.intercept)?+calibration.intercept:0;
    probability=sigmoid(slope*logit+intercept);
  }else if(calibration.method==="isotonic"&&calibration.x_thresholds?.length>1&&calibration.y_thresholds?.length===calibration.x_thresholds.length){
    const xs=calibration.x_thresholds,ys=calibration.y_thresholds;if(probability<=xs[0])probability=ys[0];else if(probability>=xs.at(-1))probability=ys.at(-1);else{
      let lo=0,hi=xs.length-1;while(hi-lo>1){const mid=(lo+hi)>>1;if(probability<xs[mid])hi=mid;else lo=mid;}
      const ratio=(probability-xs[lo])/Math.max(1e-12,xs[hi]-xs[lo]);probability=ys[lo]+ratio*(ys[hi]-ys[lo]);
    }
  }
  return clip(probability);
}
function normalisedEntropy(block){const p=normalise(block,Array(block.length).fill(1/block.length));return-p.reduce((s,v)=>s+(v>0?v*Math.log(v):0),0)/Math.log(block.length);}
function calibrateNoise(value,calibration){
  const v=clip(value),xs=calibration?.x_thresholds||[],ys=calibration?.y_thresholds||[];
  if(calibration?.method!=="isotonic"||xs.length<2||ys.length!==xs.length)return v;
  if(v<=xs[0])return clip(ys[0]);if(v>=xs.at(-1))return clip(ys.at(-1));
  let lo=0,hi=xs.length-1;while(hi-lo>1){const mid=(lo+hi)>>1;if(v<xs[mid])hi=mid;else lo=mid;}
  const t=(v-xs[lo])/Math.max(1e-12,xs[hi]-xs[lo]);return clip(ys[lo]+t*(ys[hi]-ys[lo]));
}
function physicsNoiseScore(physics,roundIndex=70,effectiveProgress=null){
  const version=final56Bundle?.training?.noise_score_version||1;
  if(version<2){const r=physicsIntegrity(physics);return r?Math.abs(r.cardCountProbabilitySum-1):1;}
  const winner=physics.slice(23,26).sort((a,b)=>b-a),gap=(winner[0]||0)-(winner[1]||0);
  if(version<3)return clip(.20*normalisedEntropy(physics.slice(0,3))+.15*normalisedEntropy(physics.slice(3,13))+.15*normalisedEntropy(physics.slice(13,23))+.35*normalisedEntropy(physics.slice(23,26))+.15*(1-gap));
  const densityGap=Math.abs(clip(physics[44])-clip(physics[45])),densityAmbiguity=1-Math.min(1,densityGap/.25);
  const uncertaintyTerm=version>=5?clip(physics[42]):normalisedEntropy(physics.slice(39,43));
  const raw=clip(.10*normalisedEntropy(physics.slice(0,3))+.075*normalisedEntropy(physics.slice(3,13))+.075*normalisedEntropy(physics.slice(13,23))+.15*normalisedEntropy(physics.slice(23,26))+.25*normalisedEntropy(physics.slice(26,39))+.15*uncertaintyTerm+.075*(1-gap)+.125*densityAmbiguity);
  const compressed=.50+.35*Math.tanh((raw-.75)/.20),uncertainty=clip(physics[42]),baseProxy=version>=5?clip(Math.max(.70*compressed+.30*uncertainty,.50+.40*uncertainty*uncertainty)):compressed;
  const influence=stageProgressV4Enabled()&&Number.isFinite(+effectiveProgress)?interpAnchors(clip(+effectiveProgress),[[0,.35],[.35,.35],[.70,.65],[1,1]]):adaptiveStageEnabled()?interpAnchors(roundIndex,[[1,.35],[40,.35],[50,.65],[70,1]]):(roundIndex<=40?.35:roundIndex<=50?.65:1);
  const proxy=clip(.50+(baseProxy-.50)*influence);
  if(version<4)return proxy;
  const calibration=final56Bundle?.training?.physics_noise_calibration||physicsBundle?.uncertainty_calibration||{};
  return calibrateNoise(proxy,calibration);
}
function buildExtended(corePB,o7,physics){
  const original=original7Vector(o7),estimatedTotal=Math.max(ESTIMATED_TOTAL_HANDS_MIN,Math.min(ESTIMATED_TOTAL_HANDS_MAX,+original[2]||60)),roundProgress=clip(Number(original[1])/estimatedTotal),uncertainty=clip(physics[42]),cardProgress=clip((+physics[43]||0)/ESTIMATED_PLAYABLE_CARDS);
  const effectiveProgress=adaptiveStageEnabled()?clip((.55+.25*uncertainty)*roundProgress+(.45-.25*uncertainty)*cardProgress):roundProgress;
  const progress=effectiveProgress**3,noise=physicsNoiseScore(physics,original[1],effectiveProgress);
  const out=[corePB,progress,...original.slice(1),...physics,noise];if(out.length!==FEATURE_DIM)throw new Error("extended dim "+out.length);return out;
}
function unpackPhysicsForecast(physics){
  if(!Array.isArray(physics)||physics.length!==PHYSICS_DIM)throw new Error("physics forecast dim mismatch");
  const physicalSemantics=physicsBundleUsesPhysicalEv(),index=physicalSemantics?PHYSICS_INDEX:LEGACY_PHYSICS_INDEX;
  const value=name=>{const v=+physics[index[name]];return Number.isFinite(v)?v:0;};
  const cardCountProbabilities={"4_cards":value("cards_p4"),"5_cards":value("cards_p5"),"6_cards":value("cards_p6")};
  const expectedNextCardCount=4*cardCountProbabilities["4_cards"]+5*cardCountProbabilities["5_cards"]+6*cardCountProbabilities["6_cards"];
  const rankExpectedConsumption=Object.fromEntries("A,2,3,4,5,6,7,8,9,10,J,Q,K".split(",").map(rank=>[rank,value("next_rank_expected_"+rank)]));
  if(!physicalSemantics){
    const suitConsumptionRatios=Object.fromEntries(["spades","hearts","diamonds","clubs"].map(suit=>[suit,value("next_suit_ratio_"+suit)]));
    return {nextCardCountProbabilities:cardCountProbabilities,expectedNextCardCount,rankExpectedConsumption,suitConsumptionRatios,physicalEv:null};
  }
  const physicalEv={banker:value("physical_ev_banker"),player:value("physical_ev_player"),gap:value("physical_ev_gap"),uncertainty:value("particle_uncertainty")};
  return {nextCardCountProbabilities:cardCountProbabilities,expectedNextCardCount,rankExpectedConsumption,physicalEv};
}
function physicsIntegrity(physics){
  if(!Array.isArray(physics)||physics.length!==PHYSICS_DIM)return null;
  const forecast=unpackPhysicsForecast(physics),sum=values=>values.reduce((total,value)=>total+(+value||0),0);
  const cardCountProbabilitySum=sum(Object.values(forecast.nextCardCountProbabilities));
  const rankExpectedConsumptionTotal=sum(Object.values(forecast.rankExpectedConsumption));
  const rankExpectedTotalGap=Math.abs(rankExpectedConsumptionTotal-forecast.expectedNextCardCount);
  const rankExpectedTotalTolerance=Math.max(.50,forecast.expectedNextCardCount*.10);
  const physicalOk=!forecast.physicalEv||(forecast.physicalEv.banker>=-1&&forecast.physicalEv.banker<=.95&&forecast.physicalEv.player>=-1&&forecast.physicalEv.player<=1&&forecast.physicalEv.gap>=-2&&forecast.physicalEv.gap<=2&&forecast.physicalEv.uncertainty>=0&&forecast.physicalEv.uncertainty<=1);
  const checks={cardCountDistribution:Math.abs(cardCountProbabilitySum-1)<=1e-4,rankConsumptionTotal:rankExpectedTotalGap<=rankExpectedTotalTolerance,physicalEvRanges:physicalOk};
  return {valid:Object.values(checks).every(Boolean),checks,cardCountProbabilitySum,expectedNextCardCount:forecast.expectedNextCardCount,rankExpectedConsumptionTotal,rankExpectedTotalGap,rankExpectedTotalTolerance,physicalEv:forecast.physicalEv||null};
}
function dataQuality(seq){const directionalRounds=directionalRoundCount(seq);return {directionalRounds,stage:directionalRounds<12?"cold":directionalRounds<20?"warm":"ready",entryEligible:directionalRounds>=12,preferredEntry:directionalRounds>=20};}
function effectiveProgressRound(roundIndex,progressWeight){
  if(!stageProgressV3Enabled()||!Number.isFinite(+progressWeight))return clip(+roundIndex||1,1,70);
  return 70*Math.cbrt(clip(+progressWeight||0));
}
function applyProbabilityBounds(rawPB,roundIndex,noiseScore,progressWeight=null){
  if(!stageProgressV3Enabled()){
    const pair=roundIndex<=40?EARLY_BOUNDS:(roundIndex>50&&noiseScore<=PHYSICS_NOISE_LOW_THRESHOLD?LATE_CLEAN_BOUNDS:DEFAULT_BOUNDS);
    return {value:clip(rawPB,pair[0],pair[1]),low:pair[0],high:pair[1]};
  }
  if(stageProgressV4Enabled()){
    const progress=Math.cbrt(clip(+progressWeight||0)),middle=clip((progress-.35)/.35),late=clip((progress-.70)/.30),clean=clip((PHYSICS_NOISE_LOW_THRESHOLD-clip(noiseScore))/PHYSICS_NOISE_LOW_THRESHOLD);
    const low=EARLY_BOUNDS[0]-.05*middle-.05*late*clean,high=EARLY_BOUNDS[1]+.05*middle+.05*late*clean;
    return {value:clip(rawPB,low,high),low,high};
  }
  const progressRound=effectiveProgressRound(roundIndex,progressWeight);
  let low=interpAnchors(progressRound,[[1,EARLY_BOUNDS[0]],[40,EARLY_BOUNDS[0]],[50,DEFAULT_BOUNDS[0]]]);
  let high=interpAnchors(progressRound,[[1,EARLY_BOUNDS[1]],[40,EARLY_BOUNDS[1]],[50,DEFAULT_BOUNDS[1]]]);
  if(progressRound>50&&noiseScore<=PHYSICS_NOISE_LOW_THRESHOLD){
    low=interpAnchors(progressRound,[[50,DEFAULT_BOUNDS[0]],[70,LATE_CLEAN_BOUNDS[0]]]);
    high=interpAnchors(progressRound,[[50,DEFAULT_BOUNDS[1]],[70,LATE_CLEAN_BOUNDS[1]]]);
  }
  return {value:clip(rawPB,low,high),low,high};
}
function dynamicEmaAlpha(roundIndex,noiseScore,config,progressWeight=null,physicsDiagnostics=null){
  if(cardcountProfile?.enabled){
    const noise=clip(noiseScore),u=clip(physicsDiagnostics?.physics_uncertainty??.5);
    return clip(profileValue("decision_alpha",roundIndex)-.04*Math.max(0,noise-.55)/.45-.03*Math.max(0,u-.5)/.5,.35,.75);
  }
  if(!adaptiveStageEnabled()){
    const stage=roundIndex<=40?"early":roundIndex>50?"late":"middle",range=stage==="early"?[.35,.45]:stage==="middle"?[.50,.60]:[.65,.75];
    const base=Number.isFinite(+config[stage+"_alpha"])?+config[stage+"_alpha"]:(range[0]+range[1])/2,gain=Math.max(0,Number.isFinite(+config.noise_gain)?+config.noise_gain:.05);
    return clip(base+gain*(.5-clip(+noiseScore||0)),range[0],range[1]);
  }
  if(stageProgressV4Enabled()){
    const progress=Math.cbrt(clip(+progressWeight||0)),base=interpAnchors(progress,[[0,.38],[.20,.41],[.35,.46],[.50,.53],[.65,.59],[.75,.63],[.85,.67],[.95,.70],[1,.72]]),noise=clip(noiseScore),adjustment=.03*Math.max(0,.5-noise)/.5-.05*Math.max(0,noise-.5)/.5;
    const diagnostics=physicsDiagnostics||lastPhysicsDiagnostics||{},directUncertainty=clip(diagnostics.physics_uncertainty??.5),physicsAdjustment=-.04*Math.max(0,directUncertainty-.50)/.50;
    return clip(base+adjustment+physicsAdjustment,.35,.75);
  }
  const early=Number.isFinite(+config.early_alpha)?+config.early_alpha:.40,middle=Number.isFinite(+config.middle_alpha)?+config.middle_alpha:.55,late=Number.isFinite(+config.late_alpha)?+config.late_alpha:.70;
  const progressRound=stageProgressV3Enabled()?effectiveProgressRound(roundIndex,progressWeight):roundIndex;
  const base=interpAnchors(progressRound,[[1,early],[20,early],[40,(early+middle)/2],[50,middle],[60,(middle+late)/2],[70,late]]);
  const gain=Math.max(0,Number.isFinite(+config.noise_gain)?+config.noise_gain:.05);
  const diagnostics=physicsDiagnostics||lastPhysicsDiagnostics||{},directUncertainty=clip(diagnostics.physics_uncertainty??.5),physicsAdjustment=-.04*Math.max(0,directUncertainty-.50)/.50;
  return clip(base+gain*(.5-clip(+noiseScore||0))+physicsAdjustment,.35,.75);
}
function applyDynamicSmoothing(clippedPB,roundIndex,noiseScore,bounds,progressWeight=null){
  const configured=final56Bundle?.smoothing||{};
  // Physics Primary V13: EMA is part of the primary Physics path, not an
  // optional XGB-selected post-process.  If model selection disables smoothing,
  // fall back to the locked Physics-primary dynamic profile.
  const config=(configured.method==="dynamic_post_clip_ema"&&configured.enabled===true)||configured.method==="causal_ema"
    ?configured
    :{method:"dynamic_post_clip_ema",enabled:true,profile:"physics_primary_v13",early_alpha:.40,middle_alpha:.55,late_alpha:.70,noise_gain:.04};
  const dynamic=config.method==="dynamic_post_clip_ema"&&config.enabled===true;
  const legacy=config.method==="causal_ema"&&(+config.strength||0)>0;
  const shoeId=getShoeId(),rows=readRows();let previous=null;
  for(let i=rows.length-1;i>=0;i--){const row=rows[i];if(row?.shoe_id!==shoeId||+row.round_index>=roundIndex)continue;
    const value=Number.isFinite(+row.physics_smoothed_p_b)?+row.physics_smoothed_p_b:(Number.isFinite(+row.primary_smoothed_p_b)?+row.primary_smoothed_p_b:NaN);
    if(Number.isFinite(value)){previous=value;break;}}
  if(previous===null)return {value:clippedPB,alpha:1,strength:0,profile:cardcountProfile?.enabled?cardcountProfile.name:config.profile||"first_round",applied:false};
  const alpha=dynamic?dynamicEmaAlpha(roundIndex,noiseScore,config,progressWeight,lastPhysicsDiagnostics):1-clip(+config.strength||0,0,.15);
  const value=clip(alpha*clippedPB+(1-alpha)*previous,bounds.low,bounds.high);
  return {value,alpha,strength:1-alpha,profile:cardcountProfile?.enabled?cardcountProfile.name:config.profile||(legacy?"legacy":"custom"),applied:true};
}
function dynamicBand(roundIndex,noiseScore,bandConfig,config){
  const stage=roundIndex<=40?"early":roundIndex<=50?"middle":"late";
  const number=(obj,key,fallback)=>typeof obj?.[key]==="number"&&Number.isFinite(obj[key])?obj[key]:fallback;
  const noise=typeof noiseScore==="number"&&Number.isFinite(noiseScore)?clip(noiseScore):1;
  const progress=clip(roundIndex/getEstimatedTotalHands());
  const low=clip(number(config,"low_noise",.60),0,.95),high=clip(number(config,"high_noise",.75),low+.001,1);
  const gate=clip((high-noise)/(high-low));
  const base=Math.max(0,number(bandConfig,stage,0));
  const progressRelief=clip(number(config,"progress_relief",.4),0,.4)*progress*gate;
  const noisePenalty=Math.max(0,noise-number(config,"noise_reference",.55))*Math.max(0,number(config,"noise_gain",.015));
  const extra=progress>number(config,"extra_progress",.65)&&noise<low?Math.max(0,number(config,"extra_relief",.0005)):0;
  const minimum=Math.max(0,number(config.minimum_by_stage,stage,number(bandConfig,"minimum",.003)));
  const maximum=Math.max(minimum,number(bandConfig,"maximum",.02));
  const finalRelief=Math.max(0,number(config.final_band_relief,stage,0))*gate;
  return {stage,progress,noise,reliefScale:gate,bandMinimum:minimum,finalBandRelief:finalRelief,
    confidenceBand:clip(base*(1-progressRelief)+noisePenalty-extra,minimum,maximum),progressRelief,noisePenalty,extra};
}
function decisionPolicy(roundIndex,noiseScore){
  const threshold=final56Bundle?.ev_thresholds||{},pick=(key,fallback)=>Number.isFinite(+threshold[key])?+threshold[key]:fallback;
  const base=roundIndex<=40?pick("early",.020):roundIndex>50?pick("late",.005):pick("middle",.010);
  const config=final56Bundle?.decision_policy||{};
  const profile=String(config.profile||final56Bundle?.decision_policy_profile||(config.enabled===true?"custom":"hard_ev"));
  if(config.enabled!==true)return {enabled:false,profile,minEv:base,activationEv:base,softBand:0,minConfidence:0,confidenceBand:0,strongMargin:0,bandMinimum:0,volumeGuard:{}};
  const value=(key,fallback)=>Number.isFinite(+config[key])?+config[key]:fallback;
  const noiseThreshold=value("noise_threshold",.78),noise=clip(+noiseScore||0),lowNoise=noise<=noiseThreshold;
  const middle=roundIndex>40&&roundIndex<=50,late=roundIndex>50;
  const relief=lowNoise?(middle?value("middle_relief",.001):late?value("late_relief",.001):0):0;
  const maxPenalty=Math.max(0,value("max_noise_ev_penalty",.001));
  const penalty=Math.min(maxPenalty,Math.max(0,noise-noiseThreshold)*(maxPenalty/Math.max(1e-9,1-noiseThreshold)));
  const minEv=Math.max(0,base-relief+penalty);
  const softBand=lowNoise?(middle?value("middle_soft_band",.001):late?value("late_soft_band",.0015):0):0;
  const bandConfig=config.confidence_band&&typeof config.confidence_band==="object"?config.confidence_band:{};
  const stage=roundIndex<=40?"early":roundIndex<=50?"middle":"late";
  const bandBase=Math.max(0,Number.isFinite(+bandConfig[stage])?+bandConfig[stage]:0);
  const bandReference=clip(Number.isFinite(+bandConfig.noise_reference)?+bandConfig.noise_reference:.50);
  const bandGain=Math.max(0,Number.isFinite(+bandConfig.noise_gain)?+bandConfig.noise_gain:0);
  const cleanLate=stage==="late"&&noise<=PHYSICS_NOISE_LOW_THRESHOLD;
  const cleanLateRelief=cleanLate?Math.max(0,Number.isFinite(+bandConfig.clean_late_relief)?+bandConfig.clean_late_relief:0):0;
  let bandMinimum=Math.max(0,Number.isFinite(+bandConfig.minimum)?+bandConfig.minimum:0);
  const bandMaximum=Math.max(bandMinimum,Number.isFinite(+bandConfig.maximum)?+bandConfig.maximum:.5);
  let confidenceBand=clip(bandBase+bandGain*Math.max(0,noise-bandReference)-cleanLateRelief,bandMinimum,bandMaximum);
  const dynamic=config.dynamic_band?.enabled===true?dynamicBand(roundIndex,noiseScore,bandConfig,config.dynamic_band):null;
  if(dynamic){confidenceBand=dynamic.confidenceBand;bandMinimum=dynamic.bandMinimum;}
  const strong=bandConfig.strong_margin&&typeof bandConfig.strong_margin==="object"?bandConfig.strong_margin:{};
  const strongMargin=Math.max(0,Number.isFinite(+strong[stage])?+strong[stage]:0);
  // Every new relief route shares the same noise gate; unknown noise is high.
  const reliefScale=dynamic?.reliefScale??1;
  const safeMinEv=dynamic?Math.max(0,base-relief*reliefScale+Math.min(maxPenalty,Math.max(0,dynamic.noise-noiseThreshold)*maxPenalty/Math.max(1e-9,1-noiseThreshold))):minEv;
  const safeSoftBand=softBand*reliefScale;
  return {enabled:true,profile,stage,progress:dynamic?.progress??clip(roundIndex/getEstimatedTotalHands()),noise:dynamic?.noise??noise,
    minEv:safeMinEv,activationEv:Math.max(0,safeMinEv-safeSoftBand),softBand:safeSoftBand,
    minConfidence:Math.max(0,value("min_confidence",.0005)),confidenceBand,strongMargin,bandMinimum,
    reliefScale,finalBandRelief:dynamic?.finalBandRelief??LOW_SKIP_FINAL_BAND_RELIEF,dynamicBand:dynamic,volumeGuard:config.volume_guard||{}};
}
function softConfidence(edge,policy){
  if(edge<=policy.activationEv)return 0;
  const premium=Math.max(0,edge-policy.minEv);
  return policy.softBand<=0?premium:Math.max(policy.minConfidence,premium,.5*Math.min(policy.softBand,edge-policy.activationEv));
}
function volumeGuardState(roundIndex,probabilityB,policy){
  const config=policy?.volumeGuard||{};
  if(policy?.enabled!==true||config.enabled!==true||(policy.reliefScale??1)<=0)return {active:false,bandRelief:0,evRelief:0};
  const window=Math.max(1,Math.floor(+config.window||16)),minimum=Math.max(0,Math.floor(+config.min_history||8));
  const shoeId=getShoeId(),rows=readRows().filter(row=>row?.shoe_id===shoeId&&+row.round_index<roundIndex&&typeof row.final_p_b==="number"&&Number.isFinite(row.final_p_b)).slice(-window);
  if(rows.length<minimum)return {active:false,bandRelief:0,evRelief:0};
  const p=clip(probabilityB),expectedProbability=value=>Math.max(clip(value),1-clip(value));
  let actions=0,expected=0,baseline=0;
  for(const row of rows){
    const priorP=clip(+row.final_p_b),priorPlayer=1-priorP;
    const priorActivation=Math.max(0,Number.isFinite(+row.activation_ev)?+row.activation_ev:policy.activationEv);
    let baseAction=(priorP*.95-priorPlayer)>priorActivation||(priorPlayer-priorP)>priorActivation;
    let expectedCorrect=expectedProbability(priorP);
    if(EV_CONSISTENCY_V1){
      const storedTie=Number.isFinite(+row.p_tie)?clip(+row.p_tie):
        Array.isArray(row.physics_48d)&&Number.isFinite(+row.physics_48d[25])?clip(+row.physics_48d[25]):0;
      const directionalMass=clip(1-storedTie),priorBankerMass=directionalMass*priorP,priorPlayerMass=directionalMass*(1-priorP);
      const priorEvBanker=priorBankerMass*.95-priorPlayerMass,priorEvPlayer=priorPlayerMass-priorBankerMass;
      baseAction=priorEvBanker>priorActivation||priorEvPlayer>priorActivation;
      expectedCorrect=Math.max(priorBankerMass,priorPlayerMass);
    }
    const action=["B","P","莊 B","閒 P"].includes(String(row.predicted_direction||""));
    if(action){actions++;expected+=expectedCorrect;}
    if(baseAction)baseline+=expectedCorrect;
  }
  const stage=roundIndex<=40?"early":roundIndex<=50?"middle":"late",rates=config.target_action_rate&&typeof config.target_action_rate==="object"?config.target_action_rate:{};
  const target=clip(Number.isFinite(+rates[stage])?+rates[stage]:0),floor=clip(Number.isFinite(+config.expected_correct_floor)?+config.expected_correct_floor:.95);
  const active=actions/rows.length<target||(baseline>0&&expected+1e-12<floor*baseline);
  const scale=policy.reliefScale??1,fraction=policy.dynamicBand?clip(Number.isFinite(+config.max_relief_fraction)?+config.max_relief_fraction:.25):1;
  const bandCap=policy.dynamicBand?policy.confidenceBand*fraction:Infinity,evCap=policy.dynamicBand?policy.activationEv*fraction:Infinity;
  return {active,target,actionRate:actions/rows.length,observations:rows.length,
    bandRelief:active?Math.min(bandCap,Math.max(0,+config.band_relief||0))*scale:0,
    evRelief:active?Math.min(evCap,Math.max(0,+config.ev_relief||0))*scale:0};
}

function directPhysicsProgressWeight(roundIndex,physics){
  const total=Math.max(ESTIMATED_TOTAL_HANDS_MIN,Math.min(ESTIMATED_TOTAL_HANDS_MAX,getEstimatedTotalHands()));
  const roundProgress=clip(roundIndex/total),uncertainty=clip(+physics?.[42]||0),cardProgress=clip((+physics?.[43]||0)/ESTIMATED_PLAYABLE_CARDS);
  const effectiveProgress=adaptiveStageEnabled()?clip((.55+.25*uncertainty)*roundProgress+(.45-.25*uncertainty)*cardProgress):roundProgress;
  return effectiveProgress**3;
}
function physicsConsistencyScore(physics){
  const pB=clip(+physics?.[23]||0),pP=clip(+physics?.[24]||0);
  const winnerSignal=pB-pP,pointSignal=Number.isFinite(+physics?.[46])?+physics[46]:0,modelEvSignal=Number.isFinite(+physics?.[41])?+physics[41]:0;
  const sign=v=>v>.015?1:v<-.015?-1:0,w=sign(winnerSignal);
  if(w===0)return .55;
  const signals=[sign(pointSignal),sign(modelEvSignal)].filter(v=>v!==0);
  if(!signals.length)return .70;
  const agree=signals.filter(v=>v===w).length;
  return clip(.45+.275*agree,0,1);
}
function calibratedPhysicsDirectionalPB(rawPB,roundIndex,uncertainty,consistency,pT=0){
  const p=clip(rawPB),u=clip(uncertainty),c=clip(consistency);
  if(roundIndex>40)return p;
  const stage=interpAnchors(roundIndex,[[1,.72],[5,.76],[10,.81],[15,.85],[20,.89],[25,.92],[30,.94],[35,.96],[40,1]]);
  let reliability=clip(stage*(1-.03*u)*(.97+.03*c),.82,1);
  if(EV_CONSISTENCY_V1)reliability=clip(reliability*(1-EARLY_TIE_PENALTY*clip(pT)),.82,1);
  return clip(.5+(p-.5)*reliability);
}
function physicsPrecisionPenalty(roundIndex,uncertainty,consistency){
  const u=clip(uncertainty),c=clip(consistency),early=roundIndex<=35?1:roundIndex<45?(45-roundIndex)/10:0;
  return LOW_SKIP_PRECISION_SCALE * (
    .0045*Math.max(0,u-.35)/.65 + .0035*(1-c) + .0015*early*(1-c)
  );
}
function directPhysicsPrimary(physics,roundIndex,macroFeatures=null){
  const pB=clip(+physics?.[23]||0),pP=clip(+physics?.[24]||0),pT=clip(+physics?.[25]||0);
  const directionalMass=Math.max(1e-9,pB+pP),rawDirectionalPB=clip(pB/directionalMass);
  const uncertainty=clip(+physics?.[42]||0),consistency=physicsConsistencyScore(physics);
  const calibratedPB=calibratedPhysicsDirectionalPB(rawDirectionalPB,roundIndex,uncertainty,consistency,pT);
  const progressWeight=directPhysicsProgressWeight(roundIndex,physics),noise=physicsNoiseScore(physics,roundIndex,Math.cbrt(progressWeight));
  const policy=decisionPolicy(roundIndex,noise);
  // Primary EV is computed only from the Direct Physics winner head.
  const baseEvBanker=pB*.95-pP,baseEvPlayer=pP-pB;
  const macro=MACRO?MACRO.adjustment(calibratedPB,macroFeatures,clip(roundIndex/getEstimatedTotalHands()),noise,{...(final56Bundle?.decision_policy?.macro_ema||{}),...(cardcountProfile?.enabled?cardcountProfile.macro:{})}):{value:calibratedPB,delta:0,reason:"module_unavailable"};
  const directionalPB=macro.value,distance=Math.abs(directionalPB-.5);
  // Soft correction of Physics EV only; tie mass and Direct MLP 48D stay fixed.
  const evBanker=baseEvBanker+1.95*directionalMass*macro.delta,evPlayer=baseEvPlayer-2*directionalMass*macro.delta;
  const modelEvBanker=Number.isFinite(+physics?.[39])?+physics[39]:evBanker,modelEvPlayer=Number.isFinite(+physics?.[40])?+physics[40]:evPlayer;
  const precisionPenalty=physicsPrecisionPenalty(roundIndex,uncertainty,consistency);
  // V14.3 Low-Skip: retain the safety terms but reduce their dominance.
  const physicsActivationEv=Math.max(0,policy.activationEv-LOW_SKIP_EV_RELIEF*(policy.reliefScale??1))+precisionPenalty;
  const physicsConfidenceBand=Math.max(
    policy.bandMinimum,
    policy.confidenceBand*LOW_SKIP_CONFIDENCE_SCALE+
      LOW_SKIP_CONSISTENCY_BAND_GAIN*(1-consistency)+
      LOW_SKIP_UNCERTAINTY_BAND_GAIN*Math.max(0,uncertainty-.50)
  );
  let candidate="Skip";
  if(distance>=physicsConfidenceBand){
    const bankerOk=evBanker>=evPlayer&&evBanker>physicsActivationEv+BANKER_EV_FLOOR;
    const playerOk=evPlayer>evBanker&&evPlayer>physicsActivationEv+PLAYER_EV_FLOOR;
    if(bankerOk)candidate="B";
    else if(playerOk)candidate="P";
  }
  const earlyTiePenaltyApplied=EV_CONSISTENCY_V1&&roundIndex<=40&&EARLY_TIE_PENALTY>0&&pT>0;
  return {pB,pP,pT,rawDirectionalPB,directionalPB,progressWeight,noise,policy,distance,uncertainty,consistency,precisionPenalty,earlyTiePenaltyApplied,
    physicsActivationEv,physicsConfidenceBand,evBanker,evPlayer,baseEvBanker,baseEvPlayer,modelEvBanker,modelEvPlayer,candidate,macro};
}

function auxiliaryCandidateFilter(candidate,corePB,xgbPB){
  if(candidate!=="B"&&candidate!=="P")return {decision:"skip",reason:"physics_candidate_skip",coreSupport:.5,xgbSupport:.5,support:.5,shrink:0};
  const coreSupport=candidate==="B"?clip(corePB):1-clip(corePB),xgbSupport=candidate==="B"?clip(xgbPB):1-clip(xgbPB);
  const support=xgbSupport; // Core/road support is diagnostic only.
  const hardConflict=xgbSupport<.35&&coreSupport<.40;
  return hardConflict
    ?{decision:"keep",reason:"aux_conflict_advisory_only",coreSupport,xgbSupport,support,shrink:1,advisoryOnly:true}
    :{decision:"keep",reason:"aux_support_advisory_only",coreSupport,xgbSupport,support,shrink:1,advisoryOnly:true};
}
// V15 read-only decision audit. Does NOT alter probability, action or any Guard.
function v15SkipDiagnostics(primary,auxFilter,decision,mode,error=""){
  if(mode!=="physics_primary"||!primary)
    return {schema_version:1,final_action:"Skip",primary_reason:"physics_unavailable",
      failed_conditions:["physics_unavailable"],stage:null,runtime_error:String(error||""),physics_pt:null,
      ev_consistency_active:!!EV_CONSISTENCY_V1,early_tie_penalty_applied:false};
  const p=primary,d=decision||{},trace=d.v15Trace||{},failed=[];
  if(p.distance+1e-12<p.physicsConfidenceBand)failed.push("physics_confidence_band");
  const physicsEvOk=(p.evBanker>=p.evPlayer&&p.evBanker>p.physicsActivationEv+BANKER_EV_FLOOR)||
    (p.evPlayer>p.evBanker&&p.evPlayer>p.physicsActivationEv+PLAYER_EV_FLOOR);
  if(!physicsEvOk)failed.push("physics_ev_gate");
  if(p.candidate==="Skip"&&failed.length===0)failed.push("physics_other");
  if(p.candidate!=="Skip"){
    if(auxFilter?.decision==="skip")failed.push("auxiliary_joint_conflict");
    if(!trace.aligned)failed.push("final_direction_alignment");
    if(trace.distance+1e-12<trace.effectiveBand)failed.push("final_confidence_band");
    if(!(trace.candidateEdge>trace.effectiveActivationEv))failed.push("final_ev_gate");
  }
  if(d.direction==="Skip"&&failed.length===0)failed.push("unclassified_skip");
  return {schema_version:1,stage:p.policy?.stage??null,round_index:trace.roundIndex??null,
    final_action:d.direction||"Skip",primary_candidate:p.candidate,
    primary_reason:d.direction!=="Skip"?"entry":failed[0],
    failed_conditions:d.direction!=="Skip"?[]:failed,
    physics_distance:p.distance,physics_band:p.physicsConfidenceBand,
    physics_ev_banker:p.evBanker,physics_ev_player:p.evPlayer,physics_activation_ev:p.physicsActivationEv,
    final_distance:trace.distance??null,final_effective_band:trace.effectiveBand??null,
    final_candidate_edge:trace.candidateEdge??null,final_activation_ev:trace.effectiveActivationEv??null,
    xgb_aux_decision:auxFilter?.decision??null,volume_guard_active:d.volumeGuardActive===true,
    physics_pt:p.pT,ev_consistency_active:!!EV_CONSISTENCY_V1,
    early_tie_penalty_applied:p.earlyTiePenaltyApplied===true};
}
function preserveCandidateDirection(candidate,probabilityB){
  if(candidate==="B")return Math.max(.5,clip(probabilityB));
  if(candidate==="P")return Math.min(.5,clip(probabilityB));
  return .5;
}

function nextHandForecast(physics){
  const unavailable={available:false,source:"direct_physics_winner_head",direction:null,probabilities:null,directional:null};
  if(!Array.isArray(physics)||physics.length!==PHYSICS_DIM)return unavailable;
  const values=physics.slice(23,26);
  if(!values.every(v=>typeof v==="number"&&Number.isFinite(v)&&v>=0))return unavailable;
  const [b,p,t]=values,total=b+p+t,mass=b+p;
  if(total<=0||mass<=0)return unavailable;
  const directional={B:b/mass,P:p/mass};
  return {available:true,source:"direct_physics_winner_head",direction:directional.B===directional.P?"Balanced":directional.B>directional.P?"B":"P",
    probabilities:{B:b/total,P:p/total,T:t/total},directional,
    uncertainty:Number.isFinite(physics[42])?clip(physics[42]):null,
    weak_signal:Math.abs(directional.B-.5)<.005};
}
function applyFinalPrediction(seq,corePrediction=null){
  // B/P/T -> Direct Physics 213D->48D -> stateless PF500 -> Physical EV -> Physics candidate
  // -> Clip/EMA -> Frozen Core + 57D XGB -> Final EV Guard -> Volume Guard.
  let particleDiagnostics=null;
  let physics=null,physicsDirect=null,physicsForecast=null,physicsIntegrityReport=null,physicalEv=null,error="";
  let resolvedCore=null,original7=null,corePB=.5,extended=null,xgbAuxPB=.5,primary=null,auxFilter=null;
  let rawPB=.5,clippedPB=.5,smoothedPB=.5,filteredPB=.5,finalPB=.5,bounds=null;
  let smoothingAlpha=1,smoothingStrength=0,smoothingProfile="off",evDecision=null,mode="physics_primary";
  const executionOrder=[];

  try{
    if(physicsBundle?.trained){
      executionOrder.push("direct_physics");
      physicsDirect=predictPhysics(seq);
      physics=physicsDirect;
      // Each estimate creates 500 new particles and replays this history only.
      // Keep Direct Physics as a safe fallback if the optional particle layer fails.
      if(PARTICLE500&&typeof PARTICLE500.estimate==="function"){
        try{
          const estimate=PARTICLE500.estimate(seq,physicsDirect,"quality",cardcountProfile?.enabled?{...cardcountProfile.pf,max_blend:profileValue("pf_max",seq.length+1),forecast_draws:cardcountProfile.anchors?.pf_draws?profileValue("pf_draws",seq.length+1):cardcountProfile.pf.forecast_draws}:{});
          if(!Array.isArray(estimate?.physics)||estimate.physics.length!==PHYSICS_DIM||!estimate.physics.every(Number.isFinite))throw new Error("invalid particle physics");
          physics=estimate.physics;particleDiagnostics=estimate.diagnostics;
          executionOrder.push("particle_filter");
        }catch(e){error="particle500:"+String(e?.message||e);}
      }
      lastPhysicsDiagnostics={...(lastPhysicsDiagnostics||{}),enabled:true,
        physics_source:particleDiagnostics?"direct_multitask_mlp_plus_particle500":"direct_multitask_mlp",particle_filter_enabled:!!particleDiagnostics,physics_uncertainty:clip(physics[42]),
        expected_consumed_cards:+physics[43]||0,physical_ev_banker:+physics[39]||0,
        physical_ev_player:+physics[40]||0,physical_ev_gap:+physics[41]||0};
      physicsForecast=unpackPhysicsForecast(physics);
      physicsIntegrityReport=physicsIntegrity(physics);
      physicalEv=physicsForecast?.physicalEv||null;
      if(physicalEv)executionOrder.push("physical_ev");
    }
  }catch(e){
    error=error?error+";physics:"+String(e?.message||e||"runtime_error"):"physics:"+String(e?.message||e||"runtime_error");
    physics=physicsDirect||null;physicalEv=null;
  }

  if(physics&&physicsBundleUsesPhysicalEv()){
    const roundIndex=Math.max(1,Math.min(70,seq.length+1));
    const macroFeatures=getMacroFeatures(seq);
    primary=directPhysicsPrimary(physics,roundIndex,macroFeatures);
    primary.macroFeatures=macroFeatures;
    executionOrder.push("physics_candidate");
    rawPB=primary.directionalPB;
    bounds=applyProbabilityBounds(rawPB,roundIndex,primary.noise,primary.progressWeight);
    clippedPB=bounds.value;
    const smoothing=applyDynamicSmoothing(clippedPB,roundIndex,primary.noise,bounds,primary.progressWeight);
    smoothedPB=smoothing.value;smoothingAlpha=smoothing.alpha;smoothingStrength=smoothing.strength;smoothingProfile=smoothing.profile;
    executionOrder.push("physics_ema");

    try{
      executionOrder.push("frozen_core");
      resolvedCore=corePrediction||CORE.hazardChoose(seq);
      original7=buildOriginal7(seq,resolvedCore);
      corePB=original7.core_p_b;

      if(final56Bundle?.trained&&modelSemanticsCompatible()){
        // Keep the auxiliary XGB on its trained Direct Physics feature distribution.
        extended=buildExtended(corePB,original7,physicsDirect||physics);
        executionOrder.push("xgb_aux_filter");
        xgbAuxPB=predictFinalProbability(final56Bundle,extended,final56Bundle.feature_names||EXTENDED_NAMES);
        auxFilter=auxiliaryCandidateFilter(primary.candidate,corePB,xgbAuxPB);
      }else{
        auxFilter=primary.candidate==="Skip"
          ?{decision:"skip",reason:"physics_candidate_skip",coreSupport:.5,xgbSupport:.5,support:.5,shrink:0}
          :{decision:"keep",reason:"xgb_unavailable_physics_primary",coreSupport:primary.candidate==="B"?corePB:1-corePB,xgbSupport:.5,support:.5,shrink:1};
        if(final56Bundle?.trained&&!modelSemanticsCompatible())error=error?error+";feature_semantics_mismatch":"feature_semantics_mismatch";
      }
    }catch(e){
      const message=String(e?.message||e||"runtime_error");error=error?error+";"+message:message;
      if(!resolvedCore)resolvedCore=corePrediction||CORE.hazardChoose(seq);
      if(!original7)original7=buildOriginal7(seq,resolvedCore);
      corePB=original7.core_p_b;
      auxFilter=primary.candidate==="Skip"
        ?{decision:"skip",reason:"physics_candidate_skip",coreSupport:.5,xgbSupport:.5,support:.5,shrink:0}
        :{decision:"keep",reason:"aux_runtime_error_physics_primary",coreSupport:primary.candidate==="B"?corePB:1-corePB,xgbSupport:.5,support:.5,shrink:1};
    }

    // The auxiliary layer must not erase or reverse the Physics probability.
    // Skip is a bet decision, not a replacement forecast of 50/50.
    filteredPB=smoothedPB;
    finalPB=filteredPB;

    executionOrder.push("final_ev_guard");
    let pPlayer=1-finalPB,evBanker=finalPB*.95-pPlayer,evPlayer=pPlayer-finalPB;
    if(EV_CONSISTENCY_V1){
      const directionalMass=clip(primary.pB+primary.pP),pBanker=directionalMass*finalPB;
      pPlayer=directionalMass*(1-finalPB);
      evBanker=pBanker*.95-pPlayer;
      evPlayer=pPlayer-pBanker;
    }
    const policy=primary.policy,candidate=primary.candidate;
    const baseBand=Math.max(policy.confidenceBand,primary.physicsConfidenceBand),baseActivation=Math.max(policy.activationEv,primary.physicsActivationEv),distance=Math.abs(finalPB-.5);
    const candidateEdge=candidate==="B"?evBanker:candidate==="P"?evPlayer:0;
    const aligned=candidate==="B"?finalPB>.5:candidate==="P"?finalPB<.5:false;
    const basePass=auxFilter?.decision!=="skip"&&candidate!=="Skip"&&aligned&&distance>=baseBand&&candidateEdge>baseActivation;

    executionOrder.push("volume_guard");
    const guard=volumeGuardState(roundIndex,finalPB,policy);
    const effectiveBand=Math.max(policy.bandMinimum,baseBand-guard.bandRelief-(policy.finalBandRelief??LOW_SKIP_FINAL_BAND_RELIEF)),effectiveActivationEv=Math.max(0,baseActivation-guard.evRelief);
    const guardedPass=auxFilter?.decision!=="skip"&&candidate!=="Skip"&&aligned&&distance>=effectiveBand&&candidateEdge>effectiveActivationEv;
    const direction=basePass||guardedPass?candidate:"Skip";
    const entryTier=direction==="Skip"?"skip":distance>=effectiveBand+policy.strongMargin?"strong":"weak";
    evDecision={
      pTie:primary.pT,pPlayer,evBanker,evPlayer,
      physicalEvBanker:primary.evBanker,physicalEvPlayer:primary.evPlayer,
      minEv:policy.minEv,activationEv:policy.activationEv,effectiveActivationEv,
      softBand:policy.softBand,confidenceBand:policy.confidenceBand,effectiveConfidenceBand:effectiveBand,
      volumeGuardActive:guard.active,entryTier,stakeMultiplier:entryTier==="weak"?.5:entryTier==="strong"?1:0,
      policyEnabled:policy.enabled,policyProfile:policy.profile,direction,
      finalDirection:direction==="B"?"莊 B":direction==="P"?"閒 P":"觀望 Skip",
      confidence:direction==="Skip"?0:softConfidence(candidateEdge,{...policy,activationEv:effectiveActivationEv}),
      basePass,guardedPass,candidate,
      v15Trace:{roundIndex,distance,baseBand,baseActivation,candidateEdge,aligned,
        effectiveBand,effectiveActivationEv,guardBandRelief:guard.bandRelief,guardEvRelief:guard.evRelief}
    };
    mode="physics_primary";
  }else{
    executionOrder.push("physics_unavailable_skip");
    resolvedCore=corePrediction||CORE.hazardChoose(seq);
    original7=buildOriginal7(seq,resolvedCore);
    corePB=original7.core_p_b;rawPB=.5;clippedPB=.5;smoothedPB=.5;filteredPB=.5;finalPB=.5;mode="physics_unavailable";
    error=error?error+";physics_primary_unavailable":"physics_primary_unavailable";
  }

  if(!resolvedCore){resolvedCore=corePrediction||CORE.hazardChoose(seq);original7=buildOriginal7(seq,resolvedCore);corePB=original7.core_p_b;}
  const direction=evDecision?.direction||"Skip",finalPP=1-finalPB;
  const confidence=evDecision?.confidence??0;
  const skipDiagnostics=v15SkipDiagnostics(primary,auxFilter,evDecision,mode,error);
  const physicalEvBanker=primary?.evBanker??physicalEv?.banker??null,physicalEvPlayer=primary?.evPlayer??physicalEv?.player??null,physicalEvGap=primary?(primary.evBanker-primary.evPlayer):(physicalEv?.gap??null),physicsUncertainty=primary?.uncertainty??physicalEv?.uncertainty??null;
  const rawForecast=nextHandForecast(physics);
  if(rawForecast.available&&particleDiagnostics)rawForecast.source="direct_physics_particle500_winner_head";
  const calibrated=LOCAL_CALIBRATION?.apply(rawForecast,seq,getShoeId(),browserCalibrationTag,localCalibrationConfig());
  const forecast=calibrated?.forecast||rawForecast;
  return {...resolvedCore,raw_next_hand_forecast:rawForecast,browser_calibration:calibrated?.status||null,next_hand_forecast:forecast,forecast_direction:forecast.direction,bet_direction:direction,direction,final_direction:evDecision?.finalDirection||(direction==="B"?"莊 B":direction==="P"?"閒 P":"觀望 Skip"),confidence,
    ev_banker:evDecision?.evBanker??null,ev_player:evDecision?.evPlayer??null,
    physical_ev_banker:physicalEvBanker,physical_ev_player:physicalEvPlayer,physical_ev_gap:physicalEvGap,physics_uncertainty:physicsUncertainty,
    min_ev:evDecision?.minEv??null,activation_ev:evDecision?.activationEv??null,effective_activation_ev:evDecision?.effectiveActivationEv??null,
    soft_band:evDecision?.softBand??0,confidence_band:evDecision?.confidenceBand??0,effective_confidence_band:evDecision?.effectiveConfidenceBand??0,
    volume_guard_active:evDecision?.volumeGuardActive??false,entry_tier:evDecision?.entryTier??"skip",stake_multiplier:evDecision?.stakeMultiplier??0,
    decision_policy_enabled:evDecision?.policyEnabled??false,decision_policy_profile:evDecision?.policyProfile??"hard_ev",probabilities:{B:finalPB,P:finalPP},
    regime:mode==="physics_primary"?(direction==="Skip"?"Direct Physics觀望":auxFilter?.decision==="downgrade"?"Direct Physics / XGB降權":"Direct Physics / XGB輔助提醒"):"模型未就緒／觀望",
    finalProbability:{version:VERSION,active:mode==="physics_primary",mode,
      skip_diagnostics:skipDiagnostics,
      particleDiagnostics,primary_source:particleDiagnostics?"direct_physics_particle500_physical_ev":"direct_physics_physical_ev",primary_candidate:primary?.candidate??"Skip",
      macro_ema:primary?.macroFeatures??null,macro_adjustment:primary?.macro??null,
      decision_stage:primary?.policy?.stage??(seq.length<40?"early":seq.length<50?"middle":"late"),decision_progress:clip((seq.length+1)/getEstimatedTotalHands()),dynamic_band:primary?.policy?.dynamicBand??null,
      physics_raw_p_b:primary?.rawDirectionalPB??null,physics_calibrated_p_b:primary?.directionalPB??null,physics_clipped_p_b:clippedPB,physics_smoothed_p_b:smoothedPB,
      physics_consistency:primary?.consistency??null,physics_precision_penalty:primary?.precisionPenalty??null,physics_activation_ev:primary?.physicsActivationEv??null,physics_confidence_band:primary?.physicsConfidenceBand??null,
      low_skip_policy:{confidence_scale:LOW_SKIP_CONFIDENCE_SCALE,ev_relief:LOW_SKIP_EV_RELIEF,final_band_relief:LOW_SKIP_FINAL_BAND_RELIEF},
      rawPB,clippedPB,smoothedPB,filteredPB,finalPB,bounds,smoothingAlpha,smoothingStrength,smoothingProfile,
      xgb_role:"advisory_only_no_veto_no_flip",xgb_aux_p_b:xgbAuxPB,core_aux_p_b:corePB,aux_filter:auxFilter,
      physical_ev_banker:physicalEvBanker,physical_ev_player:physicalEvPlayer,physical_ev_gap:physicalEvGap,physics_uncertainty:physicsUncertainty,
      execution_order:executionOrder,p_tie:evDecision?.pTie??primary?.pT??null,p_player:evDecision?.pPlayer??null,
      ev_banker:evDecision?.evBanker??null,ev_player:evDecision?.evPlayer??null,min_ev:evDecision?.minEv??null,
      activation_ev:evDecision?.activationEv??null,effective_activation_ev:evDecision?.effectiveActivationEv??null,
      soft_band:evDecision?.softBand??0,confidence_band:evDecision?.confidenceBand??0,effective_confidence_band:evDecision?.effectiveConfidenceBand??0,
      volume_guard_active:evDecision?.volumeGuardActive??false,entry_tier:evDecision?.entryTier??"skip",stake_multiplier:evDecision?.stakeMultiplier??0,
      decision_policy_enabled:evDecision?.policyEnabled??false,decision_policy_profile:evDecision?.policyProfile??"hard_ev",
      coreDirection:resolvedCore.direction,finalDirection:evDecision?.finalDirection||(direction==="B"?"莊 B":direction==="P"?"閒 P":"觀望 Skip"),
      flipped_from_physics:primary?.candidate==="B"&&direction==="P"||primary?.candidate==="P"&&direction==="B",
      original7,physics,physicsDirect,physicsForecast,physicsIntegrity:physicsIntegrityReport,physicsDiagnostics:lastPhysicsDiagnostics,dataQuality:dataQuality(seq),extended,error}};
}

function readHistory(){
  if(typeof localStorage==="undefined")return[];
  for(const key of["bgs256d_frozen_6x15_forward_v18","bgs256d_frozen_6x15_sensitive_v17","bgs256d_frozen_6x15_bigroad_v16"]){
    try{const r=JSON.parse(localStorage.getItem(key)||"null");if(r&&Array.isArray(r.history))return r.history.filter(x=>["B","P","T"].includes(x)).slice(-500);}catch(_){}
  }return[];
}
function getShoeId(){try{let id=localStorage.getItem(SHOE_KEY)||"";if(!id){id="shoe_"+Date.now().toString(36)+"_"+Math.random().toString(36).slice(2,8);localStorage.setItem(SHOE_KEY,id);}return id;}catch(_){return"browser_shoe";}}
const MACRO_KEY="bgs_macro_observations_v1",MACRO_ARCHIVE_KEY="bgs_macro_observation_shoes_v1";
let macroJournal=null,macroState=null;
function readMacroArchive(){try{const value=JSON.parse(localStorage.getItem(MACRO_ARCHIVE_KEY)||"[]");return Array.isArray(value)?value:[];}catch(_){return[];}}
function rotateShoeId(){
  try{
    const current=macroJournal||JSON.parse(localStorage.getItem(MACRO_KEY)||"null");
    if(current?.events?.length){
      const archive=readMacroArchive().filter(shoe=>shoe.shoe_id!==current.shoe_id);archive.push(current);
      let count=archive.reduce((n,shoe)=>n+(shoe.events?.length||0),0);
      while(count>MAX_TRAINING_ROWS&&archive.length>1)count-=archive.shift().events.length;
      localStorage.setItem(MACRO_ARCHIVE_KEY,JSON.stringify(archive));
    }
  }catch(_){}
  // A full archive/quota must never prevent shoe isolation/reset.
  for(const key of [SHOE_KEY,PENDING_KEY,MACRO_KEY])try{localStorage.removeItem(key);}catch(_){}
  macroJournal=null;macroState=null;
  LOCAL_CALIBRATION?.clear();
}
function saveMacroJournal(){try{localStorage.setItem(MACRO_KEY,JSON.stringify(macroJournal));}catch(_){}}
function syncMacro(history){
  if(!MACRO)return null;
  const shoeId=getShoeId();
  if(!macroJournal||macroJournal.shoe_id!==shoeId){
    let saved=null;try{saved=JSON.parse(localStorage.getItem(MACRO_KEY)||"null");}catch(_){}
    macroJournal={shoe_id:shoeId,events:saved?.shoe_id===shoeId&&Array.isArray(saved.events)?saved.events.slice(0,500):[]};
    macroState=null;
  }
  const events=macroJournal.events;
  let common=0;
  while(common<Math.min(events.length,history.length)&&events[common]?.outcome===history[common])common++;
  if(!macroState||common<events.length){
    events.splice(common);macroState=MACRO.createState(shoeId);
    for(let i=0;i<events.length;i++){
      try{events[i]=MACRO.normalizeHand(events[i]);}catch(_){events[i]=MACRO.normalizeHand({outcome:history[i]});}
      macroUpdate(macroState,events[i]);
    }
  }
  for(let i=events.length;i<history.length;i++){
    const observed=MACRO.normalizeHand({outcome:history[i]});events.push(observed);macroUpdate(macroState,observed);
  }
  saveMacroJournal();return MACRO.snapshot(macroState);
}
function getMacroFeatures(history=readHistory()){return syncMacro(history);}
function recordCompletedHand(history,observation={}){
  if(!MACRO||!history.length)return null;
  const outcome=history.at(-1);
  if(observation.outcome!==undefined&&observation.outcome!==outcome)throw new Error("observation/history mismatch");
  const observed=MACRO.normalizeHand({...observation,outcome});
  // Replaying a replacement/duplicate last hand yields the same state, not a
  // second EMA update; unknown older B/P/T events have null measurements.
  syncMacro(history.slice(0,-1));
  macroJournal.events.push(observed);macroUpdate(macroState,observed);saveMacroJournal();
  return MACRO.snapshot(macroState);
}
function undoRuntimeHistory(){
  const history=readHistory(),shoeId=getShoeId();syncMacro(history);
  LOCAL_CALIBRATION?.reconcile(history,shoeId,browserCalibrationTag);
  writeRows(readRows().filter(row=>{
    if(row?.shoe_id!==shoeId)return true;
    const count=Number(row.history_event_count??(row.round_index-1));
    return Number.isInteger(count)&&count>=0&&count<history.length&&row.history_fingerprint===history.slice(0,count).join("")&&row.actual_outcome===history[count];
  }));
  try{localStorage.removeItem(PENDING_KEY);}catch(_){}
}
function readRows(){try{const r=JSON.parse(localStorage.getItem(TRAINING_KEY)||"[]");return Array.isArray(r)?r:[];}catch(_){return[];}}
function writeRows(rows){try{localStorage.setItem(TRAINING_KEY,JSON.stringify(rows.slice(-MAX_TRAINING_ROWS)));}catch(_){}}
function cloneFiniteVector(values,dimension){return Array.isArray(values)&&values.length===dimension&&values.every(value=>Number.isFinite(+value))?values.map(value=>+value):null;}
function registerPrediction(seq,prediction){
  const r=prediction.finalProbability||{},f=r.original7||buildOriginal7(seq,prediction);
  const createdAt=Date.now(),shoeId=getShoeId(),quality=r.dataQuality||dataQuality(seq),errorCorrectionSnapshot=null;
  const pending={raw_next_hand_forecast:prediction.raw_next_hand_forecast||null,browser_calibration:prediction.browser_calibration||null,next_hand_forecast:prediction.next_hand_forecast||null,forecast_direction:prediction.forecast_direction||null,bet_direction:prediction.direction||"Skip",schema_version:SNAPSHOT_SCHEMA_VERSION,prediction_id:`${shoeId}:${createdAt}:${seq.join("")}`,shoe_id:shoeId,created_at:createdAt,
    history_fingerprint:seq.join(""),history_event_count:seq.length,directional_round_count:quality.directionalRounds,data_stage:quality.stage,
    entry_eligible:quality.entryEligible,preferred_entry:quality.preferredEntry,mode:r.mode||"core",model_versions:{final_probability:final56Bundle?.schema_version??null,physics:physicsBundle?.schema_version??null},
    core_p_b:f.core_p_b,round_index:f.round_index,estimated_total_hands:f.estimated_total_hands,remaining_ratio:f.remaining_ratio,
    sx_markov_p_same:f.sx_markov_p_same,stage:f.stage,depth:f.depth,original_7d:original7Vector(f),physics_48d:cloneFiniteVector(r.physics,PHYSICS_DIM),
    features_57d:cloneFiniteVector(r.extended,FEATURE_DIM),shoe_progress_weight:Number.isFinite(+r.extended?.[1])?+r.extended[1]:null,physics_noise_score:Number.isFinite(+r.extended?.at(-1))?+r.extended.at(-1):null,
    raw_p_b:Number.isFinite(+r.physics_raw_p_b)?+r.physics_raw_p_b:null,clipped_p_b:Number.isFinite(+r.physics_clipped_p_b)?+r.physics_clipped_p_b:null,smoothed_p_b:Number.isFinite(+r.physics_smoothed_p_b)?+r.physics_smoothed_p_b:null,
    primary_raw_p_b:Number.isFinite(+r.physics_raw_p_b)?+r.physics_raw_p_b:null,primary_smoothed_p_b:Number.isFinite(+r.physics_smoothed_p_b)?+r.physics_smoothed_p_b:null,physics_smoothed_p_b:Number.isFinite(+r.physics_smoothed_p_b)?+r.physics_smoothed_p_b:null,
    primary_candidate:r.primary_candidate||"Skip",skip_diagnostics:r.skip_diagnostics||null,xgb_aux_p_b:Number.isFinite(+r.xgb_aux_p_b)?+r.xgb_aux_p_b:null,core_aux_p_b:Number.isFinite(+r.core_aux_p_b)?+r.core_aux_p_b:null,aux_filter:r.aux_filter||null,
    macro_schema_version:1,macro_ema:r.macro_ema??null,macro_adjustment:r.macro_adjustment??null,
    decision_stage:r.decision_stage,decision_progress:r.decision_progress,dynamic_band:r.dynamic_band??null,runtime_version:VERSION,
    smoothing_alpha:Number.isFinite(+r.smoothingAlpha)?+r.smoothingAlpha:1,smoothing_strength:Number.isFinite(+r.smoothingStrength)?+r.smoothingStrength:0,smoothing_profile:r.smoothingProfile||"off",final_p_b:Number.isFinite(+r.finalPB)?+r.finalPB:null,
    probability_bounds:r.bounds?[r.bounds.low,r.bounds.high]:null,min_ev:Number.isFinite(+r.min_ev)?+r.min_ev:null,
    activation_ev:Number.isFinite(+r.activation_ev)?+r.activation_ev:null,effective_activation_ev:Number.isFinite(+r.effective_activation_ev)?+r.effective_activation_ev:null,soft_band:Number.isFinite(+r.soft_band)?+r.soft_band:0,confidence_band:Number.isFinite(+r.confidence_band)?+r.confidence_band:0,effective_confidence_band:Number.isFinite(+r.effective_confidence_band)?+r.effective_confidence_band:0,volume_guard_active:r.volume_guard_active===true,entry_tier:r.entry_tier||"core",stake_multiplier:Number.isFinite(+r.stake_multiplier)?+r.stake_multiplier:1,decision_policy_enabled:r.decision_policy_enabled===true,decision_policy_profile:r.decision_policy_profile||"hard_ev",
    predicted_direction:r.finalDirection||prediction.direction||"",physics_integrity:r.physicsIntegrity||null,
    particle_physics:r.particleDiagnostics||null,physics_direct:r.physicsDiagnostics||null,physics_direct_48d:cloneFiniteVector(r.physicsDirect,PHYSICS_DIM),
    physical_ev_banker:Number.isFinite(+r.physical_ev_banker)?+r.physical_ev_banker:null,physical_ev_player:Number.isFinite(+r.physical_ev_player)?+r.physical_ev_player:null,
    physical_ev_gap:Number.isFinite(+r.physical_ev_gap)?+r.physical_ev_gap:null,physics_uncertainty:Number.isFinite(+r.physics_uncertainty)?+r.physics_uncertainty:null,
    shoe_error_correction_version:0,pre_hand_snapshot:errorCorrectionSnapshot,shoe_error_correction:null,
    execution_order:Array.isArray(r.execution_order)?r.execution_order.slice():[]};
  try{localStorage.setItem(PENDING_KEY,JSON.stringify(pending));}catch(_){}
}
function settlePending(actualOutcome,observation=null,expectedHistory=null){
  const actual=String(actualOutcome||"").toUpperCase();
  if(actual!=="B"&&actual!=="P"&&actual!=="T")return;
  let p=null;try{p=JSON.parse(localStorage.getItem(PENDING_KEY)||"null");}catch(_){}if(!p)return;
  if(p.shoe_id!==getShoeId()||(expectedHistory&&p.history_fingerprint!==expectedHistory.join(""))){try{localStorage.removeItem(PENDING_KEY);}catch(_){}return;}
  const observed=observation&&MACRO?MACRO.normalizeHand({...observation,outcome:actual}):null;
  const row={...p,settled_at:Date.now(),actual_outcome:actual,actual_b:actual==="B"?1:actual==="P"?0:null,is_directional_label:actual!=="T",observed_hand:observed,shoe_error_correction_audit:null},rows=readRows();
  if(!rows.length||rows.at(-1)?.shoe_id!==row.shoe_id||rows.at(-1)?.history_fingerprint!==row.history_fingerprint)rows.push(row);
  writeRows(rows);try{localStorage.removeItem(PENDING_KEY);}catch(_){}
}
function exportTrainingData(){
  const physicsNames=physicsBundleUsesPhysicalEv()?PHYSICS_NAMES:LEGACY_PHYSICS_NAMES;
  const featureNames=Array.isArray(final56Bundle?.feature_names)&&final56Bundle.feature_names.length===FEATURE_DIM?final56Bundle.feature_names:["core_p_b_external","shoe_progress_weight",...ORIGINAL7_NAMES.slice(1).map(x=>"original7_"+x),...physicsNames,"physics_noise_score"];
  return JSON.stringify({schema_version:SNAPSHOT_SCHEMA_VERSION,macro_schema_version:1,target:"actual_b_binary",feature_names:featureNames,original_7d_feature_names:ORIGINAL7_NAMES,physics_48d_feature_names:physicsNames,physics_semantics:physicsBundleUsesPhysicalEv()?"physical_ev_v2":"legacy_suit_v1",rows:readRows()},null,2);
}
function downloadTrainingData(){const blob=new Blob([exportTrainingData()],{type:"application/json;charset=utf-8"}),url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="bgs_final57_training_"+Date.now()+".json";document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);}
function exportMacroObservations(){
  syncMacro(readHistory());
  const shoes=readMacroArchive().filter(shoe=>shoe.shoe_id!==macroJournal?.shoe_id);
  if(macroJournal)shoes.push(macroJournal);
  return JSON.stringify({rows:shoes.flatMap(shoe=>(shoe.events||[]).map((hand,i)=>({...hand,shoe_id:shoe.shoe_id,round_index:i+1})))},null,2);
}
function downloadMacroObservations(){const blob=new Blob([exportMacroObservations()],{type:"application/json;charset=utf-8"}),url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="bgs_observed_hands_"+Date.now()+".json";document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);}

async function fetchBundle(url){const r=await fetch(url,{cache:"no-store"});if(!r.ok)throw new Error(url+":HTTP"+r.status);return await r.json();}
async function loadModels(){
  status={physics:false,final56:false,errors:[]};
  const rs=await Promise.allSettled([fetchBundle(PHYSICS_URL),fetchBundle(FINAL56_URL),fetchBundle("cardcount_profile.json")]);
  if(rs[0].status==="fulfilled"&&rs[0].value?.model_type==="baccarat_physics_multitask_mlp"){physicsBundle=rs[0].value;status.physics=!!physicsBundle.trained;}else status.errors.push("physics_model");
  if(rs[1].status==="fulfilled"&&rs[1].value?.model_type==="xgb_final_probability_classifier"&&rs[1].value?.feature_names?.length===FEATURE_DIM){final56Bundle=rs[1].value;status.final56=!!final56Bundle.trained;}else{final56Bundle=null;status.errors.push("final57_model");}
  const candidate=rs[2].status==="fulfilled"?rs[2].value:null;
  const keys=["decision_alpha","macro_short","macro_long","pf_max","training_weight"];
  const draws=candidate?.anchors?.pf_draws;
  const validDraws=draws===undefined||(Array.isArray(draws)&&draws.length>=2&&draws.every((v,i,a)=>Array.isArray(v)&&v.length===2&&v.every(Number.isFinite)&&v[0]>=1&&v[0]<=70&&v[1]>=1&&v[1]<=5&&(!i||v[0]>a[i-1][0])));
  const valid=validDraws&&candidate?.schema_version===1&&typeof candidate.enabled==="boolean"&&typeof candidate.display_gap==="number"&&candidate.display_gap>=0&&candidate.display_gap<=.10&&keys.every(k=>Array.isArray(candidate.anchors?.[k])&&candidate.anchors[k].length>=2&&candidate.anchors[k].every((v,i,a)=>Array.isArray(v)&&v.length===2&&v.every(Number.isFinite)&&v[0]>=1&&v[0]<=70&&v[1]>0&&v[1]<=2&&(!i||v[0]>a[i-1][0])));
  const next=valid?candidate:null;
  if(JSON.stringify(next)!==JSON.stringify(cardcountProfile))macroState=null;
  cardcountProfile=next;
  const gapControl=typeof document!=="undefined"?document.getElementById("forecastGap"):null;
  if(gapControl){gapControl.disabled=!cardcountProfile?.enabled;if(displayGapOverride===null)gapControl.value=String(cardcountProfile?.enabled?cardcountProfile.display_gap:0);}
  if(!valid)status.errors.push("cardcount_profile_fallback");
  if(status.physics&&status.final56&&!modelSemanticsCompatible()){status.final56=false;status.errors.push("feature_semantics_mismatch");}
  browserCalibrationTag=status.physics&&LOCAL_CALIBRATION_API?LOCAL_CALIBRATION_API.fingerprint(VERSION,physicsBundle,final56Bundle,localCalibrationConfig()):"unready";
  renderLocalCalibrationStatus(readHistory());
  return status;
}
function saveSelection(direction){try{const old=JSON.parse(localStorage.getItem(STORAGE_KEY)||"null")||{},streak=old.last_selected===direction?Math.max(1,(+old.selection_streak||0)+1):1;localStorage.setItem(STORAGE_KEY,JSON.stringify({last_selected:direction,selection_streak:streak}));}catch(_){}}
// UI-only continuous test: never register a wager or override production guards.
let noSkipTestActive=false;
function noSkipTestDirection(f){
  return f?.available===true?(f.directional.B>=f.directional.P?"B":"P"):null;
}
function renderLocalCalibrationStatus(history,calibration=null){
  if(typeof document==="undefined")return;
  const el=document.getElementById("localCalibrationStatus");if(!el)return;
  if(!LOCAL_CALIBRATION){el.textContent="本機逐局校準未載入";return;}
  if(browserCalibrationTag==="unready"){el.textContent="本機校準：等待模型就緒";return;}
  const state=calibration||LOCAL_CALIBRATION.inspect(history,getShoeId(),browserCalibrationTag,localCalibrationConfig());
  const mode=state.storage_error?"儲存異常，暫不修正":!state.enabled?"已停用":state.applied?"小幅修正已套用":state.ready?"本局維持原始機率":"累積與逐局比對中";
  el.textContent="本機校準："+mode+"；已累積 "+state.directional_samples+" 局莊／閒結果（非完整重訓）";
}
function renderPrediction(p,n){
  renderLocalCalibrationStatus(readHistory(),p.browser_calibration);
  const el=id=>document.getElementById(id),orb=el("directionOrb");if(!orb)return;
  const f=p.next_hand_forecast,ready=f?.available===true,shownDirection=noSkipTestActive?noSkipTestDirection(f):displayDirection(f),isB=shownDirection==="B",isP=shownDirection==="P";
  el("directionText").textContent=!ready?"未就緒":isB?"莊":isP?"閒":shownDirection==="Skip"?"觀望":"均衡";
  el("directionCode").textContent=!ready?"UNAVAILABLE":isB?"BANKER":isP?"PLAYER":shownDirection==="Skip"?"SKIP":"BALANCED";
  el("confidence").textContent=ready?"莊 "+(f.directional.B*100).toFixed(2)+"% / 閒 "+(f.directional.P*100).toFixed(2)+"%":"—";
  el("regime").textContent=!ready?"模型未就緒":noSkipTestActive?"測試方向（不列入正式下注）":p.direction==="Skip"?"下注觀望（仍有機率方向）":p.direction==="B"?"下注條件：莊":"下注條件：閒";
  el("strength").textContent=ready?(f.probabilities.T*100).toFixed(2)+"%":"—";
  orb.className="direction-orb "+(!ready||!isB&&!isP?"":isB?"banker":"player");
  if(el("modePill"))el("modePill").textContent=noSkipTestActive?"全程不觀望測試":ready?(f.source==="direct_physics_particle500_winner_head"?"Physics＋PF500 機率方向":"Direct Physics 機率方向"):"模型未就緒";
  if(el("roundCount"))el("roundCount").textContent=n;
  if(el("message"))el("message").textContent="第 "+(n+1)+" 局分析完成"+(ready?(f.weak_signal?"；莊閒接近五五波":"；莊閒百分比已排除和局，和局另列")+(f.uncertainty>=.85?"；模型不確定性偏高":""):"");
  if(!noSkipTestActive&&ready&&shownDirection==="Skip"&&el("message"))el("message").textContent+="；機率差未達 "+((displayGapOverride??cardcountProfile.display_gap)*100).toFixed(1)+" 個百分點，方向觀望";
  if(noSkipTestActive&&el("message"))el("message").textContent=ready?"不觀望測試：每次報結果後自動更新下一局；不列入正式下注"+(f.directional.B===f.directional.P?"；五五波固定以莊作測試方向，無機率優勢":""):"不觀望測試：模型未就緒，暫無測試方向";
}
function installUI(){
  if(typeof document==="undefined")return;const old=document.getElementById("btnStart");if(!old)return;const btn=old.cloneNode(true);old.replaceWith(btn);
  const testButton=document.getElementById("btnNoSkipTest");
  const setTestMode=active=>{
    noSkipTestActive=active;
    if(testButton){testButton.textContent=active?"停止不觀望測試":"全程不觀望測試";testButton.setAttribute("aria-pressed",String(active));}
  };
  const refreshTest=()=>{const h=readHistory();renderPrediction(applyFinalPrediction(h),h.length);};
  if(testButton)testButton.addEventListener("click",()=>{setTestMode(!noSkipTestActive);refreshTest();});
  btn.addEventListener("click",()=>{setTestMode(false);const history=readHistory();if(!history.length){const m=document.getElementById("message");if(m)m.textContent="請先輸入牌局紀錄";return;}
    const p=applyFinalPrediction(history);saveSelection(p.direction);registerPrediction(history,p);renderPrediction(p,history.length);});
  const gapControl=document.getElementById("forecastGap");
  if(gapControl)gapControl.addEventListener("change",()=>{
    const value=Number(gapControl.value);
    if([.01,.02,.03].includes(value)){displayGapOverride=value;refreshTest();}
  });
  const fieldIds=["macroCardCount","macroPlayerScore","macroBankerScore"];
  const clearFields=()=>{for(const id of fieldIds){const e=document.getElementById(id);if(e)e.value="";}};
  // Result buttons are authoritative; optional card/point telemetry must never
  // suppress the unmodified Frozen Core click handler or its B/P/T history write.
  const historyKey="bgs256d_frozen_6x15_forward_v18";
  const historyMatches=(actual,expected)=>actual.length===expected.length&&actual.every((v,i)=>v===expected[i]);
  for(const [id,outcome] of [["btnB","B"],["btnP","P"],["btnT","T"]]){
    const button=document.getElementById(id);if(!button)continue;
    let observed=null,observationError="",beforeHistory=[],calibrationTicket=null;
    // Capture the previous history, then settle the last prediction. All optional
    // telemetry failures still let the Frozen Core receive the clicked outcome.
    button.addEventListener("click",event=>{
      beforeHistory=readHistory();
      observed=null;observationError="";calibrationTicket=null;
      const shown=Number(document.getElementById("roundCount")?.textContent);
      if(Number.isInteger(shown)&&shown!==beforeHistory.length){
        // Another tab (or failed persistent write) has changed history since this
        // page loaded. Do not let an old Frozen Core instance erase newer hands.
        event.preventDefault();event.stopImmediatePropagation();
        const message=document.getElementById("message");
        if(message){message.textContent="牌路資料已在其他頁籤更新，請重新整理後再報莊／閒／和，避免覆蓋已記錄的局數。";message.classList.add("warning");}
        return;
      }
      if(LOCAL_CALIBRATION&&browserCalibrationTag!=="unready")try{
        const prior=applyFinalPrediction(beforeHistory);
        calibrationTicket=LOCAL_CALIBRATION.prepare(beforeHistory,prior.raw_next_hand_forecast,getShoeId(),browserCalibrationTag);
      }catch(_){/* Calibration never blocks reporting the actual result. */}
      try{
        const get=id=>document.getElementById(id)?.value??"";
        observed=MACRO?MACRO.normalizeHand({outcome,card_count:get(fieldIds[0]),player_score:get(fieldIds[1]),banker_score:get(fieldIds[2])}):null;
      }catch(error){
        observationError=String(error?.message||error);
        observed=MACRO?MACRO.normalizeHand({outcome}):null;
      }
      try{syncMacro(beforeHistory);settlePending(outcome,observed,beforeHistory);}
      catch(error){observationError=observationError||"分析附加紀錄異常："+String(error?.message||error);}
    },true);
    // Runs after Frozen Core's original addOutcome handler. Verify the stored
    // history, not merely the on-screen counter (Frozen Core save catches errors).
    button.addEventListener("click",()=>{
      const expected=[...beforeHistory,outcome];
      let history=readHistory(),persistenceError="";
      if(!historyMatches(history,expected)){
        try{
          const saved=JSON.parse(localStorage.getItem(historyKey)||"null")||{};
          localStorage.setItem(historyKey,JSON.stringify({...saved,history:expected}));
          history=readHistory();
        }catch(error){persistenceError=String(error?.message||error);}
      }
      if(!historyMatches(history,expected)){
        const message=document.getElementById("message");
        if(message){message.textContent="本局儲存失敗，請勿重複按莊／閒；先匯出分析紀錄並檢查瀏覽器儲存空間。"+(persistenceError?"（"+persistenceError+"）":"");message.classList.add("warning");}
        return;
      }
      if(calibrationTicket&&calibrationTicket.model===browserCalibrationTag)LOCAL_CALIBRATION.commit(calibrationTicket,outcome,history,localCalibrationConfig());
      try{recordCompletedHand(history,observed||{});}
      catch(error){observationError=observationError||"分析附加紀錄異常："+String(error?.message||error);}
      clearFields();
      if(noSkipTestActive||LOCAL_CALIBRATION)refreshTest();
      const message=document.getElementById("message");
      if(message&&observationError){
        message.textContent="已記錄第 "+history.length+" 局；選填張數／點數已忽略："+observationError;
        message.classList.add("warning");
      }
    });
  }
  const back=document.getElementById("btnBack");if(back)back.addEventListener("click",()=>{undoRuntimeHistory();clearFields();if(noSkipTestActive||LOCAL_CALIBRATION)refreshTest();});
  const end=document.getElementById("btnEnd");if(end)end.addEventListener("click",rotateShoeId);
  if(end)end.addEventListener("click",()=>{clearFields();setTestMode(false);});
  const training=document.getElementById("btnExportTraining");if(training)training.addEventListener("click",downloadTrainingData);
  const observations=document.getElementById("btnExportObservations");if(observations)observations.addEventListener("click",downloadMacroObservations);
  syncMacro(readHistory());
}
if(typeof window!=="undefined")window.__BGS_FINAL56__={version:VERSION,getLocalCalibrationStatus:(history=readHistory())=>LOCAL_CALIBRATION?.inspect(history,getShoeId(),browserCalibrationTag,localCalibrationConfig())||null,displayDirection,profileValue,getCardcountProfile:()=>cardcountProfile,noSkipTestDirection,nextHandForecast,renderPrediction,applyFinalPrediction,predictPhysics,unpackPhysicsForecast,physicsIntegrity,dataQuality,buildOriginal7,historyVector,loadModels,setEstimatedTotalHands,getEstimatedTotalHands,directPhysicsEnabled,early35Enabled,applyProbabilityBounds,dynamicEmaAlpha,directPhysicsPrimary,auxiliaryCandidateFilter,preserveCandidateDirection,physicsConsistencyScore,calibratedPhysicsDirectionalPB,physicsPrecisionPenalty,
  registerPrediction,settlePending,exportTrainingData,downloadTrainingData,getMacroFeatures,recordCompletedHand,exportMacroObservations,rotateShoeId,undoRuntimeHistory,decisionPolicy,dynamicBand,volumeGuardState,
  getTrainingRows:()=>readRows(),getTrainingCount:()=>readRows().length,getModelStatus:()=>({...status,mode:status.physics&&status.final56?"final56":"core"})};
if(typeof window?.addEventListener==="function")window.addEventListener("storage",event=>{if(event.key===MACRO_KEY||event.key===SHOE_KEY){macroJournal=null;macroState=null;}});
loadModels();installUI();
})();
