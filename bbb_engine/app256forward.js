(() => {
"use strict";

const SHOE_DIM = 128;
const ROAD_DIM = 128;
const DIM = 256;
const BIG_ROAD_ROWS = 6;
const BIG_ROAD_COLS = 15;
const ARMS = ["P", "B"];
const ALPHA = 0.5;
const RIDGE = 1.0;
const PROB_MIN = 0.42;
const PROB_MAX = 0.58;
const SCORE_TEMP = 0.42;
const SCORE_TIE_EPS = 1e-9;
const TOTAL_CARDS = 416;
const AVG_CARDS_PER_HAND = 4.9;
const STORAGE_KEY = "bgs256d_frozen_6x15_forward_v18";
const LEGACY_KEYS = [
  "bgs256d_frozen_6x15_sensitive_v17",
  "bgs256d_frozen_6x15_bigroad_v16",
  "bgs256d_frozen_column_linucb_v15",
  "bgs256d_side_aware_column_v14_user_panel_v1",
  "bgs256d_column_geometry_v13_user_panel_v1",
  "bgs256d_stability_v12_user_panel_v1",
  "bgs256d_128plus128_frozen_direct_tech_panel_v1",
  "bgs128d_64plus64_frozen_direct_tech_panel_v1",
  "bgs64d_32plus32_frozen_direct_tech_panel_v2",
  "bgs32d_frozen_direct_tech_panel_v1"
];

const el = id => (typeof document !== "undefined" ? document.getElementById(id) : null);
const clip = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(+v) ? +v : lo));
const signed = v => clip(v, -1, 1);
const bp = seq => seq.filter(x => x === "B" || x === "P");
const sideSign = side => side === "B" ? 1 : side === "P" ? -1 : 0;

let state = { history: [], active: false, lastPrediction: null, last_selected: "", selection_streak: 0 };

function runs(seq) {
  const a = bp(seq);
  if (!a.length) return [];
  const out = [];
  let side = a[0], n = 1;
  for (let i = 1; i < a.length; i++) {
    if (a[i] === side) n++;
    else { out.push([side, n]); side = a[i]; n = 1; }
  }
  out.push([side, n]);
  return out;
}

function bankerRatio(seq, w) {
  const a = bp(seq).slice(-Math.max(1, w));
  return a.length ? a.filter(x => x === "B").length / a.length : 0.5;
}

function turnRate(seq, w) {
  const a = bp(seq).slice(-Math.max(2, w));
  if (a.length < 2) return 0.5;
  let t = 0;
  for (let i = 1; i < a.length; i++) if (a[i] !== a[i - 1]) t++;
  return t / (a.length - 1);
}

function tieRatio(seq, w = 0) {
  const a = w ? seq.slice(-w) : seq;
  return a.length ? a.filter(x => x === "T").length / a.length : 0;
}

function binaryEntropy(seq, w = 12) {
  const a = bp(seq).slice(-w);
  if (!a.length) return 1;
  const p = a.filter(x => x === "B").length / a.length, q = 1 - p;
  let e = 0;
  if (p) e -= p * Math.log2(p);
  if (q) e -= q * Math.log2(q);
  return clip(e);
}

function outcomeEntropy(seq, w = 12) {
  const a = seq.slice(-w);
  if (!a.length) return 1;
  let e = 0;
  for (const o of ["B", "P", "T"]) {
    const p = a.filter(x => x === o).length / a.length;
    if (p) e -= p * Math.log2(p);
  }
  return clip(e / Math.log2(3));
}

const balance = (seq, w) => clip(1 - Math.abs(bankerRatio(seq, w) - 0.5) * 2);

function sameTail(seq, w) {
  const a = bp(seq).slice(-w);
  return a.length < w ? 0.5 : +a.every(x => x === a[0]);
}

function alternatingTail(seq, w) {
  const a = bp(seq).slice(-w);
  if (a.length < w) return 0.5;
  for (let i = 1; i < a.length; i++) if (a[i] === a[i - 1]) return 0;
  return 1;
}

function runVolatility(seq, w = 6) {
  const h = runs(seq).slice(-w).map(x => x[1]);
  if (h.length < 2) return 0.25;
  let d = 0;
  for (let i = 1; i < h.length; i++) d += Math.abs(h[i] - h[i - 1]);
  return clip(d / (h.length - 1) / 3);
}

function runTrend(seq, w = 5) {
  const h = runs(seq).slice(-w).map(x => x[1]);
  return h.length < 2 ? 0.5 : clip(0.5 + ((h.at(-1) - h[0]) / (h.length - 1)) / 6);
}

function runStats(seq, w) {
  const h = runs(seq).slice(-w).map(x => x[1]);
  if (!h.length) return { avg: 0, max: 0, std: 0 };
  const mean = h.reduce((a, b) => a + b, 0) / h.length;
  const variance = h.reduce((s, x) => s + (x - mean) ** 2, 0) / h.length;
  return { avg: clip(mean / 8), max: clip(Math.max(...h) / 12), std: clip(Math.sqrt(variance) / 6) };
}

function cellKey(row, col) { return `${row}:${col}`; }

function buildBigRoad(seq) {
  const occupied = new Map(), cells = [], streaks = [];
  let currentSide = "", currentRow = 0, currentCol = -1, currentOriginCol = -1, streakIndex = -1, pendingTies = 0;
  const place = (side, row, col, originCol, move) => {
    const cell = { side, row, col, originCol, streakIndex, move, ties: 0, index: cells.length };
    occupied.set(cellKey(row, col), cell); cells.push(cell); streaks[streakIndex].cells.push(cell); return cell;
  };

  for (const outcome of seq) {
    if (outcome === "T") { if (cells.length) cells.at(-1).ties++; else pendingTies++; continue; }
    if (outcome !== "B" && outcome !== "P") continue;
    if (!currentSide) {
      currentSide = outcome; currentRow = 0; currentCol = 0; currentOriginCol = 0; streakIndex = 0;
      streaks.push({ side: outcome, originCol: 0, cells: [] });
      const first = place(outcome, 0, 0, 0, "start"); first.ties += pendingTies; pendingTies = 0; continue;
    }
    if (outcome !== currentSide) {
      currentSide = outcome; currentRow = 0; currentCol = currentOriginCol + 1;
      while (occupied.has(cellKey(0, currentCol))) currentCol++;
      currentOriginCol = currentCol; streakIndex++;
      streaks.push({ side: outcome, originCol: currentOriginCol, cells: [] });
      place(outcome, 0, currentCol, currentOriginCol, "new-column"); continue;
    }
    const belowRow = currentRow + 1;
    if (belowRow < BIG_ROAD_ROWS && !occupied.has(cellKey(belowRow, currentCol))) {
      currentRow = belowRow; place(outcome, currentRow, currentCol, currentOriginCol, "down");
    } else {
      let nextCol = currentCol + 1;
      while (occupied.has(cellKey(currentRow, nextCol))) nextCol++;
      currentCol = nextCol;
      place(outcome, currentRow, currentCol, currentOriginCol, belowRow >= BIG_ROAD_ROWS ? "tail-bottom" : "tail-collision");
    }
  }

  for (const streak of streaks) {
    const rows = streak.cells.map(c => c.row);
    streak.logicalLength = streak.cells.length;
    streak.verticalHeight = rows.length ? Math.max(...rows) + 1 : 0;
    streak.tailLength = streak.cells.filter(c => c.col > streak.originCol).length;
    streak.endRow = streak.cells.length ? streak.cells.at(-1).row : 0;
    streak.endCol = streak.cells.length ? streak.cells.at(-1).col : streak.originCol;
    streak.hasCollisionTail = streak.cells.some(c => c.move === "tail-collision");
    streak.hasBottomTail = streak.cells.some(c => c.move === "tail-bottom");
  }

  const maxCol = cells.length ? Math.max(...cells.map(c => c.col)) : 0;
  const viewStartCol = Math.max(0, maxCol - BIG_ROAD_COLS + 1), viewEndCol = viewStartCol + BIG_ROAD_COLS - 1;
  const visibleCells = cells.filter(c => c.col >= viewStartCol && c.col <= viewEndCol);
  const grid = Array.from({ length: BIG_ROAD_ROWS }, () => Array(BIG_ROAD_COLS).fill(null));
  for (const c of visibleCells) {
    const vc = c.col - viewStartCol;
    if (vc >= 0 && vc < BIG_ROAD_COLS) grid[c.row][vc] = c.side;
  }
  return { rows: BIG_ROAD_ROWS, cols: BIG_ROAD_COLS, cells, streaks, occupied, maxCol, viewStartCol, viewEndCol, visibleCells, grid, currentCell: cells.at(-1) || null, currentStreak: streaks.at(-1) || null };
}

function fullCompletedStreaks(road) { return road.streaks.length > 1 ? road.streaks.slice(0, -1) : []; }

function median(values) {
  if (!values.length) return 0;
  const a = [...values].sort((x, y) => x - y), m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

function modeValue(values) {
  if (!values.length) return { value: 0, support: 0 };
  const counts = new Map();
  for (const x of values) counts.set(x, (counts.get(x) || 0) + 1);
  let best = values[0], bestCount = 0;
  const med = median(values);
  for (const [value, count] of counts) {
    if (count > bestCount || (count === bestCount && Math.abs(value - med) < Math.abs(best - med))) { best = value; bestCount = count; }
  }
  return { value: best, support: bestCount / values.length };
}

function summarize(values, stdScale = 6) {
  if (!values.length) return { count:0, mean:0, med:0, mode:0, recent:0, std:0, target:0, reliability:0 };
  const mean = values.reduce((a,b)=>a+b,0)/values.length, med = median(values), mv = modeValue(values);
  const recentVals = values.slice(-4), recent = recentVals.reduce((a,b)=>a+b,0)/recentVals.length;
  const variance = values.reduce((s,x)=>s+(x-mean)**2,0)/values.length, std = Math.sqrt(variance);
  const target = 0.42*mv.value + 0.28*med + 0.30*recent;
  const reliability = clip(values.length/5) * clip(0.45+0.55*mv.support) * clip(1-std/stdScale);
  return { count:values.length, mean, med, mode:mv.value, recent, std, target, reliability };
}

function sideRoadStats(completed, side) {
  const items = completed.filter(x => x.side === side).slice(-12);
  if (!items.length) return { count:0, verticalTarget:0, logicalTarget:0, tailTarget:0, reliability:0 };
  const v = summarize(items.map(x=>x.verticalHeight),5), l = summarize(items.map(x=>x.logicalLength),8), t = summarize(items.map(x=>x.tailLength),6);
  return {
    count:items.length, verticalTarget:v.target, logicalTarget:l.target, tailTarget:t.target,
    verticalReliability:v.reliability, logicalReliability:l.reliability, tailReliability:t.reliability,
    reliability:clip(0.52*v.reliability+0.40*l.reliability+0.08*t.reliability), verticalStd:v.std, logicalStd:l.std
  };
}

function stageSurvival(completed, side, currentLogical) {
  const items = completed.filter(x => x.side === side).slice(-16);
  let reachedW=0, continuedW=0, endedW=0;
  for (let i=0;i<items.length;i++) {
    const s = items[i];
    if (s.logicalLength < currentLogical) continue;
    const recency = Math.pow(0.94, items.length-1-i);
    reachedW += recency;
    if (s.logicalLength > currentLogical) continuedW += recency;
    else endedW += recency;
  }
  const prior = 1.15, denom = reachedW + 2*prior;
  return { cont: clip((continuedW+prior)/denom), turn: clip((endedW+prior)/denom), support: clip(reachedW/4), reached: reachedW };
}

function pairRoadStats(completed, fromSide) {
  const toSide = fromSide === "B" ? "P" : "B", pairs=[];
  for (let i=0;i<completed.length-1;i++) {
    if (completed[i].side===fromSide && completed[i+1].side===toSide) pairs.push([completed[i].verticalHeight, completed[i].logicalLength, completed[i+1].verticalHeight, completed[i+1].logicalLength]);
  }
  const use = pairs.slice(-10);
  if(!use.length) return { count:0, fromVertical:0, fromLogical:0, toVertical:0, toLogical:0, reliability:0 };
  const fv=summarize(use.map(x=>x[0]),5), fl=summarize(use.map(x=>x[1]),8), tv=summarize(use.map(x=>x[2]),5), tl=summarize(use.map(x=>x[3]),8);
  return { count:use.length, fromVertical:fv.target, fromLogical:fl.target, toVertical:tv.target, toLogical:tl.target, reliability:clip(use.length/6)*clip(0.55+0.25*fv.reliability+0.20*tv.reliability) };
}

function closeness(a,b,span=3) {
  if(!b) return .5;
  return clip(1-Math.abs(a-b)/Math.max(span,b*.65+1));
}

function contextualStageStats(completed, currentIndex, currentSide, currentLogical) {
  const currentPrev1 = currentIndex > 0 ? completed[currentIndex-1] : null;
  const currentPrev2 = currentIndex > 1 ? completed[currentIndex-2] : null;
  if (!currentPrev1) return { cont:.5, turn:.5, support:0 };
  let total=0, cont=0, turn=0;
  for (let i=0;i<completed.length;i++) {
    const s=completed[i];
    if(s.side!==currentSide || s.logicalLength<currentLogical || i<1) continue;
    const p1=completed[i-1], p2=i>1?completed[i-2]:null;
    let sim = 0.72*closeness(p1.logicalLength,currentPrev1.logicalLength,3);
    if(currentPrev2 && p2) sim += 0.28*closeness(p2.logicalLength,currentPrev2.logicalLength,3.5); else sim += 0.14;
    const recency=Math.pow(0.96,completed.length-1-i), w=Math.pow(clip(sim),2)*recency;
    total+=w;
    if(s.logicalLength>currentLogical) cont+=w; else turn+=w;
  }
  const prior=.75, denom=total+2*prior;
  return { cont:clip((cont+prior)/denom), turn:clip((turn+prior)/denom), support:clip(total/3) };
}

function bigRoadGeometry(road) {
  const cells=road.visibleCells;
  if(!cells.length) return {occupancy:0,bankerRatio:.5,topOccupancy:0,bottomOccupancy:0,tailRatio:0,activeColumns:0,avgFill:0,maxFill:0,fillStd:0,profileRegularity:.5,currentRow:0,currentViewCol:0,currentInTail:0,currentAtBottom:0,blockedBelow:0};
  const counts=Array(BIG_ROAD_COLS).fill(0),profiles=Array(BIG_ROAD_COLS).fill(0); let top=0,bottom=0,tail=0,banker=0;
  for(const c of cells){const vc=c.col-road.viewStartCol;if(vc<0||vc>=BIG_ROAD_COLS)continue;counts[vc]++;profiles[vc]|=(1<<c.row);if(c.row===0)top++;if(c.row===BIG_ROAD_ROWS-1)bottom++;if(c.col>c.originCol)tail++;if(c.side==="B")banker++;}
  const active=counts.filter(x=>x>0),avg=active.length?active.reduce((a,b)=>a+b,0)/active.length:0,max=active.length?Math.max(...active):0,variance=active.length?active.reduce((s,x)=>s+(x-avg)**2,0)/active.length:0;
  const used=profiles.filter(x=>x!==0);let sim=0,n=0;
  for(let i=1;i<used.length;i++){const xor=used[i]^used[i-1];let bits=0;for(let b=0;b<BIG_ROAD_ROWS;b++)bits+=(xor>>b)&1;sim+=1-bits/BIG_ROAD_ROWS;n++;}
  const current=road.currentCell,blocked=current?(current.row>=BIG_ROAD_ROWS-1||road.occupied.has(cellKey(current.row+1,current.col))):false;
  return {occupancy:clip(cells.length/(BIG_ROAD_ROWS*BIG_ROAD_COLS)),bankerRatio:banker/cells.length,topOccupancy:clip(top/BIG_ROAD_COLS),bottomOccupancy:clip(bottom/BIG_ROAD_COLS),tailRatio:tail/cells.length,activeColumns:clip(active.length/BIG_ROAD_COLS),avgFill:clip(avg/BIG_ROAD_ROWS),maxFill:clip(max/BIG_ROAD_ROWS),fillStd:clip(Math.sqrt(variance)/BIG_ROAD_ROWS),profileRegularity:n?clip(sim/n):.5,currentRow:current?current.row:0,currentViewCol:current?current.col-road.viewStartCol:0,currentInTail:current?+(current.col>current.originCol):0,currentAtBottom:current?+(current.row===BIG_ROAD_ROWS-1):0,blockedBelow:+blocked};
}

function candidateMoveInfo(baseRoad,candidate,seq) {
  const after=buildBigRoad([...seq,candidate]),added=after.cells.length>baseRoad.cells.length?after.cells.at(-1):null;
  if(!added)return{after,row:0,col:0,viewCol:0,down:0,right:0,newColumn:0};
  return {after,row:added.row,col:added.col,viewCol:added.col-after.viewStartCol,down:+(added.move==="down"),right:+(added.move==="tail-bottom"||added.move==="tail-collision"),newColumn:+(added.move==="new-column")};
}

function gridCandidateQuality(baseRoad, afterRoad) {
  const before=bigRoadGeometry(baseRoad),after=bigRoadGeometry(afterRoad);
  const regularityGain=clip(.5+(after.profileRegularity-before.profileRegularity)*1.4);
  const smooth=clip(1-after.fillStd);
  return clip(.64*regularityGain+.36*smooth);
}

function derivedMark(h,c,row,newCol,off){if(newCol){if(c<off+1)return"";return h[c-1]===h[c-1-off]?"R":"U";}if(c<off)return"";const ref=h[c-off];return(ref>=row)===(ref>=row-1)?"R":"U";}
function buildDerivedRoads(seq){const road=buildBigRoad(seq),streaks=road.streaks,out={big_eye:[],small_road:[],cockroach_road:[]},offsets={big_eye:1,small_road:2,cockroach_road:3},heights=[];for(let i=0;i<streaks.length;i++){heights.push(streaks[i].verticalHeight);for(const[name,off]of Object.entries(offsets)){const m=derivedMark(heights,i,streaks[i].verticalHeight,true,off);if(m)out[name].push(m);}for(let r=2;r<=streaks[i].verticalHeight;r++)for(const[name,off]of Object.entries(offsets)){const m=derivedMark(heights,i,r,false,off);if(m)out[name].push(m);}}return out;}
function regularity(values,w=8){const a=values.slice(-w).filter(x=>x==="R"||x==="U");return a.length?[a.filter(x=>x==="R").length/a.length,a.length]:[.5,0];}
function derivedInfo(seq,w=8){const roads=buildDerivedRoads(seq),[be,bn]=regularity(roads.big_eye,w),[sm,sn]=regularity(roads.small_road,w),[cr,cn]=regularity(roads.cockroach_road,w),mean=(be+sm+cr)/3;return{be,sm,cr,bn,sn,cn,consensus:clip(1-(Math.abs(be-mean)+Math.abs(sm-mean)+Math.abs(cr-mean))/1.5),support:clip((bn+sn+cn)/(w*3))};}

function branchFutureQuality(seq, first) {
  const road1=buildBigRoad([...seq,first]),current=road1.currentStreak;
  if(!current)return .5;
  const completed=fullCompletedStreaks(road1), own=sideRoadStats(completed,first), stage=stageSurvival(completed,first,current.logicalLength);
  const context=contextualStageStats(completed,completed.length,first,current.logicalLength);
  const targetFit=own.count?closeness(current.logicalLength,own.logicalTarget,3):.5;
  const secondSame=buildBigRoad([...seq,first,first]),sameStreak=secondSame.currentStreak;
  const sameNextFit=own.count&&sameStreak?closeness(sameStreak.logicalLength,own.logicalTarget,3):.5;
  const pair=pairRoadStats(completed,first);
  const turnFit=pair.count?clip(.55*closeness(current.logicalLength,pair.fromLogical,3)+.45*pair.reliability):.5;
  const qContinue=clip(.42*stage.cont+.25*context.cont+.23*sameNextFit+.10*targetFit);
  const qTurn=clip(.42*stage.turn+.25*context.turn+.23*turnFit+.10*targetFit);
  return clip(.62*Math.max(qContinue,qTurn)+.38*((qContinue+qTurn)/2));
}

function bigRoadCandidates(seq) {
  const road=buildBigRoad(seq),current=road.currentStreak;
  if(!current)return{B:.5,P:.5,support:0,currentSide:"",road,geometry:bigRoadGeometry(road),forwardDirectional:0,survivalDirectional:0,contextDirectional:0,twoStepDirectional:0};
  const completed=fullCompletedStreaks(road),currentSide=current.side,opposite=currentSide==="B"?"P":"B",sign=sideSign(currentSide);
  const own=sideRoadStats(completed,currentSide),opp=sideRoadStats(completed,opposite),stage=stageSurvival(completed,currentSide,current.logicalLength),context=contextualStageStats(completed,completed.length,currentSide,current.logicalLength),pair=pairRoadStats(completed,currentSide);
  const contInfo=candidateMoveInfo(road,currentSide,seq),revInfo=candidateMoveInfo(road,opposite,seq);
  const contStreak=contInfo.after.currentStreak,nextLogical=contStreak?contStreak.logicalLength:current.logicalLength;
  const targetFitNow=own.count?closeness(current.logicalLength,own.logicalTarget,3):.5;
  const targetFitNext=own.count?closeness(nextLogical,own.logicalTarget,3):.5;
  const pairFit=pair.count?clip(.58*closeness(current.logicalLength,pair.fromLogical,3)+.42*closeness(current.verticalHeight,pair.fromVertical,2.2)):.5;
  const continueGrid=gridCandidateQuality(road,contInfo.after),reverseGrid=gridCandidateQuality(road,revInfo.after);
  const cdi=derivedInfo([...seq,currentSide],8),rdi=derivedInfo([...seq,opposite],8);
  const continueDerived=clip(.6*cdi.consensus+.4*cdi.support),reverseDerived=clip(.6*rdi.consensus+.4*rdi.support);
  const continueScore=clip(.31*stage.cont+.22*context.cont+.19*targetFitNext+.10*own.reliability+.08*continueDerived+.06*continueGrid+.04*(1-targetFitNow));
  const reverseScore=clip(.31*stage.turn+.22*context.turn+.19*targetFitNow+.14*pairFit+.06*pair.reliability+.05*reverseDerived+.03*reverseGrid);
  const scores=currentSide==="B"?{B:continueScore,P:reverseScore}:{P:continueScore,B:reverseScore};
  const twoB=branchFutureQuality(seq,"B"),twoP=branchFutureQuality(seq,"P");
  const twoStepDirectional=signed(twoB-twoP);
  const survivalDirectional=signed(sign*(stage.cont-stage.turn));
  const contextDirectional=signed(sign*(context.cont-context.turn));
  const forwardDirectional=signed(0.48*(scores.B-scores.P)+0.24*survivalDirectional+0.18*contextDirectional+0.10*twoStepDirectional);
  const support=clip(.30*stage.support+.24*context.support+.18*own.reliability+.12*pair.reliability+.16*Math.min(1,completed.length/8));
  return {...scores,support,currentSide,road,geometry:bigRoadGeometry(road),stageCont:stage.cont,stageTurn:stage.turn,stageSupport:stage.support,contextCont:context.cont,contextTurn:context.turn,contextSupport:context.support,ownTarget:own.logicalTarget||0,oppositeTarget:opp.logicalTarget||0,pairReliability:pair.reliability,pairFit,continueGrid,reverseGrid,continueMove:contInfo,reverseMove:revInfo,targetFitNow,targetFitNext,forwardDirectional,survivalDirectional,contextDirectional,twoStepDirectional,twoB,twoP};
}

function buildShoeVector(seq) {
  const used=Math.min(TOTAL_CARDS,seq.length*AVG_CARDS_PER_HAND),remaining=Math.max(0,TOTAL_CARDS-used),rr=clip(remaining/TOTAL_CARDS),pen=clip(1-rr),maturity=clip(seq.length/70),hands=clip(seq.length/(TOTAL_CARDS/AVG_CARDS_PER_HAND));
  const v=[rr,pen,rr,maturity,...Array(10).fill(1),0,0,clip(1-pen/.35),clip(1-Math.abs(pen-.5)/.35),clip((pen-.55)/.35),hands,rr,clip(Math.log1p(seq.length)/Math.log1p(TOTAL_CARDS/AVG_CARDS_PER_HAND)),tieRatio(seq),tieRatio(seq,8),tieRatio(seq,16),balance(seq,Math.max(1,bp(seq).length)),binaryEntropy(seq,12),outcomeEntropy(seq,12),outcomeEntropy(seq,24),clip(seq.length/32),1,clip(Math.sqrt(seq.length)/Math.sqrt(TOTAL_CARDS/AVG_CARDS_PER_HAND))];
  for(const w of [4,6,12,24,32])v.push(tieRatio(seq,w));
  for(const w of [4,6,8,16,24,32])v.push(binaryEntropy(seq,w));
  for(const w of [6,8,16,32])v.push(outcomeEntropy(seq,w));
  for(const w of [6,8,16,24,32])v.push(balance(seq,w));
  v.push(pen*pen,Math.sqrt(pen),rr*rr,Math.sqrt(rr),clip(1-pen/.18),clip(1-Math.abs(pen-.3)/.22),clip(1-Math.abs(pen-.62)/.24),clip((pen-.72)/.22),clip(seq.length/8),clip(seq.length/16),clip(seq.length/24),clip(seq.length/48));
  for(const w of [2,3,5,7,10,14,20,28,40,48,56,64])v.push(tieRatio(seq,w));
  for(const w of [2,3,5,7,10,14,20,28,40,48,56,64])v.push(binaryEntropy(seq,w));
  for(const w of [2,3,5,7,10,14,20,28,40,48,56,64])v.push(outcomeEntropy(seq,w));
  for(const w of [2,3,5,7,10,14,20,28,40,48,56,64])v.push(balance(seq,w));
  v.push(pen**3,rr**3,Math.sqrt(Math.sqrt(pen)),Math.sqrt(Math.sqrt(rr)),clip(1-Math.abs(pen-.125)/.125),clip(1-Math.abs(pen-.375)/.125),clip(1-Math.abs(pen-.625)/.125),clip(1-Math.abs(pen-.875)/.125),clip(seq.length/4),clip(seq.length/12),clip(seq.length/20),clip(seq.length/32),clip(.5+(tieRatio(seq,8)-tieRatio(seq,32))/2),clip(.5+(binaryEntropy(seq,8)-binaryEntropy(seq,32))/2),clip(.5+(balance(seq,8)-balance(seq,32))/2),clip(Math.log1p(seq.length)/Math.log1p(128)));
  if(v.length!==SHOE_DIM)throw new Error(`shoe mismatch ${v.length}`);
  return v;
}

const ROAD_INDEX={};
function buildRoadVector(seq) {
  const rs=runs(seq),current=rs.at(-1)||["",0],prior=i=>rs.length>i?rs.at(-1-i):["",0],d8=derivedInfo(seq,8),s4=runStats(seq,4),s8=runStats(seq,8),s12=runStats(seq,12),cand=bigRoadCandidates(seq),g=cand.geometry;
  const hs=clip(1-runVolatility(seq,6)*.55-Math.abs(turnRate(seq,8)-turnRate(seq,24))*.45),road=[];
  const push=(name,value)=>{if(!(name in ROAD_INDEX))ROAD_INDEX[name]=road.length;road.push(Number.isFinite(+value)?+value:0);};
  push("current_side_banker",sideSign(current[0]));push("current_run_norm",clip(current[1]/12));for(let i=1;i<=6;i++)push(`previous_run_${i}`,clip(prior(i)[1]/12));
  push("stage_cont_probability",cand.stageCont??.5);push("stage_turn_probability",cand.stageTurn??.5);push("context_cont_probability",cand.contextCont??.5);push("context_turn_probability",cand.contextTurn??.5);push("structure_stability",hs);push("run_volatility",runVolatility(seq,6));push("run_trend",runTrend(seq,5));
  push("derived_big_eye",d8.be);push("derived_small",d8.sm);push("derived_cockroach",d8.cr);push("derived_consensus",d8.consensus);push("derived_support",d8.support);push("same2",sameTail(seq,2));push("same3",sameTail(seq,3));push("alt4",alternatingTail(seq,4));push("alt6",alternatingTail(seq,6));
  for(const w of [2,4,6,8,10,12,16,20,24,32,48,64])push(`banker_bias_${w}`,(bankerRatio(seq,w)-.5)*2);
  for(const w of [2,4,6,8,10,12,16,20,24,32,48,64])push(`turn_rate_${w}`,turnRate(seq,w));
  for(const w of [4,8,16,24,32]){const d=derivedInfo(seq,w);push(`be_${w}`,d.be);push(`sm_${w}`,d.sm);push(`cr_${w}`,d.cr);}
  push("run_avg4",s4.avg);push("run_avg8",s8.avg);push("run_avg12",s12.avg);push("run_max8",s8.max);push("run_max12",s12.max);push("run_std8",s8.std);push("run_std12",s12.std);push("run_trend8",runTrend(seq,8));push("run_trend12",runTrend(seq,12));
  for(const w of [3,5,8,10])push(`alternating_${w}`,alternatingTail(seq,w));for(const w of [4,5,6,8,10])push(`same_${w}`,sameTail(seq,w));
  push("banker_delta_4_16",bankerRatio(seq,4)-bankerRatio(seq,16));push("banker_delta_8_32",bankerRatio(seq,8)-bankerRatio(seq,32));push("banker_delta_16_64",bankerRatio(seq,16)-bankerRatio(seq,64));push("turn_delta_4_16",turnRate(seq,4)-turnRate(seq,16));push("turn_delta_8_32",turnRate(seq,8)-turnRate(seq,32));push("turn_delta_16_64",turnRate(seq,16)-turnRate(seq,64));
  push("bigroad_candidate_B",cand.B-.5);push("bigroad_candidate_P",cand.P-.5);push("bigroad_candidate_gap_B",cand.B-cand.P);push("bigroad_support",cand.support);push("target_fit_now",cand.targetFitNow||.5);push("target_fit_next",cand.targetFitNext||.5);push("pair_reliability",cand.pairReliability||0);push("pair_fit",cand.pairFit||.5);
  push("forward_directional",cand.forwardDirectional||0);push("survival_directional",cand.survivalDirectional||0);push("context_directional",cand.contextDirectional||0);push("two_step_directional",cand.twoStepDirectional||0);push("two_step_B",(cand.twoB??.5)-.5);push("two_step_P",(cand.twoP??.5)-.5);
  push("grid_current_row",clip(g.currentRow/(BIG_ROAD_ROWS-1)));push("grid_current_col",clip(g.currentViewCol/(BIG_ROAD_COLS-1)));push("grid_visible_occupancy",g.occupancy);push("grid_visible_banker_ratio",(g.bankerRatio-.5)*2);push("grid_top_occupancy",g.topOccupancy);push("grid_bottom_occupancy",g.bottomOccupancy);push("grid_tail_ratio",g.tailRatio);push("grid_current_in_tail",g.currentInTail);push("grid_current_at_bottom",g.currentAtBottom);push("grid_blocked_below",g.blockedBelow);push("grid_continue_moves_down",cand.continueMove?.down||0);push("grid_continue_moves_right",cand.continueMove?.right||0);push("grid_continue_row",clip((cand.continueMove?.row||0)/(BIG_ROAD_ROWS-1)));push("grid_continue_col",clip((cand.continueMove?.viewCol||0)/(BIG_ROAD_COLS-1)));push("grid_reverse_col",clip((cand.reverseMove?.viewCol||0)/(BIG_ROAD_COLS-1)));push("grid_active_columns",g.activeColumns);push("grid_avg_column_fill",g.avgFill);push("grid_max_column_fill",g.maxFill);push("grid_column_fill_std",g.fillStd);push("grid_profile_regularity",g.profileRegularity);
  while(road.length<ROAD_DIM)push(`reserved_${road.length}`,0);
  if(road.length!==ROAD_DIM)throw new Error(`road mismatch ${road.length}`);
  return{vector:road,candidates:cand,bigRoad:cand.road};
}

function context256(seq){const shoe=buildShoeVector(seq),roadData=buildRoadVector(seq),vector=[...shoe,...roadData.vector];if(vector.length!==DIM)throw new Error(`context mismatch ${vector.length}`);return{vector,candidates:roadData.candidates,bigRoad:roadData.bigRoad};}

function frozenPrior(){
  const A={B:Array(DIM).fill(RIDGE),P:Array(DIM).fill(RIDGE)},b={B:Array(DIM).fill(0),P:Array(DIM).fill(0)};
  const setDirectional=(name,weight,precision=1)=>{const local=ROAD_INDEX[name];if(local===undefined)return;const i=SHOE_DIM+local;A.B[i]=precision;A.P[i]=precision;b.B[i]=weight;b.P[i]=-weight;};
  setDirectional("banker_bias_8",.042,1.20);setDirectional("banker_bias_16",.034,1.24);setDirectional("banker_bias_32",.025,1.30);
  setDirectional("banker_delta_4_16",.032,1.18);setDirectional("banker_delta_8_32",.026,1.22);setDirectional("banker_delta_16_64",.018,1.28);
  setDirectional("bigroad_candidate_gap_B",.27,.98);
  setDirectional("forward_directional",.36,.92);
  setDirectional("survival_directional",.23,.98);
  setDirectional("context_directional",.21,1.00);
  setDirectional("two_step_directional",.18,1.02);
  setDirectional("grid_visible_banker_ratio",.008,1.42);
  setDirectional("current_side_banker",.004,1.48);
  return{A,b};
}

function modelX(raw){return raw.map(v=>Number.isFinite(+v)?+v:0);}
function scoreArm(arm,x,prior){const A=prior.A[arm],b=prior.b[arm];let mean=0,uncertaintySquared=0;for(let i=0;i<DIM;i++){const a=Math.max(1e-9,A[i]);mean+=x[i]*(b[i]/a);uncertaintySquared+=(x[i]*x[i])/a;}const uncertainty=Math.sqrt(Math.max(0,uncertaintySquared));return{mean,uncertainty,score:mean+ALPHA*uncertainty,effectiveAlpha:ALPHA};}
function deterministicTie(seq){let h=0;const token="FROZEN256_BIGROAD_6X15_FORWARD_V18|"+seq.join("");for(let i=0;i<token.length;i++)h=(h*31+token.charCodeAt(i))>>>0;return h%2?"B":"P";}

function choose(seq){
  const ctx=context256(seq),x=modelX(ctx.vector),prior=frozenPrior(),scores={B:scoreArm("B",x,prior),P:scoreArm("P",x,prior)},gap=scores.B.score-scores.P.score;
  let direction,reason;if(Math.abs(gap)<=SCORE_TIE_EPS){direction=deterministicTie(seq);reason="固定歷史平手判定";}else{direction=gap>0?"B":"P";reason="256維＋6×15前瞻雙臂判斷";}
  const rawPB=1/(1+Math.exp(-Math.max(-8,Math.min(8,gap/SCORE_TEMP)))),pB=clip(rawPB,PROB_MIN,PROB_MAX),pP=1-pB,confidence=direction==="B"?pB:pP;
  const previous=state.last_selected;state.selection_streak=previous===direction?(state.selection_streak||0)+1:1;state.last_selected=direction;
  const c=ctx.candidates;let regime="混合";
  if(c.support>=.25){const currentContinuation=c.currentSide&&direction===c.currentSide;if(Math.abs(c.B-c.P)>=.045||Math.abs(c.forwardDirectional||0)>=.20)regime=currentContinuation?"大路延續":"大路反轉";else regime="大路觀察";}
  const strength=clip(.40+.32*c.support+.14*Math.abs(c.forwardDirectional||0)+.08*Math.abs(c.twoStepDirectional||0)+Math.min(.16,Math.abs(gap)*.66));
  return{direction,reason,x,scores,gap,confidence,probabilities:{B:pB,P:pP},regime,strength,candidates:c,bigRoad:{rows:BIG_ROAD_ROWS,cols:BIG_ROAD_COLS,viewStartCol:ctx.bigRoad.viewStartCol,viewEndCol:ctx.bigRoad.viewEndCol,maxCol:ctx.bigRoad.maxCol,grid:ctx.bigRoad.grid},frozen:{bootstrap:false,walk_forward:false,replay:false,settle_previous:false,update_ab:false,decay:false,alpha:ALPHA,ridge:RIDGE,context_dim:DIM}};
}

function save(){try{if(typeof localStorage!=="undefined")localStorage.setItem(STORAGE_KEY,JSON.stringify({history:state.history,last_selected:state.last_selected,selection_streak:state.selection_streak}));}catch(_){}}
function load(){try{if(typeof localStorage==="undefined")return;let text=localStorage.getItem(STORAGE_KEY);if(!text){for(const key of LEGACY_KEYS){const t=localStorage.getItem(key);if(t){text=t;break;}}}const raw=JSON.parse(text||"null");if(!raw||!Array.isArray(raw.history))return;state.history=raw.history.filter(x=>["B","P","T"].includes(x)).slice(-500);state.last_selected=["B","P"].includes(raw.last_selected)?raw.last_selected:"";state.selection_streak=Math.max(0,+raw.selection_streak||0);}catch(_){}}
function clearAllStorage(){try{if(typeof localStorage==="undefined")return;localStorage.removeItem(STORAGE_KEY);for(const key of LEGACY_KEYS)localStorage.removeItem(key);}catch(_){}}
function setMessage(text,warning=false){const box=el("message");if(!box)return;box.textContent=text;box.classList.toggle("warning",warning);}
function addOutcome(outcome){if(!["B","P","T"].includes(outcome))return;state.history.push(outcome);state.active=false;state.lastPrediction=null;save();setMessage(`已記錄第 ${state.history.length} 局`);render();}
function startAnalysis(){if(!state.history.length){setMessage("請先輸入牌局紀錄",true);return;}state.lastPrediction=choose(state.history);state.active=true;save();setMessage(`第 ${state.history.length+1} 局分析完成`);render();}
function backOne(){if(!state.history.length){setMessage("目前沒有可返回的紀錄",true);return;}state.history.pop();state.active=false;state.lastPrediction=null;save();setMessage("已返回上一局");render();}
function endAnalysis(){state={history:[],active:false,lastPrediction:null,last_selected:"",selection_streak:0};clearAllStorage();setMessage("本靴資料已清空");render();}
function renderHistory(){const box=el("historyTrack");if(!box)return;if(!state.history.length){box.innerHTML='<div class="empty-history">尚未輸入牌局</div>';return;}box.innerHTML=state.history.map((v,i)=>`<span class="history-chip ${v}" title="第 ${i+1} 局">${v}</span>`).join("");box.scrollLeft=box.scrollWidth;}
function renderPrediction(){const p=state.active?state.lastPrediction:null,orb=el("directionOrb");if(!orb)return;if(!p){el("directionText").textContent="—";el("directionCode").textContent="等待分析";el("confidence").textContent="—";el("regime").textContent="—";el("strength").textContent="—";orb.className="direction-orb idle";return;}const isB=p.direction==="B";el("directionText").textContent=isB?"莊":"閒";el("directionCode").textContent=isB?"BANKER":"PLAYER";el("confidence").textContent=(p.confidence*100).toFixed(1)+"%";el("regime").textContent=p.regime;el("strength").textContent=p.strength>=.68?"穩定":p.strength>=.52?"中等":"保守";orb.className="direction-orb "+(isB?"banker":"player");}
function render(){if(el("roundPill"))el("roundPill").textContent=`${state.history.length} 局`;if(el("modePill"))el("modePill").textContent=state.active?"分析完成":"準備中";if(el("roundCount"))el("roundCount").textContent=state.history.length;renderPrediction();renderHistory();}

if(el("btnB"))el("btnB").addEventListener("click",()=>addOutcome("B"));
if(el("btnP"))el("btnP").addEventListener("click",()=>addOutcome("P"));
if(el("btnT"))el("btnT").addEventListener("click",()=>addOutcome("T"));
if(el("btnStart"))el("btnStart").addEventListener("click",startAnalysis);
if(el("btnBack"))el("btnBack").addEventListener("click",backOne);
if(el("btnEnd"))el("btnEnd").addEventListener("click",endAnalysis);

if(typeof window!=="undefined")window.__BGS256_TEST__={runs,buildBigRoad,bigRoadGeometry,bigRoadCandidates,context256,choose,stageSurvival,contextualStageStats,branchFutureQuality,constants:{DIM,SHOE_DIM,ROAD_DIM,ALPHA,RIDGE,PROB_MIN,PROB_MAX,BIG_ROAD_ROWS,BIG_ROAD_COLS}};
load();render();
})();
