/**
 * B/P/T-only derived-road diagnostics (research display, NOT betting advice).
 * Pure UMD module: no physical posterior, card ranks, RNG or bankroll input.
 */
(function(root,factory){
 "use strict";
 const api=factory();
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.BBBDerivedRoadAux=api;
})(typeof self!=="undefined"?self:typeof window!=="undefined"?window:null,function(){
 "use strict";
 const valid=x=>x==="B"||x==="P"||x==="T";
 const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
 /**
  * Logical Big Road: one column per consecutive B/P streak, with unbounded
  * logical depth. The six-row on-screen dragon tail is merely presentation;
  * derived-road comparison must use the unwrapped logical column depths.
  * Ties do not create beads or columns.
  */
 function buildBigRoad(history){
  if(!Array.isArray(history)||history.some(x=>!valid(x)))
   throw Error("invalid_bpt_history");
  const columns=[],beads=[];let last=null;
  for(const value of history){
   if(value==="T")continue;
   if(value!==last){columns.push({side:value,depth:0});last=value;}
   const column=columns.length-1,row=columns[column].depth++;
   beads.push({column,row,side:value});
  }
  return {columns,beads,depth:beads.length,last_side:last};
 }
 /**
  * Derived roads compare Big Road shapes, NOT banker/player outcomes.
  * offset=1: Big Eye; 2: Small; 3: Cockroach.
  * A new column compares the completed lengths of columns (c-1) and
  * (c-offset-1). A continuation compares whether the older column at
  * c-offset contains the same logical row and the row just above it.
  * Both occupied / both empty => red; exactly one occupied => blue.
  * N means not enough prior columns.
  */
 function deriveColor(columns,column,row,offset){
  if(row===0){
   if(column<offset+1)return "N";
   return columns[column-1].depth===columns[column-offset-1].depth?"R":"B";
  }
  if(column<offset)return "N";
  const older=columns[column-offset].depth;
  const same=older>row,above=older>=row;
  return same===above?"R":"B";
 }
 function deriveMarks(road,offset){
  const marks=[];
  // Process beads in observed order; comparisons involve completed earlier
  // columns, which never change after their run has ended.
  for(const bead of road.beads){
   const color=deriveColor(road.columns,bead.column,bead.row,offset);
   if(color!=="N")marks.push(color);
  }
  return marks;
 }
 /**
  * Explicit, UNVALIDATED interpretation: majority R ("regularity") maps to
  * continuing the latest Big Road B/P colour; majority B ("irregularity")
  * maps to switching it. Derived-road red/blue are NOT actual B/P labels,
  * so this mapping must never modify Soft Shoe candidate, EV or wager gate.
  */
 function mapRoadSide(lastSide,red,blue){
  if(!["B","P"].includes(lastSide)||red===blue)return "NEUTRAL";
  const continuation=red>blue;
  return continuation?lastSide:lastSide==="B"?"P":"B";
 }
 function probeSnapshot(history,min){
  const road=buildBigRoad(history);
  const probes={big_eye:"N",small:"N",cockroach:"N"};
  if(road.depth<min)return {available:false,reason:"insufficient_non_tie_history",
   road,probes,road_side:"NEUTRAL",road_agreement:0,road_confidence:0};
  const labels=[["big_eye",1],["small",2],["cockroach",3]];
  for(const [name,offset] of labels){
   const marks=deriveMarks(road,offset);
   probes[name]=marks.length?marks[marks.length-1]:"N";
  }
  const votes=Object.values(probes).filter(x=>x!=="N");
  if(votes.length<2)return {available:false,
   reason:"insufficient_derived_road_columns",road,probes,
   road_side:"NEUTRAL",road_agreement:0,road_confidence:0};
  const red=votes.filter(x=>x==="R").length,blue=votes.length-red;
  const agreement=Math.max(red,blue)/votes.length;
  // Heuristic reliability descriptor, not a calibrated win probability.
  const confidence=clamp((road.depth-min+1)/(min+12),0,1)*
   (votes.length/3)*(.5+.5*agreement);
  const side=mapRoadSide(road.last_side,red,blue);
  return {available:side!=="NEUTRAL",
   ...(side==="NEUTRAL"?{reason:"no_majority"}:{}),road,probes,
   road_side:side,road_agreement:agreement,road_confidence:confidence};
 }
 function recentStability(sequence,min,window,current){
  if(!current?.available)return 0;
  const start=Math.max(0,sequence.length-window),snapshots=[];
  for(let i=start;i<sequence.length;i++){
   const snap=probeSnapshot(sequence.slice(0,i+1),min);
   if(snap.available)snapshots.push(snap.probes);
  }
  let transitions=0,switches=0;
  for(const name of ["big_eye","small","cockroach"]){
   let previous=null;
   for(const snapshot of snapshots){
    const color=snapshot[name];
    if(color!=="R"&&color!=="B")continue;
    if(previous!==null){transitions++;if(color!==previous)switches++;}
    previous=color;
   }
  }
  const temporal=transitions?1-switches/transitions:1;
  const coverage=clamp(snapshots.length/window,0,1);
  return clamp(current.road_agreement*temporal*coverage,0,1);
 }
 function learnPerformance(sequence,min,window,minSamples,floor,ceil,neutral){
  const outcomes=[];
  // Each prefix is a road forecast event. A successful Soft Shoe research
  // report always has a B/P candidate, even when its formal direction is SKIP;
  // only explicit derived-road sides are eligible for the bounded hit-rate.
  const start=Math.max(0,sequence.length-window-1);
  for(let i=start;i<sequence.length-1;i++){
   const snap=probeSnapshot(sequence.slice(0,i+1),min);
   if(!snap.available||!["B","P"].includes(snap.road_side))continue;
   outcomes.push(snap.road_side===sequence[i+1]?1:0);
  }
  const samples=outcomes.slice(-window),n=samples.length;
  if(n<minSamples)return {score:neutral,samples:n,hits:samples.reduce((a,b)=>a+b,0)};
  const hits=samples.reduce((a,b)=>a+b,0);
  // Eight pseudo-prior observations centered at 50% keep a small window calm.
  const rate=(hits+4)/(n+8);
  return {score:clamp(floor+(ceil-floor)*rate,floor,ceil),samples:n,hits};
 }
 function unavailable(reason,columns,depth,probes,adaptive){
  return {available:false,reason,big_road_columns:columns,
   big_road_depth:depth,probes,road_side:"NEUTRAL",
   road_agreement:0,road_confidence:0,adaptive,
   estimated_only:true,validated_betting_edge:false};
 }
 function analyze(history,opts={}){
  if(!Array.isArray(history)||history.some(x=>!valid(x)))
   throw Error("invalid_bpt_history");
  const min=opts.minHistory===undefined?12:opts.minHistory;
  const stabilityWindow=opts.stabilityWindow===undefined?8:opts.stabilityWindow;
  const learnWindow=opts.learnWindow===undefined?30:opts.learnWindow;
  const learnMinSamples=opts.learnMinSamples===undefined?8:opts.learnMinSamples;
  const learnFloor=opts.learnFloor===undefined ? .40:opts.learnFloor;
  const learnCeil=opts.learnCeil===undefined?1:opts.learnCeil;
  const learnNeutral=opts.learnNeutral===undefined ? .70:opts.learnNeutral;
  const adaptiveEnabled=opts.adaptiveEnabled!==false;
  if(!Number.isInteger(min)||min<1||min>72)
   throw Error("invalid_ROAD_AUX_MIN_HISTORY");
  if(!Number.isInteger(stabilityWindow)||stabilityWindow<1||stabilityWindow>72)
   throw Error("invalid_ROAD_AUX_STABILITY_WINDOW");
  if(!Number.isInteger(learnWindow)||learnWindow<1||learnWindow>72)
   throw Error("invalid_ROAD_AUX_LEARN_WINDOW");
  if(!Number.isInteger(learnMinSamples)||learnMinSamples<1||
     learnMinSamples>learnWindow)
   throw Error("invalid_ROAD_AUX_LEARN_MIN_SAMPLES");
  if(![learnFloor,learnCeil,learnNeutral].every(Number.isFinite)||
     learnFloor<0||learnCeil>1||learnFloor>learnCeil||
     learnNeutral<learnFloor||learnNeutral>learnCeil)
   throw Error("invalid_ROAD_AUX_LEARN_RANGE");
  if(typeof adaptiveEnabled!=="boolean")throw Error("invalid_ROAD_AUX_ADAPTIVE_ENABLED");
  const sequence=history.filter(x=>x!=="T");
  const snap=probeSnapshot(history,min),road=snap.road;
  const stability=recentStability(sequence,min,stabilityWindow,snap);
  const quality=snap.available?clamp(.5*snap.road_agreement+
   .3*snap.road_confidence+.2*clamp(road.depth/min,0,1),0,1):0;
  const learned=learnPerformance(sequence,min,learnWindow,learnMinSamples,
   learnFloor,learnCeil,learnNeutral);
  const adaptive={enabled:adaptiveEnabled,quality,stability,
   recency_score:learned.score,w_applied:0,learn_samples:learned.samples,
   mode:adaptiveEnabled?"adaptive":"fixed"};
  const result=snap.available?{
   available:true,big_road_columns:road.columns.length,
   big_road_depth:road.depth,probes:snap.probes,road_side:snap.road_side,
   road_agreement:snap.road_agreement,road_confidence:snap.road_confidence,
   adaptive
  }:snap.reason==="no_majority"?{
   available:false,reason:"no_majority",big_road_columns:road.columns.length,
   big_road_depth:road.depth,probes:snap.probes,road_side:"NEUTRAL",
   road_agreement:snap.road_agreement,road_confidence:snap.road_confidence,
   adaptive
  }:unavailable(snap.reason,road.columns.length,road.depth,snap.probes,adaptive);
  return {...result,estimated_only:true,validated_betting_edge:false};
 }
 return Object.freeze({buildBigRoad,deriveColor,deriveMarks,mapRoadSide,analyze});
});
