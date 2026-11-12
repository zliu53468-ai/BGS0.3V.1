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
 function unavailable(reason,columns,depth,probes){
  return {available:false,reason,big_road_columns:columns,
   big_road_depth:depth,probes,road_side:"NEUTRAL",
   road_agreement:0,road_confidence:0,
   estimated_only:true,validated_betting_edge:false};
 }
 function analyze(history,opts={}){
  const min=opts.minHistory===undefined?12:opts.minHistory;
  if(!Number.isInteger(min)||min<1||min>72)
   throw Error("invalid_ROAD_AUX_MIN_HISTORY");
  const road=buildBigRoad(history);
  const probes={big_eye:"N",small:"N",cockroach:"N"};
  if(road.depth<min)
   return unavailable("insufficient_non_tie_history",road.columns.length,road.depth,probes);
  const labels=[["big_eye",1],["small",2],["cockroach",3]];
  for(const [name,offset] of labels){
   const marks=deriveMarks(road,offset);
   probes[name]=marks.length?marks[marks.length-1]:"N";
  }
  const votes=Object.values(probes).filter(x=>x!=="N");
  if(votes.length<2)
   return unavailable("insufficient_derived_road_columns",
    road.columns.length,road.depth,probes);
  const red=votes.filter(x=>x==="R").length,blue=votes.length-red;
  const agreement=Math.max(red,blue)/votes.length;
  // Heuristic reliability descriptor, not a calibrated win probability.
  const confidence=clamp((road.depth-min+1)/(min+12),0,1)*
   (votes.length/3)*(.5+.5*agreement);
  const side=mapRoadSide(road.last_side,red,blue);
  return {available:side!=="NEUTRAL",
   ...(side==="NEUTRAL"?{reason:"no_majority"}:{}),
   big_road_columns:road.columns.length,big_road_depth:road.depth,
   probes,road_side:side,road_agreement:agreement,
   road_confidence:confidence,
   estimated_only:true,validated_betting_edge:false};
 }
 return Object.freeze({buildBigRoad,deriveColor,deriveMarks,mapRoadSide,analyze});
});
