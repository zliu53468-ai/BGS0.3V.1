"""Generate a side-invariant simulated baccarat road-shape database.

Stores continuation/switch statistics for S/C/T shape contexts instead of
absolute Banker/Player next-side counts. This prevents the pattern layer from
following whichever side currently appears more often.

Usage:
  python generate_simulated_baccarat_shapes.py --db pattern_10m.sqlite3 \
      --transitions 10000000 --max-order 24
"""
from __future__ import annotations
import argparse, json, random, time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from pattern_database_shape import initialize_database

B_PROB, P_PROB, T_PROB = 0.4586, 0.4462, 0.0952
DEFAULT_WEIGHTS: Dict[str, float] = {
    "random": 0.55, "dragon": 0.08, "chop": 0.10,
    "double_chop": 0.08, "one_two": 0.07,
    "two_one": 0.07, "cluster": 0.05,
}

def table(weights: Dict[str,float]) -> List[Tuple[str,float]]:
    total=sum(max(0.0,v) for v in weights.values()) or 1.0
    out=[]; c=0.0
    for k,v in weights.items():
        c += max(0.0,v)/total; out.append((k,c))
    out[-1]=(out[-1][0],1.0); return out

def pick(rng: random.Random, t: Sequence[Tuple[str,float]]) -> str:
    x=rng.random()
    for k,c in t:
        if x <= c: return k
    return t[-1][0]

def rand_bp(rng: random.Random) -> str:
    return "B" if rng.random() < B_PROB/(B_PROB+P_PROB) else "P"

def other(s: str) -> str: return "P" if s=="B" else "B"

def segment(rng: random.Random, regime: str, n: int, start: Optional[str]) -> List[str]:
    side=start if start in {"B","P"} and rng.random()<0.5 else rand_bp(rng)
    out: List[str]=[]
    if regime=="random": return [rand_bp(rng) for _ in range(n)]
    if regime=="chop":
        for _ in range(n): out.append(side); side=other(side)
        return out
    if regime=="double_chop":
        while len(out)<n: out += [side]*min(2,n-len(out)); side=other(side)
        return out
    if regime in {"one_two","two_one"}:
        rhythm=[1,2] if regime=="one_two" else [2,1]; i=0
        while len(out)<n:
            run=rhythm[i%2]; out += [side]*min(run,n-len(out)); side=other(side); i+=1
        return out
    if regime=="dragon":
        dominant=side
        while len(out)<n:
            run=rng.choices([3,4,5,6,7,8],[10,18,24,22,16,10],k=1)[0]
            out += [dominant]*min(run,n-len(out))
            if len(out)<n: out += [other(dominant)]*min(rng.choice([1,1,2]),n-len(out))
        return out
    while len(out)<n:
        run=rng.choices([1,2,3,4,5],[18,30,26,17,9],k=1)[0]
        out += [side]*min(run,n-len(out)); side=other(side)
    return out

def generate_shoe(rng: random.Random, min_hands:int, max_hands:int,
                  weights:Dict[str,float], min_seg:int, max_seg:int) -> List[str]:
    target=rng.randint(min_hands,max_hands); t=table(weights)
    shoe=[]; last=None
    while len(shoe)<target:
        reg=pick(rng,t); size=min(target-len(shoe),rng.randint(min_seg,max_seg))
        for bp in segment(rng,reg,size,last):
            val="T" if rng.random()<T_PROB else bp
            shoe.append(val)
            if val in {"B","P"}: last=val
    return shoe[:target]

def flush(conn, batch: Dict[str,List[int]]) -> int:
    if not batch: return 0
    conn.executemany("""
      INSERT INTO patterns(context,continue_count,switch_count) VALUES(?,?,?)
      ON CONFLICT(context) DO UPDATE SET
      continue_count=continue_count+excluded.continue_count,
      switch_count=switch_count+excluded.switch_count
    """, ((k,v[0],v[1]) for k,v in batch.items()))
    n=len(batch); batch.clear(); conn.commit(); return n

def build(db:str, transitions_target:int, max_order:int, seed:int,
          min_hands:int, max_hands:int, weights:Dict[str,float]) -> Dict[str,object]:
    path=Path(db); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): path.unlink()
    conn=initialize_database(str(path),replace_legacy=True)
    rng=random.Random(seed); batch={}; transitions=shoes=hands=flushed=0
    outcomes=Counter(); started=time.time()
    while transitions < transitions_target:
        shoe=generate_shoe(rng,min_hands,max_hands,weights,6,24)
        shoes+=1; hands+=len(shoe); outcomes.update(shoe)
        prev=None; shape=""
        for target in shoe:
            if target=="T":
                if prev is not None: shape += "T"
                continue
            if prev is None: prev=target; continue
            if transitions>=transitions_target: break
            cont=target==prev; transitions+=1
            for order in range(min(max_order,len(shape))+1):
                ctx="" if order==0 else shape[-order:]
                counts=batch.setdefault(ctx,[0,0]); counts[0 if cont else 1]+=1
            shape += "S" if cont else "C"; prev=target
        if shoes%1000==0 or len(batch)>=250000: flushed += flush(conn,batch)
    flushed += flush(conn,batch)
    meta={"schema":"shape_continue_switch_v2","source_type":"synthetic_side_invariant",
          "source_shoes":str(shoes),"transitions":str(transitions),
          "max_order":str(max_order),"seed":str(seed),"side_invariant":"1",
          "regime_weights":json.dumps(weights,sort_keys=True),
          "warning":"Synthetic cold-start data; not real casino history."}
    conn.executemany("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",meta.items()); conn.commit()
    contexts=conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
    root=conn.execute("SELECT continue_count,switch_count FROM patterns WHERE context='' ").fetchone(); conn.close()
    return {"ok":True,"db_path":str(path.resolve()),"schema":"shape_continue_switch_v2",
            "shoes":shoes,"transitions":transitions,"contexts":contexts,
            "root_counts":{"continue":root[0],"switch":root[1]} if root else None,
            "outcomes":dict(outcomes),"elapsed_seconds":round(time.time()-started,3)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--db",default="pattern_10m.sqlite3")
    ap.add_argument("--transitions",type=int,default=10_000_000); ap.add_argument("--max-order",type=int,default=24)
    ap.add_argument("--seed",type=int,default=20260713); ap.add_argument("--min-hands",type=int,default=55)
    ap.add_argument("--max-hands",type=int,default=85); args=ap.parse_args()
    print(json.dumps(build(args.db,args.transitions,args.max_order,args.seed,args.min_hands,args.max_hands,DEFAULT_WEIGHTS),ensure_ascii=False,indent=2))
if __name__=="__main__": main()
