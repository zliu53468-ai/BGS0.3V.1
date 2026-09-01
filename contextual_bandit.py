"""Panel-compatible 32D Single-Brain Contextual LinUCB core."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Mapping, Sequence
import json, math, os, time
import numpy as np

from road_model import build_standard_derived_roads
from shoe_composition import analyze_shoe_composition, fresh_counts
from shoe_constants import AVERAGE_CARDS_PER_HAND, SHOE_DECKS

ARMS = ("P", "B")
CONTEXT_DIM = 32
CONTEXT_FEATURE_NAMES = (
    "remaining_cards_ratio","penetration_ratio","estimated_hands_remaining_norm","shoe_maturity_ratio",
    "rank_A_relative_ratio","rank_2_relative_ratio","rank_3_relative_ratio","rank_4_relative_ratio",
    "rank_5_relative_ratio","rank_6_relative_ratio","rank_7_relative_ratio","rank_8_relative_ratio",
    "rank_9_relative_ratio","rank_10JQK_relative_ratio","physical_edge_proxy","shoe_information_reliability",
    "current_side_banker_binary","current_run_length_norm","previous_run_length_norm","previous2_run_length_norm",
    "recent5_banker_ratio","recent8_banker_ratio","recent12_banker_ratio","recent5_turn_rate",
    "recent8_turn_rate","recent12_turn_rate","run_length_hazard_rate","hsmm_stable_probability",
    "big_eye_regularity","small_road_regularity","cockroach_road_regularity","derived_road_consensus",
)
LINUCB_ALPHA=max(0.0,float(os.getenv("LINUCB_ALPHA","0.5") or "0.5"))
LINUCB_RIDGE=max(1e-6,float(os.getenv("LINUCB_RIDGE","1.0") or "1.0"))
LINUCB_UPDATE_WEIGHT=max(1e-3,float(os.getenv("LINUCB_UPDATE_WEIGHT","1.0") or "1.0"))
LINUCB_FORGETTING=max(.70,min(1.0,float(os.getenv("LINUCB_FORGETTING","0.90") or "0.90")))
LINUCB_ARM_ALPHA_MAX_SCALE=max(1.0,min(2.5,float(os.getenv("LINUCB_ARM_ALPHA_MAX_SCALE","1.60") or "1.60")))
LINUCB_SCORE_TIE_EPSILON=max(1e-12,float(os.getenv("LINUCB_SCORE_TIE_EPSILON","0.000001") or "0.000001"))
LINUCB_SCORE_TEMPERATURE=max(.25,min(10.0,float(os.getenv("LINUCB_SCORE_TEMPERATURE","2.0") or "2.0")))
ROAD_PRIOR_SCORE_WEIGHT=0.0
ROAD_PRIOR_PROBABILITY_SPAN=0.0
LINUCB_PROBABILITY_CORRECTION_SPAN=0.0
PROBABILITY_MIN=.42
PROBABILITY_MAX=.58
STATE_VERSION="LINUCB-2ARM-SINGLE-BRAIN-CONTEXT-16SHOE-16ROAD-32D-PANEL-V8"
ESTIMATED_CARDS_PER_ROUND=AVERAGE_CARDS_PER_HAND
_LOCK=RLock()

def _clip(v:Any,lo:float=0.,hi:float=1.)->float:
    try:n=float(v)
    except (TypeError,ValueError):return lo
    return lo if not math.isfinite(n) else max(lo,min(hi,n))

def _norm(history:Iterable[Any]|str|None)->list[str]:
    if history is None:return []
    if isinstance(history,str):
        c=history.replace("|","").replace(",","").replace(" ","").upper()
        if c and all(x in {"B","P","T"} for x in c):return list(c)[-2000:]
        items=[x for x in history.replace("|",",").split(",") if x.strip()]
    else:items=deepcopy(list(history))
    out=[]
    for item in items:
        raw=(item.get("outcome") or item.get("actual") or item.get("actual_outcome") or item.get("virtual_outcome")) if isinstance(item,Mapping) else item
        v=str(raw or "").upper().strip()
        if v in {"B","P","T"}:out.append(v)
    return out[-2000:]

def _bp(s:Sequence[str])->list[str]:return [x for x in s if x in {"B","P"}]
def _runs(s:Sequence[str])->list[tuple[str,int]]:
    a=_bp(s)
    if not a:return []
    out=[];side=a[0];n=1
    for v in a[1:]:
        if v==side:n+=1
        else:out.append((side,n));side=v;n=1
    out.append((side,n));return out

def _br(s:Sequence[str],w:int)->float:
    a=_bp(s)[-max(1,int(w)):]
    return sum(x=="B" for x in a)/len(a) if a else .5

def _tr(s:Sequence[str],w:int)->float:
    a=_bp(s)[-max(2,int(w)):]
    if len(a)<2:return .5
    return sum(a[i]!=a[i-1] for i in range(1,len(a)))/(len(a)-1)

def _reg(vals:Iterable[Any],w:int=8)->tuple[float,int]:
    a=[str(x).upper().strip() for x in list(vals)[-max(1,w):]];a=[x for x in a if x in {"R","U"}]
    return ((sum(x=="R" for x in a)/len(a),len(a)) if a else (.5,0))

def _lb(n:int)->str:
    n=max(1,int(n));return str(n) if n<=5 else "6+"
def _hzctx(side:str,cur:int,prev:Sequence[int])->list[tuple[str,str]]:
    ph=prev[-1] if prev else 0;d=[("UP" if prev[i]>prev[i-1] else "DOWN" if prev[i]<prev[i-1] else "EQUAL") for i in range(1,len(prev))]
    d1=d[-1] if d else "NA";d2=d[-2] if len(d)>=2 else "NA";c=_lb(cur);p=_lb(ph) if ph else "0"
    return [("full",f"HZF|side={side or 'NA'}|cur={c}|prev={p}|d1={d1}|d2={d2}"),("structure",f"HZS|cur={c}|prev={p}|d1={d1}|d2={d2}"),("shape",f"HZP|cur={c}|prev={p}|d1={d1}"),("length",f"HZL|cur={c}"),("global","HZG|GLOBAL")]
def _hztab(rs:Sequence[tuple[str,int]])->dict[str,dict[str,float]]:
    done=list(rs[:-1]);heights=[x[1] for x in done];tab={}
    for idx,(side,final) in enumerate(done):
        prev=heights[:idx]
        for at in range(1,max(1,final)+1):
            ev="CONTINUE" if at<final else "TURN"
            for _,k in _hzctx(side,at,prev):tab.setdefault(k,{"CONTINUE":0.,"TURN":0.})[ev]+=1.
    return tab
def _hzpost(c:Mapping[str,Any])->dict[str,float]:
    co=float(c.get("CONTINUE",0) or 0);tu=float(c.get("TURN",0) or 0);den=co+tu+6.
    return {"CONTINUE":(co+3.)/den,"TURN":(tu+3.)/den} if den>1e-12 else {"CONTINUE":.5,"TURN":.5}
def _hazard(s:Sequence[str])->float:
    rs=_runs(s)
    if not rs:return .5
    side,cur=rs[-1];heights=[x[1] for x in rs[:-1]];tab=_hztab(rs);tier="prior";prob={"CONTINUE":.5,"TURN":.5};pen=1.;ctx=_hzctx(side,cur,heights)
    for i,(name,k) in enumerate(ctx):
        c=tab.get(k,{"CONTINUE":0.,"TURN":0.});sup=c["CONTINUE"]+c["TURN"];p=_hzpost(c)
        if sup>=4:tier=name;prob=p;break
        if i<len(ctx)-1:pen*=.75
    if tier=="prior":
        g=tab.get("HZG|GLOBAL",{"CONTINUE":0.,"TURN":0.})
        if g["CONTINUE"]+g["TURN"]>0:prob=_hzpost(g)
        else:pen=0.
    cont=(1-pen)*.5+pen*prob["CONTINUE"];return _clip(1-cont)

def _entropy(s:Sequence[str],w:int=12)->float:
    a=list(s[-w:])
    if not a:return 1.
    e=0.
    for o in ("B","P","T"):
        p=sum(x==o for x in a)/len(a)
        if p>0:e-=p*math.log2(p)
    return _clip(e/math.log2(3))
def _vol(s:Sequence[str])->float:
    h=[x[1] for x in _runs(s)[-6:]]
    if len(h)<2:return .25
    return _clip((sum(abs(h[i]-h[i-1]) for i in range(1,len(h)))/(len(h)-1))/3)
def _hsmm(s:Sequence[str])->float:
    a=_tr(s,10);rs=_runs(s);cur=rs[-1][1] if rs else 0;r=_clip(cur/6);e=_entropy(s);v=_vol(s)
    p=math.exp(-((a-.25)/.24)**2-((r-.70)/.28)**2-((e-.62)/.24)**2-((v-.26)/.24)**2)
    q=math.exp(-((a-.84)/.18)**2-((r-.18)/.20)**2-((e-.70)/.23)**2-((v-.30)/.24)**2)
    t=math.exp(-((a-.52)/.28)**2-((r-.34)/.26)**2-((e-.82)/.18)**2-((v-.72)/.23)**2)
    n=math.exp(-((a-.55)/.30)**2-((r-.27)/.24)**2-((e-.94)/.11)**2-((v-.55)/.28)**2)
    w=(.25*p,.25*q,.20*t,.30*n);tot=sum(w) or 1.;return _clip((w[0]+w[1])/tot)

def _x(v:Sequence[float])->np.ndarray:return np.nan_to_num(np.asarray(v,dtype=np.float64).reshape(CONTEXT_DIM),nan=0.,posinf=2.,neginf=-1.)
@dataclass(frozen=True)
class ContextSnapshot:
    vector:np.ndarray
    metadata:dict[str,Any]

class ContextGenerator:
    def build(self,history:Iterable[Any]|str|None,shoe_context:Mapping[str,Any]|None=None)->ContextSnapshot:
        raw=_norm(deepcopy(history));bp=_bp(raw);ctx=deepcopy(dict(shoe_context or {}))
        try:decks=int(ctx.get("decks",SHOE_DECKS) or SHOE_DECKS)
        except (TypeError,ValueError):decks=SHOE_DECKS
        decks=max(1,min(16,decks));total=float(52*decks);exact=analyze_shoe_composition(ctx,default_decks=decks);counts=[]
        if exact.get("available"):
            try:counts=[float(v) for v in exact.get("remaining_counts",[])]
            except (TypeError,ValueError):counts=[]
        exact_ok=len(counts)==10
        if exact_ok:remaining=float(sum(counts));remaining_source=str(exact.get("remaining_cards_source") or exact.get("source") or "exact_remaining_counts")
        else:remaining=max(0.,total-len(raw)*float(AVERAGE_CARDS_PER_HAND));remaining_source="panel_history_round_estimate"
        remaining=max(0.,min(total,remaining));rr=_clip(remaining/total if total else 1.);pen=_clip(1-rr);maturity=_clip(len(raw)/70.);ratios=[];groups=[]
        if exact_ok:
            fresh=[float(v) for v in fresh_counts(decks)]
            for point in (1,2,3,4,5,6,7,8,9,0):
                exp=fresh[point]*rr;ratios.append(_clip(1. if exp<=1e-12 else counts[point]/exp,0.,2.))
            def gr(points):
                exp=sum(fresh[p]*rr for p in points);obs=sum(counts[p] for p in points);return _clip(1. if exp<=1e-12 else obs/exp,0.,2.)
            groups=[gr((1,2,3)),gr((4,5)),gr((6,)),gr((7,)),gr((8,)),gr((9,)),gr((0,))];ratio_source="exact_relative_to_expected_depth"
        else:ratios=[1.]*10;groups=[1.]*7;ratio_source="neutral_fallback"
        ew=(.02,.01,.01,.02,.03,.04,.04,.03,.02,-.03);edge=_clip(sum(w*(r-1) for w,r in zip(ew,ratios)),-1.,1.) if exact_ok else 0.;rel=1. if exact_ok else 0.
        rs=_runs(raw);side,run=rs[-1] if rs else ("",0);pr=rs[-2][1] if len(rs)>=2 else 0;p2=rs[-3][1] if len(rs)>=3 else 0;side_b=1. if side=="B" else 0. if side=="P" else .5;hz=_hazard(raw);stable=_hsmm(raw)
        d=build_standard_derived_roads(deepcopy(bp));be=list(d.get("big_eye") or []);sm=list(d.get("small_road") or []);cr=list(d.get("cockroach_road") or []);ber,bn=_reg(be);smr,sn=_reg(sm);crr,cn=_reg(cr);mean=(ber+smr+crr)/3;cons=_clip(1-(abs(ber-mean)+abs(smr-mean)+abs(crr-mean))/1.5);cs=sn+cn;small_c=(smr*sn+crr*cn)/cs if cs else .5
        marks=[str(r[-1]).upper() for r in (be,sm,cr) if r and str(r[-1]).upper() in {"R","U"}];binary=1. if marks and sum(x=="R" for x in marks)*2>=len(marks) else 0.
        shoe=[rr,pen,rr,maturity,*ratios,edge,rel];road=[side_b,_clip(run/8),_clip(pr/8),_clip(p2/8),_br(raw,5),_br(raw,8),_br(raw,12),_tr(raw,5),_tr(raw,8),_tr(raw,12),hz,stable,ber,smr,crr,cons];vec=np.nan_to_num(np.asarray([*shoe,*road],dtype=np.float64),nan=0.,posinf=2.,neginf=-1.)
        if vec.shape!=(32,):raise RuntimeError(f"context dimension mismatch: {vec.shape}")
        return ContextSnapshot(vec,{"raw_round_count":len(raw),"bp_round_count":len(bp),"tie_count":sum(x=="T" for x in raw),"remaining_cards":remaining,"remaining_ratio":rr,"penetration_ratio":pen,"estimated_hands_remaining_norm":rr,"shoe_maturity_ratio":maturity,"remaining_cards_source":remaining_source,"soft_remaining_cards_ignored_for_panel_compatibility":(not exact_ok and bool(ctx.get("remaining_cards"))),"exact_composition_available":exact_ok,"rank_ratio_source":ratio_source,"rank_ratios_a_to_10jqk":ratios,"shoe_group_ratios":{"A23":groups[0],"45":groups[1],"6":groups[2],"7":groups[3],"8":groups[4],"9":groups[5],"10JQK":groups[6]},"physical_edge_proxy":edge,"shoe_information_reliability":rel,"combinatorial_advantage_offset":0.,"probabilistic_shoe_reliability":0.,"hsmm_stable_probability":stable,"hazard_rate":hz,"hazard_formula":"panel_proxy","hsmm_formula":"panel_proxy","derived_road_regularity_binary":binary,"derived_latest_marks":marks,"run_length":run,"run_length_norm":_clip(run/8),"shoe_decks":decks,"previous_run_length":pr,"previous2_run_length":p2,"current_side":side,"current_side_banker_binary":side_b,"recent5_banker_ratio":_br(raw,5),"recent8_banker_ratio":_br(raw,8),"recent12_banker_ratio":_br(raw,12),"recent5_turn_rate":_tr(raw,5),"recent8_turn_rate":_tr(raw,8),"recent12_turn_rate":_tr(raw,12),"big_eye_regularity":ber,"small_road_regularity":smr,"cockroach_road_regularity":crr,"small_cockroach_regularity":small_c,"derived_road_consensus":cons,"context_layout":"16_shoe_plus_16_road_32d","context_compatibility":"standalone_32d_panel","shoe_feature_values":shoe,"road_feature_values":road,"formal_direction_source":"contextual_linucb","single_brain":True,"external_direction_votes_enabled":False,"anti_echo_external_penalty":False})

def _state_path()->Path:
    c=[];configured=str(os.getenv("LINUCB_STATE_FILE","") or "").strip()
    if configured:c.append(Path(configured).expanduser())
    c += [Path("/var/data/contextual_linucb_state.json"),Path(__file__).resolve().parent/"data"/"contextual_linucb_state.json",Path("/tmp/contextual_linucb_state.json")]
    for p in c:
        try:p.parent.mkdir(parents=True,exist_ok=True);q=p.parent/f".linucb_write_{time.time_ns()}";q.write_text("ok");q.unlink(missing_ok=True);return p
        except OSError:pass
    return Path("/tmp/contextual_linucb_state.json")
STATE_FILE=_state_path()
def _new_arm():return {"A":(np.eye(32)*LINUCB_RIDGE).tolist(),"b":np.zeros(32).tolist(),"n":0,"effective_n":0.}
def _new_scope():
    now=int(time.time());return {"arms":{a:_new_arm() for a in ARMS},"pending":{},"updates":0,"last_selected":"","selection_streak":0,"panel_bootstrap_done":False,"bootstrap_rounds":0,"bootstrap_source_rounds":0,"bootstrap_history_fingerprint":"","created_at":now,"updated_at":now}
def _read():
    try:
        p=json.loads(STATE_FILE.read_text())
        if not isinstance(p,dict):raise ValueError
    except Exception:p={}
    if p.get("version")!=STATE_VERSION or p.get("dim")!=32:p={}
    return {"version":STATE_VERSION,"dim":32,"alpha":LINUCB_ALPHA,"ridge":LINUCB_RIDGE,"forgetting":LINUCB_FORGETTING,"scopes":p.get("scopes") if isinstance(p.get("scopes"),dict) else {}}
def _write(p:Mapping[str,Any]):
    t=STATE_FILE.with_suffix(STATE_FILE.suffix+".tmp");t.write_text(json.dumps(dict(p),ensure_ascii=False));t.replace(STATE_FILE)
def make_scope_key(*,user_id:str="",venue:str="",room:str="",shoe_id:str="")->str:
    raw="|".join((str(user_id or "").strip(),str(venue or "").upper().strip(),str(room or "").strip(),str(shoe_id or "").strip()));return sha256((raw or "GLOBAL").encode()).hexdigest()[:24]
def _fp(h:Sequence[str])->str:return sha256("".join(h).encode()).hexdigest()[:24]
def _arrays(s:Mapping[str,Any])->tuple[np.ndarray,np.ndarray]:
    try:
        A=np.asarray(s.get("A"),dtype=np.float64).reshape(32,32);b=np.asarray(s.get("b"),dtype=np.float64).reshape(32)
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):raise ValueError
        return A,b
    except Exception:return np.eye(32)*LINUCB_RIDGE,np.zeros(32)

class ContextualLinUCB:
    def __init__(self,alpha:float=LINUCB_ALPHA):self.alpha=max(0.,float(alpha));self.generator=ContextGenerator()
    def _score(self,state:Mapping[str,Any],xv:np.ndarray,scale:float)->dict[str,float]:
        xv=_x(xv);A,b=_arrays(state)
        try:theta=np.linalg.solve(A,b);sx=np.linalg.solve(A,xv)
        except np.linalg.LinAlgError:A=A+np.eye(32)*LINUCB_RIDGE;theta=np.linalg.solve(A,b);sx=np.linalg.solve(A,xv)
        mean=float(xv@theta);unc=float(math.sqrt(max(0.,xv@sx)));ea=self.alpha*max(.5,min(2.5,float(scale)));return {"score":mean+ea*unc,"mean":mean,"uncertainty":unc,"effective_alpha":ea,"raw_n":float(state.get("n",0) or 0),"effective_n":float(state.get("effective_n",state.get("n",0)) or 0)}
    def _decay(self,scope:dict[str,Any]):
        I=np.eye(32)*LINUCB_RIDGE;arms=scope.setdefault("arms",{})
        for a in ARMS:
            s=dict(arms.get(a) or _new_arm());A,b=_arrays(s);s["A"]=(I+LINUCB_FORGETTING*(A-I)).tolist();s["b"]=(LINUCB_FORGETTING*b).tolist();s["effective_n"]=LINUCB_FORGETTING*float(s.get("effective_n",s.get("n",0)) or 0);arms[a]=s
    def _update_scope(self,scope:dict[str,Any],*,action:str,context_vector:Sequence[float],actual_outcome:str)->dict[str,Any]:
        action=str(action or "").upper().strip();actual=str(actual_outcome or "").upper().strip()
        if action not in ARMS or actual not in {"B","P","T"}:return {"updated":False,"reason":"invalid_feedback"}
        xv=_x(context_vector);self._decay(scope);scope["updates"]=int(scope.get("updates",0) or 0)+1;scope["updated_at"]=int(time.time())
        if actual=="T":return {"updated":True,"action":action,"actual_outcome":"T","reward":0.,"directional_sample_applied":False,"forgetting":LINUCB_FORGETTING,"reason":"tie_reward_zero_no_directional_information"}
        reward=(.95 if action=="B" else 1.) if action==actual else -1.;s=dict(scope.get("arms",{}).get(action) or _new_arm());A,b=_arrays(s);A=A+LINUCB_UPDATE_WEIGHT*np.outer(xv,xv);b=b+LINUCB_UPDATE_WEIGHT*reward*xv;s.update({"A":A.tolist(),"b":b.tolist(),"n":int(s.get("n",0) or 0)+1,"effective_n":float(s.get("effective_n",0) or 0)+1});scope.setdefault("arms",{})[action]=s;return {"updated":True,"action":action,"actual_outcome":actual,"reward":reward,"directional_sample_applied":True,"update_weight":LINUCB_UPDATE_WEIGHT,"forgetting":LINUCB_FORGETTING,"context_l2_normalized":False,"single_brain_update":True}
    def update(self,*,scope_key:str,action:str,context_vector:Sequence[float],actual_outcome:str,clear_pending:bool=True)->dict[str,Any]:
        with _LOCK:
            root=_read();scope=deepcopy(dict(root["scopes"].get(scope_key) or _new_scope()));r=self._update_scope(scope,action=action,context_vector=context_vector,actual_outcome=actual_outcome);r.update({"diagnostic_only":False,"formal_model":"contextual_linucb"})
            if clear_pending:scope["pending"]={}
            root["scopes"][scope_key]=scope;_write(root);return r
    def _pending(self,scope:dict[str,Any],raw:Sequence[str])->dict[str,Any]:
        p=deepcopy(dict(scope.get("pending") or {}))
        if not p:return {"updated":False,"reason":"no_pending_prediction"}
        n=int(p.get("raw_round_count",0) or 0)
        if len(raw)<=n:return {"updated":False,"reason":"no_new_resolved_round"}
        if _fp(raw[:n])!=str(p.get("history_fingerprint") or ""):scope["pending"]={};return {"updated":False,"reason":"history_reset_or_misaligned","previous_len":n,"current_len":len(raw)}
        r=self._update_scope(scope,action=str(p.get("action") or ""),context_vector=p.get("context_vector") or [],actual_outcome=raw[n]);scope["pending"]={};r.update({"history_aligned":True,"resolved_history_index":n,"history_rounds_after_append":len(raw)});return r
    def _tie(self,scope:Mapping[str,Any],raw:Sequence[str])->tuple[str,str]:
        arms=dict(scope.get("arms") or {});bn=float((arms.get("B") or {}).get("effective_n",0) or 0);pn=float((arms.get("P") or {}).get("effective_n",0) or 0)
        if abs(bn-pn)>1e-9:return ("B" if bn<pn else "P"),"tie_less_sampled_arm"
        last=str(scope.get("last_selected") or "").upper().strip()
        if last in ARMS:return ("P" if last=="B" else "B"),"tie_opposite_previous_arm"
        token=sha256(("LOCAL_32D|"+"".join(raw)).encode()).digest()[0];return ("B" if token%2 else "P"),"tie_deterministic_history_hash"
    def _choose(self,scope:Mapping[str,Any],xv:np.ndarray,raw:Sequence[str]):
        n=len(_bp(raw));base=1.35 if n<8 else 1.15 if n<15 else 1.;arms=dict(scope.get("arms") or {});eff={a:max(0.,float((arms.get(a) or {}).get("effective_n",(arms.get(a) or {}).get("n",0)) or 0)) for a in ARMS};tot=sum(eff.values());scores={}
        for a in ARMS:
            imb=math.sqrt(max(1.,tot+2)/max(1.,eff[a]+1));scale=base*_clip(imb,.85,LINUCB_ARM_ALPHA_MAX_SCALE);item=self._score(arms.get(a,{}),xv,scale);item.update({"linucb_score":item["score"],"alpha_scale":scale,"external_score_component":0.});scores[a]=item
        gap=float(scores["B"]["score"]-scores["P"]["score"])
        if abs(gap)<=LINUCB_SCORE_TIE_EPSILON:direction,reason=self._tie(scope,raw)
        else:direction=("B" if gap>0 else "P");reason="linucb_ucb_score_argmax"
        return scores,eff,tot,direction,reason,gap
    def _remember(self,scope:dict[str,Any],direction:str,repeat:bool=False)->int:
        prev=str(scope.get("last_selected") or "").upper().strip();st=int(scope.get("selection_streak",0) or 0);st=st if repeat and prev==direction else st+1 if prev==direction else 1;scope.update({"last_selected":direction,"selection_streak":st,"updated_at":int(time.time())});return st
    def _bootstrap(self,scope:dict[str,Any],raw:Sequence[str],ctx:Mapping[str,Any])->dict[str,Any]:
        if scope.get("panel_bootstrap_done"):return {"applied":False,"reason":"bootstrap_already_done","bootstrap_rounds":int(scope.get("bootstrap_rounds",0) or 0)}
        updates=directional=ties=0
        for i in range(1,len(raw)):
            prefix=list(raw[:i]);snap=self.generator.build(prefix,ctx);_,_,_,direction,_,_=self._choose(scope,_x(snap.vector),prefix);self._remember(scope,direction);r=self._update_scope(scope,action=direction,context_vector=snap.vector,actual_outcome=raw[i]);updates+=int(bool(r.get("updated")));directional+=int(bool(r.get("directional_sample_applied")));ties+=int(raw[i]=="T")
        scope.update({"pending":{},"panel_bootstrap_done":True,"bootstrap_rounds":max(0,len(raw)-1),"bootstrap_source_rounds":len(raw),"bootstrap_history_fingerprint":_fp(raw),"updated_at":int(time.time())});return {"applied":True,"reason":"panel_walk_forward_bootstrap","bootstrap_rounds":max(0,len(raw)-1),"updates":updates,"directional_updates":directional,"tie_updates":ties,"source_rounds":len(raw)}
    def predict(self,*,history:Iterable[Any]|str|None,shoe_context:Mapping[str,Any]|None,scope_key:str)->dict[str,Any]:
        raw=_norm(deepcopy(history));ctx=deepcopy(dict(shoe_context or {}));snap=self.generator.build(raw,ctx);rawx=snap.vector.copy();xv=_x(rawx);fp=_fp(raw)
        with _LOCK:
            root=_read();scope=deepcopy(dict(root["scopes"].get(scope_key) or _new_scope()));boot=self._bootstrap(scope,raw,ctx)
            if boot.get("applied"):feedback={"updated":bool(boot.get("updates")),"reason":"panel_walk_forward_bootstrap","bootstrap":deepcopy(boot),"diagnostic_only":False,"formal_model":"contextual_linucb"}
            else:feedback=self._pending(scope,raw);feedback.update({"diagnostic_only":False,"formal_model":"contextual_linucb"})
            pending=deepcopy(dict(scope.get("pending") or {}));repeat=pending.get("history_fingerprint")==fp and pending.get("raw_round_count")==len(raw);scores,eff,tot,direction,reason,gap=self._choose(scope,xv,raw);rawpb=1/(1+math.exp(-max(-8.,min(8.,gap/LINUCB_SCORE_TEMPERATURE))));pb=_clip(rawpb,PROBABILITY_MIN,PROBABILITY_MAX);pp=1-pb;probs={"B":pb,"P":pp,"T":0.};conf=pb if direction=="B" else pp;st=self._remember(scope,direction,repeat)
            snap.metadata.update({"selection_streak":st,"linucb_direction_weight":1.,"road_prior_direction_weight":0.,"road_forecaster_direction_weight":0.,"derived_road_direction_weight":0.,"geometry_direction_weight":0.,"anti_echo_direction_weight":0.,"panel_bootstrap":deepcopy(boot)});scope["pending"]={"action":direction,"context_vector":[float(v) for v in rawx],"raw_round_count":len(raw),"history_fingerprint":fp,"created_at":int(time.time())};root["scopes"][scope_key]=scope;_write(root)
        return {"model":"contextual_linucb_single_brain","version":STATE_VERSION,"legacy_state_version":STATE_VERSION,"direction":direction,"selected_arm":direction,"arm_index":1 if direction=="B" else 0,"probabilities":probs,"selected_win_probability":conf,"confidence":conf,"context_vector":[float(v) for v in rawx],"model_context_vector":[float(v) for v in xv],"context_feature_names":list(CONTEXT_FEATURE_NAMES),"context_dim":32,"context_metadata":deepcopy(snap.metadata),"road_prior":{"diagnostic_only":True,"direction_weight":0.,"banker_probability":.5,"player_probability":.5},"road_prior_probability":{"B":.5,"P":.5},"road_forecaster":{"available":False,"diagnostic_only":True,"formal_direction_weight":0.},"features_used":dict(zip(CONTEXT_FEATURE_NAMES,[float(v) for v in rawx])),"effective_support":tot,"uncertainty":scores[direction]["uncertainty"],"linucb_probability_correction":0.,"linucb_direction_weight":1.,"learning_reliability":_clip(tot/10.),"scores":scores,"score_gap":gap,"score_semantics":"contextual_linucb_ucb_scores_only","alpha":self.alpha,"ridge":LINUCB_RIDGE,"forgetting":LINUCB_FORGETTING,"feedback_update":feedback,"bootstrap_update":deepcopy(boot),"panel_bootstrap_applied":bool(boot.get("applied")),"scope_key":scope_key,"arms":list(ARMS),"selection_reason":reason,"selection_streak":st,"effective_arm_samples":eff,"history_round_count":len(raw),"bp_history_round_count":len(_bp(raw)),"history_fingerprint":fp,"short_shoe_target_rounds":"50-70","formal_context_source":"single_brain_32d_panel_compatible_context","formal_direction_source":"contextual_linucb","road_context_direction_weight":0.,"card_composition_direction_weight":0.,"probability_semantics":"bounded_logistic_mapping_of_linucb_ucb_score_gap","cold_start_uses_road_prior":False,"shoe_context_used_for_formal_direction":True,"shoe_context_used_as_features":True,"shoe_context_independent_vote":False,"external_road_vote_enabled":False,"anti_echo_external_penalty":False,"panel_compatible":True,"anti_lock":{"enabled":False,"method":"none_external_feedback_only","tie_is_non_directional":True,"old_v1_v2_v3_v4_v5_v6_v7_state_reused":False}}

_DEFAULT_BANDIT=ContextualLinUCB()
def predict_bandit(*,history:Iterable[Any]|str|None,shoe_context:Mapping[str,Any]|None,scope_key:str)->dict[str,Any]:return _DEFAULT_BANDIT.predict(history=deepcopy(history),shoe_context=deepcopy(dict(shoe_context or {})),scope_key=str(scope_key or ""))
def update_bandit(*,scope_key:str,action:str,context_vector:Sequence[float],actual_outcome:str,clear_pending:bool=True)->dict[str,Any]:return _DEFAULT_BANDIT.update(scope_key=str(scope_key or ""),action=action,context_vector=deepcopy(list(context_vector)),actual_outcome=actual_outcome,clear_pending=clear_pending)
__all__=["ARMS","CONTEXT_DIM","CONTEXT_FEATURE_NAMES","ContextGenerator","ContextualLinUCB","ESTIMATED_CARDS_PER_ROUND","SHOE_DECKS","LINUCB_ALPHA","LINUCB_ARM_ALPHA_MAX_SCALE","LINUCB_FORGETTING","LINUCB_RIDGE","LINUCB_SCORE_TIE_EPSILON","LINUCB_UPDATE_WEIGHT","PROBABILITY_MIN","PROBABILITY_MAX","ROAD_PRIOR_PROBABILITY_SPAN","ROAD_PRIOR_SCORE_WEIGHT","STATE_VERSION","make_scope_key","predict_bandit","update_bandit"]
