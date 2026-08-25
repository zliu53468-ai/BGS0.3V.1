"""BGS CUSUM-LinUCB with 29D next-hand predictive Markov context."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any,Dict,Iterable,List,Mapping,Optional,Sequence
import json,math,time
import numpy as np
from road_model import build_road_context

ARMS=("B","P")
MODEL_VERSION="CUSUM-LINUCB-V1.3-MARKOV29-NEXT-HAND-DYNAMIC-RESET-NO-OBSERVE"
STATE_SCHEMA_VERSION="CUSUM-LINUCB-STATE-V4-MARKOV29-NEXT-HAND"
ROAD_FEATURE_NAMES=("bias","history_maturity","global_banker_balance","recent3_banker_balance","recent8_banker_balance","current_streak_direction","current_streak_length","alternation6","alternation12","transition_acceleration","streak_break_signal","long_dragon_tail_pressure","observed_tie_rate","road_planning_balance","road_recent_balance","road_confidence","road_agreement","big_eye_saturation","small_road_saturation","cockroach_road_saturation","derived_road_consensus")
MARKOV_PROBABILITY_NAMES=("markov_p_b_given_b","markov_p_p_given_b","markov_p_b_given_p","markov_p_p_given_p","markov_p_b_given_bb","markov_p_b_given_pp","markov_p_p_given_bb","markov_p_p_given_pp")
MARKOV_FEATURE_NAMES=("markov_next_edge_order1","markov_next_edge_order2","markov_next_edge_order3","markov_next_edge_blended","markov_next_support_order1","markov_next_support_order2","markov_next_support_order3","markov_next_flip_pressure")
FEATURE_NAMES=ROAD_FEATURE_NAMES+MARKOV_FEATURE_NAMES; CONTEXT_DIM=len(FEATURE_NAMES)
CUSUM_ALPHA=.65; CUSUM_L2=1.; CUSUM_FORGETTING_FACTOR=.985; CUSUM_DRIFT_V=.15; CUSUM_THRESHOLD_H=4.5
CUSUM_MIN_OBSERVATIONS=8; CUSUM_VACUUM_HANDS=5; CUSUM_FORCE_OBSERVE_HANDS=0; PREQUENTIAL_WARMUP_BP=6; HISTORY_REPLAY_LIMIT=120
MARKOV_ALPHA=1.; MARKOV_WINDOW_SIZE=36; MARKOV_SUPPORT_PRIOR=6.; MARKOV_MAX_ORDER=3
TIE_PRIOR=.095156; TIE_PRIOR_STRENGTH=40.; _LOCK=RLock(); BASE_DIR=Path(__file__).resolve().parent

def _state_file():
    for p in (Path("/var/data/contextual_bandit_state_cusum_v1.json"),BASE_DIR/"data"/"contextual_bandit_state_cusum_v1.json",Path("/tmp/bgs_contextual_bandit_state_cusum_v1.json")):
        try:
            p.parent.mkdir(parents=True,exist_ok=True); q=p.parent/f".cusum_probe_{time.time_ns()}"; q.write_text("ok",encoding="utf-8"); q.unlink(missing_ok=True); return p
        except OSError: pass
    raise RuntimeError("No writable CUSUM state path")
CMAB_STATE_FILE=_state_file()

def _clip(v,lo=-1.,hi=1.):
    try:x=float(v)
    except (TypeError,ValueError):return 0.
    return max(lo,min(hi,x)) if math.isfinite(x) else 0.

def _clean(values):
    out=[]
    for item in values:
        raw=item.get("outcome") if isinstance(item,Mapping) else item; v=str(raw or "").upper().strip()
        if v in {"B","P","T"}:out.append(v)
    return out[-2000:]

class MarkovFeatureExtractor:
    """Variable-order B/P Markov predictor for the *next* hand.

    Uses active suffix contexts at orders 1, 2 and 3. This explicitly models BP/PB
    alternation states as well as BBB/PPP streak states. All conditional estimates use
    Laplace smoothing and support-aware backoff. reset() stays synchronized with CUSUM.
    """
    FEATURE_NAMES=MARKOV_FEATURE_NAMES; PROBABILITY_NAMES=MARKOV_PROBABILITY_NAMES
    def __init__(self,alpha=MARKOV_ALPHA,window_size=MARKOV_WINDOW_SIZE,support_prior=MARKOV_SUPPORT_PRIOR,max_order=MARKOV_MAX_ORDER):
        self.alpha=max(1e-9,float(alpha)); self.window_size=max(4,int(window_size)); self.support_prior=max(1e-9,float(support_prior)); self.max_order=max(1,min(3,int(max_order))); self._values=[]
    @staticmethod
    def _encode(v):
        if isinstance(v,(int,np.integer)) and int(v) in (0,1):return int(v)
        s=str(v or "").upper().strip(); return 1 if s=="B" else 0 if s=="P" else None
    @staticmethod
    def _ctx_text(ctx):return "".join("B" if x else "P" for x in ctx)
    def reset(self):self._values=[]
    def update(self,v):
        x=self._encode(v)
        if x is None:return
        self._values.append(x)
        if len(self._values)>self.window_size:self._values=self._values[-self.window_size:]
    def extend(self,values):
        for v in values:self.update(v)
    def _smooth(self,b,p):
        d=float(b+p)+2*self.alpha; return (b+self.alpha)/d,(p+self.alpha)/d
    def _reliability(self,n,order=1):
        n=max(0.,float(n)); prior=self.support_prior+1.5*max(0,int(order)-1); return n/(n+prior)
    def _active(self,order):
        vals=list(self._values); order=max(1,min(self.max_order,int(order)))
        if len(vals)<order:return {"order":order,"context":"","banker_count":0,"player_count":0,"support":0,"p_b":.5,"p_p":.5,"reliability":0.}
        ctx=tuple(vals[-order:]); b=p=0
        for i in range(order,len(vals)):
            if tuple(vals[i-order:i])!=ctx:continue
            if vals[i]==1:b+=1
            else:p+=1
        pb,pp=self._smooth(b,p); n=b+p
        return {"order":order,"context":self._ctx_text(ctx),"banker_count":b,"player_count":p,"support":n,"p_b":pb,"p_p":pp,"reliability":self._reliability(n,order)}
    def next_prediction(self):
        st={o:self._active(o) for o in (1,2,3)}; blended=.5
        for o in (1,2,3):
            r=float(st[o]["reliability"]); blended+=r*(float(st[o]["p_b"])-blended)
        blended=max(1e-6,min(1-1e-6,blended)); miss=1.
        for o in (1,2,3):miss*=1-float(st[o]["reliability"])
        conf=1-miss; last=self._values[-1] if self._values else None
        flip=.5 if last is None else (1-blended if last==1 else blended)
        return {"banker_probability":blended,"player_probability":1-blended,"selected_arm":"B" if blended>=.5 else "P","confidence":conf,"flip_probability":flip,"continue_probability":1-flip,"orders":{str(k):dict(v) for k,v in st.items()}}
    def _legacy(self):
        v=list(self._values); bb=bp=pb=pp=0
        for a,c in zip(v,v[1:]):
            if a==1 and c==1:bb+=1
            elif a==1:bp+=1
            elif c==1:pb+=1
            else:pp+=1
        bab=pab=bpp=ppp=0
        for a,b,c in zip(v,v[1:],v[2:]):
            if a==b==1:
                if c==1:bab+=1
                else:pab+=1
            elif a==b==0:
                if c==1:bpp+=1
                else:ppp+=1
        pbb,ppb=self._smooth(bb,bp); pbp,ppp1=self._smooth(pb,pp); pbbb,ppbb=self._smooth(bab,pab); pbpp,pppp=self._smooth(bpp,ppp)
        return [pbb,ppb,pbp,ppp1,pbbb,pbpp,ppbb,pppp]
    def extract_probabilities(self):return self._legacy()
    def extract_features(self):
        p=self.next_prediction(); o1,o2,o3=p["orders"]["1"],p["orders"]["2"],p["orders"]["3"]
        e1=(2*o1["p_b"]-1)*o1["reliability"]; e2=(2*o2["p_b"]-1)*o2["reliability"]; e3=(2*o3["p_b"]-1)*o3["reliability"]
        eb=(2*p["banker_probability"]-1)*p["confidence"]; flip=(2*p["flip_probability"]-1)*p["confidence"]
        return [e1,e2,e3,eb,o1["reliability"],o2["reliability"],o3["reliability"],flip]
    def feature_dict(self):return dict(zip(self.FEATURE_NAMES,self.extract_features()))
    def probability_dict(self):return dict(zip(self.PROBABILITY_NAMES,self.extract_probabilities()))
    def to_state(self):return {"alpha":self.alpha,"window_size":self.window_size,"support_prior":self.support_prior,"max_order":self.max_order,"values":list(self._values),"sample_count":len(self._values)}
    @classmethod
    def from_state(cls,state):
        x=cls(state.get("alpha",MARKOV_ALPHA),state.get("window_size",MARKOV_WINDOW_SIZE),state.get("support_prior",MARKOV_SUPPORT_PRIOR),state.get("max_order",MARKOV_MAX_ORDER))
        try:x.extend(list(state.get("values") or []))
        except Exception:x.reset()
        return x
    @classmethod
    def from_sequence(cls,values,alpha=MARKOV_ALPHA,window_size=MARKOV_WINDOW_SIZE,support_prior=MARKOV_SUPPORT_PRIOR,max_order=MARKOV_MAX_ORDER):
        x=cls(alpha,window_size,support_prior,max_order); x.extend(values); return x

def _balance(seq,n=None):
    s=list(seq[-n:] if n else seq); return 0. if not s else _clip((sum(x=="B" for x in s)/len(s)-.5)*2)
def _transition_rate(seq,n):
    s=list(seq[-n:]); return 0. if len(s)<2 else sum(a!=b for a,b in zip(s,s[1:]))/(len(s)-1)
def _alternation(seq,n):return _clip((_transition_rate(seq,n)-.5)*2)
def _streak(seq):
    if not seq:return "",0
    side=seq[-1]; n=1
    for x in reversed(seq[:-1]):
        if x!=side:break
        n+=1
    return side,n
def _streak_break(seq):
    s=list(seq)
    if len(s)<4 or s[-1]==s[-2]:return 0.
    old=s[-2]; n=1
    for x in reversed(s[:-2]):
        if x!=old:break
        n+=1
    return 0. if n<3 else (1. if s[-1]=="B" else -1.)*min(1.,n/6.)
def _road_saturation(road,name):
    p=road.get("full_road_analysis")
    if not isinstance(p,Mapping):
        m=road.get("models"); p=m.get("full_road") if isinstance(m,Mapping) else {}
    st=dict(p.get("derived_stats") or {}).get(name) if isinstance(p,Mapping) else None
    if not isinstance(st,Mapping):return 0.
    b=_clip(st.get("balance",0),0,1); c=_clip(st.get("recent_continuation",.5),0,1); return max(b,abs(2*c-1))
def _prob_balance(v):
    try:p=float(v)
    except (TypeError,ValueError):p=.5
    return _clip((p-.5)*2)

def build_context_vector(history,road_context=None,markov_features=None):
    raw=_clean(history); bp=[x for x in raw if x in ARMS]; road=dict(road_context or {}); side,run=_streak(bp); sign=1. if side=="B" else -1. if side=="P" else 0.
    tie=sum(x=="T" for x in raw)/max(1,len(raw)); dis=_clip(road.get("recent_model_disagreement",road.get("model_disagreement",.2)),0,1)
    big=_road_saturation(road,"big_eye"); small=_road_saturation(road,"small_road"); cock=_road_saturation(road,"cockroach_road"); mean=(big+small+cock)/3
    consensus=_clip(mean*(1-(abs(big-small)+abs(small-cock)+abs(cock-big))/3),0,1)
    rv=[1.,min(1.,len(bp)/60),_balance(bp),_balance(bp,3),_balance(bp,8),sign,min(1.,run/8),_alternation(bp,6),_alternation(bp,12),_clip(_transition_rate(bp,6)-_transition_rate(bp,14)),_streak_break(bp),sign*min(1.,max(0,run-3)/5),_clip(tie/.2,0,1),_prob_balance(road.get("planning_probability",.5)),_prob_balance(road.get("recent_probability",.5)),_clip(road.get("confidence_score",0),0,1),_clip(1-min(1,dis/.2),0,1),big,small,cock,consensus]
    if len(rv)!=len(ROAD_FEATURE_NAMES):raise RuntimeError("Road context dimension mismatch")
    mv=MarkovFeatureExtractor.from_sequence(bp).extract_features() if markov_features is None else [float(x) for x in markov_features]
    if len(mv)!=8 or not all(math.isfinite(x) and -1<=x<=1 for x in mv):raise ValueError("markov_features must contain 8 finite values in [-1,1]")
    out=rv+mv
    if len(out)!=CONTEXT_DIM:raise RuntimeError("CUSUM context dimension mismatch")
    return [round(_clip(x),10) for x in out]

def _vec(c):
    x=np.asarray(list(c),dtype=np.float64)
    if x.shape!=(CONTEXT_DIM,) or not np.all(np.isfinite(x)):raise ValueError(f"context must be finite {CONTEXT_DIM}-vector")
    return np.clip(x,-1,1)
def _softmax(b,p):
    z=np.asarray([b,p],dtype=float)/.85; z-=np.max(z); e=np.exp(np.clip(z,-40,40)); e/=max(1e-12,float(e.sum())); return {"B":float(e[0]),"P":float(e[1])}
def _tie_prob(raw):
    p=(sum(x=="T" for x in raw)+TIE_PRIOR*TIE_PRIOR_STRENGTH)/(len(raw)+TIE_PRIOR_STRENGTH); return max(.04,min(.18,float(p)))

class CUSUMLinUCB:
    def __init__(self,alpha=CUSUM_ALPHA,l2=CUSUM_L2,forgetting_factor=CUSUM_FORGETTING_FACTOR,cusum_h=CUSUM_THRESHOLD_H,cusum_v=CUSUM_DRIFT_V,min_cusum_observations=CUSUM_MIN_OBSERVATIONS,vacuum_hands=CUSUM_VACUUM_HANDS):
        self.alpha=float(alpha); self.l2=max(1e-9,float(l2)); self.forgetting_factor=max(.8,min(1.,float(forgetting_factor))); self.cusum_h=max(.5,float(cusum_h)); self.cusum_v=max(0.,float(cusum_v)); self.min_cusum_observations=max(2,int(min_cusum_observations)); self.vacuum_hands=max(1,int(vacuum_hands))
        self.total_observations=self.observations_since_reset=self.reset_count=0; self.g_plus=self.g_minus=self.last_residual=self.last_expected_reward=self.last_observed_reward=0.; self.last_reset={}; self.applied_event_ids=[]; self.markov_extractor=MarkovFeatureExtractor(); self._fresh_matrices()
    def _fresh_matrices(self):
        I=np.eye(CONTEXT_DIM)*self.l2; z=np.zeros(CONTEXT_DIM); self.arms={a:{"A":I.copy(),"b":z.copy(),"updates":0,"reward_sum":0.} for a in ARMS}; self.context_information={"A":I.copy(),"updates":0}
    @staticmethod
    def _pinv(A):
        A=.5*(A+A.T)
        try:i=np.linalg.pinv(A,rcond=1e-10,hermitian=True)
        except TypeError:i=np.linalg.pinv(A,rcond=1e-10)
        return i if np.all(np.isfinite(i)) else np.linalg.pinv(A+np.eye(A.shape[0])*1e-6,rcond=1e-8)
    def arm_metrics(self,arm,context):
        x=_vec(context); s=self.arms[arm]; inv=self._pinv(np.asarray(s["A"],float)); th=inv@np.asarray(s["b"],float); mean=float(th@x); var=max(0.,float(x@inv@x)); std=math.sqrt(var); bonus=self.alpha*std
        return {"expected_reward":mean,"mean_reward":mean,"variance":var,"uncertainty":std,"ucb_bonus":bonus,"ucb_score":mean+bonus,"updates":int(s["updates"]),"reward_sum":float(s["reward_sum"])}
    def predict_context(self,context):
        x=_vec(context); m={a:self.arm_metrics(a,x) for a in ARMS}; b,p=m["B"]["ucb_score"],m["P"]["ucb_score"]; sel="B" if b>=p else "P"; inv=self._pinv(np.asarray(self.context_information["A"],float)); var=max(0.,float(x@inv@x))
        return {"selected_arm":sel,"metrics":m,"conditional_probabilities":_softmax(b,p),"shared_uncertainty":{"variance":var,"uncertainty":math.sqrt(var),"updates":int(self.context_information["updates"])}}
    def _cusum(self,obs,exp):
        r=float(obs-exp); self.last_residual=r; self.last_observed_reward=float(obs); self.last_expected_reward=float(exp); self.g_plus=max(0.,self.g_plus+r-self.cusum_v); self.g_minus=max(0.,self.g_minus-r-self.cusum_v); ready=self.observations_since_reset>=self.min_cusum_observations and self.total_observations>=self.min_cusum_observations; plus=ready and self.g_plus>self.cusum_h; minus=ready and self.g_minus>self.cusum_h
        return {"residual":r,"g_plus":self.g_plus,"g_minus":self.g_minus,"ready":ready,"alarm":bool(plus or minus),"alarm_side":"positive" if plus else "negative" if minus else "","threshold_h":self.cusum_h,"drift_v":self.cusum_v}
    def reset_model(self,reason="cusum_change_point",alarm_side="",residual=None):
        self.reset_count+=1; e={"triggered":True,"reason":reason,"alarm_side":alarm_side,"residual":float(self.last_residual if residual is None else residual),"g_plus_before_reset":self.g_plus,"g_minus_before_reset":self.g_minus,"threshold_h":self.cusum_h,"drift_v":self.cusum_v,"at_total_observation":self.total_observations,"reset_count":self.reset_count,"timestamp":int(time.time())}
        self._fresh_matrices(); self.markov_extractor.reset(); self.observations_since_reset=0; self.g_plus=self.g_minus=0.; self.last_reset=e; return dict(e)
    def _update(self,x,rewards):
        I=np.eye(CONTEXT_DIM)*self.l2; outer=np.outer(x,x); lam=self.forgetting_factor
        for arm,r in rewards.items():
            s=self.arms[arm]; A=np.asarray(s["A"],float); b=np.asarray(s["b"],float); A=lam*(.5*(A+A.T))+(1-lam)*I+outer; b=lam*b+float(r)*x; s.update(A=.5*(A+A.T),b=b,updates=int(s["updates"])+1,reward_sum=float(s["reward_sum"])+float(r))
        A=np.asarray(self.context_information["A"],float); A=lam*(.5*(A+A.T))+(1-lam)*I+outer; self.context_information={"A":.5*(A+A.T),"updates":int(self.context_information["updates"])+1}
    def observe(self,context,actual_outcome,selected_arm=""):
        actual=str(actual_outcome or "").upper().strip()
        if actual not in ARMS:return {"updated":False,"reason":"tie_or_invalid_outcome"}
        x=_vec(context); pred=self.predict_context(x); chosen=str(selected_arm or pred["selected_arm"]).upper(); chosen=chosen if chosen in ARMS else pred["selected_arm"]; exp=_clip(pred["metrics"][chosen]["expected_reward"]); obs=1. if chosen==actual else -1.; c=self._cusum(obs,exp); reset={}
        if c["alarm"]:reset=self.reset_model("cusum_residual_change_point",c["alarm_side"],c["residual"])
        self._update(x,{a:(1. if a==actual else -1.) for a in ARMS}); self.markov_extractor.update(actual); self.total_observations+=1; self.observations_since_reset+=1
        return {"updated":True,"actual_outcome":actual,"selected_arm":chosen,"observed_reward":obs,"expected_reward":exp,"cusum":c,"reset_triggered":bool(reset),"reset_event":reset}
    def observe_reward(self,context,selected_arm,reward):
        arm=str(selected_arm).upper(); x=_vec(context); exp=_clip(self.arm_metrics(arm,x)["expected_reward"]); r=_clip(reward); c=self._cusum(r,exp); reset={}
        if c["alarm"]:reset=self.reset_model("cusum_selected_arm_reward_change_point",c["alarm_side"],c["residual"])
        self._update(x,{arm:r}); self.total_observations+=1; self.observations_since_reset+=1; return {"updated":True,"selected_arm":arm,"reward":r,"expected_reward":exp,"cusum":c,"reset_triggered":bool(reset),"reset_event":reset}
    def risk_status(self,context):
        sh=self.predict_context(context)["shared_uncertainty"]; u=float(sh["uncertainty"]); info=1/(1+u); ref=self.observations_since_reset if self.reset_count else self.total_observations; maturity=min(1.,ref/12); conf=_clip(.1+.55*info+.35*maturity,0,.9); vacuum=bool(self.reset_count and self.observations_since_reset<=self.vacuum_hands)
        if vacuum:
            conf=min(conf,min(.48,.08+.08*self.observations_since_reset)); weight=min(.18,.04+.03*self.observations_since_reset); bet=.35 if self.observations_since_reset<=2 else .45 if self.observations_since_reset==3 else .5 if self.observations_since_reset==4 else .6
        elif self.total_observations<PREQUENTIAL_WARMUP_BP:weight,bet=.08,.5
        else:weight,bet=min(.45,.15+.35*conf),min(1.,.55+.5*conf)
        return {"confidence_score":conf,"post_reset_vacuum_active":vacuum,"vacuum_hands_required":self.vacuum_hands,"observations_since_reset":self.observations_since_reset,"force_observe":False,"bet_multiplier":bet,"ensemble_weight_suggestion":weight,"uncertainty":u,"variance":float(sh["variance"]),"maturity":maturity}
    def to_state(self):
        return {"version":MODEL_VERSION,"context_dim":CONTEXT_DIM,"alpha":self.alpha,"l2":self.l2,"forgetting_factor":self.forgetting_factor,"cusum_h":self.cusum_h,"cusum_v":self.cusum_v,"min_cusum_observations":self.min_cusum_observations,"vacuum_hands":self.vacuum_hands,"total_observations":self.total_observations,"observations_since_reset":self.observations_since_reset,"reset_count":self.reset_count,"g_plus":self.g_plus,"g_minus":self.g_minus,"last_residual":self.last_residual,"last_expected_reward":self.last_expected_reward,"last_observed_reward":self.last_observed_reward,"last_reset":self.last_reset,"applied_event_ids":self.applied_event_ids[-5000:],"markov_extractor":self.markov_extractor.to_state(),"arms":{a:{"A":np.asarray(self.arms[a]["A"]).tolist(),"b":np.asarray(self.arms[a]["b"]).tolist(),"updates":int(self.arms[a]["updates"]),"reward_sum":float(self.arms[a]["reward_sum"])} for a in ARMS},"context_information":{"A":np.asarray(self.context_information["A"]).tolist(),"updates":int(self.context_information["updates"])}}
    @classmethod
    def from_state(cls,state):
        if not isinstance(state,Mapping) or state.get("version")!=MODEL_VERSION or int(state.get("context_dim",0) or 0)!=CONTEXT_DIM:return cls()
        m=cls(state.get("alpha",CUSUM_ALPHA),state.get("l2",CUSUM_L2),state.get("forgetting_factor",CUSUM_FORGETTING_FACTOR),state.get("cusum_h",CUSUM_THRESHOLD_H),state.get("cusum_v",CUSUM_DRIFT_V),state.get("min_cusum_observations",CUSUM_MIN_OBSERVATIONS),state.get("vacuum_hands",CUSUM_VACUUM_HANDS))
        try:
            for a in ARMS:
                A=np.asarray(state["arms"][a]["A"],float); b=np.asarray(state["arms"][a]["b"],float)
                if A.shape!=(CONTEXT_DIM,CONTEXT_DIM) or b.shape!=(CONTEXT_DIM,):raise ValueError
                m.arms[a]={"A":.5*(A+A.T),"b":b,"updates":int(state["arms"][a].get("updates",0)),"reward_sum":float(state["arms"][a].get("reward_sum",0))}
            A=np.asarray(state["context_information"]["A"],float)
            if A.shape!=(CONTEXT_DIM,CONTEXT_DIM):raise ValueError
            m.context_information={"A":.5*(A+A.T),"updates":int(state["context_information"].get("updates",0))}; m.total_observations=int(state.get("total_observations",0)); m.observations_since_reset=int(state.get("observations_since_reset",0)); m.reset_count=int(state.get("reset_count",0)); m.g_plus=float(state.get("g_plus",0)); m.g_minus=float(state.get("g_minus",0)); m.last_residual=float(state.get("last_residual",0)); m.last_expected_reward=float(state.get("last_expected_reward",0)); m.last_observed_reward=float(state.get("last_observed_reward",0)); m.last_reset=dict(state.get("last_reset") or {}); m.applied_event_ids=[str(x) for x in list(state.get("applied_event_ids") or [])][-5000:]; m.markov_extractor=MarkovFeatureExtractor.from_state(dict(state.get("markov_extractor") or {}))
        except Exception:return cls()
        return m

def _safe_road(history):
    try:return dict(build_road_context(history,initial_image_count=len(history),manual_count=0) or {})
    except Exception:return {}
def _replay(raw):
    hist=list(raw)[-HISTORY_REPLAY_LIMIT:]; m=CUSUMLinUCB(); prefix=[]; bp_before=0; resets=[]; replayed=0
    for actual in hist:
        if actual in ARMS:
            if bp_before>=PREQUENTIAL_WARMUP_BP:
                ctx=build_context_vector(prefix,road_context=_safe_road(prefix),markov_features=m.markov_extractor.extract_features()); up=m.observe(ctx,actual)
                if up.get("reset_triggered"):resets.append(dict(up.get("reset_event") or {}))
                replayed+=1
            else:m.markov_extractor.update(actual)
        prefix.append(actual); bp_before+=int(actual in ARMS)
    return m,{"raw_round_count":len(hist),"bp_training_samples":replayed,"reset_count":m.reset_count,"reset_events":resets[-20:],"history_fingerprint":sha256("".join(hist).encode()).hexdigest()[:24],"mode":"prequential_cusum_dynamic_linucb_markov29_next_hand","markov_window_size":m.markov_extractor.window_size,"markov_sample_count":len(m.markov_extractor.to_state()["values"])}

def predict_bandit(history,road_context=None,venue="",room="",user_id="",run_seed=None):
    del run_seed; raw=_clean(history); road=dict(road_context or {}) or _safe_road(raw); m,replay=_replay(raw); mv=m.markov_extractor.extract_features(); mp=m.markov_extractor.probability_dict(); mn=m.markov_extractor.next_prediction(); ctx=build_context_vector(raw,road_context=road,markov_features=mv); pred=m.predict_context(ctx); risk=m.risk_status(ctx); sel=pred["selected_arm"]; cond=pred["conditional_probabilities"]; tie=_tie_prob(raw); mass=1-tie; probs={"B":cond["B"]*mass,"P":cond["P"]*mass,"T":tie}; sh=pred["shared_uncertainty"]; fp=sha256(json.dumps({"history":"".join(raw),"venue":venue.upper(),"room":room,"context":ctx},sort_keys=True).encode()).hexdigest()[:24]; cus={"g_plus":m.g_plus,"g_minus":m.g_minus,"h":m.cusum_h,"v":m.cusum_v,"last_residual":m.last_residual,"last_expected_reward":m.last_expected_reward,"last_observed_reward":m.last_observed_reward,"reset_count":m.reset_count,"last_reset":m.last_reset}
    return {"ok":True,"engine":"CUSUM_LINUCB_DYNAMIC_CONTEXTUAL_BANDIT","model_version":MODEL_VERSION,"model_core":"cusum_linucb_markov29_next_hand_dynamic_reset_no_observe","prediction_fingerprint":fp,"road_support":road,"probabilities":probs,"bandit_learning_probabilities":probs,"banker_rate":round(probs["B"]*100,2),"player_rate":round(probs["P"]*100,2),"tie_rate":round(probs["T"]*100,2),"selected_arm":sel,"base_bandit_direction":sel,"recommend":sel,"recommend_text":"莊" if sel=="B" else "閒","action":sel,"action_text":"莊" if sel=="B" else "閒","internal_recommend":sel,"internal_action":sel,"next_round_direction":sel,"next_round_direction_text":"莊" if sel=="B" else "閒","signal_allowed":True,"signal_status_code":"CUSUM_LINUCB_DIRECTION","direction_source":"cusum_linucb","direction_edge":abs(cond["B"]-cond["P"]),"confidence_score":risk["confidence_score"],"confidence":risk["confidence_score"],"quality_score":risk["confidence_score"],"post_reset_vacuum_active":risk["post_reset_vacuum_active"],"force_observe":False,"hands_since_reset":risk["observations_since_reset"],"ensemble_weight_suggestion":risk["ensemble_weight_suggestion"],"bet_multiplier":risk["bet_multiplier"],"risk_control":risk,"cusum":cus,"reset_triggered":bool(m.last_reset),"uncertainty":sh["uncertainty"],"variance":sh["variance"],"variance_safe":True,"unknown_region_active":risk["post_reset_vacuum_active"],"is_extreme_unseen":False,"hard_brake_active":False,"uncertainty_braking":{"active":False,"is_extreme_unseen":False,"variance":sh["variance"],"action_space_variance":sh["variance"],"action_space_std":sh["uncertainty"],"variance_safe":True,"post_reset_vacuum_active":risk["post_reset_vacuum_active"],"confidence_score":risk["confidence_score"],"bet_multiplier":risk["bet_multiplier"],"observe_required":False,"cusum":cus},"markov_features":dict(zip(MARKOV_FEATURE_NAMES,mv)),"markov_probabilities":mp,"markov_next_prediction":mn,"markov_state":m.markov_extractor.to_state(),"bandit_context":ctx,"context_vector":ctx,"context_feature_names":list(FEATURE_NAMES),"bandit_scores":pred["metrics"],"bandit_state":{"alpha":CUSUM_ALPHA,"l2":CUSUM_L2,"forgetting_factor":CUSUM_FORGETTING_FACTOR,"context_dim":CONTEXT_DIM,"total_updates":m.total_observations,"observations_since_reset":m.observations_since_reset,"reset_count":m.reset_count,"cusum_h":CUSUM_THRESHOLD_H,"cusum_v":CUSUM_DRIFT_V,"vacuum_hands":CUSUM_VACUUM_HANDS,"force_observe_hands":0,"history_replay":replay,"state_file":str(CMAB_STATE_FILE),"markov_alpha":MARKOV_ALPHA,"markov_window_size":MARKOV_WINDOW_SIZE,"markov_support_prior":MARKOV_SUPPORT_PRIOR,"markov_max_order":MARKOV_MAX_ORDER},"adaptive_ensemble":{"active":False,"suggested_share":risk["ensemble_weight_suggestion"],"reason":"predictor.py performs final fusion"},"venue":venue,"room":room,"user_id":user_id,"input_required":False,"probability_semantics":"normalized_model_score_not_guaranteed_outcome_probability"}

def _uid(user_id):return sha256((str(user_id or "").strip() or "__anonymous__").encode()).hexdigest()[:24]
def _read_store():
    try:
        d=json.loads(CMAB_STATE_FILE.read_text(encoding="utf-8"))
        if d.get("schema_version")==STATE_SCHEMA_VERSION and isinstance(d.get("users"),dict):return d
    except Exception:pass
    return {"schema_version":STATE_SCHEMA_VERSION,"version":MODEL_VERSION,"context_dim":CONTEXT_DIM,"users":{}}
def _write_store(d):
    x=dict(d); x.update(schema_version=STATE_SCHEMA_VERSION,version=MODEL_VERSION,context_dim=CONTEXT_DIM,updated_at=int(time.time())); tmp=CMAB_STATE_FILE.with_suffix(CMAB_STATE_FILE.suffix+".tmp"); tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8"); tmp.replace(CMAB_STATE_FILE)
def update_bandit(context,selected_arm,reward,event_id="",actual_outcome="",update_weight=1.,user_id="",prediction_probabilities=None):
    del update_weight,prediction_probabilities; arm=str(selected_arm).upper(); event=str(event_id or "")
    if arm not in ARMS:raise ValueError("selected_arm must be B or P")
    with _LOCK:
        store=_read_store(); key=_uid(user_id); m=CUSUMLinUCB.from_state(dict(store["users"].get(key) or {}))
        if event and event in m.applied_event_ids:return {"updated":False,"reason":"duplicate_event","event_id":event}
        actual=str(actual_outcome).upper()
        if actual in ARMS:r=m.observe(context,actual,selected_arm=arm)
        elif reward is not None and math.isfinite(float(reward)):r=m.observe_reward(context,arm,float(reward))
        else:return {"updated":False,"reason":"tie_or_skipped_reward","event_id":event}
        if event:m.applied_event_ids=(m.applied_event_ids+[event])[-5000:]
        store["users"][key]=m.to_state(); _write_store(store)
    return {**r,"event_id":event,"model_version":MODEL_VERSION,"reset_count":m.reset_count,"hands_since_reset":m.observations_since_reset}
def get_bandit_summary(user_id=""):
    with _LOCK:m=CUSUMLinUCB.from_state(dict(_read_store()["users"].get(_uid(user_id)) or {}))
    return {"version":MODEL_VERSION,"context_dim":CONTEXT_DIM,"feature_names":list(FEATURE_NAMES),"total_updates":m.total_observations,"observations_since_reset":m.observations_since_reset,"reset_count":m.reset_count,"markov_features":m.markov_extractor.feature_dict(),"markov_probabilities":m.markov_extractor.probability_dict(),"markov_next_prediction":m.markov_extractor.next_prediction(),"markov_state":m.markov_extractor.to_state(),"cusum":{"g_plus":m.g_plus,"g_minus":m.g_minus,"h":m.cusum_h,"v":m.cusum_v,"last_residual":m.last_residual,"last_reset":m.last_reset},"arms":{a:{"updates":m.arms[a]["updates"],"reward_sum":m.arms[a]["reward_sum"]} for a in ARMS},"state_file":str(CMAB_STATE_FILE)}

class ContextualBanditEngine:
    def predict(self,history,**kwargs):return predict_bandit(history,**kwargs)
    def update(self,**kwargs):return update_bandit(**kwargs)
    def summary(self,user_id=""):return get_bandit_summary(user_id)
    def reset_model(self,user_id="",reason="manual_reset"):
        with _LOCK:
            s=_read_store(); k=_uid(user_id); m=CUSUMLinUCB.from_state(dict(s["users"].get(k) or {})); e=m.reset_model(reason); s["users"][k]=m.to_state(); _write_store(s); return e

DECISION_STRATEGY_ARMS=("math_only","ev_road_blend","conservative"); DECISION_STRATEGY_CONTEXT_DIM=34
def build_decision_strategy_context(history,**kwargs):del kwargs; return [1.,min(1.,len(_clean(history))/80)]+[0.]*(DECISION_STRATEGY_CONTEXT_DIM-2)
def select_decision_strategy(history,**kwargs):return {"version":"DECISION-STRATEGY-COMPAT-CUSUM-V1","selected_arm":"conservative","profile":{"kelly_multiplier":.5},"context":build_decision_strategy_context(history),"eligible_exact_composition":False,"reason":"compatibility only"}
def update_decision_strategy(**kwargs):return {"updated":False,"reason":"legacy_strategy_disabled","event_id":str(kwargs.get("event_id") or "")}
class DecisionStrategyBanditEngine:
    def select(self,history,**kwargs):return select_decision_strategy(history,**kwargs)
    def update(self,**kwargs):return update_decision_strategy(**kwargs)

__all__=["ARMS","MODEL_VERSION","ROAD_FEATURE_NAMES","MARKOV_PROBABILITY_NAMES","MARKOV_FEATURE_NAMES","FEATURE_NAMES","CONTEXT_DIM","MARKOV_ALPHA","MARKOV_WINDOW_SIZE","MARKOV_SUPPORT_PRIOR","MARKOV_MAX_ORDER","CUSUM_ALPHA","CUSUM_L2","CUSUM_FORGETTING_FACTOR","CUSUM_DRIFT_V","CUSUM_THRESHOLD_H","CUSUM_MIN_OBSERVATIONS","CUSUM_VACUUM_HANDS","CUSUM_FORCE_OBSERVE_HANDS","MarkovFeatureExtractor","CUSUMLinUCB","ContextualBanditEngine","DecisionStrategyBanditEngine","DECISION_STRATEGY_ARMS","DECISION_STRATEGY_CONTEXT_DIM","build_context_vector","build_decision_strategy_context","predict_bandit","update_bandit","get_bandit_summary","select_decision_strategy","update_decision_strategy"]
