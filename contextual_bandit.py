from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import json, math, time
import numpy as np
from road_model import build_road_context

ARMS=("B","P"); RULE_ARMS=("FOLLOW","BREAK")
MODEL_VERSION="CUSUM-LINUCB-V1.1-RULE-FOLLOW-BREAK"
STATE_SCHEMA_VERSION="CUSUM-LINUCB-STATE-V1.1"
FEATURE_NAMES=("bias","history_maturity","global_banker_balance","recent3_banker_balance","recent8_banker_balance","current_streak_direction","current_streak_length","alternation6","alternation12","transition_acceleration","streak_break_signal","long_dragon_tail_pressure","observed_tie_rate","road_planning_balance","road_recent_balance","road_confidence","road_agreement","big_eye_saturation","small_road_saturation","cockroach_road_saturation","derived_road_consensus","structural_rule_active","structural_rule_reliability","structural_rule_direction")
CONTEXT_DIM=len(FEATURE_NAMES)
CUSUM_ALPHA=.65; CUSUM_L2=1.; CUSUM_FORGETTING_FACTOR=.985; CUSUM_DRIFT_V=.15; CUSUM_THRESHOLD_H=4.5
CUSUM_MIN_OBSERVATIONS=8; CUSUM_VACUUM_HANDS=5; CUSUM_FORCE_OBSERVE_HANDS=3; PREQUENTIAL_WARMUP_BP=6; HISTORY_REPLAY_LIMIT=120
RULE_MIN_OBSERVATIONS=6; TIE_PRIOR=.095156; TIE_PRIOR_STRENGTH=40.
_LOCK=RLock(); BASE_DIR=Path(__file__).resolve().parent

def _state_file():
    for p in (Path('/var/data/contextual_bandit_state_cusum_v1_1.json'),BASE_DIR/'data'/'contextual_bandit_state_cusum_v1_1.json',Path('/tmp/bgs_contextual_bandit_state_cusum_v1_1.json')):
        try:
            p.parent.mkdir(parents=True,exist_ok=True); q=p.parent/f'.probe_{time.time_ns()}'; q.write_text('ok'); q.unlink(missing_ok=True); return p
        except OSError: pass
    raise RuntimeError('No writable CUSUM state path')
CMAB_STATE_FILE=_state_file()

def _clip(v,lo=-1.,hi=1.):
    try: x=float(v)
    except (TypeError,ValueError): return 0.
    return max(lo,min(hi,x)) if math.isfinite(x) else 0.

def _clean(values):
    out=[]
    for item in values:
        raw=item.get('outcome') if isinstance(item,Mapping) else item; v=str(raw or '').upper().strip()
        if v in {'B','P','T'}: out.append(v)
    return out[-2000:]

def _balance(seq,n=None):
    s=list(seq[-n:] if n else seq); return 0. if not s else _clip((sum(x=='B' for x in s)/len(s)-.5)*2)

def _transition(seq,n):
    s=list(seq[-n:]); return 0. if len(s)<2 else sum(a!=b for a,b in zip(s,s[1:]))/(len(s)-1)

def _streak(seq):
    if not seq:return '',0
    side=seq[-1]; n=1
    for x in reversed(seq[:-1]):
        if x!=side: break
        n+=1
    return side,n

def _streak_break(seq):
    s=list(seq)
    if len(s)<4 or s[-1]==s[-2]: return 0.
    old=s[-2]; n=1
    for x in reversed(s[:-2]):
        if x!=old:break
        n+=1
    return 0. if n<3 else (1. if s[-1]=='B' else -1.)*min(1.,n/6.)

def _road_sat(road,name):
    p=road.get('full_road_analysis')
    if not isinstance(p,Mapping):
        m=road.get('models'); p=m.get('full_road') if isinstance(m,Mapping) else {}
    st=dict(p.get('derived_stats') or {}).get(name) if isinstance(p,Mapping) else None
    if not isinstance(st,Mapping):return 0.
    b=_clip(st.get('balance',0),0,1); c=_clip(st.get('recent_continuation',.5),0,1); return max(b,abs(2*c-1))

def _rule(road):
    r=road.get('structural_regime')
    if not isinstance(r,Mapping):
        m=road.get('models'); r=m.get('structural_regime') if isinstance(m,Mapping) else {}
    d=dict(r or {}); side=str(d.get('direction') or '').upper(); active=bool(d.get('active')) and side in ARMS
    return {'active':active,'name':str(d.get('name') or 'mixed'),'direction':side if active else '','reliability':_clip(d.get('reliability',0),0,1),'support':int(d.get('support',0) or 0),'reason':str(d.get('reason') or '')}

def build_context_vector(history,*,road_context=None):
    raw=_clean(history); bp=[x for x in raw if x in ARMS]; road=dict(road_context or {}); side,run=_streak(bp); sign=1. if side=='B' else -1. if side=='P' else 0.
    dis=_clip(road.get('recent_model_disagreement',road.get('model_disagreement',.2)),0,1); big=_road_sat(road,'big_eye'); small=_road_sat(road,'small_road'); cock=_road_sat(road,'cockroach_road'); mean=(big+small+cock)/3; consensus=_clip(mean*(1-(abs(big-small)+abs(small-cock)+abs(cock-big))/3),0,1); rule=_rule(road); rd=1. if rule['direction']=='B' else -1. if rule['direction']=='P' else 0.
    def pb(v):
        try:p=float(v)
        except (TypeError,ValueError):p=.5
        return _clip((p-.5)*2)
    vals=[1.,min(1.,len(bp)/60),_balance(bp),_balance(bp,3),_balance(bp,8),sign,min(1.,run/8),_clip((_transition(bp,6)-.5)*2),_clip((_transition(bp,12)-.5)*2),_clip(_transition(bp,6)-_transition(bp,14)),_streak_break(bp),sign*min(1.,max(0,run-3)/5),_clip((sum(x=='T' for x in raw)/max(1,len(raw)))/.2,0,1),pb(road.get('planning_probability',.5)),pb(road.get('recent_probability',.5)),_clip(road.get('confidence_score',0),0,1),_clip(1-min(1,dis/.2),0,1),big,small,cock,consensus,1. if rule['active'] else 0.,rule['reliability'],rd]
    if len(vals)!=CONTEXT_DIM:raise RuntimeError('context dimension mismatch')
    return [round(_clip(v),10) for v in vals]

def _vec(x):
    v=np.asarray(list(x),dtype=float)
    if v.shape!=(CONTEXT_DIM,) or not np.all(np.isfinite(v)):raise ValueError(f'context must be finite {CONTEXT_DIM}-vector')
    return np.clip(v,-1,1)

def _soft(a,b,t=.85):
    z=np.asarray([a,b],dtype=float)/t; z-=np.max(z); e=np.exp(np.clip(z,-40,40)); e/=max(1e-12,float(e.sum())); return float(e[0]),float(e[1])

def _tie(raw):
    p=(sum(x=='T' for x in raw)+TIE_PRIOR*TIE_PRIOR_STRENGTH)/(len(raw)+TIE_PRIOR_STRENGTH); return max(.04,min(.18,float(p)))

class CUSUMLinUCB:
    def __init__(self,alpha=CUSUM_ALPHA,l2=CUSUM_L2,forgetting_factor=CUSUM_FORGETTING_FACTOR,cusum_h=CUSUM_THRESHOLD_H,cusum_v=CUSUM_DRIFT_V,min_cusum_observations=CUSUM_MIN_OBSERVATIONS,vacuum_hands=CUSUM_VACUUM_HANDS):
        self.alpha=float(alpha); self.l2=max(1e-9,float(l2)); self.forgetting_factor=max(.8,min(1.,float(forgetting_factor))); self.cusum_h=max(.5,float(cusum_h)); self.cusum_v=max(0.,float(cusum_v)); self.min_cusum_observations=max(2,int(min_cusum_observations)); self.vacuum_hands=max(1,int(vacuum_hands)); self.total_observations=0; self.observations_since_reset=0; self.rule_observations=0; self.rule_observations_since_reset=0; self.reset_count=0; self.g_plus=self.g_minus=self.last_residual=self.last_expected_reward=self.last_observed_reward=0.; self.last_reset={}; self.applied_event_ids=[]; self._fresh()
    def _fresh(self):
        I=np.eye(CONTEXT_DIM)*self.l2; z=np.zeros(CONTEXT_DIM); self.arms={a:{'A':I.copy(),'b':z.copy(),'updates':0,'reward_sum':0.} for a in ARMS}; self.rule_arms={a:{'A':I.copy(),'b':z.copy(),'updates':0,'reward_sum':0.} for a in RULE_ARMS}; self.context_information={'A':I.copy(),'updates':0}; self.rule_context_information={'A':I.copy(),'updates':0}
    @staticmethod
    def _pinv(A):
        A=.5*(A+A.T)
        try:r=np.linalg.pinv(A,rcond=1e-10,hermitian=True)
        except TypeError:r=np.linalg.pinv(A,rcond=1e-10)
        return r if np.all(np.isfinite(r)) else np.linalg.pinv(A+np.eye(A.shape[0])*1e-6,rcond=1e-8)
    def _metrics(self,store,arm,x):
        x=_vec(x); s=store[arm]; inv=self._pinv(np.asarray(s['A'],float)); theta=inv@np.asarray(s['b'],float); mean=float(theta@x); var=max(0.,float(x@inv@x)); std=math.sqrt(var); bonus=self.alpha*std; return {'expected_reward':mean,'mean_reward':mean,'variance':var,'uncertainty':std,'ucb_bonus':bonus,'ucb_score':mean+bonus,'updates':int(s['updates']),'reward_sum':float(s['reward_sum'])}
    def arm_metrics(self,arm,context):return self._metrics(self.arms,arm,context)
    def predict_context(self,context):
        x=_vec(context); m={a:self.arm_metrics(a,x) for a in ARMS}; b,p=m['B']['ucb_score'],m['P']['ucb_score']; sel='B' if b>=p else 'P'; bp,pp=_soft(b,p); inv=self._pinv(np.asarray(self.context_information['A'],float)); var=max(0.,float(x@inv@x)); return {'selected_arm':sel,'metrics':m,'conditional_probabilities':{'B':bp,'P':pp},'shared_uncertainty':{'variance':var,'uncertainty':math.sqrt(var),'updates':int(self.context_information['updates'])}}
    def predict_rule(self,context,rule):
        r=dict(rule or {})
        if not (r.get('active') and r.get('direction') in ARMS):return {'active':False,'decision':'NO_RULE','decision_direction':'','rule_direction':'','follow_probability':.5,'break_probability':.5,'confidence':0.,'mature':False,'observations':self.rule_observations}
        m={a:self._metrics(self.rule_arms,a,context) for a in RULE_ARMS}; fs,bs=m['FOLLOW']['ucb_score'],m['BREAK']['ucb_score']; fp,bp=_soft(fs,bs,.75); dec='FOLLOW' if fs>=bs else 'BREAK'; rd=r['direction']; dd=rd if dec=='FOLLOW' else ('P' if rd=='B' else 'B'); maturity=min(1.,self.rule_observations_since_reset/RULE_MIN_OBSERVATIONS); conf=max(0.,min(.95,.55*abs(fp-bp)+.45*maturity)); return {'active':True,'decision':dec,'rule_name':r.get('name','mixed'),'rule_direction':rd,'decision_direction':dd,'follow_score':fs,'break_score':bs,'follow_probability':fp,'break_probability':bp,'confidence':conf,'mature':self.rule_observations_since_reset>=RULE_MIN_OBSERVATIONS,'observations':self.rule_observations,'observations_since_reset':self.rule_observations_since_reset,'metrics':m}
    def _update_store(self,store,info,x,rewards):
        x=_vec(x); I=np.eye(CONTEXT_DIM)*self.l2; outer=np.outer(x,x); lam=self.forgetting_factor
        for arm,r in rewards.items():
            s=store[arm]; A=np.asarray(s['A'],float); b=np.asarray(s['b'],float); A=lam*.5*(A+A.T)+(1-lam)*I+outer; b=lam*b+float(r)*x; s.update(A=.5*(A+A.T),b=b,updates=int(s['updates'])+1,reward_sum=float(s['reward_sum'])+float(r))
        A=np.asarray(info['A'],float); A=lam*.5*(A+A.T)+(1-lam)*I+outer; info['A']=.5*(A+A.T); info['updates']=int(info['updates'])+1
    def _cusum(self,observed,expected):
        res=float(observed-expected); self.last_residual=res; self.last_observed_reward=float(observed); self.last_expected_reward=float(expected); self.g_plus=max(0.,self.g_plus+res-self.cusum_v); self.g_minus=max(0.,self.g_minus-res-self.cusum_v); ready=self.observations_since_reset>=self.min_cusum_observations and self.total_observations>=self.min_cusum_observations; plus=ready and self.g_plus>self.cusum_h; minus=ready and self.g_minus>self.cusum_h; return {'residual':res,'g_plus':self.g_plus,'g_minus':self.g_minus,'ready':ready,'alarm':bool(plus or minus),'alarm_side':'positive' if plus else 'negative' if minus else '','threshold_h':self.cusum_h,'drift_v':self.cusum_v}
    def reset_model(self,reason='cusum_change_point',alarm_side='',residual=None):
        self.reset_count+=1; ev={'triggered':True,'reason':reason,'alarm_side':alarm_side,'residual':float(self.last_residual if residual is None else residual),'g_plus_before_reset':self.g_plus,'g_minus_before_reset':self.g_minus,'threshold_h':self.cusum_h,'drift_v':self.cusum_v,'at_total_observation':self.total_observations,'reset_count':self.reset_count,'timestamp':int(time.time())}; self._fresh(); self.observations_since_reset=0; self.rule_observations_since_reset=0; self.g_plus=self.g_minus=0.; self.last_reset=ev; return dict(ev)
    def observe(self,context,actual_outcome,selected_arm='',rule=None):
        actual=str(actual_outcome or '').upper()
        if actual not in ARMS:return {'updated':False,'reason':'tie_or_invalid_outcome'}
        x=_vec(context); pred=self.predict_context(x); chosen=str(selected_arm or pred['selected_arm']).upper(); chosen=chosen if chosen in ARMS else pred['selected_arm']; expected=_clip(pred['metrics'][chosen]['expected_reward']); observed=1. if chosen==actual else -1.; c=self._cusum(observed,expected); reset={}
        if c['alarm']:reset=self.reset_model('cusum_residual_change_point',c['alarm_side'],c['residual'])
        self._update_store(self.arms,self.context_information,x,{a:(1. if a==actual else -1.) for a in ARMS}); self.total_observations+=1; self.observations_since_reset+=1; ru={'updated':False}
        rr=dict(rule or {})
        if rr.get('active') and rr.get('direction') in ARMS:
            label='FOLLOW' if actual==rr['direction'] else 'BREAK'; self._update_store(self.rule_arms,self.rule_context_information,x,{a:(1. if a==label else -1.) for a in RULE_ARMS}); self.rule_observations+=1; self.rule_observations_since_reset+=1; ru={'updated':True,'label':label}
        return {'updated':True,'actual_outcome':actual,'selected_arm':chosen,'observed_reward':observed,'expected_reward':expected,'cusum':c,'reset_triggered':bool(reset),'reset_event':reset,'rule_update':ru}
    def observe_reward(self,context,selected_arm,reward):
        arm=str(selected_arm).upper(); x=_vec(context); expected=_clip(self.arm_metrics(arm,x)['expected_reward']); r=_clip(reward); c=self._cusum(r,expected); reset={}
        if c['alarm']:reset=self.reset_model('cusum_selected_arm_reward_change_point',c['alarm_side'],c['residual'])
        self._update_store(self.arms,self.context_information,x,{arm:r}); self.total_observations+=1; self.observations_since_reset+=1; return {'updated':True,'selected_arm':arm,'reward':r,'expected_reward':expected,'cusum':c,'reset_triggered':bool(reset),'reset_event':reset}
    def risk_status(self,context):
        s=self.predict_context(context)['shared_uncertainty']; u=float(s['uncertainty']); info=1/(1+u); ref=self.observations_since_reset if self.reset_count else self.total_observations; maturity=min(1.,ref/12); conf=_clip(.1+.55*info+.35*maturity,0,.9); vacuum=bool(self.reset_count and self.observations_since_reset<=self.vacuum_hands); conf=min(conf,min(.48,.08+.08*self.observations_since_reset)) if vacuum else conf; observe=bool(vacuum and self.observations_since_reset<=CUSUM_FORCE_OBSERVE_HANDS)
        if observe:w,bet=0.,0.
        elif vacuum:w,bet=min(.18,.04+.03*self.observations_since_reset),(.35 if self.observations_since_reset==4 else .6)
        elif self.total_observations<PREQUENTIAL_WARMUP_BP:w,bet=.08,.5
        else:w,bet=min(.45,.15+.35*conf),min(1.,.55+.5*conf)
        return {'confidence_score':conf,'post_reset_vacuum_active':vacuum,'vacuum_hands_required':self.vacuum_hands,'observations_since_reset':self.observations_since_reset,'force_observe':observe,'bet_multiplier':bet,'ensemble_weight_suggestion':w,'uncertainty':u,'variance':float(s['variance']),'maturity':maturity}
    def to_state(self):
        pack=lambda d:{a:{'A':np.asarray(d[a]['A']).tolist(),'b':np.asarray(d[a]['b']).tolist(),'updates':int(d[a]['updates']),'reward_sum':float(d[a]['reward_sum'])} for a in d}
        return {'version':MODEL_VERSION,'context_dim':CONTEXT_DIM,'alpha':self.alpha,'l2':self.l2,'forgetting_factor':self.forgetting_factor,'cusum_h':self.cusum_h,'cusum_v':self.cusum_v,'min_cusum_observations':self.min_cusum_observations,'vacuum_hands':self.vacuum_hands,'total_observations':self.total_observations,'observations_since_reset':self.observations_since_reset,'rule_observations':self.rule_observations,'rule_observations_since_reset':self.rule_observations_since_reset,'reset_count':self.reset_count,'g_plus':self.g_plus,'g_minus':self.g_minus,'last_residual':self.last_residual,'last_expected_reward':self.last_expected_reward,'last_observed_reward':self.last_observed_reward,'last_reset':self.last_reset,'applied_event_ids':self.applied_event_ids[-5000:],'arms':pack(self.arms),'rule_arms':pack(self.rule_arms),'context_information':{'A':np.asarray(self.context_information['A']).tolist(),'updates':int(self.context_information['updates'])},'rule_context_information':{'A':np.asarray(self.rule_context_information['A']).tolist(),'updates':int(self.rule_context_information['updates'])}}
    @classmethod
    def from_state(cls,state):
        if not isinstance(state,Mapping) or int(state.get('context_dim',0) or 0)!=CONTEXT_DIM:return cls()
        m=cls(state.get('alpha',CUSUM_ALPHA),state.get('l2',CUSUM_L2),state.get('forgetting_factor',CUSUM_FORGETTING_FACTOR),state.get('cusum_h',CUSUM_THRESHOLD_H),state.get('cusum_v',CUSUM_DRIFT_V),state.get('min_cusum_observations',CUSUM_MIN_OBSERVATIONS),state.get('vacuum_hands',CUSUM_VACUUM_HANDS))
        try:
            for key,names,target in [('arms',ARMS,m.arms),('rule_arms',RULE_ARMS,m.rule_arms)]:
                for a in names:
                    d=state[key][a]; A=np.asarray(d['A'],float); b=np.asarray(d['b'],float)
                    if A.shape!=(CONTEXT_DIM,CONTEXT_DIM) or b.shape!=(CONTEXT_DIM,):raise ValueError
                    target[a]={'A':.5*(A+A.T),'b':b,'updates':int(d.get('updates',0)),'reward_sum':float(d.get('reward_sum',0))}
            for key in ['context_information','rule_context_information']:
                d=state[key]; A=np.asarray(d['A'],float); setattr(m,key,{'A':.5*(A+A.T),'updates':int(d.get('updates',0))})
            for k in ['total_observations','observations_since_reset','rule_observations','rule_observations_since_reset','reset_count']:setattr(m,k,int(state.get(k,0)))
            for k in ['g_plus','g_minus','last_residual','last_expected_reward','last_observed_reward']:setattr(m,k,float(state.get(k,0)))
            m.last_reset=dict(state.get('last_reset') or {}); m.applied_event_ids=[str(x) for x in state.get('applied_event_ids',[])][-5000:]
        except Exception:return cls()
        return m

def _safe_road(history):
    try:return dict(build_road_context(history,initial_image_count=len(history),manual_count=0) or {})
    except Exception:return {}

def _replay(raw):
    hist=list(raw)[-HISTORY_REPLAY_LIMIT:]; m=CUSUMLinUCB(); prefix=[]; bp=0; resets=[]; samples=rules=0
    for actual in hist:
        if actual in ARMS and bp>=PREQUENTIAL_WARMUP_BP:
            road=_safe_road(prefix); ctx=build_context_vector(prefix,road_context=road); up=m.observe(ctx,actual,rule=_rule(road)); samples+=1; rules+=int(up.get('rule_update',{}).get('updated',False))
            if up.get('reset_triggered'):resets.append(dict(up.get('reset_event') or {}))
        prefix.append(actual); bp+=int(actual in ARMS)
    return m,{'raw_round_count':len(hist),'bp_training_samples':samples,'rule_training_samples':rules,'reset_count':m.reset_count,'reset_events':resets[-20:],'history_fingerprint':sha256(''.join(hist).encode()).hexdigest()[:24],'mode':'prequential_cusum_rule_follow_break'}

def predict_bandit(history,*,road_context=None,venue='',room='',user_id='',run_seed=None):
    del run_seed
    raw=_clean(history); road=dict(road_context or {}) or _safe_road(raw); model,replay=_replay(raw); ctx=build_context_vector(raw,road_context=road); pred=model.predict_context(ctx); risk=model.risk_status(ctx); rule=_rule(road); rp=model.predict_rule(ctx,rule); selected=pred['selected_arm']; cond=pred['conditional_probabilities']; directional=dict(cond)
    if rp.get('active') and rp.get('mature'):
        selected=rp['decision_direction']; follow_p,break_p=rp['follow_probability'],rp['break_probability']; directional={'B':follow_p,'P':break_p} if rp['rule_direction']=='B' else {'B':break_p,'P':follow_p}
    tie=_tie(raw); mass=1-tie; probs={'B':directional['B']*mass,'P':directional['P']*mass,'T':tie}; observe=bool(risk['force_observe']); action='O' if observe else selected; fingerprint=sha256(json.dumps({'history':''.join(raw),'venue':venue.upper(),'room':room,'context':ctx},sort_keys=True).encode()).hexdigest()[:24]; cusum={'g_plus':model.g_plus,'g_minus':model.g_minus,'h':model.cusum_h,'v':model.cusum_v,'last_residual':model.last_residual,'last_expected_reward':model.last_expected_reward,'last_observed_reward':model.last_observed_reward,'reset_count':model.reset_count,'last_reset':model.last_reset}; dec=str(rp.get('decision') or 'NO_RULE')
    return {'ok':True,'engine':'CUSUM_LINUCB_RULE_FOLLOW_BREAK','model_version':MODEL_VERSION,'model_core':'cusum_linucb_dynamic_reset_rule_follow_break','prediction_fingerprint':fingerprint,'road_support':road,'probabilities':probs,'banker_rate':round(probs['B']*100,2),'player_rate':round(probs['P']*100,2),'tie_rate':round(probs['T']*100,2),'selected_arm':selected,'base_bandit_direction':pred['selected_arm'],'recommend':action,'action':action,'recommend_text':'觀望' if observe else '莊' if selected=='B' else '閒','action_text':'觀望' if observe else '莊' if selected=='B' else '閒','internal_recommend':selected,'internal_action':selected,'next_round_direction':selected,'next_round_direction_text':'莊' if selected=='B' else '閒','signal_allowed':not observe,'signal_status_code':'CUSUM_POST_RESET_VACUUM_OBSERVE' if observe else 'RULE_FOLLOW_BREAK_DIRECTION','direction_source':'rule_follow_break' if rp.get('active') and rp.get('mature') else 'cusum_linucb','direction_edge':abs(directional['B']-directional['P']),'confidence_score':risk['confidence_score'],'confidence':risk['confidence_score'],'quality_score':risk['confidence_score'],'post_reset_vacuum_active':risk['post_reset_vacuum_active'],'force_observe':observe,'hands_since_reset':risk['observations_since_reset'],'ensemble_weight_suggestion':risk['ensemble_weight_suggestion'],'bet_multiplier':risk['bet_multiplier'],'risk_control':risk,'cusum':cusum,'reset_triggered':bool(model.last_reset),'uncertainty':pred['shared_uncertainty']['uncertainty'],'variance':pred['shared_uncertainty']['variance'],'unknown_region_active':risk['post_reset_vacuum_active'],'hard_brake_active':observe,'rule_state':{**rp,'confirmed_rule':rule,'rule_decision':dec,'follow_rule':dec=='FOLLOW','break_rule':dec=='BREAK'},'rule_decision':dec,'rule_name':rule['name'],'rule_direction':rp.get('rule_direction',''),'rule_follow_probability':float(rp.get('follow_probability',.5)),'rule_break_probability':float(rp.get('break_probability',.5)),'rule_decision_direction':rp.get('decision_direction',''),'rule_model_mature':bool(rp.get('mature')),'rule_observations':int(rp.get('observations',model.rule_observations)),'context_vector':ctx,'bandit_context':ctx,'context_feature_names':list(FEATURE_NAMES),'bandit_scores':pred['metrics'],'bandit_state':{'context_dim':CONTEXT_DIM,'total_updates':model.total_observations,'observations_since_reset':model.observations_since_reset,'rule_observations':model.rule_observations,'rule_observations_since_reset':model.rule_observations_since_reset,'reset_count':model.reset_count,'history_replay':replay,'state_file':str(CMAB_STATE_FILE)},'venue':venue,'room':room,'user_id':user_id,'input_required':False,'probability_semantics':'normalized_model_score_not_guaranteed_outcome_probability'}

def _uid(user_id):return sha256((str(user_id or '').strip() or '__anonymous__').encode()).hexdigest()[:24]
def _read_store():
    try:
        d=json.loads(CMAB_STATE_FILE.read_text())
        if d.get('schema_version')==STATE_SCHEMA_VERSION and isinstance(d.get('users'),dict):return d
    except Exception:pass
    return {'schema_version':STATE_SCHEMA_VERSION,'version':MODEL_VERSION,'context_dim':CONTEXT_DIM,'users':{}}
def _write_store(d):
    x=dict(d); x.update(schema_version=STATE_SCHEMA_VERSION,version=MODEL_VERSION,context_dim=CONTEXT_DIM,updated_at=int(time.time())); tmp=CMAB_STATE_FILE.with_suffix(CMAB_STATE_FILE.suffix+'.tmp'); tmp.write_text(json.dumps(x,ensure_ascii=False)); tmp.replace(CMAB_STATE_FILE)
def update_bandit(*,context,selected_arm,reward,event_id='',actual_outcome='',update_weight=1.,user_id='',prediction_probabilities=None):
    del update_weight,prediction_probabilities
    arm=str(selected_arm).upper(); event=str(event_id or '')
    if arm not in ARMS:raise ValueError('selected_arm must be B or P')
    with _LOCK:
        store=_read_store(); key=_uid(user_id); m=CUSUMLinUCB.from_state(dict(store['users'].get(key) or {}))
        if event and event in m.applied_event_ids:return {'updated':False,'reason':'duplicate_event','event_id':event}
        actual=str(actual_outcome).upper(); result=m.observe(context,actual,selected_arm=arm) if actual in ARMS else m.observe_reward(context,arm,float(reward)) if reward is not None and math.isfinite(float(reward)) else {'updated':False,'reason':'tie_or_skipped_reward','event_id':event}
        if not result.get('updated'):return result
        if event:m.applied_event_ids=(m.applied_event_ids+[event])[-5000:]
        store['users'][key]=m.to_state(); _write_store(store)
    return {**result,'event_id':event,'model_version':MODEL_VERSION,'reset_count':m.reset_count,'hands_since_reset':m.observations_since_reset}
def get_bandit_summary(user_id=''):
    with _LOCK:m=CUSUMLinUCB.from_state(dict(_read_store()['users'].get(_uid(user_id)) or {}))
    return {'version':MODEL_VERSION,'context_dim':CONTEXT_DIM,'feature_names':list(FEATURE_NAMES),'total_updates':m.total_observations,'observations_since_reset':m.observations_since_reset,'rule_observations':m.rule_observations,'rule_observations_since_reset':m.rule_observations_since_reset,'reset_count':m.reset_count,'state_file':str(CMAB_STATE_FILE)}
class ContextualBanditEngine:
    def predict(self,history,**kwargs):return predict_bandit(history,**kwargs)
    def update(self,**kwargs):return update_bandit(**kwargs)
    def summary(self,user_id=''):return get_bandit_summary(user_id)
    def reset_model(self,user_id='',reason='manual_reset'):
        with _LOCK:
            s=_read_store(); k=_uid(user_id); m=CUSUMLinUCB.from_state(dict(s['users'].get(k) or {})); ev=m.reset_model(reason); s['users'][k]=m.to_state(); _write_store(s); return ev
DECISION_STRATEGY_ARMS=('math_only','ev_road_blend','conservative'); DECISION_STRATEGY_CONTEXT_DIM=34
def build_decision_strategy_context(history,**kwargs):del kwargs; return [1.,min(1.,len(_clean(history))/80)]+[0.]*(DECISION_STRATEGY_CONTEXT_DIM-2)
def select_decision_strategy(history,**kwargs):return {'version':'DECISION-STRATEGY-COMPAT-CUSUM-V1','selected_arm':'conservative','profile':{'kelly_multiplier':.5},'context':build_decision_strategy_context(history),'eligible_exact_composition':False,'reason':'compatibility only'}
def update_decision_strategy(**kwargs):return {'updated':False,'reason':'legacy_strategy_disabled','event_id':str(kwargs.get('event_id') or '')}
class DecisionStrategyBanditEngine:
    def select(self,history,**kwargs):return select_decision_strategy(history,**kwargs)
    def update(self,**kwargs):return update_decision_strategy(**kwargs)
__all__=['ARMS','RULE_ARMS','MODEL_VERSION','FEATURE_NAMES','CONTEXT_DIM','CUSUM_ALPHA','CUSUM_L2','CUSUM_FORGETTING_FACTOR','CUSUM_DRIFT_V','CUSUM_THRESHOLD_H','CUSUM_MIN_OBSERVATIONS','CUSUM_VACUUM_HANDS','CUSUM_FORCE_OBSERVE_HANDS','RULE_MIN_OBSERVATIONS','CUSUMLinUCB','ContextualBanditEngine','DecisionStrategyBanditEngine','DECISION_STRATEGY_ARMS','DECISION_STRATEGY_CONTEXT_DIM','build_context_vector','build_decision_strategy_context','predict_bandit','update_bandit','get_bandit_summary','select_decision_strategy','update_decision_strategy']
