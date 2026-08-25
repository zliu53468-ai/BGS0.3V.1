from __future__ import annotations
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import secrets
from adaptive_ensemble import adapt_prediction
from contextual_bandit import MODEL_VERSION as CUSUM_MODEL_VERSION, predict_bandit
from road_model import build_road_context

ADAPTIVE_MODEL_VARIANT='V35.1_RULE_FOLLOW_BREAK_CUSUM_LINUCB'
MAX_CUSUM_ENSEMBLE_WEIGHT=.65; RULE_GATE_MIN_WEIGHT=.50; RULE_GATE_MAX_WEIGHT=.65
DB_HOLDOUT={'status':'removed','replacement':'FULL_ROAD_ADAPTIVE_PLUS_CUSUM_RULE_FOLLOW_BREAK','note':'正式方向使用牌路結構、Adaptive 與 CUSUM-LinUCB 跟規律/斷規律決策。'}

def _normalize_outcome_history(values):
    out=[]
    for item in values:
        raw=(item.get('outcome') or item.get('actual') or item.get('actual_outcome') or item.get('virtual_outcome')) if isinstance(item,Mapping) else item; v=str(raw or '').upper().strip()
        if v in {'B','P','T'}:out.append(v)
    return out[-2000:]

def _bandit_learning_scope(user_id='',venue='',room=''):
    raw='|'.join((str(user_id or '__anonymous__'),str(venue or '').upper().strip(),str(room or '').strip())); return '__cusum_scope__:'+sha256(raw.encode()).hexdigest()[:32]

def _normalize_probabilities(v):
    if not isinstance(v,Mapping):return {'B':.455,'P':.455,'T':.09}
    d={}
    for k in ('B','P','T'):
        try:d[k]=max(0.,float(v.get(k,0) or 0))
        except (TypeError,ValueError):d[k]=0.
    s=sum(d.values()); return {k:d[k]/s for k in d} if s>1e-12 else {'B':.455,'P':.455,'T':.09}

def _conditional_banker(v):
    p=_normalize_probabilities(v); s=p['B']+p['P']; return .5 if s<=1e-12 else p['B']/s

def _road_seed_prediction(road,history):
    try:b=float(road.get('banker_probability',.5) or .5)
    except (TypeError,ValueError):b=.5
    try:p=float(road.get('player_probability',1-b) or 0)
    except (TypeError,ValueError):p=1-b
    s=b+p; b,p=(.5,.5) if s<=1e-12 else (b/s,p/s)
    try:t=max(0.,min(.3,float(road.get('observed_tie_rate',0) or 0)))
    except (TypeError,ValueError):t=0.
    mass=1-t; direction=str(road.get('direction') or '').upper(); direction=direction if direction in {'B','P'} else ('B' if b>=p else 'P'); hist=_normalize_outcome_history(history)
    return {'model_version':'FULL_ROAD_ADAPTIVE_V35.1','model_variant':ADAPTIVE_MODEL_VARIANT,'prediction_fingerprint':sha256('|'.join(hist).encode()).hexdigest()[:24],'probabilities':{'B':mass*b,'P':mass*p,'T':t},'raw_probabilities':{'B':mass*b,'P':mass*p,'T':t},'banker_rate':round(mass*b*100,2),'player_rate':round(mass*p*100,2),'tie_rate':round(t*100,2),'action':direction,'recommend':direction,'internal_action':direction,'internal_recommend':direction,'next_round_direction':direction}

def _conf(r,default=.5):
    for k in ('ensemble_confidence','confidence','quality_score'):
        try:v=float(r.get(k,0) or 0)
        except (TypeError,ValueError):continue
        if v>0:return max(0.,min(1.,v))
    return default

def _gate(bandit):
    raw=bandit.get('rule_state'); s=dict(raw) if isinstance(raw,Mapping) else {}; dec=str(s.get('rule_decision') or s.get('decision') or bandit.get('rule_decision') or 'NO_RULE').upper(); dd=str(s.get('decision_direction') or bandit.get('rule_decision_direction') or '').upper(); rd=str(s.get('rule_direction') or bandit.get('rule_direction') or '').upper(); active=bool(s.get('active')) and rd in {'B','P'}; mature=bool(s.get('mature',bandit.get('rule_model_mature',False)))
    def f(k,default):
        try:return max(0.,min(1.,float(s.get(k,bandit.get('rule_'+k,default)) or default)))
        except (TypeError,ValueError):return default
    return {'active':active,'mature':mature,'decision':dec if dec in {'FOLLOW','BREAK'} else 'NO_RULE','decision_direction':dd if dd in {'B','P'} else '','rule_direction':rd,'rule_name':str(s.get('rule_name') or bandit.get('rule_name') or 'mixed'),'confidence':f('confidence',0.),'follow_probability':f('follow_probability',.5),'break_probability':f('break_probability',.5),'observations':int(s.get('observations',bandit.get('rule_observations',0)) or 0)}

def _fuse_adaptive_and_cusum(adaptive_prediction,bandit_prediction):
    result=dict(adaptive_prediction or {}); bandit=dict(bandit_prediction or {}); risk=dict(bandit.get('risk_control') or {}); gate=_gate(bandit); ap=_normalize_probabilities(result.get('adaptive_only_probabilities') if isinstance(result.get('adaptive_only_probabilities'),Mapping) else result.get('probabilities')); bp=_normalize_probabilities(bandit.get('probabilities')); ab=_conditional_banker(ap); bb=_conditional_banker(bp)
    try:req=float(bandit.get('ensemble_weight_suggestion',risk.get('ensemble_weight_suggestion',0)) or 0)
    except (TypeError,ValueError):req=0.
    rule_active=bool(gate['active'] and gate['mature'] and gate['decision_direction'] in {'B','P'})
    if rule_active:req=max(req,min(RULE_GATE_MAX_WEIGHT,RULE_GATE_MIN_WEIGHT+.15*gate['confidence']))
    bw=max(0.,min(MAX_CUSUM_ENSEMBLE_WEIGHT,req)); force=bool(bandit.get('force_observe') or risk.get('force_observe')); vacuum=bool(bandit.get('post_reset_vacuum_active') or risk.get('post_reset_vacuum_active'))
    if force:bw=0.
    rw=1-bw; cb=max(1e-6,min(1-1e-6,rw*ab+bw*bb)); tie=max(0.,min(.3,ap['T'])); mass=1-tie; banker=mass*cb; player=mass*(1-cb); ad=str(result.get('adaptive_only_direction') or result.get('action') or result.get('recommend') or '').upper(); ad=ad if ad in {'B','P'} else ('B' if ab>=.5 else 'P'); bd=str(bandit.get('selected_arm') or bandit.get('next_round_direction') or '').upper(); bd=bd if bd in {'B','P'} else ('B' if bb>=.5 else 'P'); final=ad if abs(cb-.5)<=1e-12 else ('B' if cb>.5 else 'P')
    bc=max(0.,min(1.,float(bandit.get('confidence_score',risk.get('confidence_score',0)) or 0))); confidence=rw*_conf(result)+bw*bc; confidence=max(confidence,min(.9,.45+.45*gate['confidence'])) if rule_active else confidence; confidence=min(confidence,.25 if force else .5) if vacuum else confidence
    try:bet=max(0.,min(1.,float(bandit.get('bet_multiplier',risk.get('bet_multiplier',1)) or 0)))
    except (TypeError,ValueError):bet=1.
    if force:bet=0.
    text='跟規律' if gate['decision']=='FOLLOW' else '斷規律' if gate['decision']=='BREAK' else '目前無確認規律'
    result.update({'model_version':f'FULL-ROAD-ADAPTIVE+CUSUM-RULE::{CUSUM_MODEL_VERSION}','model_variant':ADAPTIVE_MODEL_VARIANT,'decision_pipeline':'full_road_rule_detection_then_follow_break_gate_then_adaptive_cusum_fusion','probabilities':{'B':banker,'P':player,'T':tie},'banker_rate':round(banker*100,2),'player_rate':round(player*100,2),'tie_rate':round(tie*100,2),'adaptive_only_direction':ad,'bandit_only_direction':bd,'contextual_bandit_enabled':True,'contextual_bandit_update_enabled':True,'cusum_linucb_enabled':True,'rule_follow_break_enabled':True,'rule_follow_break_active':rule_active,'rule_decision':gate['decision'],'rule_decision_text':text,'rule_name':gate['rule_name'],'rule_direction':gate['rule_direction'],'rule_decision_direction':gate['decision_direction'],'rule_follow_probability':gate['follow_probability'],'rule_break_probability':gate['break_probability'],'rule_gate_confidence':gate['confidence'],'rule_gate_observations':gate['observations'],'ucb_influenced_final_direction':bw>0,'post_reset_vacuum_active':vacuum,'force_observe':force,'ensemble_confidence':confidence,'confidence':confidence,'quality_score':confidence,'bet_multiplier':bet,'cusum_bandit':bandit,'ensemble_scheduler':{'road_weight':rw,'cusum_linucb_weight':bw,'requested_cusum_weight':req,'max_cusum_weight':MAX_CUSUM_ENSEMBLE_WEIGHT,'rule_gate_active':rule_active,'rule_decision':gate['decision'],'rule_gate_confidence':gate['confidence'],'post_reset_vacuum_active':vacuum,'force_observe':force,'reason':'CUSUM 剛重置：暫停正式訊號' if force else f'確認規律：模型判斷{text}' if rule_active else '沒有成熟的跟規律/斷規律判斷，沿用 Adaptive + CUSUM 融合'}})
    adaptive=dict(result.get('adaptive_ensemble') or {}); adaptive.update({'active':True,'mode':'adaptive_road_plus_cusum_rule_follow_break','rule_follow_break_enabled':True,'rule_follow_break_active':rule_active,'rule_decision':gate['decision'],'cusum_linucb_weight':bw,'road_weight':rw,'overall_confidence':confidence,'bet_multiplier':bet}); result['adaptive_ensemble']=adaptive; final_text='莊' if final=='B' else '閒'
    if force:result.update({'recommend':'O','recommend_text':'觀望','action':'O','action_text':'觀望／CUSUM 重置探索期','internal_recommend':'O','internal_action':'O','next_round_direction':final,'next_round_direction_text':final_text,'signal_allowed':False,'signal_status_code':'CUSUM_POST_RESET_VACUUM_OBSERVE','signal_status_text':'CUSUM 重置後探索期','signal_reason':'變點已觸發硬重置；暫停讓舊規律直接影響正式方向。','direction_source':'cusum_post_reset_vacuum_observe','hard_brake_active':True,'is_extreme_unseen':True})
    else:result.update({'recommend':final,'recommend_text':final_text,'action':final,'action_text':final_text,'internal_recommend':final,'internal_action':final,'next_round_direction':final,'next_round_direction_text':final_text,'signal_allowed':True,'signal_status_code':'RULE_FOLLOW_BREAK_DECISION' if rule_active else 'ADAPTIVE_CUSUM_LINUCB_FUSION','signal_status_text':text if rule_active else 'Full Road Adaptive + CUSUM-LinUCB 動態融合','signal_reason':f"{gate['rule_name']} 規律方向 {gate['rule_direction']}；模型判斷{text}，下一手方向 {final_text}。" if rule_active else f'Adaptive 權重 {rw:.3f}；CUSUM-LinUCB 權重 {bw:.3f}。','direction_source':'rule_follow_break_gate' if rule_active else 'adaptive_cusum_linucb_fusion','hard_brake_active':False,'is_extreme_unseen':False})
    result['direction_edge']=abs(2*cb-1); result['direction_edge_percent']=round(result['direction_edge']*100,4); return result

class ShadowBacktestController:
    def __init__(self):self.shadow_buffer=[]
    @staticmethod
    def stream_key(user_id='',venue='',room='',shoe_id=''):return sha256('|'.join((str(user_id or '__anonymous__'),str(venue or '').upper().strip(),str(room or '').strip(),str(shoe_id or '').strip())).encode()).hexdigest()[:24]
    def apply(self,history,prediction,stream_key='__default__'):
        del stream_key; self.shadow_buffer=[x for x in history if x in {'B','P'}][-3:]; r=dict(prediction or {}); r.setdefault('shadow_buffer',list(self.shadow_buffer)); return r
ShortTermTakeoverController=ShadowBacktestController; _SHADOW_CONTROLLER=ShadowBacktestController(); _SHORT_TERM_CONTROLLER=_SHADOW_CONTROLLER

def predict(history=None,venue='',room='',shoe_id='',user_id='',run_seed=None,shoe_context=None,road_context=None):
    vals=[] if history is None else [p for p in history.replace('|',',').split(',') if p.strip()] if isinstance(history,str) else list(history); cleaned=_normalize_outcome_history(vals); road=dict(road_context or {}); road=road if isinstance(road.get('models'),Mapping) else build_road_context(cleaned,seed=run_seed); seed=_road_seed_prediction(road,cleaned); seed['road_support']=dict(road); seed['component_probabilities']=dict(road.get('component_probabilities') or {}); adaptive=adapt_prediction(seed,venue=venue,room=room,shoe_id=shoe_id); scope=_bandit_learning_scope(user_id,venue,room); bandit=predict_bandit(cleaned,road_context=road,venue=venue,room=room,user_id=scope,run_seed=run_seed); result=_fuse_adaptive_and_cusum(adaptive,bandit); mf=str(result.get('prediction_fingerprint') or ''); bf=str(bandit.get('prediction_fingerprint') or ''); result['model_prediction_fingerprint']=mf; result['prediction_fingerprint']=sha256('|'.join((mf,bf,str(venue).upper().strip(),str(room).strip(),str(shoe_id or '__unspecified_shoe__'))).encode()).hexdigest()[:24]; result.update({'shoe_id':str(shoe_id or ''),'bandit_learning_user_id':scope,'bandit_scope_mode':'user_venue_room_hashed','bandit_shoe_isolated':False,'shoe_event_isolated':True,'composition_quality':'not_applicable_road_cusum','remaining_counts_source':'not_used','shoe_context_ignored':bool(shoe_context),'road_quality_ok':bool(road.get('quality_ok',road.get('recognition_quality_ok',True))),'road_support':dict(road),'component_probabilities':dict(road.get('component_probabilities') or {}),'input_required':False,'probability_semantics':'direction_score_not_guaranteed_outcome_probability'}); return result

def run_virtual_round(session,run_seed=None):
    from particle_filter_points import counts_from_shoe,deal_ordered_hand
    shoe=[int(c) for c in list(session.get('virtual_shoe') or [])]
    if len(shoe)<6:raise ValueError('虛擬牌靴不足，請重新建立牌靴。')
    hist=_normalize_outcome_history(list(session.get('round_history') or [])); seed=int(run_seed if run_seed is not None else secrets.randbits(32))&0xffffffff; prediction=predict(hist,str(session.get('venue') or ''),str(session.get('room') or ''),str(session.get('shoe_id') or ''),str(session.get('user_id') or ''),seed); hand,remain=deal_ordered_hand(shoe); data=hand.as_dict(); side=str(prediction.get('action') or '').upper(); actual=str(hand.outcome or '').upper(); verdict='OBSERVE' if side=='O' else 'TIE_SKIPPED' if actual=='T' else 'HIT' if side==actual else 'MISS'; prediction.update({'ok':True,'mode':'virtual_shoe_rule_follow_break_compatibility','virtual_hand':data,'virtual_outcome':actual,'virtual_outcome_text':data['outcome_text'],'verdict':verdict,'verdict_text':{'HIT':'命中','MISS':'未命中','TIE_SKIPPED':'和局不計','OBSERVE':'觀望／不計勝負'}[verdict],'cards_consumed':int(hand.cards_used),'remaining_cards_after':len(remain),'remaining_counts_after':counts_from_shoe(remain),'round_number':int(session.get('hand_number',0) or 0)+1,'warmup_rounds':int(session.get('warmup_rounds',0) or 0),'bandit_learning_applied':True}); return {'prediction':prediction,'hand':data,'remaining_shoe':remain}
def parse_point_observation(value):del value; return None
__all__=['DB_HOLDOUT','ShadowBacktestController','ShortTermTakeoverController','parse_point_observation','predict','run_virtual_round']
