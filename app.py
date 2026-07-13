"""LINE point-input baccarat bot with venue/room/trial panels."""
from __future__ import annotations
import base64, hashlib, hmac, json, os, re, traceback, urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
import store
from predictor import predict, parse_point_observation, reset_uid_model

BASE_DIR = Path(__file__).resolve().parent
CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN','').strip()
CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET','').strip()
TAIPEI_TZ = timezone(timedelta(hours=8))
TRIAL_MINUTES = int(os.getenv('TRIAL_MINUTES','30'))
ADMIN_LINE_URL = os.getenv('ADMIN_LINE_URL','https://line.me/R/ti/p/%40jins888')
ACCESS_FILE = Path(os.getenv('ACCESS_DATA_FILE', str(BASE_DIR/'data'/'access_control.json')))
VENUES = [('OB','歐博真人'),('DG','DG真人'),('MT','MT真人'),('T9','T9真人'),('SA','SA真人'),('DB','DB真人')]
PERMANENT_CODES={'aaa1688003','aaa1888007','aaa1000889'}
MONTHLY_CODES={'aaa13002','aaa15001','aaa199801'}
TEMP_CODES={'aaaa1999152','aaa345556','aaa987743'}
ALL_CODES=PERMANENT_CODES|MONTHLY_CODES|TEMP_CODES
app=FastAPI(title='Baccarat Point Particle Filter Bot',version='1.0.0')

def now(): return datetime.now(TAIPEI_TZ)
def iso(x): return x.astimezone(TAIPEI_TZ).isoformat(timespec='seconds')
def parse_dt(v):
    try:
        d=datetime.fromisoformat(str(v)); return d if d.tzinfo else d.replace(tzinfo=TAIPEI_TZ)
    except Exception:return None

def load_access():
    try:
        if not ACCESS_FILE.exists(): return {}
        with ACCESS_FILE.open('r',encoding='utf-8') as f: return json.load(f)
    except Exception:return {}
def save_access(d):
    ACCESS_FILE.parent.mkdir(parents=True,exist_ok=True); t=ACCESS_FILE.with_suffix('.tmp')
    with t.open('w',encoding='utf-8') as f: json.dump(d,f,ensure_ascii=False,indent=2)
    t.replace(ACCESS_FILE)
def status(uid):
    r=load_access().get(uid,{})
    if r.get('permanent'): return {'active':True,'label':'永久版','remaining':None}
    exp=parse_dt(r.get('access_expires_at') or r.get('trial_expires_at'))
    if exp and exp>now(): return {'active':True,'label':r.get('plan','試用'),'remaining':int((exp-now()).total_seconds())}
    if r.get('used_trial'): return {'active':False,'expired':True,'label':'已到期','remaining':0}
    return {'active':False,'trial_available':True,'label':'尚未開始試用','remaining':TRIAL_MINUTES*60}
def ensure(uid):
    s=status(uid)
    if s.get('active'): return s
    if s.get('trial_available'):
        d=load_access(); r=d.get(uid,{})|{'used_trial':True,'plan':'trial','trial_expires_at':iso(now()+timedelta(minutes=TRIAL_MINUTES))}; d[uid]=r; save_access(d); return status(uid)
    raise PermissionError('expired')
def activate(uid,code):
    d=load_access(); r=d.get(uid,{})
    if code in PERMANENT_CODES:r|={'permanent':True,'used_trial':True,'plan':'permanent'}
    elif code in MONTHLY_CODES:r|={'permanent':False,'used_trial':True,'plan':'monthly','access_expires_at':iso(now()+timedelta(days=30))}
    elif code in TEMP_CODES:r|={'permanent':False,'used_trial':True,'plan':'temporary','access_expires_at':iso(now()+timedelta(minutes=30))}
    else: raise ValueError('開通碼錯誤')
    d[uid]=r; save_access(d); return status(uid)
def remain(v):
    if v is None:return '永久'
    m=max(0,int(v))//60; return f'{m}分鐘'

def verify(body,sig):
    if not CHANNEL_SECRET:return True
    if not sig:return False
    ex=base64.b64encode(hmac.new(CHANNEL_SECRET.encode(),body,hashlib.sha256).digest()).decode(); return hmac.compare_digest(ex,sig)
def reply(token,msgs):
    if not CHANNEL_ACCESS_TOKEN: print(json.dumps(msgs,ensure_ascii=False)); return
    requests.post('https://api.line.me/v2/bot/message/reply',headers={'Authorization':f'Bearer {CHANNEL_ACCESS_TOKEN}','Content-Type':'application/json'},json={'replyToken':token,'messages':msgs[:5]},timeout=8)
def txt(t):return {'type':'text','text':str(t)[:5000]}
def action(label,act,**kw):return {'type':'button','style':'primary','height':'sm','action':{'type':'postback','label':label,'data':urllib.parse.urlencode({'action':act,**kw})}}
def flex(title,body,buttons=None):
    c=[{'type':'text','text':title,'weight':'bold','size':'xl','color':'#FFD000'},{'type':'text','text':body,'wrap':True,'margin':'md','color':'#FFFFFF'}]
    if buttons:c.append({'type':'box','layout':'vertical','spacing':'sm','margin':'lg','contents':buttons})
    return {'type':'flex','altText':title,'contents':{'type':'bubble','size':'mega','body':{'type':'box','layout':'vertical','backgroundColor':'#111111','paddingAll':'18px','contents':c}}}
def venue_panel(uid):return flex('AI 點數粒子模型',f'UID權限：{status(uid)["label"]}｜請選擇遊戲館。',[action(name,'venue',venue=code) for code,name in VENUES])
def room_panel(code):return flex('請輸入房間',f'已選擇 {dict(VENUES).get(code,code)}。請直接輸入房間名稱或桌號。')
def point_panel(uid,s,notice=''):
    obs=s.get('observations') or []; last='、'.join([f'閒{x["player"]}莊{x["banker"]}' for x in obs[-8:]]) or '尚無'
    body=f'{notice}\n館別：{s.get("venue") or "-"}｜房間：{s.get("room") or "-"}\n已輸入 {len(obs)} 局\n最近：{last}\n\n請輸入：閒6莊5（也支援 P6B5 或 6,5）'
    return flex('輸入閒莊點數',body,[action('開始AI判斷','predict'),action('上一步','undo'),action('清除本靴','reset'),action('結束分析','end')])
def result_panel(uid,s):
    p=s.get('last_prediction') or {}
    body=f'第 {len(s.get("observations") or [])+1} 局\n莊 {p.get("banker_rate",0):.1f}%\n閒 {p.get("player_rate",0):.1f}%\n和 {p.get("tie_rate",0):.1f}%\n\n推薦：{p.get("recommend_text","-")}\n{p.get("signal_level","")}｜{p.get("reason","")}'
    return flex('下一局點數模擬',body,[action('繼續輸入點數','panel'),action('結束分析','end')])
def expired():return {'type':'flex','altText':'試用已到期','contents':{'type':'bubble','body':{'type':'box','layout':'vertical','backgroundColor':'#111111','paddingAll':'18px','contents':[{'type':'text','text':'試用已到期','weight':'bold','size':'xl','color':'#FFD000'},{'type':'button','style':'primary','color':'#06C755','margin':'lg','action':{'type':'uri','label':'聯繫管理員','uri':ADMIN_LINE_URL}}]}}}

def predict_session(uid):
    s=store.get_session(uid) or store.new_session(uid); ensure(uid)
    p=predict(s.get('observations') or [],venue=s.get('venue',''),room=s.get('room',''),shoe_id=s.get('shoe_id',''),user_id=uid)
    s['last_prediction']=p; return store.upsert_session(uid,s)

@app.api_route('/',methods=['GET','HEAD'])
def root():return PlainTextResponse('OK')
@app.api_route('/health',methods=['GET','HEAD'])
def health():return JSONResponse({'ok':True,'version':'point-pf-1.0'})
@app.post('/webhook')
async def webhook(request:Request):
    body=await request.body()
    if not verify(body,request.headers.get('X-Line-Signature')):return JSONResponse({'ok':False},status_code=401)
    payload=json.loads(body.decode() or '{}')
    for e in payload.get('events',[]):
        token=e.get('replyToken',''); src=e.get('source') or {}; uid=src.get('userId') or src.get('groupId') or 'anonymous'
        try:
            if e.get('type')=='follow': reply(token,[venue_panel(uid)]); continue
            if e.get('type')=='message' and (e.get('message') or {}).get('type')=='text':
                text=str((e.get('message') or {}).get('text') or '').strip()
                if text in ALL_CODES: activate(uid,text); reply(token,[txt('✅ 開通成功'),venue_panel(uid)]); continue
                if text in {'開始','開始分析','選館'}: reply(token,[venue_panel(uid)]); continue
                s=store.get_session(uid) or store.new_session(uid)
                if s.get('venue') and not s.get('room'):
                    s['room']=text; s=store.upsert_session(uid,s); reply(token,[point_panel(uid,s,'已設定房間')]); continue
                ob=parse_point_observation(text)
                if ob:
                    try: ensure(uid)
                    except PermissionError: reply(token,[expired()]); continue
                    s=store.add_point_observation(uid,ob['player'],ob['banker']); reply(token,[point_panel(uid,s,'已新增點數')]); continue
                if text in {'預測','AI','開始AI判斷'}:
                    try:s=predict_session(uid); reply(token,[result_panel(uid,s)])
                    except PermissionError:reply(token,[expired()])
                    continue
                reply(token,[txt('請輸入「開始分析」，或輸入例如：閒6莊5')]); continue
            if e.get('type')=='postback':
                q={k:v[0] for k,v in urllib.parse.parse_qs((e.get('postback') or {}).get('data','')).items()}; a=q.get('action')
                if a=='venue':
                    s=store.get_session(uid) or store.new_session(uid); s.update({'venue':q.get('venue',''),'room':''}); store.upsert_session(uid,s); reply(token,[room_panel(q.get('venue',''))])
                elif a=='predict':
                    try:s=predict_session(uid); reply(token,[result_panel(uid,s)])
                    except PermissionError:reply(token,[expired()])
                elif a=='panel':reply(token,[point_panel(uid,store.get_session(uid) or store.new_session(uid))])
                elif a=='undo':
                    s=store.undo_round(uid); reset_uid_model(uid,s.get('venue',''),s.get('room',''),s.get('shoe_id','')); reply(token,[point_panel(uid,s,'已刪除上一局')])
                elif a=='reset':
                    old=store.get_session(uid) or {}; reset_uid_model(uid,old.get('venue',''),old.get('room',''),old.get('shoe_id','')); s=store.clear_history(uid); reply(token,[point_panel(uid,s,'已清除本靴')])
                elif a=='end':reply(token,[flex('本靴分析已結束','需要下一靴時請輸入「開始分析」。')])
        except Exception as ex:
            traceback.print_exc(); reply(token,[txt(f'操作失敗：{ex}')])
    return JSONResponse({'ok':True})
@app.post('/callback')
async def callback(request:Request):return await webhook(request)
