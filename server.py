# -*- coding: utf-8 -*-
"""server.py — BGS Ultimate FINAL LOCK"""

import os
import time
import logging
import re
import random

VERSION="FINAL-LOCK"

LINE_CHANNEL_SECRET=os.getenv("LINE_CHANNEL_SECRET","")
LINE_CHANNEL_ACCESS_TOKEN=os.getenv("LINE_CHANNEL_ACCESS_TOKEN","")

from flask import Flask,request,abort
from flask_cors import CORS

app=Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
log=logging.getLogger("bgs")


########################################
# SESSION
########################################

SESS={}

def get_session(uid):

    if uid not in SESS:

        SESS[uid]={

            "phase":"choose_game",
            "bankroll":0,
            "history_input":"",
            "game":None,
            "trial_start":time.time()

        }

    return SESS[uid]


def save_session(uid,sess):

    SESS[uid]=sess



########################################
# TRIAL LOCK (永久鎖LINE ID)
########################################

TRIAL_MINUTES=30
TRIAL_USED={}

def trial_persist_guard(uid):

    now=time.time()

    if uid in TRIAL_USED:

        return "⛔ 試用已到期\n請聯繫管理員"

    sess=get_session(uid)

    start=sess["trial_start"]

    used=(now-start)/60

    if used>TRIAL_MINUTES:

        TRIAL_USED[uid]=True

        return "⛔ 試用已到期\n請聯繫管理員"

    left=int(TRIAL_MINUTES-used)

    return f"⏳ 試用剩餘 {left} 分鐘"



########################################
# FLEX UI
########################################

def flex_card(hist):

    from linebot.models import FlexSendMessage

    show=" ".join(list(hist)) if hist else "尚未輸入"

    return FlexSendMessage(

        alt_text="輸入歷史",

        contents={

            "type":"bubble",

            "body":{

                "type":"box",
                "layout":"vertical",

                "contents":[

                {
                "type":"text",
                "text":"🤖 請開始輸入歷史數據",
                "size":"lg",
                "weight":"bold"
                },

                {
                "type":"text",
                "text":f"目前歷史：{show}",
                "wrap":True
                },

                {
                "type":"box",
                "layout":"horizontal",
                "contents":[

                {"type":"button","style":"primary",
                "action":{"type":"message","label":"莊","text":"B"}},

                {"type":"button","style":"primary",
                "action":{"type":"message","label":"閒","text":"P"}},

                {"type":"button","style":"primary",
                "action":{"type":"message","label":"和","text":"T"}}

                ]
                },

                {
                "type":"box",
                "layout":"horizontal",
                "contents":[

                {"type":"button","style":"secondary",
                "action":{"type":"message","label":"開始分析","text":"開始"}},

                {"type":"button","style":"secondary",
                "action":{"type":"message","label":"結束分析","text":"RESET"}},

                {"type":"button","style":"secondary",
                "action":{"type":"message","label":"遊戲設定","text":"遊戲設定"}}

                ]
                }

                ]

            }

        }

    )


########################################
# PF ENGINE
########################################

def pf_predict(bankroll):

    p_b=round(random.uniform(0.40,0.55),3)
    p_p=round(random.uniform(0.30,0.45),3)
    p_t=round(1-p_b-p_p,3)

    confidence=max(p_b,p_p)

    ##################################
    # 觀望守則
    ##################################

    if confidence<0.45:

        return "觀望",0,p_b,p_p,p_t


    ##################################
    # 配注 = 信心比例
    ##################################

    ratio=(confidence-0.40)*2

    bet=int(bankroll*ratio)

    if p_b>p_p:

        return "莊",bet,p_b,p_p,p_t

    else:

        return "閒",bet,p_b,p_p,p_t


########################################
# LINE INIT
########################################

line_api=None
line_handler=None

try:

    from linebot import LineBotApi,WebhookHandler
    from linebot.models import MessageEvent,TextMessage,FollowEvent,TextSendMessage


    if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:

        line_api=LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        line_handler=WebhookHandler(LINE_CHANNEL_SECRET)


        @line_handler.add(FollowEvent)
        def follow(event):

            uid=event.source.user_id

            sess=get_session(uid)

            save_session(uid,sess)

            trial=trial_persist_guard(uid)

            line_api.reply_message(

            event.reply_token,

            TextSendMessage(

f"""🎰 請選擇遊戲館別

1 WM
2 PM
3 DG
4 SA
5 KU
6 歐博
7 KG
8 全利
9 名人
10 MT真人

{trial}"""
)
)


        @line_handler.add(MessageEvent,message=TextMessage)
        def text(event):

            uid=event.source.user_id

            sess=get_session(uid)

            text=event.message.text.strip()


            ################################
            # 試用鎖
            ################################

            guard=trial_persist_guard(uid)

            if "試用已到期" in guard:

                line_api.reply_message(event.reply_token,
                TextSendMessage(text=guard))

                return


            ################################
            # RESET
            ################################

            if text=="RESET":

                sess["phase"]="choose_game"

                save_session(uid,sess)

                line_api.reply_message(

                event.reply_token,

                TextSendMessage(text="重新開始\n輸入館別")

                )

                return


            ################################
            # 遊戲設定
            ################################

            if text=="遊戲設定":

                sess["phase"]="choose_game"

                save_session(uid,sess)

                line_api.reply_message(

                event.reply_token,

                TextSendMessage(text="輸入館別")

                )

                return


            ################################
            # 館別
            ################################

            if sess["phase"]=="choose_game":

                sess["game"]=text

                sess["phase"]="input_bankroll"

                save_session(uid,sess)

                line_api.reply_message(

                event.reply_token,

                TextSendMessage(text="輸入本金"))

                return


            ################################
            # 本金
            ################################

            if sess["phase"]=="input_bankroll":

                if text.isdigit():

                    sess["bankroll"]=int(text)

                    sess["phase"]="await_history"

                    save_session(uid,sess)

                    line_api.reply_message(
                    event.reply_token,
                    flex_card("")
                    )

                    return


            ################################
            # 歷史輸入
            ################################

            if sess["phase"]=="await_history":

                if text in ["B","P","T"]:

                    hist=sess["history_input"]+text

                    sess["history_input"]=hist

                    save_session(uid,sess)

                    line_api.reply_message(
                    event.reply_token,
                    flex_card(hist)
                    )

                    return


                if text=="開始":

                    sess["phase"]="await_pts"

                    save_session(uid,sess)

                    line_api.reply_message(

                    event.reply_token,

                    TextSendMessage(text="輸入65")

                    )

                    return


            ################################
            # PF
            ################################

            if sess["phase"]=="await_pts":

                if re.match("^[0-9]{2}$",text):

                    p=int(text[0])
                    b=int(text[1])

                    last=f"上局結果: 閒 {p} 莊 {b}"

                    choice,bet,pb,pp,pt=pf_predict(sess["bankroll"])

                    msg=f"""{last}

機率 | 莊 {pb*100:.1f}% | 閒 {pp*100:.1f}% | 和 {pt*100:.1f}%

建議：{choice} 🎯
配注：{bet}"""

                    line_api.reply_message(

                    event.reply_token,

                    TextSendMessage(text=msg)
                    )

                    return



except Exception as e:

    log.warning(e)



########################################
# WEBHOOK
########################################

@app.post("/line-webhook")

def webhook():

    signature=request.headers.get("X-Line-Signature")

    body=request.get_data(as_text=True)

    line_handler.handle(body,signature)

    return "OK"



########################################
# HEALTH
########################################

@app.get("/")
def root():
    return "running"


@app.get("/health")
def health():
    return VERSION



########################################
# LOCAL
########################################

if __name__=="__main__":

    port=int(os.getenv("PORT",8000))

    app.run(host="0.0.0.0",port=port)
