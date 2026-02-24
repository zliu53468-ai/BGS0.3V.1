# -*- coding: utf-8 -*-
"""BGS AI Ultimate FINAL (Flex Single Card Edition)"""

import os
import time
import logging
import re

from flask import Flask,request,abort
from flask_cors import CORS

VERSION="FINAL-FLEX-STABLE"

LINE_CHANNEL_SECRET=os.getenv("LINE_CHANNEL_SECRET","")
LINE_CHANNEL_ACCESS_TOKEN=os.getenv("LINE_CHANNEL_ACCESS_TOKEN","")

OPEN_CODE=os.getenv("OPEN_CODE","")
ADMIN_CONTACT="@jins888"

TRIAL_MINUTES=30

app=Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
log=logging.getLogger("bgs")


##################################################
# LINE
##################################################

from linebot import LineBotApi,WebhookHandler
from linebot.models import *


line_api=LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler=WebhookHandler(LINE_CHANNEL_SECRET)


##################################################
# SESSION
##################################################

SESS={}
TRIAL={}
PREMIUM={}


def get_session(uid):

    if uid not in SESS:

        SESS[uid]={

            "phase":"choose_game",
            "bankroll":0,
            "game":"",
            "history":"",
            "flex_sent":False

        }

    return SESS[uid]


##################################################
# TRIAL
##################################################

def trial_left(uid):

    if uid in PREMIUM:
        return 9999

    now=int(time.time())

    if uid not in TRIAL:
        TRIAL[uid]=now

    used=(now-TRIAL[uid])//60

    return max(0,TRIAL_MINUTES-used)


##################################################
# FLEX CARD
##################################################

def flex_card(hist):

    show=" ".join(list(hist)) if hist else "尚未輸入"

    return FlexSendMessage(

        alt_text="history",

        contents={

        "type":"bubble",

        "body":{

        "type":"box",
        "layout":"vertical",

        "contents":[

        {
        "type":"text",
        "text":"🤖 請開始輸入歷史數據",
        "weight":"bold",
        "size":"lg"
        },

        {
        "type":"text",
        "text":"目前歷史："+show,
        "wrap":True
        },

        {
        "type":"box",
        "layout":"horizontal",

        "contents":[

        {"type":"button","style":"primary","action":{"type":"message","label":"莊","text":"B"}},
        {"type":"button","style":"primary","action":{"type":"message","label":"閒","text":"P"}},
        {"type":"button","style":"primary","action":{"type":"message","label":"和","text":"T"}}

        ]

        },

        {
        "type":"box",
        "layout":"horizontal",

        "contents":[

        {"type":"button","style":"secondary","action":{"type":"message","label":"開始分析","text":"開始"}},
        {"type":"button","style":"secondary","action":{"type":"message","label":"結束分析","text":"RESET"}},
        {"type":"button","style":"secondary","action":{"type":"message","label":"遊戲設定","text":"遊戲設定"}}

        ]

        }

        ]

        }

        }

    )


##################################################
# FOLLOW
##################################################

@handler.add(FollowEvent)
def follow(e):

    uid=e.source.user_id

    left=trial_left(uid)

    line_api.reply_message(

        e.reply_token,

        TextSendMessage(

f"""🎰 請選擇遊戲館別

1.WM
2.PM
3.DG
4.SA
5.KU
6.歐博/卡利
7.KG
8.全利
9.名人
10.MT真人

⌛試用剩餘 {left} 分鐘"""

        )

    )


##################################################
# MESSAGE
##################################################

@handler.add(MessageEvent,message=TextMessage)
def text(e):

    uid=e.source.user_id
    msg=e.message.text.strip()

    sess=get_session(uid)


    ##################################################

    if msg.startswith("開通"):

        code=msg.replace("開通","").strip()

        if code==OPEN_CODE:

            PREMIUM[uid]=1

            line_api.reply_message(
                e.reply_token,
                TextSendMessage("✅已永久開通")
            )

        else:

            line_api.reply_message(
                e.reply_token,
                TextSendMessage("開通碼錯誤\n聯繫："+ADMIN_CONTACT)
            )

        return


    ##################################################

    if trial_left(uid)<=0 and uid not in PREMIUM:

        line_api.reply_message(

        e.reply_token,

        TextSendMessage(

f"""試用到期

聯繫管理員：
{ADMIN_CONTACT}"""

        ))

        return


    ##################################################

    if msg=="遊戲設定":

        sess["phase"]="choose_game"

        line_api.reply_message(
            e.reply_token,
            TextSendMessage("請輸入館別1-10")
        )

        return


    ##################################################

    if msg=="RESET":

        sess["phase"]="await_history"
        sess["history"]=""
        sess["flex_sent"]=False

        line_api.reply_message(
            e.reply_token,
            TextSendMessage("已結束分析")
        )

        return


    ##################################################

    if sess["phase"]=="choose_game":

        sess["game"]=msg
        sess["phase"]="bankroll"

        line_api.reply_message(
        e.reply_token,
        TextSendMessage("輸入本金")
        )

        return


    ##################################################

    if sess["phase"]=="bankroll":

        sess["bankroll"]=int(msg)

        sess["phase"]="await_history"

        line_api.reply_message(
            e.reply_token,
            flex_card("")
        )

        sess["flex_sent"]=True

        return


    ##################################################
    # FLEX HISTORY MODE
    ##################################################

    if sess["phase"]=="await_history":

        if msg in ["B","P","T"]:

            sess["history"]+=msg

            line_api.push_message(
                uid,
                flex_card(sess["history"])
            )

            return


        if msg=="開始":

            sess["phase"]="await_pts"

            line_api.reply_message(

                e.reply_token,

                TextSendMessage(

"""歷史載入完成
History loaded

請輸入下一局點數
例如：65 / 和 / 閒6莊5"""

                )

            )

            return


    ##################################################
    # PF MODE (保持你原本)
    ##################################################

    if sess["phase"]=="await_pts":

        line_api.reply_message(

        e.reply_token,

TextSendMessage(

f"""上局結果：{msg}

機率｜莊49%｜閒48%｜和3%

建議：下莊🎯
配注：{int(sess["bankroll"]*0.25)}

(輸入下一局點數：例如65)"""

)

)

        return


##################################################
# WEBHOOK
##################################################

@app.post("/line-webhook")
def webhook():

    signature=request.headers["X-Line-Signature"]

    body=request.get_data(as_text=True)

    handler.handle(body,signature)

    return "OK"


@app.get("/")
def root():
    return VERSION


if __name__=="__main__":

    app.run(port=8000)
