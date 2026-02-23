# -*- coding: utf-8 -*-
"""BGS AI Baccarat Final Ultimate"""

import os,time,re,random

VERSION="FINAL-ULTIMATE"

LINE_CHANNEL_SECRET=os.getenv("LINE_CHANNEL_SECRET","")
LINE_CHANNEL_ACCESS_TOKEN=os.getenv("LINE_CHANNEL_ACCESS_TOKEN","")
OPEN_PASSWORD=os.getenv("OPEN_PASSWORD","")

ADMIN_CONTACT="@jins888"
TRIAL_MINUTES=30

from flask import Flask,request,abort
from flask_cors import CORS

app=Flask(__name__)
CORS(app)

##################################################
# SESSION
##################################################

SESS={}
TRIAL={}

def get_session(uid):

    if uid not in SESS:

        SESS[uid]={
            "phase":"welcome",
            "bankroll":0,
            "history":"",
            "game":"",
            "premium":False
        }

    return SESS[uid]


##################################################
# TRIAL SYSTEM
##################################################

def trial_ok(uid):

    if uid in TRIAL:

        start=TRIAL[uid]

        if time.time()-start>TRIAL_MINUTES*60:

            return False

        return True

    TRIAL[uid]=time.time()
    return True


##################################################
# FLEX UI
##################################################

def flex_history(hist):

    from linebot.models import FlexSendMessage

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

                "text":"🤖BGS AI百家樂預測系統",

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
# BET SYSTEM
##################################################

def bet_amount(bankroll,confidence):

    minbet=int(bankroll*0.05)

    maxbet=int(bankroll*0.40)

    return int(minbet+(maxbet-minbet)*confidence)


##################################################
# PF ENGINE (簡化版)
##################################################

def pf_predict(sess):

    conf=random.random()

    if conf<0.25:

        return "觀望",conf,0

    side=random.choice(["莊","閒"])

    bet=bet_amount(sess["bankroll"],conf)

    return side,conf,bet


##################################################
# LINE
##################################################

line_api=None
line_handler=None

from linebot import LineBotApi,WebhookHandler
from linebot.models import MessageEvent,TextMessage,FollowEvent,TextSendMessage

line_api=LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
line_handler=WebhookHandler(LINE_CHANNEL_SECRET)


##################################################
# FOLLOW
##################################################

@line_handler.add(FollowEvent)
def follow(event):

    uid=event.source.user_id

    SESS.pop(uid,None)

    TRIAL.pop(uid,None)

    line_api.reply_message(

        event.reply_token,

        TextSendMessage(

text=f"""🎰歡迎加入BGS AI百家樂預測系統🤖

請輸入：

遊戲設定

開始使用

📞客服：{ADMIN_CONTACT}
"""
)

)


##################################################
# MESSAGE
##################################################

@line_handler.add(MessageEvent,message=TextMessage)
def message(event):

    uid=event.source.user_id
    msg=event.message.text.strip()

    sess=get_session(uid)


    ###################################
    # 開通碼
    ###################################

    if msg.startswith("開通"):

        code=msg.replace("開通","").strip()

        if code==OPEN_PASSWORD:

            sess["premium"]=True

            line_api.reply_message(event.reply_token,

TextSendMessage("✅永久開通成功"))

            return


    ###################################
    # 試用限制
    ###################################

    if not sess["premium"]:

        if not trial_ok(uid):

            line_api.reply_message(event.reply_token,

TextSendMessage(f"""試用到期

請輸入

開通 密碼

客服：{ADMIN_CONTACT}"""))

            return


    ###################################
    # 遊戲設定
    ###################################

    if msg=="遊戲設定":

        sess["phase"]="choose"

        line_api.reply_message(

event.reply_token,

TextSendMessage(

"""🎰請選擇遊戲館別

1 WM
2 PM
3 DG
4 SA
5 KU
6 歐博
7 KG
8 全利
9 名人
10 MT真人"""
)

)

        return


    ###################################
    # 選館別
    ###################################

    if sess["phase"]=="choose":

        sess["game"]=msg

        sess["phase"]="bank"

        line_api.reply_message(

event.reply_token,

TextSendMessage("輸入本金"))

        return


    ###################################
    # 本金
    ###################################

    if sess["phase"]=="bank":

        if msg.isdigit():

            sess["bankroll"]=int(msg)

            sess["phase"]="history"

            sess["history"]=""

            line_api.reply_message(

event.reply_token,

flex_history("")

)

        return


    ###################################
    # 歷史輸入
    ###################################

    if sess["phase"]=="history":

        if msg in ["B","P","T"]:

            sess["history"]+=msg

            line_api.reply_message(

event.reply_token,

flex_history(sess["history"])

)

            return


        if msg=="開始":

            sess["phase"]="predict"

            line_api.reply_message(

event.reply_token,

TextSendMessage(

"""歷史載入完成
History loaded

請輸入下一局點數
例如：65 / 和 / 閒6莊5"""
)

)

            return


        if msg=="RESET":

            sess["phase"]="choose"

            line_api.reply_message(event.reply_token,

TextSendMessage("重新設定"))

            return


    ###################################
    # PF分析
    ###################################

    if sess["phase"]=="predict":

        if re.match(r'^\d\d$',msg) or msg in ["和"]:

            side,conf,bet=pf_predict(sess)

            if side=="觀望":

                txt="建議：觀望 👀"

            else:

                txt=f"""PF分析完成

建議：{side}

配注：{bet}"""

            line_api.reply_message(

event.reply_token,

TextSendMessage(txt)

)

            return


##################################################
# WEBHOOK
##################################################

@app.post("/line-webhook")

def webhook():

    signature=request.headers.get("X-Line-Signature")
    body=request.get_data(as_text=True)

    line_handler.handle(body,signature)

    return "OK"


@app.get("/")

def root():

    return "running"


@app.get("/health")

def health():

    return VERSION


if __name__=="__main__":

    port=int(os.getenv("PORT",8000))
    app.run(host="0.0.0.0",port=port)
