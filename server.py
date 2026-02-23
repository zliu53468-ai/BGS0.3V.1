# -*- coding: utf-8 -*-
"""server.py — BGS Ultimate Final Production Stable"""

import os
import time
import logging
import re
from typing import Optional, Dict, Any

VERSION = "ultimate-final-stable"

LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")

from flask import Flask, request, abort
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bgs")

##################################################
# DEDUPE SYSTEM
##################################################

DEDUPE_TTL = 60
DEDUPE_CACHE = {}

def _dedupe_event(event_id):

    if not event_id:
        return True

    now = int(time.time())

    if event_id in DEDUPE_CACHE:
        if now - DEDUPE_CACHE[event_id] < DEDUPE_TTL:
            return False

    DEDUPE_CACHE[event_id] = now
    return True


def _extract_line_event_id(event):

    try:
        return event.message.id
    except:
        return None


##################################################
# SESSION SYSTEM
##################################################

SESS = {}

def get_session(uid):

    if uid not in SESS:

        SESS[uid] = {

            "phase": "choose_game",
            "bankroll": 0,
            "history_input": "",
            "game": None

        }

    return SESS[uid]


def save_session(uid,sess):

    SESS[uid] = sess


##################################################
# TRIAL STUB
##################################################

def trial_persist_guard(uid):
    return None

def is_premium(uid):
    return False

def set_premium(uid,flag=True):
    pass

TRIAL_MINUTES = 30
ADMIN_CONTACT = "-"


##################################################
# PF STUB
##################################################

def reset_pf_for_uid(uid):
    pass


##################################################
# QUICK REPLY STUB
##################################################

def quick_predict():
    return None


##################################################
# REPLY SYSTEM
##################################################

def _reply(api,token,text):

    from linebot.models import TextSendMessage

    api.reply_message(
        token,
        TextSendMessage(text=text)
    )


##################################################
# FLEX UI
##################################################

def flex_history_card():

    from linebot.models import FlexSendMessage

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
                        "text":"先輸入莊/閒/和；按開始分析",
                        "wrap":True
                    },

                    {
                        "type":"box",
                        "layout":"horizontal",
                        "contents":[

                            {
                                "type":"button",
                                "style":"primary",
                                "action":{
                                    "type":"message",
                                    "label":"莊",
                                    "text":"B"
                                }
                            },

                            {
                                "type":"button",
                                "style":"primary",
                                "action":{
                                    "type":"message",
                                    "label":"閒",
                                    "text":"P"
                                }
                            },

                            {
                                "type":"button",
                                "style":"primary",
                                "action":{
                                    "type":"message",
                                    "label":"和",
                                    "text":"T"
                                }
                            }

                        ]

                    },

                    {
                        "type":"box",
                        "layout":"horizontal",
                        "contents":[

                            {
                                "type":"button",
                                "style":"secondary",
                                "action":{
                                    "type":"message",
                                    "label":"開始分析",
                                    "text":"開始"
                                }
                            },

                            {
                                "type":"button",
                                "style":"secondary",
                                "action":{
                                    "type":"message",
                                    "label":"結束分析",
                                    "text":"RESET"
                                }
                            },

                            {
                                "type":"button",
                                "style":"secondary",
                                "action":{
                                    "type":"message",
                                    "label":"遊戲設定",
                                    "text":"遊戲設定"
                                }
                            }

                        ]

                    }

                ]

            }

        }

    )


def flex_history_card_with_history(hist):

    from linebot.models import FlexSendMessage

    show = " ".join(list(hist)) if hist else "尚未輸入"

    card = flex_history_card()

    card.contents["body"]["contents"].insert(

        1,

        {
            "type":"text",
            "text":f"目前歷史：{show}",
            "wrap":True
        }

    )

    return card


##################################################
# LINE INIT
##################################################

line_api=None
line_handler=None

try:

    from linebot import LineBotApi,WebhookHandler
    from linebot.models import MessageEvent,TextMessage,FollowEvent,FlexSendMessage

    if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:

        line_api=LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        line_handler=WebhookHandler(LINE_CHANNEL_SECRET)


        @line_handler.add(FollowEvent)
        def follow(event):

            uid=event.source.user_id

            sess=get_session(uid)

            sess["phase"]="choose_game"

            save_session(uid,sess)

            _reply(line_api,event.reply_token,

"""🎰 請選擇遊戲館別

1 WM
2 PM
3 DG
4 SA
5 KU
6 歐博
7 KG
8 全利
9 名人
10 MT真人""")



        @line_handler.add(MessageEvent,message=TextMessage)
        def text(event):

            uid=event.source.user_id

            if not _dedupe_event(_extract_line_event_id(event)):
                return

            text=event.message.text.strip()

            sess=get_session(uid)


            #################################

            if sess["phase"]=="choose_game":

                sess["game"]=text
                sess["phase"]="input_bankroll"

                save_session(uid,sess)

                _reply(line_api,event.reply_token,"輸入籌碼")

                return


            #################################

            if sess["phase"]=="input_bankroll":

                if text.isdigit():

                    sess["bankroll"]=int(text)
                    sess["phase"]="await_history"

                    save_session(uid,sess)

                    line_api.reply_message(

                        event.reply_token,

                        flex_history_card()

                    )

                    return


            #################################

            if sess["phase"]=="await_history":

                if text in ["B","P","T"]:

                    hist=sess["history_input"]

                    hist+=text

                    sess["history_input"]=hist

                    save_session(uid,sess)

                    line_api.reply_message(
                        event.reply_token,
                        flex_history_card_with_history(hist)
                    )

                    return


                if text=="開始":

                    sess["phase"]="await_pts"

                    save_session(uid,sess)

                    _reply(line_api,event.reply_token,
                           "已開始分析 請輸入65")

                    return


except Exception as e:

    log.warning(e)


##################################################
# WEBHOOK
##################################################

@app.post("/line-webhook")

def webhook():

    signature=request.headers.get("X-Line-Signature")

    body=request.get_data(as_text=True)

    if line_handler is None:
        abort(400)

    line_handler.handle(body,signature)

    return "OK"



##################################################
# HEALTH
##################################################

@app.get("/")

def root():

    return "running"


@app.get("/health")

def health():

    return VERSION


##################################################
# LOCAL
##################################################

if __name__=="__main__":

    port=int(os.getenv("PORT",8000))

    app.run(host="0.0.0.0",port=port)
