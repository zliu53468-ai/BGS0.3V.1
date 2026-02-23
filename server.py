# -*- coding: utf-8 -*-
"""BGS FINAL LOCK VERSION"""

import os
import time
import logging
import re

VERSION="FINAL-LOCK"

LINE_CHANNEL_SECRET=os.getenv("LINE_CHANNEL_SECRET","")
LINE_CHANNEL_ACCESS_TOKEN=os.getenv("LINE_CHANNEL_ACCESS_TOKEN","")

# 開通碼使用 Render 環境變數
OPEN_PASSWORD=os.getenv("OPEN_PASSWORD","")

# 官方客服
ADMIN_CONTACT="@jins888"

TRIAL_MINUTES=30

from flask import Flask,request,abort
from flask_cors import CORS

app=Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
log=logging.getLogger("bgs")

##################################################
# SESSION
##################################################

SESS={}
TRIAL_DB={}
PREMIUM_DB={}

def get_session(uid):

    if uid not in SESS:

        SESS[uid]={

            "phase":"idle",
            "game":None,
            "bankroll":0,
            "history_input":""

        }

    return SESS[uid]


##################################################
# 試用系統（LINE ID永久鎖）
##################################################

def trial_left(uid):

    if uid in PREMIUM_DB:
        return 9999

    now=int(time.time())

    if uid not in TRIAL_DB:
        TRIAL_DB[uid]=now

    used=(now-TRIAL_DB[uid])//60

    return max(0,TRIAL_MINUTES-used)


def trial_guard(uid):

    if uid in PREMIUM_DB:
        return None

    left=trial_left(uid)

    if left<=0:

        return f"""⛔試用已到期

🔐輸入：
開通 密碼

📞聯繫官方LINE：
{ADMIN_CONTACT}
"""

    return None


##################################################
# Dedup
##################################################

DEDUP={}

def dedupe(eid):

    if not eid:
        return True

    now=time.time()

    if eid in DEDUP:

        if now-DEDUP[eid]<60:
            return False

    DEDUP[eid]=now

    return True


##################################################
# Flex UI
##################################################

def flex_history(hist=""):

    from linebot.models import FlexSendMessage

    show=" ".join(hist) if hist else "尚未輸入"

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


##################################################
# PF輸出（舊版格式）
##################################################

def pf_output(p,b):

 banker=round(40+p%10,1)
 player=round(40+b%10,1)
 tie=round(100-banker-player,1)

 conf=max(banker,player)

 if conf<45:
  return "觀望"

 if banker>player:
  bet="莊"
 else:
  bet="閒"

 amount=int(conf*10)

 return f"""上局結果：閒 {p} 莊 {b}

機率 | 莊 {banker}% | 閒 {player}%
| 和 {tie}%

建議：下{bet} 🎯
配注：{amount}

(輸入下一局點數：例如65)
"""


##################################################
# LINE
##################################################

line_api=None
line_handler=None

try:

 from linebot import LineBotApi,WebhookHandler
 from linebot.models import *

 line_api=LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
 line_handler=WebhookHandler(LINE_CHANNEL_SECRET)


 @line_handler.add(MessageEvent,message=TextMessage)
 def msg(event):

  uid=event.source.user_id

  if not dedupe(event.message.id):
   return

  text=event.message.text.strip()

  sess=get_session(uid)


  ################ 開通 ################

  if text.startswith("開通"):

   pw=text.split()[-1]

   if pw==OPEN_PASSWORD:

    PREMIUM_DB[uid]=1

    line_api.reply_message(event.reply_token,
    TextSendMessage(text="✅已永久開通"))

    return


  ################ 試用 ################

  guard=trial_guard(uid)

  if guard:

   line_api.reply_message(event.reply_token,
   TextSendMessage(text=guard))

   return


  ################ 遊戲設定 ################

  if text=="遊戲設定":

   sess["phase"]="choose"

   line_api.reply_message(event.reply_token,
   TextSendMessage(text=f"""請選擇遊戲館別

1. WM
2. PM
3. DG
4. SA
5. KU
6. 歐博/卡利
7. KG
8. 全利
9. 名人
10. MT真人

試用剩餘 {trial_left(uid)} 分鐘（共30分鐘）
"""))

   return


  ################ RESET ################

  if text in ["RESET","結束分析"]:

   SESS.pop(uid,None)

   line_api.reply_message(event.reply_token,
   TextSendMessage(text="已重置\n輸入遊戲設定開始"))

   return


  ################ 館別 ################

  if sess["phase"]=="choose":

   sess["game"]=text
   sess["phase"]="bankroll"

   line_api.reply_message(event.reply_token,
   TextSendMessage(text=f"🎰已選擇：{text}\n請輸入初始籌碼"))

   return


  ################ 籌碼 ################

  if sess["phase"]=="bankroll":

   if text.isdigit():

    sess["bankroll"]=int(text)
    sess["phase"]="history"

    line_api.reply_message(event.reply_token,
    flex_history())

    return


  ################ 歷史 ################

  if sess["phase"]=="history":

   if text in ["B","P","T"]:

    sess["history_input"]+=text

    line_api.reply_message(event.reply_token,
    flex_history(sess["history_input"]))

    return


   if text=="開始":

    sess["phase"]="pf"

    line_api.reply_message(event.reply_token,
    TextSendMessage(text="""歷史載入完成
History loaded

請輸入下一局點數
例如：65 / 和 / 閒6莊5"""))

    return


  ################ PF ################

  if sess["phase"]=="pf":

   if re.match("^[0-9]{2}$",text):

    p=int(text[0])
    b=int(text[1])

    line_api.reply_message(event.reply_token,
    TextSendMessage(text=pf_output(p,b)))

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
