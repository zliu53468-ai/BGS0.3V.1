# -*- coding: utf-8 -*-
"""
GRU Deep + Meta Model
Realtime Baccarat Engine
"""

import numpy as np
from collections import deque

LOOKBACK = 40


class GRUModel:

    def __init__(self):

        self.history = deque(maxlen=LOOKBACK)


    def update(self,outcome):

        self.history.append(outcome)


    def load_history(self,seq):

        self.history.clear()

        for s in seq[-LOOKBACK:]:
            self.history.append(s)


    def undo(self):

        if len(self.history)>0:
            self.history.pop()


    def predict(self):

        if len(self.history)<6:

            return np.array(
                [0.4586,0.4462,0.0952],
                dtype=np.float32
            )


        h=np.array(self.history)

        b=np.sum(h==0)
        p=np.sum(h==1)
        t=np.sum(h==2)

        total=len(h)

        pB=b/total
        pP=p/total
        pT=t/total


        last=h[-8:]

        b_run=np.sum(last==0)
        p_run=np.sum(last==1)


        if b_run>p_run:

            pB+=0.06

        elif p_run>b_run:

            pP+=0.06


        probs=np.array([pB,pP,pT])

        probs=probs/probs.sum()

        return probs.astype(np.float32)


    # ===== META MODEL =====

    def meta_state(self):

        if len(self.history)<12:

            return "CHAOS"


        h=list(self.history)

        switches=0

        for i in range(1,len(h)):

            if h[i]!=h[i-1]:
                switches+=1


        switch_rate=switches/len(h)


        runs=[]

        cur=1

        for i in range(1,len(h)):

            if h[i]==h[i-1]:
                cur+=1
            else:
                runs.append(cur)
                cur=1

        runs.append(cur)

        avg_run=np.mean(runs)


        if avg_run>=2.4:

            return "TREND"

        if switch_rate>0.65:

            return "CHOP"


        return "CHAOS"
