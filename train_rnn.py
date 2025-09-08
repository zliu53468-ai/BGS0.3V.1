#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Tiny GRU RNN (3-class: B/P/T) aligned with server v15

CSV schema:
user_id,ts,history_before,label

ENV (all optional):
- TRAIN_DATA_PATH   default /data/logs/rounds.csv
- RNN_OUT_PATH      default /data/models/rnn.pt
- FEAT_WIN          default 40          # RNN 吃更長一點上下文；server 推論用全歷史沒問題
- VAL_SPLIT         default 0.15
- EPOCHS            default 30
- BATCH_SIZE        default 128
- LR                default 1e-3
- HIDDEN            default 32
- SEED              default 42
"""

import os, csv, random
from typing import List, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/data/logs/rounds.csv")
RNN_OUT_PATH    = os.getenv("RNN_OUT_PATH", "/data/models/rnn.pt")
os.makedirs(os.path.dirname(RNN_OUT_PATH), exist_ok=True)

FEAT_WIN   = int(os.getenv("FEAT_WIN", "40"))
VAL_SPLIT  = float(os.getenv("VAL_SPLIT", "0.15"))
EPOCHS     = int(os.getenv("EPOCHS", "30"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
LR         = float(os.getenv("LR", "0.001"))
HIDDEN     = int(os.getenv("HIDDEN", "32"))
SEED       = int(os.getenv("SEED", "42"))

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

LUT = {'B':0,'P':1,'T':2}
INV = {0:'B',1:'P',2:'T'}

def load_rows(path:str):
    rows=[]
    if not os.path.exists(path):
        print(f"[ERROR] data file not found: {path}")
        return rows
    with open(path,"r",encoding="utf-8") as f:
        r=csv.reader(f)
        for line in r:
            if not line or len(line)<4: continue
            try:
                ts=int(line[1])
            except Exception:
                continue
            hist=(line[2] or "").strip().upper()
            lab =(line[3] or "").strip().upper()
            if lab in LUT:
                rows.append((ts,hist,lab))
    rows.sort(key=lambda x:x[0])
    return rows

def encode_seq(seq:str)->List[int]:
    return [LUT[ch] for ch in seq if ch in LUT]

def onehot(i:int)->List[float]:
    v=[0.0,0.0,0.0]; v[i]=1.0; return v

class SeqDataset(Dataset):
    def __init__(self, rows, K:int):
        self.samples=[]
        for _, hist, lab in rows:
            ids=encode_seq(hist)
            if not ids:  # 沒歷史就跳過
                continue
            # 取最後 K 手作為輸入序列（padding 在前）
            take=ids[-K:]
            pad=K-len(take)
            x = ([2]*pad + take)  # 用 2('T') 當作 padding 類別，與 onehot 相容
            self.samples.append( (x, LUT[lab]) )
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        # 轉 onehot 時序張量 (K,3)
        X = torch.tensor([onehot(t) for t in x], dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return X, y

class TinyRNN(nn.Module):
    def __init__(self, in_dim=3, hidden=HIDDEN, out_dim=3):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, batch_first=True)
        self.fc  = nn.Linear(hidden, out_dim)
    def forward(self, x):
        o,_ = self.rnn(x)        # (B,T,H)
        h   = o[:,-1,:]          # last step
        return self.fc(h)        # (B,3)

def time_split(rows, val_ratio:float):
    n=len(rows)
    k=max(1, int(n*(1.0-val_ratio)))
    return rows[:k], rows[k:]

def class_weights(y: np.ndarray)->torch.Tensor:
    cnt=np.bincount(y, minlength=3).astype(np.float64)
    cnt[cnt==0]=1.0
    inv=1.0/cnt
    w=inv*(3.0/np.sum(inv))
    return torch.tensor(w, dtype=torch.float32)

def main():
    rows=load_rows(TRAIN_DATA_PATH)
    if len(rows)<200:
        print(f"[WARN] few samples: {len(rows)}; training anyway.")
    tr,va = time_split(rows, VAL_SPLIT)

    ds_tr = SeqDataset(tr, FEAT_WIN)
    ds_va = SeqDataset(va, FEAT_WIN)
    if len(ds_tr)==0:
        print("[ERROR] no training samples"); return

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, drop_last=False) if len(ds_va)>0 else None

    model = TinyRNN()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    # 以類別權重處理不平衡
    y_all = np.array([y for _,_,y in tr], dtype=object)  # 先做個簡單提取
    y_all = np.array([LUT[str(y)] for y in y_all if str(y) in LUT], dtype=np.int64)
    cw    = class_weights(y_all) if len(y_all)>0 else torch.tensor([1.0,1.0,1.0])
    crit  = nn.CrossEntropyLoss(weight=cw)

    best_loss = float("inf"); best_state=None; no_improve=0
    EARLY = 6

    for ep in range(1, EPOCHS+1):
        model.train(); tr_loss=0.0
        for X,y in dl_tr:
            opt.zero_grad()
            logits = model(X)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tr_loss += float(loss.detach().cpu().item()) * X.size(0)
        tr_loss /= max(1,len(ds_tr))

        va_loss=None
        if dl_va is not None and len(ds_va)>0:
            model.eval(); s=0.0
            with torch.no_grad():
                for X,y in dl_va:
                    logits = model(X)
                    loss = crit(logits, y)
                    s += float(loss.detach().cpu().item()) * X.size(0)
            va_loss = s / max(1,len(ds_va))

        if va_loss is None:
            print(f"[EP {ep}] train_loss={tr_loss:.4f}")
        else:
            print(f"[EP {ep}] train_loss={tr_loss:.4f}  valid_loss={va_loss:.4f}")
            if va_loss < best_loss - 1e-4:
                best_loss = va_loss
                best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
                no_improve=0
            else:
                no_improve+=1
                if no_improve>=EARLY:
                    print(f"[EARLY STOP] no improve {EARLY} epochs.")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), RNN_OUT_PATH)
    print(f"[OK] saved RNN model -> {RNN_OUT_PATH}")

if __name__=="__main__":
    main()
