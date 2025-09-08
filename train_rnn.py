#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Tiny RNN (GRU) for 3-class B/P/T with sequence one-hot.
ENV:
- TRAIN_DATA_PATH  /data/logs/rounds.csv
- RNN_OUT_PATH     /data/models/rnn.pt
- FEAT_WIN         60   # RNN 序列較長較有利
- EPOCHS           12
- LR               0.003
- BATCH_SIZE       64
"""
import os, csv, random
from typing import List, Tuple
import numpy as np

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/data/logs/rounds.csv")
RNN_OUT_PATH    = os.getenv("RNN_OUT_PATH", "/data/models/rnn.pt")
os.makedirs(os.path.dirname(RNN_OUT_PATH), exist_ok=True)

FEAT_WIN  = int(os.getenv("FEAT_WIN","60"))
EPOCHS    = int(os.getenv("EPOCHS","12"))
LR        = float(os.getenv("LR","0.003"))
BATCH     = int(os.getenv("BATCH_SIZE","64"))
LUT={'B':0,'P':1,'T':2}

import torch
import torch.nn as tnn
device = torch.device("cpu")

def read_rows(path:str):
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        r=csv.reader(f)
        for line in r:
            if len(line)<4: continue
            hist=(line[2] or "").strip().upper()
            lab =(line[3] or "").strip().upper()
            if lab in LUT:
                rows.append((hist, lab))
    return rows

def seq_to_tensor(seq:str, K:int)->torch.Tensor:
    seq=[ch for ch in seq if ch in LUT]
    seq=seq[-K:]
    X=[]
    for ch in seq:
        one=[0.0,0.0,0.0]; one[LUT[ch]]=1.0
        X.append(one)
    if not X:
        X=[[0.0,0.0,0.0]]
    return torch.tensor(X, dtype=torch.float32)

class TinyRNN(tnn.Module):
    def __init__(self, in_dim=3, hidden=32, out_dim=3):
        super().__init__()
        self.rnn = tnn.GRU(in_dim, hidden, batch_first=True)
        self.fc  = tnn.Linear(hidden, out_dim)
    def forward(self, x):
        # x: [B,T,3] 可變長，這裡取最後時序輸出
        pad = tnn.utils.rnn.pack_sequence(x, enforce_sorted=False)
        out, _ = self.rnn(pad)
        out,_ = tnn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        last = out[torch.arange(len(out)), [len(t)-1 for t in x], :]
        return self.fc(last)

def collate(batch):
    X=[seq_to_tensor(hist, FEAT_WIN) for hist,_ in batch]
    y=torch.tensor([LUT[lab] for _,lab in batch], dtype=torch.long)
    return X, y

def main():
    rows=read_rows(TRAIN_DATA_PATH)
    random.shuffle(rows)
    n=len(rows); k=max(1,int(n*0.85))
    tr,va=rows[:k], rows[k:]
    model=TinyRNN().to(device)
    opt=torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn=tnn.CrossEntropyLoss()

    def run_epoch(data, train=True):
        model.train(train)
        total=0.0; correct=0; cnt=0
        for i in range(0, len(data), BATCH):
            batch=data[i:i+BATCH]
            X,y=collate(batch)
            X=[t.to(device) for t in X]; y=y.to(device)
            if train: opt.zero_grad()
            logits=model(X)
            loss=loss_fn(logits, y)
            if train:
                loss.backward(); opt.step()
            total+=float(loss.item())*len(y)
            pred=logits.argmax(dim=1)
            correct+=int((pred==y).sum().item()); cnt+=len(y)
        return total/max(1,cnt), correct/max(1,cnt)

    best=(1e9,0.0); best_state=None
    for ep in range(1,EPOCHS+1):
        tr_loss, tr_acc = run_epoch(tr, True)
        va_loss, va_acc = run_epoch(va, False) if va else (tr_loss, tr_acc)
        print(f"Epoch {ep:02d}: tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} va_loss={va_loss:.4f} va_acc={va_acc:.3f}")
        if va_loss < best[0]:
            best=(va_loss, va_acc); best_state=model.state_dict()

    if best_state is None: best_state=model.state_dict()
    torch.save(best_state, RNN_OUT_PATH)
    print("[OK] RNN saved ->", RNN_OUT_PATH)

if __name__=="__main__":
    main()
