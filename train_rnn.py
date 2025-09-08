#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Tiny RNN (GRU) for 3-class B/P/T
RNN 僅使用序列 one-hot（不含大路靜態特徵），與 server 推論一致
Env:
- TRAIN_DATA_PATH  (default /data/logs/rounds.csv)
- RNN_OUT_PATH     (default /data/models/rnn.pt)
- EPOCHS           (default 8)
- BATCH_SIZE       (default 128)
- FEAT_WIN         (default 60)  # RNN 可放大窗口；推論會吃全序列
"""

import os, csv, random
from typing import List, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH","/data/logs/rounds.csv")
RNN_OUT_PATH    = os.getenv("RNN_OUT_PATH",  "/data/models/rnn.pt")
os.makedirs(os.path.dirname(RNN_OUT_PATH), exist_ok=True)

EPOCHS     = int(os.getenv("EPOCHS","8"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE","128"))
FEAT_WIN   = int(os.getenv("FEAT_WIN","60"))   # 訓練截尾長度
CLASS_ORDER=("B","P","T"); LUT={'B':0,'P':1,'T':2}

def seq_to_tensor(seq: List[str], K:int)->np.ndarray:
    tail = seq[-K:] if K>0 else seq
    X=[]
    for ch in tail:
        X.append([1.0 if ch==c else 0.0 for c in CLASS_ORDER])
    return np.array(X, dtype=np.float32)

class SeqDataset(Dataset):
    def __init__(self, rows: List[Tuple[List[str], int]], K:int):
        self.rows=rows; self.K=K
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        seq, y = self.rows[idx]
        x = seq_to_tensor(seq, self.K)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

class TinyRNN(nn.Module):
    def __init__(self, in_dim=3, hidden=64, out_dim=3):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, batch_first=True)
        self.fc  = nn.Linear(hidden, out_dim)
    def forward(self, x):
        # x: (B, T, 3)
        out, _ = self.rnn(x)
        h = out[:, -1, :]
        return self.fc(h)

def load_rows(path:str)->List[Tuple[List[str], int]]:
    rows=[]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path,"r",encoding="utf-8") as f:
        r=csv.reader(f)
        for line in r:
            if not line or len(line)<4: continue
            hist=(line[2] or "").strip().upper()
            lab =(line[3] or "").strip().upper()
            if lab not in LUT: continue
            seq=[ch for ch in hist if ch in LUT]
            if len(seq)==0: continue
            rows.append((seq, LUT[lab]))
    # 時序排序
    return rows

def time_split(rows: List[Tuple[List[str], int]], val_ratio:float=0.15):
    n=len(rows); k=max(1,int(n*(1.0-val_ratio)))
    return rows[:k], rows[k:]

def pad_collate(batch):
    # 以 batch 內最長長度做 zero-pad
    xs, ys = zip(*batch)
    maxT = max(x.shape[0] for x in xs)
    Xp = torch.zeros(len(xs), maxT, xs[0].shape[1], dtype=torch.float32)
    for i,x in enumerate(xs):
        Xp[i, -x.shape[0]:, :] = x
    Y = torch.stack(ys)
    return Xp, Y

def main():
    rows = load_rows(TRAIN_DATA_PATH)
    if len(rows)<200:
        print(f"[WARN] few samples: {len(rows)}")
    # 轉成 (history_before 的序列 → label) 訓練樣本
    tr, va = time_split(rows, 0.15)

    # 截尾（僅訓練效率用；推論時 server 會吃整串）
    def clip_rows(rs):
        out=[]
        for seq, y in rs:
            seq_clip = seq[-FEAT_WIN:] if FEAT_WIN>0 else seq
            out.append((seq_clip, y))
        return out

    tr = clip_rows(tr)
    va = clip_rows(va)

    ds_tr = SeqDataset(tr, FEAT_WIN)
    ds_va = SeqDataset(va, FEAT_WIN)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyRNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_va = 1e9; best_state=None
    for ep in range(1, EPOCHS+1):
        model.train(); loss_sum=0.0; n=0
        for X,Y in dl_tr:
            X=X.to(device); Y=Y.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = crit(logits, Y)
            loss.backward(); opt.step()
            loss_sum += loss.item()*X.size(0); n+=X.size(0)
        tr_loss = loss_sum/max(1,n)

        # valid
        model.eval(); va_loss=0.0; m=0
        with torch.no_grad():
            for X,Y in dl_va:
                X=X.to(device); Y=Y.to(device)
                logits = model(X)
                loss = crit(logits, Y)
                va_loss += loss.item()*X.size(0); m+=X.size(0)
        va_loss /= max(1,m)
        print(f"[E{ep}] train={tr_loss:.4f} valid={va_loss:.4f}")

        if va_loss < best_va:
            best_va = va_loss
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}

    if best_state is None:
        best_state = {k:v.cpu() for k,v in model.state_dict().items()}
    torch.save(best_state, RNN_OUT_PATH)
    print(f"[OK] saved RNN -> {RNN_OUT_PATH}")

if __name__=="__main__":
    main()
