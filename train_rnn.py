#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train TinyRNN (3-class: B/P/T) from CSV logged by server.py

Input CSV format (by /line-webhook auto logging):
user_id,ts,history_before,label
Uxxx,  1693900000, BPPB,  B

Env:
- TRAIN_DATA_PATH  (default /mnt/data/logs/rounds.csv)
- RNN_OUT_PATH     (default /opt/models/rnn.pt)  # 與 server.py 的 RNN_PATH 一致
- EPOCHS=20, BATCH=64, HIDDEN=32, MAXLEN=60, LR=0.001, WEIGHT_T=2.5
- VAL_SPLIT=0.15 (time-based split)
"""
import os, csv, math, random
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/mnt/data/logs/rounds.csv")
RNN_OUT_PATH    = os.getenv("RNN_OUT_PATH", "/opt/models/rnn.pt")
os.makedirs(os.path.dirname(RNN_OUT_PATH), exist_ok=True)

EPOCHS  = int(os.getenv("EPOCHS", "20"))
BATCH   = int(os.getenv("BATCH", "64"))
HIDDEN  = int(os.getenv("HIDDEN", "32"))
MAXLEN  = int(os.getenv("MAXLEN", "60"))
LR      = float(os.getenv("LR", "0.001"))
VAL_SPLIT = float(os.getenv("VAL_SPLIT", "0.15"))
WEIGHT_T  = float(os.getenv("WEIGHT_T", "2.5"))  # T 類別加權，處理不平衡

LUT = {'B':0,'P':1,'T':2}
INV = {0:'B',1:'P',2:'T'}

def load_rows(path: str) -> List[Tuple[int,str,str]]:
    """回傳 list[(ts, history_before, label)]，按 ts 排序"""
    rows = []
    if not os.path.exists(path):
        print(f"[WARN] data file not found: {path}")
        return rows
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        for line in r:
            if not line or len(line) < 4: continue
            user_id, ts, hist, lab = line[0], line[1], line[2], line[3]
            if not lab or lab.upper() not in LUT: continue
            rows.append( (int(ts), hist.strip().upper(), lab.strip().upper()) )
    rows.sort(key=lambda x: x[0])
    return rows

def seq_to_onehot(seq: str, maxlen: int) -> torch.Tensor:
    """
    將 'BPTBP' -> [T, maxlen, 3] one-hot（左側補零），長度不足補零
    """
    arr = []
    for ch in seq[-maxlen:]:  # 只截取最後 maxlen
        v = [0.0,0.0,0.0]
        if ch in LUT:
            v[LUT[ch]] = 1.0
        arr.append(v)
    # 左側補零到 maxlen
    while len(arr) < maxlen:
        arr = [[0.0,0.0,0.0]] + arr
    return torch.tensor(arr, dtype=torch.float32)

class Roadset(Dataset):
    def __init__(self, rows: List[Tuple[int,str,str]], maxlen:int):
        self.samples = rows
        self.maxlen = maxlen
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        ts, hist, lab = self.samples[idx]
        x = seq_to_onehot(hist, self.maxlen)  # [maxlen,3]
        y = LUT[lab]
        return x, y

class TinyRNN(nn.Module):
    def __init__(self, in_dim=3, hidden=HIDDEN, out_dim=3):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, batch_first=True)
        self.fc  = nn.Linear(hidden, out_dim)
    def forward(self, x):
        # x: [B,T,3]
        out, _ = self.rnn(x)
        logit = self.fc(out[:, -1, :])  # 取最後時刻
        return logit

def time_split(rows, val_ratio=0.15):
    n = len(rows)
    k = max(1, int(n*(1.0-val_ratio)))
    return rows[:k], rows[k:]

def main():
    rows = load_rows(TRAIN_DATA_PATH)
    if len(rows) < 100:
        print(f"[WARN] too few rows: {len(rows)} (need >=100).")
    train_rows, val_rows = time_split(rows, VAL_SPLIT)
    train_ds = Roadset(train_rows, MAXLEN)
    val_ds   = Roadset(val_rows,   MAXLEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyRNN().to(device)
    # 類別權重：對 T 加重，降低不平衡影響
    class_weights = torch.tensor([1.0, 1.0, WEIGHT_T], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    patience = max(3, int(EPOCHS/5))
    bad = 0

    for ep in range(1, EPOCHS+1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_dl:
            x = x.to(device)  # [B,T,3]
            y = y.to(device)  # [B]
            optim.zero_grad()
            logit = model(x)
            loss = criterion(logit, y)
            loss.backward()
            optim.step()
            total += float(loss.item()) * x.size(0); n += x.size(0)
        train_loss = total / max(1,n)

        # val
        model.eval()
        vtotal, vn = 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device); y = y.to(device)
                logit = model(x)
                loss = criterion(logit, y)
                vtotal += float(loss.item()) * x.size(0); vn += x.size(0)
        val_loss = vtotal / max(1,vn)
        print(f"[{ep:02d}/{EPOCHS}] train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            bad = 0
            torch.save(model.state_dict(), RNN_OUT_PATH)
            print(f"  -> saved to {RNN_OUT_PATH}")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    print("Done.")

if __name__ == "__main__":
    main()
