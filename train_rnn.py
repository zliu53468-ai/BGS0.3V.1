#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train TinyRNN (GRU) for 3-class B/P/T.

Env (optional):
- TRAIN_DATA_PATH   default /data/logs/rounds.csv
- MODEL_DIR         default /data/models
- RNN_OUT_PATH      default {MODEL_DIR}/rnn.pt
- FEAT_WIN          default 20  (sequence length)
- VAL_SPLIT         default 0.15
- EPOCHS            default 30
- BATCH_SIZE        default 64
- LR                default 1e-3
- HIDDEN            default 32
- PATIENCE          default 5
"""

import os, csv, math, random
from typing import List, Tuple
import numpy as np

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/data/logs/rounds.csv")
MODEL_DIR       = os.getenv("MODEL_DIR", "/data/models")
RNN_OUT_PATH    = os.getenv("RNN_OUT_PATH", os.path.join(MODEL_DIR, "rnn.pt"))
os.makedirs(MODEL_DIR, exist_ok=True)

FEAT_WIN   = int(os.getenv("FEAT_WIN", "20"))
VAL_SPLIT  = float(os.getenv("VAL_SPLIT", "0.15"))
EPOCHS     = int(os.getenv("EPOCHS", "30"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
LR         = float(os.getenv("LR", "1e-3"))
HIDDEN     = int(os.getenv("HIDDEN", "32"))
PATIENCE   = int(os.getenv("PATIENCE", "5"))

LUT = {"B":0, "P":1, "T":2}

def load_rows(path:str):
    rows = []
    if not os.path.exists(path):
        print(f"[ERROR] data file not found: {path}")
        return rows
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        for line in r:
            if not line or len(line) < 4: continue
            try:
                ts   = int(line[1])
                hist = (line[2] or "").strip().upper()
                lab  = (line[3] or "").strip().upper()
            except Exception:
                continue
            if lab in LUT:
                rows.append((ts, hist, lab))
    rows.sort(key=lambda x: x[0])
    return rows

def seq_to_tensor(seq:str, K:int) -> np.ndarray:
    seq = [ch for ch in seq if ch in LUT]
    take = seq[-K:]
    X = []
    for ch in take:
        one = [0.0,0.0,0.0]; one[LUT[ch]] = 1.0
        X.append(one)
    need = K - len(X)
    if need > 0:
        X = [[0.0,0.0,0.0]]*need + X
    return np.array(X, dtype=np.float32)  # shape: (K,3)

def build_xy(rows, K:int):
    X, y = [], []
    for _, hist, lab in rows:
        X.append(seq_to_tensor(hist, K))
        y.append(LUT[lab])
    return np.stack(X, axis=0), np.array(y, dtype=np.int64)

def time_split(rows, val_ratio:float):
    n = len(rows)
    k = max(1, int(n*(1.0 - val_ratio)))
    return rows[:k], rows[k:]

def class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=3).astype(np.float64)
    counts[counts==0] = 1.0
    inv = 1.0 / counts
    w = inv * (3.0 / inv.sum())
    return w.astype(np.float32)  # length=3

def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    rows = load_rows(TRAIN_DATA_PATH)
    if len(rows) < 200:
        print(f"[WARN] few samples: {len(rows)}; training anyway.")

    tr, va = time_split(rows, VAL_SPLIT)
    Xtr, ytr = build_xy(tr, FEAT_WIN)
    Xva, yva = build_xy(va, FEAT_WIN) if len(va)>0 else (None, None)

    w = class_weights(ytr)
    class_w = torch.tensor(w, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class TinyRNN(nn.Module):
        def __init__(self, hidden=HIDDEN):
            super().__init__()
            self.rnn = nn.GRU(input_size=3, hidden_size=hidden, batch_first=True)
            self.fc  = nn.Linear(hidden, 3)
        def forward(self, x):
            out, _ = self.rnn(x)      # x: (B,K,3)
            return self.fc(out[:,-1,:])

    model = TinyRNN().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.CrossEntropyLoss(weight=class_w.to(device))

    def make_loader(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)

    tr_loader = make_loader(Xtr, ytr, True)
    va_loader = make_loader(Xva, yva, False) if Xva is not None else None

    best_loss = float("inf"); patience = PATIENCE; best_state = None

    for ep in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(tr_loader.dataset)

        if va_loader is not None:
            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = crit(logits, yb)
                    va_loss += loss.item() * xb.size(0)
            va_loss /= len(va_loader.dataset)
        else:
            va_loss = tr_loss

        print(f"[E{ep:02d}] train={tr_loss:.4f} valid={va_loss:.4f}")

        if va_loss + 1e-6 < best_loss:
            best_loss = va_loss
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            patience = PATIENCE
        else:
            patience -= 1
            if patience <= 0:
                print("[EarlyStop] no improvement.")
                break

    if best_state is None:
        best_state = model.state_dict()
    torch.save(best_state, RNN_OUT_PATH)
    print(f"[OK] saved RNN model -> {RNN_OUT_PATH}")

if __name__ == "__main__":
    main()
