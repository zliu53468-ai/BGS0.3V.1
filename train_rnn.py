#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train TinyRNN (GRU) aligned with server v15.3
- Input: sequence of one-hot(B,P,T), padded/truncated to FEAT_WIN
- Label: next outcome class (B/P/T) from CSV rows (history_before -> label)
- Saves state_dict to RNN_PATH
"""

import os, csv, random
from typing import List, Tuple
import numpy as np

import torch
import torch.nn as tnn
import torch.optim as optim

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/data/logs/rounds.csv")
RNN_OUT_PATH    = os.getenv("RNN_OUT_PATH",   "/data/models/rnn.pt")
os.makedirs(os.path.dirname(RNN_OUT_PATH), exist_ok=True)

FEAT_WIN   = int(os.getenv("FEAT_WIN", "20"))
VAL_SPLIT  = float(os.getenv("VAL_SPLIT", "0.15"))
BATCH_SIZE = int(os.getenv("RNN_BS", "64"))
EPOCHS     = int(os.getenv("RNN_EPOCHS", "15"))
LR         = float(os.getenv("RNN_LR", "0.003"))
MAX_HISTORY= int(os.getenv("MAX_HISTORY", "400"))

LUT = {'B':0,'P':1,'T':2}
CLASS_ORDER = ("B","P","T")

class TinyRNN(tnn.Module):
    def __init__(self, in_dim=3, hidden=16, out_dim=3):
        super().__init__()
        self.rnn = tnn.GRU(in_dim, hidden, batch_first=True)
        self.fc  = tnn.Linear(hidden, out_dim)
    def forward(self, x):
        o, _ = self.rnn(x)
        return self.fc(o[:, -1, :])

def load_rows(path: str):
    rows = []
    if not os.path.exists(path):
        print(f"[ERROR] data file not found: {path}")
        return rows
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        for line in r:
            if not line or len(line) < 4: continue
            try:
                ts = int(line[1])
            except Exception:
                continue
            hist = (line[2] or "").strip().upper()
            lab  = (line[3] or "").strip().upper()
            if lab in LUT:
                rows.append((ts, hist, lab))
    rows.sort(key=lambda x: x[0])
    return rows

def hist_to_tensor(hist: str, K: int) -> torch.Tensor:
    seq = [ch for ch in hist if ch in LUT]
    if len(seq) > MAX_HISTORY:
        seq = seq[-MAX_HISTORY:]
    take = seq[-K:]
    pad = K - len(take)
    out = []
    # left-pad with zeros
    for _ in range(pad):
        out.append([0.0, 0.0, 0.0])
    for ch in take:
        one = [0.0,0.0,0.0]; one[LUT[ch]] = 1.0
        out.append(one)
    return torch.tensor(out, dtype=torch.float32)  # (K,3)

def build_xy(rows, K: int):
    X, y = [], []
    for _, hist, lab in rows:
        X.append(hist_to_tensor(hist, K))
        y.append(LUT[lab])
    X = torch.stack(X, dim=0)  # (N,K,3)
    y = torch.tensor(y, dtype=torch.long)
    return X, y

def time_split(rows, val_ratio: float):
    n = len(rows)
    k = max(1, int(n*(1.0 - val_ratio)))
    return rows[:k], rows[k:]

def class_weights(y: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(y, minlength=3).float()
    counts[counts==0] = 1.0
    inv = 1.0 / counts
    w = inv[y]
    w *= (len(w) / w.sum())
    # use per-class weights
    per_class = inv * (3.0 / inv.sum())
    return per_class

def batch_iter(X, y, bs: int, shuffle=True):
    idx = list(range(len(y)))
    if shuffle: random.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i+bs]
        yield X[j], y[j]

def main():
    rows = load_rows(TRAIN_DATA_PATH)
    if len(rows) < 50:
        print(f"[WARN] few samples: {len(rows)}; training anyway.")
    tr, va = time_split(rows, VAL_SPLIT)
    Xtr, ytr = build_xy(tr, FEAT_WIN)
    Xva, yva = build_xy(va, FEAT_WIN) if va else (None, None)

    model = TinyRNN(in_dim=3, hidden=16, out_dim=3)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = tnn.CrossEntropyLoss(weight=class_weights(ytr))

    best_va = float("inf")
    patience = max(3, EPOCHS//3)
    bad = 0

    for ep in range(1, EPOCHS+1):
        model.train()
        total = 0.0
        for xb, yb in batch_iter(Xtr, ytr, BATCH_SIZE, shuffle=True):
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        tr_loss = total / len(ytr)

        model.eval()
        with torch.no_grad():
            if Xva is not None:
                logits = model(Xva)
                va_loss = tnn.functional.cross_entropy(logits, yva).item()
            else:
                va_loss = tr_loss

        print(f"[EP {ep}] train={tr_loss:.4f} valid={va_loss:.4f}")

        if va_loss + 1e-6 < best_va:
            best_va = va_loss; bad = 0
            torch.save(model.state_dict(), RNN_OUT_PATH)
            print(f"  -> checkpoint saved to {RNN_OUT_PATH}")
        else:
            bad += 1
            if bad >= patience:
                print("Early stop.")
                break

    # final save (in case very small data / no val improvement caught)
    if not os.path.exists(RNN_OUT_PATH):
        torch.save(model.state_dict(), RNN_OUT_PATH)
    print(f"[OK] saved RNN -> {RNN_OUT_PATH}")

if __name__ == "__main__":
    main()
