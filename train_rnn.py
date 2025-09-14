# train_rnn.py — Tiny GRU for next-hand prediction (CPU friendly)

import os, csv, math, random
from typing import List, Tuple
import numpy as np

SEED = int(os.getenv("SEED","42")); random.seed(SEED); np.random.seed(SEED)

try:
    import torch, torch.nn as nn
    TORCH_OK = True
except Exception:
    TORCH_OK = False

MAP = {"B":0,"P":1,"T":2,"莊":0,"閒":1,"和":2}
def parse_seq(s:str)->List[int]:
    s=(s or "").strip().upper()
    if not s: return []
    toks=s.split()
    seq=list(s) if len(toks)==1 else toks
    out=[]
    for ch in seq:
        if ch in MAP: out.append(MAP[ch])
    return out

def load_rows(path:str)->List[Tuple[List[int], int]]:
    rows=[]
    if not os.path.exists(path):
        print(f"[WARN] train file not found: {path}"); return rows
    with open(path,"r",encoding="utf-8") as f:
        r=csv.DictReader(f)
        has_next=("next" in (r.fieldnames or []))
        for row in r:
            hist=parse_seq(row.get("history",""))
            if not hist: continue
            if has_next:
                nxt=MAP.get((row.get("next","") or "").strip().upper(), None)
                if nxt is not None: rows.append((hist, int(nxt)))
            else:
                # 自動切片：用每個位置的下一手當標籤
                for i in range(1, len(hist)):
                    rows.append((hist[:i], hist[i]))
    return rows

def one_hot_seq(seq: List[int], max_len:int)->np.ndarray:
    sub = seq[-max_len:] if len(seq)>max_len else seq[:]
    L=len(sub); oh=np.zeros((L,3), dtype=np.float32)
    for i,v in enumerate(sub):
        oh[i,v]=1.0
    return oh

def batchify(samples, max_len):
    X=[]; y=[]
    for s, nxt in samples:
        X.append(one_hot_seq(s, max_len))
        y.append(int(nxt))
    # pad to same length
    Lmax=max(len(x) for x in X)
    Xpad=[]
    for x in X:
        pad = np.zeros((Lmax-len(x),3), dtype=np.float32)
        Xpad.append(np.vstack([pad, x]))  # left pad
    return np.stack(Xpad,0), np.array(y, dtype=np.int64)

def main():
    if not TORCH_OK:
        print("[ERROR] PyTorch not installed. Please install torch for training RNN.")
        return

    TRAIN_PATH=os.getenv("TRAIN_PATH","data/train.csv")
    RNN_OUT_PATH=os.getenv("RNN_OUT_PATH","data/models/rnn.pt")
    MAX_RNN_LEN=int(os.getenv("MAX_RNN_LEN","256"))
    HIDDEN=int(os.getenv("RNN_HIDDEN","32"))
    EPOCHS=int(os.getenv("EPOCHS","8"))
    LR=float(os.getenv("LR","0.003"))
    BATCH=int(os.getenv("BATCH","64"))

    data=load_rows(TRAIN_PATH)
    if not data:
        print("[ERROR] no training data."); return
    random.shuffle(data)
    n=int(0.9*len(data))
    train=data[:n]; val=data[n:]

    Xtr, ytr = batchify(train, MAX_RNN_LEN)
    Xva, yva = batchify(val, MAX_RNN_LEN)

    import torch, torch.nn as nn, torch.optim as optim
    device=torch.device("cpu")

    class TinyRNN(nn.Module):
        def __init__(self, in_dim=3, hid=HIDDEN, out_dim=3):
            super().__init__()
            self.gru = nn.GRU(input_size=in_dim, hidden_size=hid, num_layers=1, batch_first=True)
            self.fc  = nn.Linear(hid, out_dim)
        def forward(self, x):
            o,_=self.gru(x); return self.fc(o[:,-1,:])

    model=TinyRNN().to(device)
    opt=optim.Adam(model.parameters(), lr=LR)
    crit=nn.CrossEntropyLoss()

    def run_epoch(X, y, train_mode=True):
        model.train(train_mode)
        bs=BATCH; N=X.shape[0]; tot=0; corr=0
        idx=list(range(N))
        if train_mode: random.shuffle(idx)
        for i in range(0, N, bs):
            j=idx[i:i+bs]
            xb=torch.from_numpy(X[j]).to(device)
            yb=torch.from_numpy(y[j]).to(device)
            if train_mode: opt.zero_grad()
            logits=model(xb)
            loss=crit(logits, yb)
            if train_mode:
                loss.backward(); opt.step()
            tot += float(loss.item())*len(j)
            pred=logits.argmax(-1)
            corr+=int((pred==yb).sum().item())
        return tot/N, corr/N

    for ep in range(1, EPOCHS+1):
        tr_loss, tr_acc = run_epoch(Xtr, ytr, True)
        va_loss, va_acc = run_epoch(Xva, yva, False)
        print(f"[{ep}/{EPOCHS}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

    os.makedirs(os.path.dirname(RNN_OUT_PATH), exist_ok=True)
    torch.save(model.state_dict(), RNN_OUT_PATH)
    print(f"[OK] saved RNN to {RNN_OUT_PATH}")

if __name__=="__main__":
    main()
