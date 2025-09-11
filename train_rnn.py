# train_rnn.py
import os, csv, math, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

MAP = {"B":0,"P":1,"T":2}

FEAT_WIN = int(os.getenv("FEAT_WIN","40"))
BATCH    = int(os.getenv("RNN_BATCH","64"))
EPOCHS   = int(os.getenv("RNN_EPOCHS","30"))
LR       = float(os.getenv("RNN_LR","1e-3"))
OUT_PATH = os.getenv("RNN_OUT_PATH","/data/models/rnn.pt")
SEED     = int(os.getenv("SEED","42"))

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def parse_history(s):
    s=(s or "").strip().upper()
    toks=s.split(); seq=list(s) if len(toks)==1 else toks
    out=[]
    for ch in seq:
        if ch in MAP: out.append(MAP[ch])
    return out

def one_hot_seq(seq, win):
    sub = seq[-win:] if len(seq)>win else seq[:]
    pad = [-1]*max(0, win-len(sub))
    final = (pad+sub)[-win:]
    oh=[]
    for v in final:
        a=[0,0,0]
        if v in (0,1,2): a[v]=1
        oh.append(a)
    return np.array(oh, dtype=np.float32)   # [win,3]

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X=X; self.y=y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long)

def load_dataset():
    X=[]; y=[]
    if os.path.exists("data/train.csv"):
        with open("data/train.csv", newline='', encoding="utf-8") as f:
            for row in csv.DictReader(f):
                hist=row.get("history") or row.get("seq") or row.get("sequence")
                nxt=row.get("next") or row.get("label") or row.get("y")
                if not hist or not nxt: continue
                seq=parse_history(hist); nxt=nxt.strip().upper()[:1]
                if nxt not in MAP: continue
                X.append(one_hot_seq(seq, FEAT_WIN))
                y.append(MAP[nxt])
    if os.path.exists("data/train.txt"):
        with open("data/train.txt", encoding="utf-8") as f:
            for line in f:
                if "->" not in line: continue
                left,right=line.split("->",1)
                seq=parse_history(left)
                nxt=right.strip().upper()[:1]
                if nxt not in MAP: continue
                X.append(one_hot_seq(seq, FEAT_WIN))
                y.append(MAP[nxt])
    if not X: raise SystemExit("No data found.")
    X=np.stack(X,0)   # [N,win,3]
    y=np.array(y, dtype=np.int64)
    return X,y

class TinyRNN(nn.Module):
    def __init__(self, in_dim=3, hid=64, out_dim=3):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hid, num_layers=1, batch_first=True)
        self.fc  = nn.Linear(hid, out_dim)
    def forward(self, x):
        o,_ = self.gru(x)    # [B,T,H]
        last = o[:, -1, :]
        return self.fc(last)

def main():
    X,y=load_dataset()
    n=len(X); tr=int(n*0.85)
    Xtr, ytr = X[:tr], y[:tr]
    Xva, yva = X[tr:], y[tr:] if tr<n else (X[:0], y[:0])

    tr_ds=SeqDataset(Xtr,ytr); va_ds=SeqDataset(Xva,yva)
    tr_dl=DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    va_dl=DataLoader(va_ds, batch_size=BATCH, shuffle=False) if len(va_ds)>0 else None

    model=TinyRNN()
    opt=torch.optim.AdamW(model.parameters(), lr=LR)
    crit=nn.CrossEntropyLoss()

    best_loss=1e9; patience=6; bad=0
    for ep in range(1, EPOCHS+1):
        model.train(); tl=0.0; nbt=0
        for xb,yb in tr_dl:
            opt.zero_grad()
            logits=model(xb)
            loss=crit(logits, yb)
            loss.backward()
            opt.step()
            tl += float(loss); nbt += 1
        tl /= max(1,nbt)

        vl=0.0; nvb=0
        if va_dl:
            model.eval()
            with torch.no_grad():
                for xb,yb in va_dl:
                    logits=model(xb)
                    loss=crit(logits,yb)
                    vl += float(loss); nvb += 1
            vl /= max(1,nvb)
        else:
            vl = tl

        print(f"Epoch {ep:02d} | train {tl:.4f} | valid {vl:.4f}")
        if vl < best_loss - 1e-4:
            best_loss = vl; bad = 0
            os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
            torch.save(model.state_dict(), OUT_PATH)
            print(f"  -> saved {OUT_PATH}")
        else:
            bad += 1
            if bad >= patience:
                print("Early stop.")
                break

if __name__=="__main__":
    main()
