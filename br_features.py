# br_features.py
from typing import Any, Dict, List, Tuple
import os, math

CLASS_ORDER = ("B","P","T")

def map_to_big_road(seq: List[str], rows:int=6, cols:int=20) -> Tuple[List[List[str]], Dict[str,Any]]:
    grid=[["" for _ in range(cols)] for _ in range(rows)]
    if not seq:
        return grid, {"cur_run":0, "col_depth":0, "blocked":False, "r":0, "c":0, "early_dragon_hint":False}

    r=c=0; last=None
    for ch in seq:
        if last is None:
            grid[r][c]=ch; last=ch; continue
        if ch==last:
            if r+1<rows and grid[r+1][c]=="":
                r+=1
            else:
                c=min(cols-1, c+1)
                while c<cols and grid[r][c]!="":
                    c=min(cols-1, c+1)
                if c>=cols: c=cols-1
        else:
            last=ch
            c=min(cols-1, c+1); r=0
            while c<cols and grid[r][c]!="":
                c=min(cols-1, c+1)
            if c>=cols: c=cols-1
        if grid[r][c]=="": grid[r][c]=ch

    # depth
    cur_depth=0
    for rr in range(rows):
        if grid[rr][c]!="": cur_depth=rr+1

    def last_run_len(s: List[str])->int:
        if not s: return 0
        ch=s[-1]; i=len(s)-2; n=1
        while i>=0 and s[i]==ch: n+=1; i-=1
        return n

    blocked = (cur_depth>=rows) or (r==rows-1) or (r+1<rows and grid[r+1][c]!="" and last==grid[r][c])

    feats = {
        "cur_run": last_run_len(seq),
        "col_depth": cur_depth,
        "blocked": blocked,
        "r": r, "c": c,
        "early_dragon_hint": (cur_depth>=3 and features_like_early_dragon(seq))
    }
    return grid, feats

def features_like_early_dragon(seq: List[str]) -> bool:
    k=min(6, len(seq))
    if k<4: return False
    tail=seq[-k:]
    most=max(tail.count("B"), tail.count("P"))
    return (most>=k-1)

def bp_only(seq: List[str]) -> List[str]:
    return [x for x in seq if x in ("B","P")]

def run_hist(seq_bp: List[str]) -> Dict[int,int]:
    hist: Dict[int,int]={}
    if not seq_bp: return hist
    cur=1
    for i in range(1,len(seq_bp)):
        if seq_bp[i]==seq_bp[i-1]: cur+=1
        else:
            hist[cur]=hist.get(cur,0)+1
            cur=1
    hist[cur]=hist.get(cur,0)+1
    return hist

def hazard_from_hist(L:int, hist:Dict[int,int]) -> float:
    if L<=0: return 0.0
    a = float(os.getenv("HZD_ALPHA","0.5"))
    ge = sum(v for k,v in hist.items() if k>=L)
    end= hist.get(L, 0)
    return (end + a) / (ge + a*max(1,len(hist)))

def mean_revert_score(seq: List[str]) -> Tuple[float, str]:
    b = seq.count("B"); p = seq.count("P")
    tot = max(1, b+p)
    diff = (b-p)/tot
    side = "P" if diff>0 else ("B" if diff<0 else "")
    return abs(diff), side

# ----- Prob utils -----
def exp_decay_freq(seq: List[str], gamma: float = None) -> List[float]:
    if not seq: return [1/3,1/3,1/3]
    if gamma is None: gamma = float(os.getenv("EW_GAMMA","0.96"))
    wB=wP=wT=0.0; w=1.0
    for r in reversed(seq):
        if r=="B": wB += w
        elif r=="P": wP += w
        else: wT += w
        w *= gamma
    alpha = float(os.getenv("LAPLACE","0.5"))
    wB+=alpha; wP+=alpha; wT+=alpha
    S = wB+wP+wT
    return [wB/S, wP/S, wT/S]

def recent_freq(seq: List[str], win: int) -> List[float]:
    if not seq: return [1/3,1/3,1/3]
    cut = seq[-win:] if win>0 else seq
    a = float(os.getenv("LAPLACE","0.5"))
    nB=cut.count("B")+a; nP=cut.count("P")+a; nT=cut.count("T")+a
    tot=max(1,len(cut))+3*a
    return [nB/tot, nP/tot, nT/tot]

def is_zigzag(seq: List[str], k:int=6)->bool:
    s = seq[-k:] if len(seq)>=k else seq
    if len(s)<4: return False
    alt = all(s[i]!=s[i-1] for i in range(1,len(s)))
    if alt: return True
    if len(s)%2==0:
        pairs=[s[i:i+2] for i in range(0,len(s),2)]
        if all(len(p)==2 and p[0]==p[1] for p in pairs):
            if all(pairs[i][0]!=pairs[i-1][0] for i in range(1,len(pairs))):
                return True
    return False
