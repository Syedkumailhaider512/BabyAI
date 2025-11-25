# utils.py — small helpers

import os, re, datetime as dt, math
import numpy as np

def sanitize_label(s: str) -> str:
    s = re.sub(r"\W+", "_", s.strip())
    return s or "item"

def clamp(v, a, b):
    return max(a, min(b, v))

def now_iso():
    return dt.datetime.utcnow().isoformat() + "Z"

def parse_iso(ts: str):
    if not ts: return None
    return dt.datetime.fromisoformat(ts.replace("Z",""))

def next_indexed_name(prefix, label, ext):
    base=f"{prefix}{label}_"; idx=1
    while True:
        name=f"{base}{idx:03d}.{ext}"
        if not os.path.exists(name): return name
        idx+=1

# simple 3D positions for viz
def positions_3d(n=100, layout="helix"):
    if layout == "grid":
        xs,ys,zs=[],[],[]
        for i in range(10):
            for j in range(10):
                xs.append(i-4.5); ys.append(j-4.5); zs.append(0.2*math.sin(i/2)+0.2*math.cos(j/2))
        return np.array(xs),np.array(ys),np.array(zs)
    t=np.linspace(0,4*np.pi,n); x=np.cos(t)*4.0; y=np.sin(t)*4.0; z=np.linspace(-2.0,2.0,n)
    return x,y,z
