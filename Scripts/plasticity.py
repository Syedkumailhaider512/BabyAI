# plasticity.py — spikes, STDP, delta, eligibility traces

import numpy as np
from config import ETA_HEBB, A_PLUS, A_MINUS, DELTA_T, TAU_PLUS, TAU_MINUS, MIX_STDP, ETA_DELTA, L2_DECAY
from config import ELIG_TAU_S, ELIG_SCALE, MAIN_SPIKE_LOW

def to_spikes_from_top_row(row, low=MAIN_SPIKE_LOW):
    return (row >= low).astype(np.float32)

def sigmoid(x, gain=3.0):
    x = np.clip(x, -50, 50)
    return 1.0/(1.0+np.exp(-gain*x))

def hebb_stdp_from_spikes(S, scale=1.0):
    T,N=S.shape; W=np.zeros((N,N),np.float32)
    for t in range(T):
        idx=np.where(S[t]==1)[0]
        if len(idx)>=2:
            W[np.ix_(idx,idx)] += (ETA_HEBB*scale)
            np.fill_diagonal(W,0.0)
    eplus=np.exp(-DELTA_T/TAU_PLUS); eminus=np.exp(-DELTA_T/TAU_MINUS)
    for t in range(T-1):
        pre=np.where(S[t]==1)[0]; post=np.where(S[t+1]==1)[0]
        for i in pre:
            for j in post:
                W[i,j] += (A_PLUS  * eplus  * scale)
                W[j,i] -= (A_MINUS * eminus * scale)
    W=np.clip(W,0.0,None)
    if W.max()>1e-12: W/=W.max()
    return W

def delta_rule_update(W,S, scale=1.0, gain=3.0):
    T,N=S.shape
    for t in range(T-1):
        s=S[t].reshape(N,1); target=S[t+1].reshape(N,1)
        yhat=sigmoid(W@s, gain=gain)
        grad=(target-yhat)@s.T
        W += (ETA_DELTA * scale) * grad
        W *= (1.0 - L2_DECAY)
    return np.clip(W,0.0,1.0)

def eligibility_traces_from_spikes(S, dt_s):
    T,N=S.shape
    lam = np.exp(-dt_s/ELIG_TAU_S)
    E = np.zeros((N,N), np.float32)
    for t in range(T-1, -1, -1):
        s = S[t].astype(np.float32)
        E = lam*E + ELIG_SCALE * np.outer(s, s)
    if E.max()>1e-12: E/=E.max()
    return E
