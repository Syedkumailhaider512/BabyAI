# brain.py — persistence, time-based decay, prototypes (K<=3), updates

import os, json, datetime as dt
import numpy as np
import pandas as pd
from config import BRAIN_JSON, MASTER_CSV, EDGE_PRUNE, EMA_ALPHA, LAMBDA_FORGET, PRUNE_EPS, MAX_PROTOS_PER_LABEL
from config import SYNTH_GAIN
from utils import now_iso, parse_iso, sanitize_label
from dtw import dtw_align_to_proto, time_warp_to_canon
from plasticity import to_spikes_from_top_row, hebb_stdp_from_spikes, delta_rule_update, eligibility_traces_from_spikes

def load_brain():
    if os.path.exists(BRAIN_JSON):
        with open(BRAIN_JSON,"r",encoding="utf-8") as f: return json.load(f)
    return {}

def save_brain(brain):
    with open(BRAIN_JSON,"w",encoding="utf-8") as f: json.dump(brain,f,indent=2)

def W_from_sparse(edges, N):
    W=np.zeros((N,N),np.float32)
    for i,j,w in edges: W[int(i),int(j)]=float(w)
    return W

def sparse_from_W(W,thresh=0.0):
    N=W.shape[0]; edges=[]
    for i in range(N):
        for j in range(N):
            w=float(W[i,j])
            if w>=thresh: edges.append([int(i),int(j),round(w,6)])
    return edges

def _ensure_label_entry(brain, label, N_feat, canon_frames):
    e = brain.get(label)
    if e is None:
        e = {
            "count": 0, "updated_at": None, "last_decay_at": None,
            "prototypes": [],
            "last_strip": None, "avg_duration": None,
            "W_sparse": [], "N_feat": N_feat, "canon_frames": canon_frames,
            "threshold_low": 0.85,
            "speak_model": {"gamma":1.0,"tilt":0.0,"noise":0.0,"smooth":0.0,
                            "attack_ms":10,"release_ms":20,"gain":SYNTH_GAIN,
                            "tries":0,"last_error":None,"best_error":None}
        }
        brain[label] = e
    return e

def _time_decay_W(entry):
    if not entry["W_sparse"]: return
    last = parse_iso(entry.get("last_decay_at") or entry.get("updated_at"))
    if not last:
        entry["last_decay_at"] = now_iso();
        return
    dt_sec = (dt.datetime.utcnow() - last).total_seconds()
    if dt_sec <= 0: return
    W = W_from_sparse(entry["W_sparse"], entry["N_feat"])
    W *= np.exp(-LAMBDA_FORGET * dt_sec)
    W[W < PRUNE_EPS] = 0.0
    entry["W_sparse"] = sparse_from_W(W, EDGE_PRUNE)
    entry["last_decay_at"] = now_iso()

def _pick_or_create_proto(entry, Xc):
    if not entry["prototypes"]:
        entry["prototypes"].append({"P": Xc.tolist(), "count": 1})
        return 0, Xc
    dists = []
    for k,p in enumerate(entry["prototypes"]):
        P = np.array(p["P"], float)
        d = np.mean((Xc - P)**2)
        dists.append((d,k))
    d,kbest = min(dists)
    Pbest = np.array(entry["prototypes"][kbest]["P"], float)
    if d > 0.15 and len(entry["prototypes"]) < MAX_PROTOS_PER_LABEL:
        entry["prototypes"].append({"P": Xc.tolist(), "count": 1})
        return len(entry["prototypes"])-1, Xc
    return kbest, Pbest

def update_label(brain, label, X_raw, duration_s, curiosity, N_feat, canon_frames, MAIN_SPIKE_LOW, f0_hz=None):
    label = sanitize_label(label)
    entry = _ensure_label_entry(brain, label, N_feat, canon_frames)

    # ----- update per-label F0 (EMA) -----
    sm = entry.get("speak_model", {})
    if f0_hz is not None:
        prev = sm.get("f0_hz")
        sm["f0_hz"] = float((1.0-0.2)*prev + 0.2*float(f0_hz)) if prev is not None else float(f0_hz)
    entry["speak_model"] = sm

    # Decay weights by elapsed time
    _time_decay_W(entry)

    # Align
    if entry["prototypes"]:
        Pmean = np.mean([np.array(p["P"], float) for p in entry["prototypes"]], axis=0)
        Xc = dtw_align_to_proto(X_raw, Pmean)
    else:
        Xc = time_warp_to_canon(X_raw, canon_frames)

    # Choose / update prototype
    idx, Psel = _pick_or_create_proto(entry, Xc)
    P_new = (1.0-EMA_ALPHA)*Psel + EMA_ALPHA*Xc
    entry["prototypes"][idx]["P"] = P_new.tolist()
    entry["prototypes"][idx]["count"] = int(entry["prototypes"][idx].get("count",0)+1)

    # Spikes
    S = (Xc >= MAIN_SPIKE_LOW).astype(np.float32)

    # Weights
    W = W_from_sparse(entry.get("W_sparse", []), N_feat)

    # Eligibility traces
    E = eligibility_traces_from_spikes(S, dt_s=0.01)

    # Curiosity
    pscale = float(curiosity["gates"].get("plasticity_scale", 1.0)) if curiosity else 1.0
    dopamine = float(curiosity["summary"]["C_peak"] if curiosity else 0.0)

    # STDP + Delta
    W_stdp = hebb_stdp_from_spikes(S, scale=pscale)
    W = (1.0 - 0.3)*W + 0.3*W_stdp
    W = delta_rule_update(W, S, scale=pscale)
    W += 0.2 * dopamine * E
    W = np.clip(W, 0.0, 1.0)

    W[W < PRUNE_EPS] = 0.0
    entry["W_sparse"] = sparse_from_W(W, EDGE_PRUNE)

    # Misc
    entry["last_strip"] = X_raw.tolist()
    entry["avg_duration"] = float(duration_s if entry.get("avg_duration") is None
                                  else (1.0-EMA_ALPHA)*entry["avg_duration"] + EMA_ALPHA*duration_s)
    entry["count"] = int(entry.get("count",0)+1)
    entry["updated_at"] = now_iso()
    if curiosity:
        entry["last_curiosity"] = curiosity
    brain[label] = entry
    return entry

def get_best_prototype(brain, label):
    e = brain.get(label)
    if not e or not e.get("prototypes"):
        raise ValueError(f"No prototype for '{label}'. Record first.")
    idx = int(np.argmax([p.get("count",1) for p in e["prototypes"]]))
    P = np.array(e["prototypes"][idx]["P"], float)
    dur = float(e.get("avg_duration", P.shape[0]*0.1))
    sm = e.get("speak_model", {})
    return P, dur, sm
