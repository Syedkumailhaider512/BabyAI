# curiosity.py — live curiosity engine + offline summary (with safety gating)

import numpy as np
from dataclasses import dataclass
from config import GUI_FRAME_MS, LOUDNESS_MAX_DB, CURIOSITY_THR, DOPAMINE_THR

@dataclass
class CuriosityConfig:
    frame_rate: float
    loudness_max_db: float = LOUDNESS_MAX_DB
    curiosity_threshold: float = CURIOSITY_THR
    dopamine_threshold: float = DOPAMINE_THR

class AuditoryCuriosity:
    """
    Lightweight predictor: per-bin AR(3) with LMS updates.
    Curiosity = prediction error × Goldilocks (entropy) × deviance.
    """
    def __init__(self, cfg: CuriosityConfig, freqs=None):
        self.cfg = cfg
        self.p = 3
        self.w = None
        self.buf = None
        self.ema = None
        self.alpha = 0.1
        self.mu = 0.01

    def _init(self, N):
        self.w = np.zeros((N, self.p), float)
        self.buf = np.zeros((self.p, N), float)
        self.ema = np.zeros(N, float)

    def _roughness(self, x):
        return float(np.clip(np.mean(np.abs(np.diff(x))) * 5.0, 0.0, 1.0))

    def process_frame(self, mags, t_sec):
        x = np.asarray(mags).astype(float)
        N = x.shape[0]
        if self.w is None: self._init(N)

        amp = np.sqrt((x**2).mean()) + 1e-9
        x_n = x / (x.sum() + 1e-12)

        phi = self.buf
        y_hat = (self.w * phi.T).sum(axis=1)
        err = x - y_hat
        self.w += self.mu * (err.reshape(-1,1) * phi.T)

        self.buf[1:] = self.buf[:-1]
        self.buf[0] = x

        se = np.mean((err**2)) / (np.mean(x**2) + 1e-9)
        S_t = float(np.clip(se, 0.0, 4.0) / 4.0)

        H = -np.sum(x_n * np.log2(x_n + 1e-12))
        H /= np.log2(N)
        H_t = float(H)
        gold = 1.0 - 4.0*(H_t-0.5)**2

        self.ema = (1.0-0.05)*self.ema + 0.05*(x / (np.linalg.norm(x)+1e-9))
        ema_n = self.ema / (np.linalg.norm(self.ema)+1e-9)
        x_u = x / (np.linalg.norm(x)+1e-9)
        cos = float(np.dot(ema_n, x_u))
        D_t = float(np.clip(1.0 - cos, 0.0, 1.0))

        C_raw = S_t * gold * (0.5 + 0.5*D_t)
        C_t = float(np.clip(C_raw, 0.0, 1.0))

        L_db = 20.0*np.log10(amp + 1e-9) + 60.0
        R_norm = self._roughness(x)
        avoid = (L_db > self.cfg.loudness_max_db) or (R_norm > 0.85)

        dopamine_boost = C_t >= self.cfg.dopamine_threshold and not avoid
        plasticity_gain = 0.3 + 0.7 * C_t
        if avoid: plasticity_gain = 0.2

        return {
            "t": float(t_sec),
            "C_t": C_t,
            "S_t": S_t,
            "H_t": H_t,
            "D_t": D_t,
            "L_db": float(L_db),
            "R_norm": float(R_norm),
            "gates": {
                "avoid": bool(avoid),
                "dopamine_boost": bool(dopamine_boost),
                "plasticity_gain": float(plasticity_gain),
                "lc_mode": "phasic" if C_t>self.cfg.curiosity_threshold else "tonic"
            }
        }

def offline_summary(series, curiosity_threshold):
    if not series:
        return {"series": [], "summary": {}, "gates": {"plasticity_scale":0.2, "any_avoid":False, "any_dopamine":False, "mode":"tonic"}}
    avoid_count = sum(int(s["gates"]["avoid"]) for s in series)
    dop_count   = sum(int(s["gates"]["dopamine_boost"]) for s in series)
    cur_count   = sum(int(s["C_t"] > curiosity_threshold) for s in series)
    C_vals = [s["C_t"] for s in series]
    S_vals = [s["S_t"] for s in series]
    H_vals = [s["H_t"] for s in series]
    D_vals = [s["D_t"] for s in series]
    L_vals = [s["L_db"] for s in series]
    R_vals = [s["R_norm"] for s in series]
    avoid_ratio = avoid_count/len(series)
    gains = [s["gates"]["plasticity_gain"] for s in series if not s["gates"]["avoid"]]
    plasticity_scale = float(np.mean(gains)) if gains else 0.2
    if avoid_ratio>0.4: plasticity_scale = min(plasticity_scale, 0.4)
    summary = {
        "C_mean": float(np.mean(C_vals)), "C_peak": float(np.max(C_vals)),
        "S_mean": float(np.mean(S_vals)), "H_mean": float(np.mean(H_vals)), "D_mean": float(np.mean(D_vals)),
        "L_db_mean": float(np.mean(L_vals)), "R_mean": float(np.mean(R_vals)),
        "avoid_ratio": float(avoid_ratio),
        "dopamine_ratio": float(dop_count/len(series)),
        "curiosity_ratio": float(cur_count/len(series)),
    }
    gates = {
        "plasticity_scale": plasticity_scale,
        "any_avoid": avoid_count>0,
        "any_dopamine": dop_count>0,
        "mode": "phasic" if cur_count/len(series)>0.2 else "tonic"
    }
    # downsample to ≤60 points for JSON
    if len(series)>60:
        idx = np.linspace(0,len(series)-1,60).astype(int).tolist()
        series = [series[i] for i in idx]
    return {"series": series, "summary": summary, "gates": gates}
