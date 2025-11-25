# -*- coding: utf-8 -*-
"""
audition_babyai.py
Auditory Core for BabyAI (PyCharm-friendly, no librosa, no scipy)

Features:
  - STFT/ISTFT (numpy only)
  - Energy-based VAD for best segment
  - Feature extraction to fixed N dims (default N=100), per-frame normalization
  - Curiosity engine (novelty vs running EMA)
  - f0 estimation (autocorrelation)
  - Per-label prototype storage (JSON), canon frames resampling
  - Simple speech synthesis: sine source at learned f0 with smooth envelope

Dependencies:
  - numpy

Author: ChatGPT (for Kumail)
"""

from __future__ import annotations
import os, json, time, math, wave, struct
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np

# ---------------------------
# Constants / Config
# ---------------------------

SR: int = 16000                # sample rate
GUI_FRAME_MS: int = 10         # "frame" step for curiosity timing (10 ms)
TARGET_DIM: int = 100          # feature dimension (frequency bins)
CANON_FRAMES: int = 64         # canonical number of frames in prototype
MAIN_SPIKE_LOW: float = 0.2    # low spike threshold (for future use)
CURIOSITY_THR: float = 0.5     # threshold for "high curiosity"

BRAIN_PATH: str = os.environ.get("BABYAI_AUD_BRAIN", "/mnt/data/brain_audition.json")

# ---------------------------
# Utilities
# ---------------------------

def _l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(x) + eps)
    return (x / n).astype(np.float32)

def sanitize_label(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_")

# ---------------------------
# STFT / ISTFT (numpy-only)
# ---------------------------

def stft(y: np.ndarray, n_fft: int = 512, hop: int = 160, win: int = 400) -> np.ndarray:
    """Return complex STFT: (freq_bins, frames)."""
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if y.size < win:
        pad = np.zeros((win,), dtype=np.float32)
        pad[:y.size] = y
        y = pad
    window = np.hanning(win).astype(np.float32)
    frames = 1 + (len(y) - win) // hop
    out = []
    for i in range(frames):
        start, end = i * hop, i * hop + win
        frame = y[start:end] * window
        spec = np.fft.rfft(frame, n=n_fft)
        out.append(spec)
    return np.stack(out, axis=1)  # (freq, time)

def istft(S: np.ndarray, hop: int = 160, win: int = 400) -> np.ndarray:
    """ISTFT for complex spectrogram (freq, time)."""
    n_fft = (S.shape[0] - 1) * 2
    window = np.hanning(win).astype(np.float32)
    frames = S.shape[1]
    out_len = hop * (frames - 1) + win
    y = np.zeros((out_len,), dtype=np.float32)
    win_sum = np.zeros((out_len,), dtype=np.float32)
    for i in range(frames):
        spec = S[:, i]
        frame = np.fft.irfft(spec, n=n_fft).astype(np.float32)
        start, end = i * hop, i * hop + win
        y[start:end] += frame[:win] * window
        win_sum[start:end] += window
    nz = win_sum > 1e-8
    y[nz] /= win_sum[nz]
    return y

# ---------------------------
# Feature Extraction
# ---------------------------

def extract_features(y: np.ndarray, n_fft: int = 512, hop: int = 160, win: int = 400, out_dim: int = TARGET_DIM) -> np.ndarray:
    """
    Return T×N features (time frames × pooled frequency bins).
    - Magnitude STFT
    - Log-compression
    - Per-frame L2 norm
    - Pooled to out_dim via average grouping
    """
    S = stft(y, n_fft=n_fft, hop=hop, win=win)            # (freq, time)
    mag = np.abs(S).astype(np.float32) + 1e-8
    mag = np.log1p(mag)                                   # log compression
    # Pool frequency axis to out_dim
    f, t = mag.shape
    if f >= out_dim:
        stride = f // out_dim
        pooled = mag[:stride * out_dim, :].reshape(out_dim, stride, t).mean(axis=1)
    else:
        pooled = np.zeros((out_dim, t), dtype=np.float32)
        pooled[:f, :] = mag
    # Per-frame L2 norm
    X = []
    for i in range(pooled.shape[1]):
        X.append(_l2(pooled[:, i]))
    X = np.stack(X, axis=0)  # (time, out_dim)
    return X

# ---------------------------
# VAD (pick best voiced segment)
# ---------------------------

# ... keep the rest of your audition_babyai.py as you have it ...
# Replace only the vad_best_segment with this robust version:

def vad_best_segment(y: np.ndarray, sr: int = SR, frame_ms: int = 20, min_ms: int = 300, max_ms: int = 1500) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Energy-based VAD to pick one best segment. Robust to short clips.
    Returns (segment, meta).
    """
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    frame = max(1, int(sr * frame_ms / 1000.0))
    energies = [float(np.mean(y[i:i+frame] ** 2)) for i in range(0, max(1, len(y) - frame + 1), frame)]
    if len(energies) == 0:
        return y, {"start": 0, "end": len(y), "frames": 1, "energy_sum": 0.0}

    # Smooth energies
    win = max(1, int(50 / frame_ms))
    ker = np.ones((win,), dtype=np.float32) / float(win)
    sm = np.convolve(np.array(energies, dtype=np.float32), ker, mode="same")

    # Window search; guard against too-short sequences
    min_frames = max(1, int(min_ms / frame_ms))
    max_frames = max(min_frames, int(max_ms / frame_ms))

    best_i, best_w, best_e = 0, min_frames, -1.0
    found = False
    csum = np.cumsum(np.pad(sm, (1, 0)))

    for w in range(min_frames, max_frames + 1):
        if len(sm) < w:
            continue
        sums = csum[w:] - csum[:-w]  # sliding sum
        if sums.size == 0:
            continue
        j = int(np.argmax(sums))
        e = float(sums[j])
        if e > best_e:
            best_e, best_i, best_w = e, j, w
            found = True

    if not found:
        # fallback: whole audio
        return y, {"start": 0, "end": len(y), "frames": len(energies), "energy_sum": float(np.sum(sm))}

    start = best_i * frame
    end = min(len(y), (best_i + best_w) * frame)
    return y[start:end], {"start": start, "end": end, "frames": best_w, "energy_sum": best_e}


# ---------------------------
# Curiosity (novelty vs EMA)
# ---------------------------

@dataclass
class CuriosityConfig:
    frame_rate: float = 1000.0 / GUI_FRAME_MS  # frames per second for timing
    ema_beta: float = 0.1

class AuditoryCuriosity:
    def __init__(self, cfg: CuriosityConfig = CuriosityConfig()):
        self.cfg = cfg
        self.ema: Optional[np.ndarray] = None

    def process_frame(self, x: np.ndarray, t_sec: float) -> float:
        """
        x: 1D feature (TARGET_DIM)
        return: novelty in [0, 1]
        """
        x = _l2(x)
        if self.ema is None:
            self.ema = x.copy()
            return 1.0
        self.ema = (1.0 - self.cfg.ema_beta) * self.ema + self.cfg.ema_beta * x
        # novelty ~ 1 - cosine
        n = 1.0 - float(np.dot(_l2(x), _l2(self.ema)))
        n = max(0.0, min(1.0, n))
        return n

def offline_summary(series: List[float], thr: float = CURIOSITY_THR) -> Dict[str, Any]:
    if not series:
        return {"mean": 0.0, "max": 0.0, "above_thr": 0}
    arr = np.array(series, dtype=np.float32)
    return {"mean": float(arr.mean()), "max": float(arr.max()), "above_thr": int(np.sum(arr > thr))}

# ---------------------------
# f0 (autocorrelation) & speech
# ---------------------------

def estimate_f0_autocorr(y: np.ndarray, sr: int = SR, fmin: float = 70.0, fmax: float = 400.0) -> float:
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    y = y - float(np.mean(y))
    y = y / (float(np.std(y)) + 1e-8)
    n = len(y)
    # autocorr via FFT
    s = np.fft.rfft(y, n=2*n)
    ac = np.fft.irfft(s * np.conj(s))[:n]
    ac /= (ac[0] + 1e-8)
    # search lag window
    lag_min = max(1, int(sr / fmax))
    lag_max = min(n-1, int(sr / fmin))
    if lag_min >= lag_max:
        return 0.0
    lag = lag_min + int(np.argmax(ac[lag_min:lag_max]))
    f0 = float(sr / lag) if lag > 0 else 0.0
    return f0

def _write_wav(path: str, y: np.ndarray, sr: int = SR) -> None:
    """Write mono float32 [-1,1] as 16-bit PCM WAV (no scipy)."""
    y = np.clip(y, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sr)
        for v in y:
            w.writeframes(struct.pack("<h", int(v * 32767.0)))

def default_voice_profile() -> Dict[str, Any]:
    return {"sr": SR, "amp": 0.2}

def speak_and_save(label: str, P_canon: np.ndarray, avg_dur: float = 1.0,
                   speak_model: Dict[str, Any] = None, vp: Dict[str, Any] = None,
                   play: bool = False) -> Tuple[str, Dict[str, Any]]:
    """
    Simple synthesizer: sine source at estimated f0 with Hann envelope.
    """
    label = sanitize_label(label)
    sr = (vp or {}).get("sr", SR)
    amp = (vp or {}).get("amp", 0.2)
    dur = max(0.25, float(avg_dur))
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False).astype(np.float32)
    # Derive f0 from prototype "centroid" (dummy if not present)
    f0 = 180.0
    if isinstance(P_canon, np.ndarray) and P_canon.size > 0:
        # Map prototype index of max energy → f0 range
        idx = int(np.argmax(P_canon))
        f0 = 90.0 + 300.0 * (idx / max(1, P_canon.size - 1))
    sig = amp * np.sin(2.0 * np.pi * f0 * t)
    env = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.linspace(0, 1, sig.size, endpoint=False))
    y = (sig * env).astype(np.float32)
    out = f"/mnt/data/speech_{label}_{int(time.time())}.wav"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    _write_wav(out, y, sr=sr)
    return out, {"f0": f0, "dur": dur}

# ---------------------------
# Brain (per-label prototypes)
# ---------------------------

def load_brain(path: str = BRAIN_PATH) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_brain(brain: Dict[str, Any], path: str = BRAIN_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(brain, f)

def _resample_time_mean(X: np.ndarray, out_frames: int = CANON_FRAMES) -> np.ndarray:
    """Resample time frames to out_frames via piecewise average."""
    T, N = X.shape
    if T == out_frames:
        return X.copy()
    # map each target frame to a range in original frames
    idxs = np.linspace(0, T, out_frames + 1)
    Y = []
    for i in range(out_frames):
        a = int(idxs[i]); b = int(max(a + 1, idxs[i+1]))
        seg = X[a:b]
        Y.append(seg.mean(axis=0) if seg.size else X[min(a, T-1)])
    return np.stack(Y, axis=0)

def update_label(
    brain: Dict[str, Any],
    label: str,
    X_raw: np.ndarray,
    duration_s: float,
    curiosity: Dict[str, Any],
    N_feat: int,
    canon_frames: int = CANON_FRAMES,
    MAIN_SPIKE_LOW: float = MAIN_SPIKE_LOW,
    f0_hz: float = 0.0
) -> Tuple[Dict[str, Any], None, None, Dict[str, Any]]:
    """
    Update the persistent brain store for a label.
    Stores:
      - prototypes: list of {"P": [floats...]} where P is (canon_frames, N_feat) mean (flattened row-major)
      - avg_duration
      - speak_model (placeholder)
      - last_curiosity
      - f0
    """
    lab = sanitize_label(label)
    entry = brain.get(lab, {
        "prototypes": [],
        "avg_duration": float(duration_s),
        "speak_model": {},
        "last_curiosity": {},
        "f0_hz": float(f0_hz)
    })

    # Canonical proto as time-mean → (N_feat,) then store row-major (canon_frames × N_feat) mean per time bin
    Xr = _resample_time_mean(X_raw, out_frames=canon_frames)  # (canon_frames, N)
    P = Xr.mean(axis=0)                                       # (N,)
    entry["prototypes"].append({"P": P.tolist(), "created_at": time.time()})
    # Update averages
    entry["avg_duration"] = 0.9 * float(entry.get("avg_duration", duration_s)) + 0.1 * float(duration_s)
    entry["last_curiosity"] = curiosity
    if f0_hz > 0.0:
        entry["f0_hz"] = float(f0_hz)

    brain[lab] = entry
    info = {"label": lab, "num_prototypes": len(entry["prototypes"])}
    return entry, None, None, info
