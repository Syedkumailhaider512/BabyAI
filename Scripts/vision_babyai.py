# -*- coding: utf-8 -*-
"""
vision_babyai.py
Biologically-inspired Vision Core for BabyAI

Pipeline:
  Retina DoG (ON/OFF) → LGN divisive norm → V1 Gabor energy bank
  → Optional IT invariance (small rot/scale pooling) → fixed-length feature

Also includes VisionBrain:
  - Per-label EMA prototypes
  - Novelty-based auto-dopamine (RPE-like)
  - Time-based exponential decay (forgetting) for prototypes
  - JSON persistence

Dependencies:
  - numpy
  - opencv-python  (pip install opencv-python)

Author: ChatGPT (for Kumail)
"""

from __future__ import annotations
import os, json, time, math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

import numpy as np

try:
    import cv2
except Exception as e:
    raise ImportError(
        "OpenCV is required for vision_babyai.py. Install with: pip install opencv-python\n"
        f"Original error: {e}"
    )

# ---------------------------
# Parameters
# ---------------------------

@dataclass
class RetinaParams:
    sigma_on: float = 1.2
    sigma_off: float = 2.4

@dataclass
class V1Params:
    num_orient: int = 8            # orientations
    kernel_sizes: Tuple[int, ...] = (7, 11, 15)  # odd sizes → spatial frequencies

@dataclass
class ITParams:
    use_it: bool = True
    rotations_deg: Tuple[float, ...] = (-15.0, -7.0, 0.0, 7.0, 15.0)
    scales: Tuple[float, ...] = (0.9, 1.0, 1.1)
    pool: str = "max"  # {"max", "mean"}

# ---------------------------
# Utils
# ---------------------------

_DEF_FEAT_DIM = 512  # final vector size

def _l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(x) + eps)
    return (x / n).astype(np.float32)

def _cos(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.dot(_l2(a, eps), _l2(b, eps)))

def _to_gray01(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB/Gray ndarray to grayscale float32 in [0,1]."""
    if img_bgr.ndim == 2:
        gray = img_bgr.astype(np.float32)
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rng = gray.max() - gray.min()
    if rng < 1e-8:
        return np.zeros_like(gray, dtype=np.float32)
    return (gray - gray.min()) / rng

def _retina_dog(gray01: np.ndarray, s_on: float, s_off: float) -> Tuple[np.ndarray, np.ndarray]:
    """Difference of Gaussians → ON & OFF channels."""
    on  = cv2.GaussianBlur(gray01, (0, 0), s_on)
    off = cv2.GaussianBlur(gray01, (0, 0), s_off)
    on_ch  = np.clip(on - off, 0, 1).astype(np.float32)
    off_ch = np.clip(off - on, 0, 1).astype(np.float32)
    return on_ch, off_ch

def _lgn_divisive_norm(ch: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    mu = float(ch.mean())
    sd = float(ch.std()) + eps
    return ((ch - mu) / sd).astype(np.float32)

def _v1_gabor_energy(base: np.ndarray, num_orient: int, kernel_sizes: Tuple[int, ...]) -> np.ndarray:
    """
    Apply a bank of Gabor filters (OpenCV kernels) and return energy maps stacked (C,H,W).
    """
    H, W = base.shape
    thetas = np.linspace(0.0, np.pi, num_orient, endpoint=False)
    maps: List[np.ndarray] = []
    for ksz in kernel_sizes:
        lam = max(4.0, 0.6 * ksz)
        sigma = 0.5 * ksz
        gamma = 0.5
        for th in thetas:
            kern = cv2.getGaborKernel((ksz, ksz), sigma, th, lam, gamma, psi=0, ktype=cv2.CV_32F)
            resp = cv2.filter2D(base, cv2.CV_32F, kern)
            maps.append(np.abs(resp).astype(np.float32))
    stack = np.stack(maps, axis=0) if maps else np.zeros((1, H, W), dtype=np.float32)
    return stack

def _spatial_pool_to_dim(x: np.ndarray, out_dim: int = _DEF_FEAT_DIM) -> np.ndarray:
    """
    Global-average pool spatial dims → channel vector, then resize/pad to fixed dim.
    """
    if x.ndim == 3:
        c, h, w = x.shape
        v = x.reshape(c, -1).mean(axis=1).astype(np.float32)
    else:
        v = x.reshape(1, -1).mean(axis=1).astype(np.float32)

    if v.size > out_dim:
        stride = v.size // out_dim
        v = v[:stride * out_dim].reshape(out_dim, stride).mean(axis=1)
    elif v.size < out_dim:
        pad = np.zeros((out_dim,), dtype=np.float32)
        pad[:v.size] = v
        v = pad
    return _l2(v)

# ---------------------------
# Public feature function
# ---------------------------

def v1_feature_from_bgr(
    img_bgr: np.ndarray,
    ret_params: RetinaParams = RetinaParams(),
    v1_params:  V1Params  = V1Params(),
    it_params:  Optional[ITParams] = None,
    out_dim: int = _DEF_FEAT_DIM,
) -> np.ndarray:
    """
    Full pipeline → fixed-length feature vector (float32).
    If it_params.use_it, do small rot/scale pooling (IT-like invariance).
    """
    gray = _to_gray01(img_bgr)

    def one_pass(g: np.ndarray) -> np.ndarray:
        on, off = _retina_dog(g, ret_params.sigma_on, ret_params.sigma_off)
        on = _lgn_divisive_norm(on); off = _lgn_divisive_norm(off)
        base = 0.5 * (on + off)
        v1 = _v1_gabor_energy(base, v1_params.num_orient, v1_params.kernel_sizes)
        return _spatial_pool_to_dim(v1, out_dim)

    if not it_params or not it_params.use_it:
        return one_pass(gray)

    # IT-like pooling
    feats: List[np.ndarray] = []
    H, W = gray.shape
    for a in it_params.rotations_deg:
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), float(a), 1.0)
        g_rot = cv2.warpAffine(gray, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        for s in it_params.scales:
            g_s = cv2.resize(g_rot, None, fx=float(s), fy=float(s), interpolation=cv2.INTER_LINEAR)
            g_s = cv2.resize(g_s, (W, H), interpolation=cv2.INTER_AREA)
            feats.append(one_pass(g_s))

    F = np.stack(feats, axis=0) if feats else one_pass(gray)[None, :]
    pooled = F.max(axis=0) if (it_params.pool == "max") else F.mean(axis=0)
    return _l2(pooled)

# ---------------------------
# VisionBrain (learning + persistence)
# ---------------------------

class VisionBrain:
    """
    Stores per-label prototypes; supports novelty → auto-dopamine; time decay; JSON persistence.
    brain.json schema:
      {
        "labels": {
          "label": {
             "proto": [floats...],
             "count": int,
             "updated_at": float
          },
          ...
        },
        "last_decay_ts": float
      }
    """

    def __init__(self, path: str = "./vision_brain.json", beta: float = 0.2, tau_proto_seconds: float = 1800.0):
        self.path = path
        self.beta = float(beta)
        self.tau_proto_seconds = float(tau_proto_seconds)
        self.labels: Dict[str, Dict[str, Any]] = {}
        self.last_decay_ts: float = time.time()
        self._load()

    # --- persistence
    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.labels = data.get("labels", {})
            self.last_decay_ts = float(data.get("last_decay_ts", time.time()))
        else:
            self._save()

    def _save(self) -> None:
        data = {"labels": self.labels, "last_decay_ts": self.last_decay_ts}
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    # --- decay
    def _apply_time_decay(self) -> None:
        now = time.time()
        dt = max(0.0, now - self.last_decay_ts)
        if dt > 0.0 and self.tau_proto_seconds > 0.0:
            factor = math.exp(-dt / self.tau_proto_seconds)
            for lab in list(self.labels.keys()):
                proto = np.array(self.labels[lab]["proto"], dtype=np.float32)
                proto = (proto * factor).astype(np.float32)
                self.labels[lab]["proto"] = proto.tolist()
        self.last_decay_ts = now

    # --- auto dopamine from novelty (1 - cosine)
    def _auto_dopamine(self, feat: np.ndarray) -> float:
        if not self.labels:
            return 1.0
        sims = []
        for lab, rec in self.labels.items():
            p = np.array(rec["proto"], dtype=np.float32)
            sims.append(_cos(feat, p))
        conf = max(sims) if sims else 0.0
        novelty = max(0.0, 1.0 - conf)
        # Keep within [0,1.5] to mirror GUI ranges
        return float(max(0.0, min(1.5, novelty + 0.1)))

    # --- main API
    def record_image(
        self,
        img_bgr: np.ndarray,
        label: str,
        ret_params: RetinaParams = RetinaParams(),
        v1_params:  V1Params  = V1Params(),
        auto_dopamine: bool = True,
        it_params: Optional[ITParams] = ITParams(use_it=True),
        store_dir: Optional[str] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Run pipeline, compute dopamine, update prototype (EMA), persist.
        Returns: (dopamine_used, info_dict)
        """
        self._apply_time_decay()
        feat = v1_feature_from_bgr(img_bgr, ret_params, v1_params, it_params, out_dim=_DEF_FEAT_DIM)

        # novelty → DA
        da = self._auto_dopamine(feat) if auto_dopamine else 1.0

        # update EMA prototype
        lab = str(label).strip().lower()
        rec = self.labels.get(lab)
        if rec is None:
            rec = {"proto": feat.tolist(), "count": 0, "updated_at": time.time()}
            self.labels[lab] = rec
        else:
            proto = np.array(rec["proto"], dtype=np.float32)
            proto = (1.0 - self.beta) * proto + self.beta * feat
            rec["proto"] = _l2(proto).tolist()
            rec["updated_at"] = time.time()
        rec["count"] = int(rec.get("count", 0)) + 1

        # (optional) dump intermediate maps if requested
        if store_dir:
            try:
                os.makedirs(store_dir, exist_ok=True)
                # Save a small preview: the input gray
                gray = _to_gray01(img_bgr)
                prev = (gray * 255.0).astype(np.uint8)
                cv2.imwrite(os.path.join(store_dir, f"{lab}_{rec['count']:04d}.png"), prev)
            except Exception:
                pass

        self._save()
        info = {"label": lab, "count": rec["count"], "feature_dim": int(feat.size)}
        return float(da), info
