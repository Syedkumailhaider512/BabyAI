# multisensory_gui.py — BabyAI Gradio GUI aligned with Tk pipeline
# Deps: pip install gradio plotly opencv-python numpy

import os, io, json, time, math, contextlib, wave
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import gradio as gr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Optional OpenCV (preferred); fallback to PIL for image I/O ---
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False
    from PIL import Image

# --- Plotly for interactive charts ---
import plotly.graph_objs as go

# --- Your project modules (Tk pipeline parity) ---
try:
    import config as CFG
except Exception:
    class CFG:
        BRAIN_JSON = os.path.join(os.getcwd(), "brain.json")
        CANON_FRAMES = 64
        USE_MEL_FRONTEND = False

try:
    import frontend as FE
except Exception:
    FE = None

try:
    import dtw as DTW
except Exception:
    DTW = None

try:
    import synthesis as SYN
except Exception:
    SYN = None

# Optional: 3D viz & recognition helpers you already have
try:
    from plotly_viz_and_rec import (
        voice_helix_fig,
        brain_3d_fig,
        auto_dopamine,
        robust_recognize_from_image,
        _softmax_conf,
    )
    HAS_VIZ_HELPERS = True
except Exception:
    HAS_VIZ_HELPERS = False

# Optional: your vision encoder
try:
    import vision_babyai as V
except Exception:
    V = None


# =========================
# Image resize (keep aspect)
# =========================
def _resize_keep_aspect(bgr: np.ndarray, target_w: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w == 0 or h == 0:
        return bgr
    scale = float(target_w) / float(w)
    new_h = max(1, int(round(h * scale)))
    if HAS_CV2:
        return cv2.resize(bgr, (target_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        from PIL import Image as _PILImage
        rgb = bgr[:, :, ::-1]
        im = _PILImage.fromarray(rgb)
        im = im.resize((target_w, new_h), resample=_PILImage.Resampling.LANCZOS)
        return np.array(im)[:, :, ::-1].copy()

IMG_TARGET_W = 500  # display width (px), height auto by aspect


# =========================
# Paths / persistence
# =========================
MEM_DIR = os.path.join(os.getcwd(), "mem")
os.makedirs(MEM_DIR, exist_ok=True)

BRAIN_PATH = getattr(CFG, "BRAIN_JSON", os.path.join(os.getcwd(), "brain.json"))

def _load_brain() -> Dict[str, Any]:
    if os.path.isfile(BRAIN_PATH):
        try:
            with open(BRAIN_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"labels": {}, "meta": {"version": 1}}

def _save_brain(B: Dict[str, Any]):
    tmp = BRAIN_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(B, f)
    os.replace(tmp, BRAIN_PATH)


# =========================
# Utils
# =========================
def _to_bgr(img_path: str) -> np.ndarray:
    if HAS_CV2:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        return _resize_keep_aspect(img, IMG_TARGET_W)
    else:
        from PIL import Image as _PILImage
        im = _PILImage.open(img_path).convert("RGB")
        w, h = im.size
        scale = float(IMG_TARGET_W) / float(w)
        new_h = max(1, int(round(h * scale)))
        im = im.resize((IMG_TARGET_W, new_h), resample=_PILImage.Resampling.LANCZOS)
        return np.array(im)[:, :, ::-1].copy()  # RGB->BGR

def _to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if HAS_CV2 else bgr[:, :, ::-1].copy()

def _l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).ravel()
    n = float(np.linalg.norm(x) + eps)
    return (x / n).astype(np.float32)

def _resample_time(X: np.ndarray, T: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == T:
        return X
    t_in = np.linspace(0.0, 1.0, X.shape[0])
    t_out = np.linspace(0.0, 1.0, T)
    out = np.stack([np.interp(t_out, t_in, X[:, j]) for j in range(X.shape[1])], axis=1)
    return out.astype(np.float32)


# =========================
# Audio pipeline (Tk parity)
# =========================
def dev_features_from_audio(audio_tuple) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Returns (X_raw[T,N], sr, X_canon[Tc,N]) using the *same* pipeline as gui_main.py:
      - FE.vad_best_segment(y, sr)
      - FE.extract_features(seg)  # uses config.USE_MEL_FRONTEND + SR internally
      - DTW (if available) to canonical frames CFG.CANON_FRAMES
    """
    if audio_tuple is None:
        return np.zeros((0, 0), np.float32), 16000, np.zeros((0, 0), np.float32)

    sr, wave_np = audio_tuple
    y = np.asarray(wave_np, dtype=np.float32).ravel()

    # VAD trim
    if FE and hasattr(FE, "vad_best_segment"):
        try:
            seg, _ = FE.vad_best_segment(y, sr=sr)
            if seg is None or seg.size == 0:
                seg = y
        except Exception:
            seg = y
    else:
        seg = y

    # FE.extract_features(seg) — signature w/out sr kwarg
    if FE and hasattr(FE, "extract_features"):
        try:
            X = FE.extract_features(seg)
        except Exception:
            T = max(1, int(len(seg) / max(1, int(sr/100))))
            N = 100
            X = np.abs(np.random.randn(T, N)).astype(np.float32) * 0.01
    else:
        T = max(1, int(len(seg) / max(1, int(sr/100))))
        N = 100
        X = np.abs(np.random.randn(T, N)).astype(np.float32) * 0.01

    # Canonicalize in time (DTW or resample)
    T_canon = int(getattr(CFG, "CANON_FRAMES", 64))
    if DTW is not None and hasattr(DTW, "align_to_canonical"):
        try:
            Xc = DTW.align_to_canonical(X, T_canon)
        except Exception:
            Xc = _resample_time(X, T_canon)
    else:
        Xc = _resample_time(X, T_canon)

    return X, sr, Xc


# =========================
# Vision encoding (proto)
# =========================
def encode_vision_feature(bgr: np.ndarray) -> np.ndarray:
    if V is not None and hasattr(V, "encode"):
        try:
            return _l2(V.encode(bgr))
        except Exception:
            pass
    # Fallback: simple HOG-ish downsample
    if HAS_CV2:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    else:
        from PIL import Image as _PILImage
        gray = bgr.mean(axis=2).astype(np.float32) / 255.0
        small = np.array(
            _PILImage.fromarray((gray*255).astype(np.uint8)).resize((32,32))
        ).astype(np.float32)/255.0
    feat = np.concatenate([small.mean(axis=0), small.mean(axis=1), small.ravel()])
    return _l2(feat)


# =========================
# Plotly helpers (2D)
# =========================
def plotly_line(y: np.ndarray, title: str, x_title: str = "Index", y_title: str = "Value") -> go.Figure:
    y = np.asarray(y, dtype=np.float32).ravel()
    x = np.arange(y.size)
    fig = go.Figure(data=[go.Scatter(x=x, y=y, mode="lines")])
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, margin=dict(l=10, r=10, b=40, t=40))
    return fig

def plotly_area(y: np.ndarray, title: str) -> go.Figure:
    y = np.asarray(y, dtype=np.float32).ravel()
    x = np.arange(y.size)
    fig = go.Figure(data=[go.Scatter(x=x, y=y, fill="tozeroy", mode="lines")])
    fig.update_layout(title=title, xaxis_title="Band", yaxis_title="Energy", margin=dict(l=10, r=10, b=40, t=40))
    return fig


# =========================
# Co-firing spike animation (Plotly)
# =========================
def cofiring_spike_frames(v: np.ndarray, X: np.ndarray, spike_v: float = 0.15, spike_a: float = 0.2) -> go.Figure:
    v = _l2(np.asarray(v, dtype=np.float32))
    if X.ndim != 2 or X.size == 0:
        return go.Figure()
    T, N = X.shape
    # simple spikes: thresholded features
    v_spk = (v > spike_v).astype(np.float32)

    frames = []
    for t in range(T):
        a = _l2(X[t])
        a_spk = (a > spike_a).astype(np.float32)
        M = np.outer(v_spk, a_spk)
        frames.append(go.Frame(data=[go.Heatmap(z=M, colorscale="Inferno", zmin=0, zmax=1)], name=f"t{t}"))

    fig = go.Figure(
        data=[go.Heatmap(z=np.outer(v_spk, (X[0] > spike_a).astype(np.float32)), colorscale="Inferno", zmin=0, zmax=1)],
        layout=go.Layout(
            title="Co-firing spikes over time",
            xaxis=dict(title="Audio units"),
            yaxis=dict(title="Vision units"),
            updatemenus=[dict(type="buttons", showactive=True, y=1.05, x=0.05, xanchor="left",
                              buttons=[dict(label="Play", method="animate",
                                            args=[None, dict(frame=dict(duration=60, redraw=True),
                                                             fromcurrent=True, mode="immediate")]),
                                       dict(label="Pause", method="animate",
                                            args=[[None], dict(frame=dict(duration=0),
                                                               mode="immediate")])])],
            margin=dict(l=10, r=10, b=40, t=40)
        ),
        frames=frames
    )
    return fig


# =========================
# Vision IT (Matplotlib)
# =========================
def _to_gray01(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        gray = img_bgr.astype(np.float32)
    else:
        if HAS_CV2:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            from PIL import Image as _PILImage
            im = _PILImage.fromarray(img_bgr[:, :, ::-1])
            gray = np.array(im.convert("L"), dtype=np.float32)
    rng = gray.max() - gray.min()
    if rng < 1e-8:
        return np.zeros_like(gray, dtype=np.float32)
    return (gray - gray.min()) / rng

def _gabor_kernel(ksize: int, theta: float, lam: float, sigma: float, gamma: float = 0.5) -> np.ndarray:
    ax = np.arange(ksize, dtype=np.float32) - (ksize - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    x =  xx * np.cos(theta) + yy * np.sin(theta)
    y = -xx * np.sin(theta) + yy * np.cos(theta)
    g = np.exp(-(x**2 + (gamma**2)*(y**2)) / (2*(sigma**2))) * np.cos(2*np.pi*x/lam)
    g -= g.mean()
    g /= (np.linalg.norm(g) + 1e-8)
    return g.astype(np.float32)

def vision_it_panel(img_bgr: np.ndarray,
                    num_orient: int = 8,
                    kernel_sizes: Tuple[int, ...] = (7, 11, 15),
                    title: str = "Vision — RGC & V1 (orientation × scale)") -> plt.Figure:
    gray = _to_gray01(img_bgr)
    if HAS_CV2:
        on  = cv2.GaussianBlur(gray, (0,0), 1.2)
        off = cv2.GaussianBlur(gray, (0,0), 2.4)
    else:
        on = gray; off = gray
    on  = np.clip(on - off, 0, 1)
    off = np.clip(off - on, 0, 1)
    base = 0.5*(on+off)

    thetas = np.linspace(0.0, np.pi, num_orient, endpoint=False)
    maps: List[np.ndarray] = []
    for k in kernel_sizes:
        lam = max(4.0, 0.6*k); sigma = 0.5*k
        for th in thetas:
            kern = _gabor_kernel(k, th, lam=lam, sigma=sigma)
            if HAS_CV2:
                resp = cv2.filter2D(base, cv2.CV_32F, kern)
            else:
                resp = base
            maps.append(np.abs(resp))

    n_rows = 2 + len(kernel_sizes)
    n_cols = max(4, num_orient)
    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(title, fontsize=14)

    ax = fig.add_subplot(n_rows, 2, 1)
    ax.imshow(on, cmap="magma"); ax.set_title("RGC ON"); ax.axis("off")
    ax = fig.add_subplot(n_rows, 2, 2)
    ax.imshow(off, cmap="magma"); ax.set_title("RGC OFF"); ax.axis("off")

    idx_local = 0
    for si, k in enumerate(kernel_sizes):
        row = si + 3
        for oi in range(num_orient):
            m = maps[idx_local]; idx_local += 1
            ax = fig.add_subplot(n_rows, n_cols, (row-1)*n_cols + oi + 1)
            ax.imshow(m, cmap="magma")
            ax.set_title(f"k={k} θ={int(thetas[oi]*180/np.pi)}°", fontsize=8)
            ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# =========================
# Persistent brain ops
# =========================
def _store_audio_proto(B: Dict[str, Any], lab: str, a_feat: np.ndarray, sr: int):
    B["labels"].setdefault(lab, {}).setdefault("audio", {})
    B["labels"][lab]["audio"]["feat"] = np.asarray(a_feat, dtype=np.float32).tolist()
    B["labels"][lab]["audio"]["sr"] = int(sr)

def _get_audio_proto(B: Dict[str, Any], lab: str) -> Tuple[Optional[np.ndarray], int]:
    node = B.get("labels", {}).get(lab, {}).get("audio", None)
    if not node: return None, 16000
    a = np.asarray(node.get("feat", []), dtype=np.float32)
    sr = int(node.get("sr", 16000))
    return (a if a.size else None), sr

def _store_vision_proto(B: Dict[str, Any], lab: str, v: np.ndarray, ema: float = 0.9):
    B["labels"].setdefault(lab, {}).setdefault("vision", {})
    p = np.asarray(B["labels"][lab]["vision"].get("proto", []), dtype=np.float32)
    v = _l2(v)
    if p.size:
        p = _l2(ema * p + (1 - ema) * v)
    else:
        p = v
    B["labels"][lab]["vision"]["proto"] = p.tolist()

def _get_vision_proto(B: Dict[str, Any], lab: str) -> Optional[np.ndarray]:
    p = np.asarray(B.get("labels", {}).get(lab, {}).get("vision", {}).get("proto", []), dtype=np.float32)
    return p if p.size else None


# =========================
# Global state
# =========================
IMGS_BGR: List[np.ndarray] = []
DISPLAY_RGB: List[np.ndarray] = []
BRAIN = _load_brain()


# =========================
# Callbacks
# =========================
def add_images(paths: List[str]):
    global IMGS_BGR, DISPLAY_RGB
    if not paths:
        return "No images selected.", [], "0"
    added = 0
    for p in paths:
        try:
            bgr = _to_bgr(p)
            IMGS_BGR.append(bgr)
            DISPLAY_RGB.append(_to_rgb(bgr))
            added += 1
        except Exception as e:
            print("Add error:", e)
    return f"Loaded {added} image(s). (display width={IMG_TARGET_W}px, aspect preserved)", DISPLAY_RGB, str(max(0, len(IMGS_BGR) - added))

def enroll_concept(idx_str: str, audio_tuple, label: str, tau_val: float, temp: float):
    if not IMGS_BGR:
        return "Add images first.", None, None, None, None
    if idx_str is None or not str(idx_str).isdigit():
        return "Click an image in the gallery to select it.", None, None, None, None
    idx = int(idx_str)
    if idx < 0 or idx >= len(IMGS_BGR):
        return f"Index out of range (0..{len(IMGS_BGR)-1}).", None, None, None, None
    lab = (label or "").strip()
    if not lab:
        return "Enter a label/name for this concept.", None, None, None, None
    if audio_tuple is None:
        return "Record a reference voice for this concept.", None, None, None, None

    # Audio dev features
    X, sr, Xc = dev_features_from_audio(audio_tuple)
    if X.size == 0:
        return "Audio too short/silent.", None, None, None, None
    a_feat = X.mean(axis=0)

    # Vision feature
    v_feat = encode_vision_feature(IMGS_BGR[idx])

    # Auto-DA (safe across dims)
    if HAS_VIZ_HELPERS:
        DA = auto_dopamine(v_feat, a_feat)
    else:
        L = min(v_feat.size, a_feat.size)
        v_p = v_feat if v_feat.size == L else _resample_time(v_feat[None, :], L)[0]
        a_p = a_feat if a_feat.size == L else _resample_time(a_feat[None, :], L)[0]
        DA = float(np.clip(0.5 * (1.0 + float(np.dot(_l2(v_p), _l2(a_p)))), 0.0, 1.0))

    # Update persistent store
    _store_vision_proto(BRAIN, lab, v_feat, ema=0.9)
    _store_audio_proto(BRAIN, lab, a_feat, sr)
    _save_brain(BRAIN)

    # Visuals
    vfig = plotly_line(v_feat, f"Vision features — {lab}")
    afig = plotly_area(a_feat, f"Audio bands — {lab}")
    cfig = cofiring_spike_frames(v_feat, Xc)

    return f"Enrolled '{lab}'. Auto-DA={DA:.2f}. Saved to brain.", vfig, afig, cfig, json.dumps({"frames": int(Xc.shape[0]), "bands": int(Xc.shape[1])}, indent=2)

def recognize_image(idx_str: str, tau_val: float, temp: float, ask_confirm: bool):
    if not IMGS_BGR:
        return "Add images first.", None, "{}", None
    if idx_str is None or not str(idx_str).isdigit():
        return "Click an image to select.", None, "{}", None
    idx = int(idx_str)
    if idx < 0 or idx >= len(IMGS_BGR):
        return f"Index out of range.", None, "{}", None

    q = encode_vision_feature(IMGS_BGR[idx])

    # Compare to vision prototypes
    labels = list(BRAIN.get("labels", {}).keys())
    sims = {}
    for lab in labels:
        vp = _get_vision_proto(BRAIN, lab)
        if vp is not None:
            L = min(len(vp), len(q))
            sims[lab] = float(np.dot(_l2(vp[:L]), _l2(q[:L])))

    if HAS_VIZ_HELPERS:
        lab, conf, scores = _softmax_conf(sims, temp=float(temp))
    else:
        if sims:
            labs = list(sims.keys()); vals = np.array([sims[k] for k in labs], dtype=np.float32)
            z = (vals - vals.max()) / max(float(temp), 1e-6)
            p = np.exp(z); p /= (p.sum() + 1e-8)
            j = int(np.argmax(p)); lab = labs[j]; conf = float(p[j])
            scores = {labs[i]: float(p[i]) for i in range(len(labs))}
        else:
            lab, conf, scores = None, float("nan"), {}

    if lab is None or not np.isfinite(conf) or conf < float(tau_val):
        return f"Abstained (conf={conf:.3f} < τ={float(tau_val):.2f}).", plotly_line(q, "Vision features (query)"), json.dumps(scores, indent=2), None

    msg = f"Prediction: {lab} (conf={conf:.3f}, τ={float(tau_val):.2f})"
    if ask_confirm:
        msg += " — confirm below to strengthen."
    return msg, plotly_line(q, "Vision features (query)"), json.dumps(scores, indent=2), lab

def confirm_recognition(idx_str: str, predicted_lab: str):
    """Strengthen the vision prototype for the predicted label with EMA."""
    if not predicted_lab:
        return "Nothing to confirm."
    if not IMGS_BGR:
        return "Add images first."
    if idx_str is None or not str(idx_str).isdigit():
        return "Click an image to select."
    idx = int(idx_str)
    if idx < 0 or idx >= len(IMGS_BGR):
        return "Index out of range."
    v = encode_vision_feature(IMGS_BGR[idx])
    _store_vision_proto(BRAIN, predicted_lab, v, ema=0.95)  # slightly stronger EMA (more conservative)
    _save_brain(BRAIN)
    return f"Reinforced '{predicted_lab}' and stored image in brain."

def _mel2hz(m): return 700.0 * (10.0**(m/2595.0) - 1.0)
def _hz2mel(f): return 2595.0 * np.log10(1.0 + (f/700.0))
def _mel_centers(n_bands: int, fmin: float, fmax: float):
    mmin, mmax = _hz2mel(fmin), _hz2mel(fmax)
    m = np.linspace(mmin, mmax, num=n_bands)
    return _mel2hz(m)

def _sine_from_bands(a_feat: np.ndarray, sr: int = 16000, dur: float = 0.9) -> np.ndarray:
    a = np.asarray(a_feat, dtype=np.float32).ravel()
    if a.size == 0:
        return np.zeros(int(sr * dur), dtype=np.float32)
    idx = np.argsort(-a)[:min(8, a.size)]
    amps = np.maximum(a[idx], 0.0) ** 0.6
    amps = amps / (amps.max() + 1e-8)
    freqs = _mel_centers(a.size, 90.0, 6000.0)[idx]
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False).astype(np.float32)
    y = np.zeros_like(t, dtype=np.float32)
    for w, f in zip(amps, freqs):
        for h in (1, 2, 3):
            y += (w / (h**1.2)) * np.sin(2*np.pi*(f*h)*t).astype(np.float32)
    pk = float(np.max(np.abs(y))) if y.size else 0.0
    if pk > 1e-6:
        y = y / pk * (10 ** (-3/20.0))
    fade = int(0.02 * sr)
    if fade > 0 and y.size > 2*fade:
        win = np.linspace(0, 1, fade).astype(np.float32)
        y[:fade] *= win
        y[-fade:] *= win[::-1]
    return y

def _save_wav(path: str, y: np.ndarray, sr: int = 16000):
    y = np.asarray(y, dtype=np.float32)
    y = np.clip(y, -1.0, 1.0)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((y * 32767.0).astype(np.int16).tobytes())

def speak_selected(idx_str: str, tau_val: float, temp: float, force_top: bool):
    if not IMGS_BGR:
        return "Add images first.", None
    if idx_str is None or not str(idx_str).isdigit():
        return "Click an image to select.", None
    idx = int(idx_str)
    if idx < 0 or idx >= len(IMGS_BGR):
        return f"Index out of range.", None

    # Decide label via recognition
    msg, _, scores_json, lab = recognize_image(idx_str, tau_val, temp, ask_confirm=False)
    if lab is None and not force_top:
        return msg, None
    if lab is None and force_top:
        scores = json.loads(scores_json) if scores_json else {}
        if not scores:
            return "No labels learned yet; enroll first.", None
        lab = max(scores.items(), key=lambda kv: kv[1])[0]

    out_path = None
    a_feat, sr = _get_audio_proto(BRAIN, lab)

    # Try synthesizer
    if SYN is not None and hasattr(SYN, "speak_and_save") and a_feat is not None:
        try:
            avg_dur = 0.9
            ts = int(time.time())
            candidate = os.path.join(MEM_DIR, f"speech_{lab}_{ts}.wav")
            ret = SYN.speak_and_save(lab, a_feat, avg_dur, None, None, play=False)
            if isinstance(ret, (bytes, bytearray)):
                with open(candidate, "wb") as f:
                    f.write(ret)
                out_path = candidate
            elif isinstance(ret, str):
                out_path = ret
            else:
                # ensure a file exists at least
                yfb = _sine_from_bands(a_feat, sr=sr, dur=avg_dur)
                _save_wav(candidate, yfb, sr=sr)
                out_path = candidate
        except Exception:
            out_path = None

    # Fallback: procedural sine synth from bands
    if out_path is None:
        if a_feat is None:
            return f"'{lab}' has no audio prototype; enroll with voice first.", None
        ts = int(time.time())
        out_path = os.path.join(MEM_DIR, f"speech_{lab}_{ts}.wav")
        y = _sine_from_bands(a_feat, sr=sr, dur=0.9)
        _save_wav(out_path, y, sr=sr)

    # Quick sanity
    try:
        with contextlib.closing(wave.open(out_path, "rb")) as wf:
            nframes = wf.getnframes(); fr = wf.getframerate(); sw = wf.getsampwidth()
            dur = float(nframes / max(fr, 1))
    except Exception:
        dur = 0.0
    return f"Synthesized '{lab}' → {out_path} (dur={dur:.2f}s)", out_path


# =========================
# Viz renderers for tabs (Figures only)
# =========================
def render_vision_panel(idx_str: str):
    if not IMGS_BGR: return None
    if idx_str is None or not str(idx_str).isdigit(): return None
    idx = int(idx_str)
    if idx < 0 or idx >= len(IMGS_BGR): return None
    return vision_it_panel(IMGS_BGR[idx])

def render_voice_helix_plot(audio_tuple):
    if not HAS_VIZ_HELPERS:
        return None
    X, sr, Xc = dev_features_from_audio(audio_tuple)
    if X.size == 0:
        return None
    return voice_helix_fig(X, title="Neural Voice Helix")

def render_brain_plot(idx_str: str, audio_tuple):
    if not HAS_VIZ_HELPERS:
        return None
    if not IMGS_BGR or audio_tuple is None:
        return None
    if idx_str is None or not str(idx_str).isdigit():
        return None
    idx = int(idx_str)
    if idx < 0 or idx >= len(IMGS_BGR):
        return None
    X, sr, Xc = dev_features_from_audio(audio_tuple)
    if X.size == 0:
        return None
    v = encode_vision_feature(IMGS_BGR[idx])
    return brain_3d_fig(v, X, topk_edges=80, threshold=0.12)

def generate_all(idx_str: str, audio_tuple):
    # IT
    it_fig = render_vision_panel(idx_str)
    # Helix
    helix_fig = render_voice_helix_plot(audio_tuple)
    helix_msg = "OK" if helix_fig is not None else "Please record voice above."
    # Brain
    brain_fig = render_brain_plot(idx_str, audio_tuple)
    brain_msg = "OK" if brain_fig is not None else "Need image + recorded voice above."
    return it_fig, helix_msg, helix_fig, brain_msg, brain_fig


# =========================
# UI
# =========================
with gr.Blocks(title="👶🧠 BabyAI — Multisensory Cross-Modal Brain") as demo:
    gr.Markdown("# 👶🧠 BabyAI — Multisensory Cross-Modal Brain")
    gr.Markdown(
        "Enroll a person by selecting images and recording a reference voice. "
        "Recognize any image, confirm to strengthen, and **Speak** the associated voice.\n\n"
        "**T (temperature):** softness of softmax over similarities. Lower T ⇒ sharper/confident.\n\n"
        "**τ (threshold):** minimum confidence to accept; if conf < τ ⇒ abstain (open-set)."
    )

    with gr.Row():
        files = gr.Files(label="Add images (multiple)", file_count="multiple", type="filepath")
        btn_add = gr.Button("Add to Gallery")

    load_status = gr.Textbox(label="Load status", interactive=False)
    # Scrollable gallery
    gallery = gr.Gallery(label="Gallery", columns=5, height=500, allow_preview=True)
    idx = gr.Textbox(label="Selected index (auto)", value="0")

    with gr.Accordion("Controls", open=True):
        with gr.Row():
            label_box = gr.Textbox(label="Concept label (e.g., 'Kumail')")
            audio = gr.Audio(label="Record voice (dev bands)", sources=["microphone"], type="numpy")
        with gr.Row():
            tau = gr.Slider(-1.0, 1.0, value=0.18, step=0.01, label="Recognition threshold τ",
                            info="Minimum accepted confidence; below this the system abstains.")
            temp = gr.Slider(0.01, 0.5, value=0.07, step=0.01, label="Confidence temperature T",
                             info="Softmax temperature over similarity scores. Lower = sharper.")
            force_top = gr.Checkbox(label="Force Speak (ignore τ)", value=False)

    with gr.Row():
        btn_enroll = gr.Button("Enroll concept (selected image + voice)")
        btn_rec = gr.Button("Recognize image")
        btn_confirm = gr.Button("Confirm recognition (strengthen)")
        btn_say = gr.Button("Speak associated voice")

    status = gr.Textbox(label="Status", interactive=False)
    vfig = gr.Plot(label="Vision features (Plotly)")
    afig = gr.Plot(label="Audio features (Plotly)")
    cfig = gr.Plot(label="Co-firing spikes (animated)")

    scores_json = gr.Code(label="Scores (JSON, top-k / normalized)")
    predicted_lab = gr.Textbox(label="Predicted label (for confirm)", interactive=False)

    synth_audio = gr.Audio(label="Synthesized audio", type="filepath")

    # --- Viz Tabs using MAIN idx + MAIN audio ---
    with gr.Tabs():
        with gr.Tab("Vision IT"):
            btn_vis = gr.Button("Generate")
            vis_plot = gr.Plot(label="RGC + V1 (IT pooled)")
        with gr.Tab("Voice Helix (3D)"):
            btn_helix = gr.Button("Generate")
            helix_status = gr.Textbox(label="Helix status", interactive=False)
            helix_plot = gr.Plot(label="Interactive Helix (Plotly 3D)")
        with gr.Tab("3D Brain (Interactive)"):
            btn_brain = gr.Button("Generate")
            brain_status = gr.Textbox(label="Brain status", interactive=False)
            brain_plot = gr.Plot(label="Interactive 3D Brain")

    # Optional global convenience button
    btn_all = gr.Button("✨ Generate All (IT + Helix + 3D Brain)")

    # --- Wiring that must come after components exist ---
    btn_add.click(add_images, [files], [load_status, gallery, idx])

    def on_gallery_select(evt: gr.SelectData):
        return str(evt.index)
    gallery.select(on_gallery_select, None, [idx])

    btn_enroll.click(enroll_concept,
                     [idx, audio, label_box, tau, temp],
                     [status, vfig, afig, cfig, scores_json])

    btn_rec.click(recognize_image,
                  [idx, tau, temp, gr.State(False)],
                  [status, vfig, scores_json, predicted_lab])

    btn_confirm.click(confirm_recognition, [idx, predicted_lab], [status])

    btn_say.click(speak_selected, [idx, tau, temp, force_top], [status, synth_audio])

    # Vision IT uses main idx
    btn_vis.click(render_vision_panel, [idx], [vis_plot])

    # Helix uses main audio (wrapper to add status)
    def _helix_dispatch(a):
        fig = render_voice_helix_plot(a)
        return ("OK", fig) if fig is not None else ("Please record voice above.", None)
    btn_helix.click(_helix_dispatch, [audio], [helix_status, helix_plot])

    # Brain uses main idx + main audio
    def _brain_dispatch(i, a):
        fig = render_brain_plot(i, a)
        return ("OK", fig) if fig is not None else ("Need image + recorded voice above.", None)
    btn_brain.click(_brain_dispatch, [idx, audio], [brain_status, brain_plot])

    # One-click "Generate All"
    btn_all.click(generate_all, [idx, audio], [vis_plot, helix_status, helix_plot, brain_status, brain_plot])

if __name__ == "__main__":
    demo.launch(allowed_paths=[os.getcwd(), MEM_DIR])
