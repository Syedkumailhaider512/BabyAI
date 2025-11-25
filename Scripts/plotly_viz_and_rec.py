# plotly_viz_and_rec.py
# Deps: pip install plotly numpy
from typing import Iterable, Optional, Tuple, Dict, List
import numpy as np
import plotly.graph_objs as go


# =========================
#  Utility math
# =========================
def _l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).ravel()
    n = float(np.linalg.norm(x) + eps)
    return (x / n).astype(np.float32)

def _cos(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = _l2(a, eps); b = _l2(b, eps)
    return float(np.dot(a, b))

# --- NEW: robust pooling/align helpers ---
def _avgpool_to(x: np.ndarray, m: int) -> np.ndarray:
    """Average-pool 1D vector to length m (handles any length, keeps energy order-free)."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if m <= 0:
        return np.zeros(0, dtype=np.float32)
    if x.size == m:
        return x.astype(np.float32)
    # split into m nearly-equal chunks and average
    chunks = np.array_split(x, m)
    pooled = np.array([c.mean() if c.size else 0.0 for c in chunks], dtype=np.float32)
    return pooled

def _align_pair(a: np.ndarray, b: np.ndarray, prefer: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (a', b') with matched dimensionality via average pooling.
    If 'prefer' is set, pool both to that length; else pool to min(len(a), len(b)).
    """
    la, lb = int(np.asarray(a).size), int(np.asarray(b).size)
    if la == 0 or lb == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    m = int(prefer) if (prefer and prefer > 0) else min(la, lb)
    return _avgpool_to(a, m), _avgpool_to(b, m)



# =========================
#  Plotly 3D: Voice Helix
# =========================
def voice_helix_fig(X: np.ndarray,
                    title: str = "Neural Voice Helix",
                    top_bands: int = 6,
                    turns: float = 2.5,
                    frame_ms: int = 50) -> go.Figure:
    """
    X: (T, N) time×feature matrix from your audition pipeline.
    Returns a Plotly Figure with a 'Play/Pause' animation cursor.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.size == 0:
        return go.Figure()

    T, N = X.shape
    energy = X.mean(axis=1)
    energy_n = (energy - energy.min()) / (energy.ptp() + 1e-8)

    theta = np.linspace(0, 2*np.pi*turns, T)
    z = np.linspace(0, 1.0, T)
    r = 0.6 + 0.35*energy_n
    x = r*np.cos(theta); y = r*np.sin(theta)

    traces: List[go.Scatter3d] = []
    traces.append(go.Scatter3d(
        x=x, y=y, z=z, mode="lines",
        line=dict(width=6, color=energy_n, colorscale="Turbo"),
        name="Backbone"
    ))

    band_energy = X.mean(axis=0)
    band_idx = np.argsort(-band_energy)[:max(1, min(top_bands, N))]
    for b in band_idx:
        amp = X[:, b]
        amp_n = (amp - amp.min())/(amp.ptp()+1e-8)
        rb = 0.2 + 0.2*amp_n
        xb = (r+rb)*np.cos(theta + b*0.05)
        yb = (r+rb)*np.sin(theta + b*0.05)
        traces.append(go.Scatter3d(
            x=xb, y=yb, z=z, mode="lines",
            line=dict(width=3, color=amp_n, colorscale="Viridis"),
            name=f"Band {b}"
        ))

    frames = []
    for t in range(T):
        frames.append(go.Frame(
            data=[go.Scatter3d(
                x=[x[t]], y=[y[t]], z=[z[t]],
                mode="markers", marker=dict(size=8, color="white"),
                name="cursor")]
        ))

    fig = go.Figure(
        data=traces + [go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]],
                                    mode="markers", marker=dict(size=8, color="white"),
                                    name="cursor")],
        layout=go.Layout(
            title=f"{title} — {T} frames",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Time"),
            margin=dict(l=0, r=0, b=0, t=40),
            updatemenus=[dict(
                type="buttons", showactive=True, y=1.05, x=0.05, xanchor="left",
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                          fromcurrent=True, mode="immediate")]),
                    dict(label="Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0),
                                            mode="immediate")])
                ]
            )]
        ),
        frames=frames
    )
    return fig


# =========================
#  Plotly 3D: Brain Graph
# =========================
def _hemisphere_points(n: int, side: str = "left", radius: float = 1.0) -> np.ndarray:
    if n <= 0: return np.zeros((0,3), dtype=np.float32)
    k = np.arange(1, 2*n+1)
    phi = (1 + 5**0.5) / 2
    theta = 2*np.pi * (k/phi % 1.0)
    z = 1 - 2*k/(2*n+1)
    r = np.sqrt(1 - z*z)
    x = r*np.cos(theta); y = r*np.sin(theta)
    pts = np.stack([x,y,z], axis=1)*radius
    if side == "left":
        pts[:,0] = -np.abs(pts[:,0]) - 0.5*radius
    else:
        pts[:,0] =  np.abs(pts[:,0]) + 0.5*radius
    return pts[:n].astype(np.float32)

def brain_3d_fig(v: np.ndarray, X: np.ndarray,
                 topk_edges: int = 80,
                 threshold: float = 0.12,
                 title: str = "3D Brain — Cross-modal Co-firing") -> go.Figure:
    """
    v: (D_v,) vision feature vector for an image.
    X: (T, D_a) time×feature matrix from audio.
    Draws two hemispheres (vision left, audition right) and animates top co-firing edges over time.
    """
    v = _l2(v)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.size == 0:
        return go.Figure()
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    T, N = X.shape
    a_mean = _l2(X.mean(axis=0))

    vmax = min(128, v.size); amax = min(128, N)
    v_idx = np.argsort(-np.abs(v))[:vmax]
    a_idx = np.argsort(-np.abs(a_mean))[:amax]
    v_small = v[v_idx]; a_small = a_mean[a_idx]
    L = _hemisphere_points(vmax, side="left",  radius=1.2)
    R = _hemisphere_points(amax, side="right", radius=1.2)

    v_trace = go.Scatter3d(
        x=L[:,0], y=L[:,1], z=L[:,2], mode="markers",
        marker=dict(size=6 + 12*(np.abs(v_small)/(np.max(np.abs(v_small))+1e-8)),
                    color=v_small, colorscale="Plasma", opacity=0.95),
        name="Vision"
    )
    a_trace = go.Scatter3d(
        x=R[:,0], y=R[:,1], z=R[:,2], mode="markers",
        marker=dict(size=6 + 12*(np.abs(a_small)/(np.max(np.abs(a_small))+1e-8)),
                    color=a_small, colorscale="Viridis", opacity=0.95),
        name="Audition"
    )

    frames = []
    for t in range(T):
        a_t = _l2(X[t])[a_idx]
        S = np.outer(v_small, a_t).ravel()
        if topk_edges < S.size:
            idx = np.argpartition(-S, topk_edges)[:topk_edges]
        else:
            idx = np.arange(S.size)
        idx = idx[np.argsort(-S[idx])]

        xs, ys, zs = [], [], []
        for k in idx:
            vs = k // amax; as_ = k % amax
            w = S[k]
            if w < threshold: break
            xs += [L[vs,0], R[as_,0], None]
            ys += [L[vs,1], R[as_,1], None]
            zs += [L[vs,2], R[as_,2], None]

        edge_trace = go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(width=2, color="orange"), opacity=0.6,
            name=f"Edges t={t}"
        )
        frames.append(go.Frame(data=[edge_trace], name=f"t{t}"))

    fig = go.Figure(
        data=[v_trace, a_trace, go.Scatter3d(x=[], y=[], z=[], mode="lines",
                                             line=dict(width=2, color="orange"), name="Edges")],
        layout=go.Layout(
            title=title,
            scene=dict(xaxis_title="Left: Vision • Right: Audition"),
            margin=dict(l=0, r=0, b=0, t=40), showlegend=True,
            updatemenus=[dict(type="buttons", showactive=True, y=1.05, x=0.05, xanchor="left",
                              buttons=[
                                  dict(label="Play", method="animate",
                                       args=[None, dict(frame=dict(duration=60, redraw=True),
                                                        fromcurrent=True, mode="immediate")]),
                                  dict(label="Pause", method="animate",
                                       args=[[None], dict(frame=dict(duration=0),
                                                          mode="immediate")])
                              ])]
        ),
        frames=frames
    )
    return fig


# =========================
#   Auto Dopamine (DA)
# =========================
def auto_dopamine(v_feat: np.ndarray,
                  a_feat: np.ndarray,
                  vision_memory: Optional[Iterable[np.ndarray]] = None,
                  w_agree: float = 0.7,
                  w_novel: float = 0.3) -> float:
    """
    DA ∈ [0,1] combining:
      - cross-modal agreement (cosine between *aligned* v vs. a)
      - novelty (1 - max cosine to any prior *aligned* vision memory vector)
    Dimensionalities are reconciled via average pooling.
    """
    # 1) Agreement (pool both to a common size; prefer audio len if it's 100)
    a = np.asarray(a_feat, dtype=np.float32).ravel()
    v = np.asarray(v_feat, dtype=np.float32).ravel()
    prefer = a.size if a.size > 0 else None
    v_aligned, a_aligned = _align_pair(v, a, prefer=prefer)
    if v_aligned.size == 0 or a_aligned.size == 0:
        agreement = 0.5  # neutral if something is empty
    else:
        agreement = 0.5 * (_cos(v_aligned, a_aligned) + 1.0)  # map [-1,1] → [0,1]

    # 2) Novelty against memory (align each memory vector to v_aligned’s length)
    novelty = 0.5  # neutral default
    if vision_memory is not None:
        best = 0.0
        for mvec in vision_memory:
            ma, _ = _align_pair(mvec, v, prefer=v_aligned.size)
            if ma.size == 0:
                continue
            sim = _cos(ma, v_aligned)
            if sim > best:
                best = sim
        novelty = float(np.clip(1.0 - best, 0.0, 1.0))

    da = float(np.clip(w_agree * agreement + w_novel * novelty, 0.0, 1.0))
    return da



# =========================
#  Robust recognition gate
# =========================
def _softmax_conf(scores: Dict[str, float], temp: float = 0.07) -> Tuple[Optional[str], float, Dict[str, float]]:
    """
    Temperature-scaled softmax over raw similarity scores.
    Returns (label, confidence in [0,1], normalized_scores)
    """
    if not scores:
        return None, float("nan"), {}
    labels = list(scores.keys())
    vals = np.array([scores[k] for k in labels], dtype=np.float32)

    # shift for numerical stability
    z = (vals - vals.max()) / max(temp, 1e-6)
    e = np.exp(z - z.max())
    p = e / (e.sum() + 1e-8)

    j = int(np.argmax(p))
    label = labels[j]; conf = float(p[j])
    norm_scores = {labels[i]: float(p[i]) for i in range(len(labels))}
    return label, conf, norm_scores

def robust_recognize_from_image(workspace,
                                img_bgr: np.ndarray,
                                tau: float = 0.0,
                                temp: float = 0.07,
                                topk_return: int = 5) -> Tuple[Optional[str], float, Dict[str, float]]:
    """
    Wraps workspace.predict_from_image with:
      - fallback if workspace returns None/low-confidence
      - temperature calibration to get stable conf ∈ [0,1]
      - τ thresholding (open-set)
    """
    lab, conf, scores = workspace.predict_from_image(img_bgr)

    # If model didn't return a label or conf, re-decide from scores.
    if lab is None or conf is None or not np.isfinite(conf):
        lab, conf, scores = _softmax_conf(scores or {}, temp=temp)

    # Keep a compact top-k score dict for UI/debug
    if scores:
        top_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:topk_return]
        scores = dict(top_items)

    # τ gating (open-set abstention)
    if lab is None or conf < float(tau):
        return None, float(conf if np.isfinite(conf) else "nan"), scores

    return lab, float(conf), scores
