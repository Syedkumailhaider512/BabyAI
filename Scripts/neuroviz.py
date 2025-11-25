# neuroviz.py  —  Creative visualizations for BabyAI (PyCharm-friendly)
# Requires: numpy, matplotlib, plotly, (opencv-python optional)
# Install:  pip install plotly matplotlib imageio opencv-python

from __future__ import annotations
import os, io, math
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False
    from PIL import Image

import plotly.graph_objs as go


# ---------------------------
# Helpers
# ---------------------------

def _l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).ravel()
    n = float(np.linalg.norm(x) + eps)
    return (x / n).astype(np.float32)

def _to_gray01(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        gray = img_bgr.astype(np.float32)
    else:
        if HAS_CV2:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            im = Image.fromarray(img_bgr[:, :, ::-1])
            gray = np.array(im.convert("L"), dtype=np.float32)
    rng = gray.max() - gray.min()
    if rng < 1e-8: return np.zeros_like(gray, dtype=np.float32)
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


# ---------------------------
# Vision: IT-style panel
# ---------------------------

def vision_it_panel(img_bgr: np.ndarray,
                    num_orient: int = 8,
                    kernel_sizes: Tuple[int, ...] = (7, 11, 15),
                    title: str = "Vision — RGC & V1 (orientation × scale)") -> plt.Figure:
    """
    Produces a Matplotlib figure with:
      - RGC ON/OFF (from DoG)
      - A grid of V1 energy maps across orientations/scales
    Returns a Matplotlib Figure for gr.Plot.
    """
    gray = _to_gray01(img_bgr)
    on  = cv2.GaussianBlur(gray, (0,0), 1.2)
    off = cv2.GaussianBlur(gray, (0,0), 2.4)
    on  = np.clip(on - off, 0, 1)
    off = np.clip(off - on, 0, 1)
    base = 0.5*(on+off)

    thetas = np.linspace(0.0, np.pi, num_orient, endpoint=False)
    maps: List[np.ndarray] = []
    for k in kernel_sizes:
        lam = max(4.0, 0.6*k); sigma = 0.5*k
        for th in thetas:
            kern = _gabor_kernel(k, th, lam=lam, sigma=sigma)
            resp = cv2.filter2D(base, cv2.CV_32F, kern)
            maps.append(np.abs(resp))

    n_rows = 2 + len(kernel_sizes)  # 1 row RGC (2 images merged), 1 blank spacer, then scales
    n_cols = max(4, num_orient)
    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(title, fontsize=14)

    # Row 0: RGC ON/OFF
    ax = fig.add_subplot(n_rows, 2, 1)
    ax.imshow(on, cmap="magma"); ax.set_title("RGC ON"); ax.axis("off")
    ax = fig.add_subplot(n_rows, 2, 2)
    ax.imshow(off, cmap="magma"); ax.set_title("RGC OFF"); ax.axis("off")

    # Rows for scales
    idx = 0
    for si, k in enumerate(kernel_sizes):
        row = si + 3
        for oi in range(num_orient):
            m = maps[idx]; idx += 1
            ax = fig.add_subplot(n_rows, n_cols, (row-1)*n_cols + oi + 1)
            ax.imshow(m, cmap="magma")
            ax.set_title(f"V1: k={k} θ={int(thetas[oi]*180/np.pi)}°", fontsize=8)
            ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------
# Voice: 3D Helix (interactive Plotly)
# ---------------------------

def voice_helix_html(X: np.ndarray,
                     title: str = "Voice Helix",
                     top_bands: int = 6,
                     turns: float = 2.5) -> str:
    """
    X: (T, N) feature frames (already normalized per frame).
    Returns a self-contained Plotly HTML string (interactive).
    """
    T, N = X.shape
    energy = X.mean(axis=1)
    energy_n = (energy - energy.min()) / (energy.ptp() + 1e-8)

    # Helix backbone (x,y,z) — time along z
    theta = np.linspace(0, 2*np.pi*turns, T)
    z = np.linspace(0, 1.0, T)
    r = 0.6 + 0.35*energy_n
    x = r*np.cos(theta)
    y = r*np.sin(theta)

    data = []
    # Backbone
    data.append(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        line=dict(width=6, color=energy_n, colorscale="Turbo"),
        name="Backbone"
    ))

    # Highlight top-B bands as ribbons (thin lines)
    band_energy = X.mean(axis=0)
    band_idx = np.argsort(-band_energy)[:max(1, min(top_bands, N))]
    for b in band_idx:
        amp = X[:, b]
        amp_n = (amp - amp.min())/(amp.ptp()+1e-8)
        rb = 0.2 + 0.2*amp_n
        xb = (r+rb)*np.cos(theta + b*0.05)
        yb = (r+rb)*np.sin(theta + b*0.05)
        data.append(go.Scatter3d(
            x=xb, y=yb, z=z,
            mode="lines",
            line=dict(width=3, color=amp_n, colorscale="Viridis"),
            name=f"Band {b}"
        ))

    # Moving "cursor" (frame indicator)
    frames = []
    for t in range(T):
        frames.append(go.Frame(data=[
            go.Scatter3d(
                x=[x[t]], y=[y[t]], z=[z[t]],
                mode="markers",
                marker=dict(size=8, color="white"),
                name="cursor"
            )
        ]))

    fig = go.Figure(
        data=data + [go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]],
                                  mode="markers", marker=dict(size=8, color="white"), name="cursor")],
        layout=go.Layout(
            title=f"{title} — {T} frames",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Time"),
            margin=dict(l=0, r=0, b=0, t=40),
            updatemenus=[dict(
                type="buttons",
                showactive=True,
                y=1.05,
                x=0.05,
                xanchor="left",
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True, mode="immediate")]),
                    dict(label="Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0), mode="immediate")])
                ]
            )]
        ),
        frames=frames
    )
    return fig.to_html(include_plotlyjs="inline", full_html=False)


# ---------------------------
# 3D Brain Graph (interactive Plotly with frames)
# ---------------------------

def _hemisphere_points(n: int, side: str = "left", radius: float = 1.0) -> np.ndarray:
    """
    Distribute n points on left/right hemisphere of a sphere.
    side: "left" => x <= 0, "right" => x >= 0
    """
    if n <= 0: return np.zeros((0,3), dtype=np.float32)
    # Fibonacci sphere then cut half
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

def brain_3d_html(v: np.ndarray, X: np.ndarray,
                  topk_edges: int = 80,
                  threshold: float = 0.15,
                  title: str = "3D Brain — cross-modal co-firing") -> str:
    """
    v: (V,) normalized vision vector
    X: (T, N) per-frame audio features (per-frame normalized)
    Creates a 3D interactive Plotly figure with:
      - vision hemisphere nodes (left) sized by |v|
      - audio hemisphere nodes (right) sized by |mean(X)|
      - edges per frame connecting top-k co-firings in outer(v, X[t])
    Returns HTML string for embedding (gr.HTML).
    """
    v = _l2(v); V = v.size
    X = np.asarray(X, dtype=np.float32)
    X = (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8))
    T, N = X.shape
    a_mean = _l2(X.mean(axis=0))
    # pick top nodes for clarity
    vmax = min(128, V)
    amax = min(128, N)
    v_idx = np.argsort(-np.abs(v))[:vmax]
    a_idx = np.argsort(-np.abs(a_mean))[:amax]
    v_small = v[v_idx]
    a_small = a_mean[a_idx]
    L = _hemisphere_points(vmax, side="left", radius=1.2)
    R = _hemisphere_points(amax, side="right", radius=1.2)

    # node traces
    v_trace = go.Scatter3d(
        x=L[:,0], y=L[:,1], z=L[:,2],
        mode="markers",
        marker=dict(
            size=6 + 12*(np.abs(v_small)/ (np.max(np.abs(v_small))+1e-8)),
            color=v_small, colorscale="Plasma", opacity=0.95
        ),
        name="Vision"
    )
    a_trace = go.Scatter3d(
        x=R[:,0], y=R[:,1], z=R[:,2],
        mode="markers",
        marker=dict(
            size=6 + 12*(np.abs(a_small)/ (np.max(np.abs(a_small))+1e-8)),
            color=a_small, colorscale="Viridis", opacity=0.95
        ),
        name="Audition"
    )

    # frames: edges per time frame
    frames = []
    for t in range(T):
        a_t = _l2(X[t])[a_idx]
        # co-firing scores
        S = np.outer(v_small, a_t)  # (vmax, amax)
        S_flat = S.ravel()
        if topk_edges < S_flat.size:
            idx = np.argpartition(-S_flat, topk_edges)[:topk_edges]
        else:
            idx = np.arange(S_flat.size)
        idx = idx[np.argsort(-S_flat[idx])]
        # build line segments with None separators
        xs, ys, zs, cs = [], [], [], []
        for k in idx:
            vs = k // amax
            as_ = k % amax
            w = S_flat[k]
            if w < threshold: break
            xs += [L[vs,0], R[as_,0], None]
            ys += [L[vs,1], R[as_,1], None]
            zs += [L[vs,2], R[as_,2], None]
            cs.append(w)
        edge_trace = go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=2, color="orange"),
            opacity=0.6,
            name=f"Edges t={t}"
        )
        frames.append(go.Frame(data=[edge_trace], name=f"t{t}"))

    fig = go.Figure(
        data=[v_trace, a_trace, go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(width=2, color="orange"), name="Edges")],
        layout=go.Layout(
            title=title,
            scene=dict(xaxis_title="Left: Vision • Right: Audition", yaxis_title="", zaxis_title=""),
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=True,
            updatemenus=[dict(
                type="buttons",
                showactive=True,
                y=1.05,
                x=0.05,
                xanchor="left",
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True, mode="immediate")]),
                    dict(label="Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0), mode="immediate")])
                ]
            )]
        ),
        frames=frames
    )
    return fig.to_html(include_plotlyjs="inline", full_html=False)
