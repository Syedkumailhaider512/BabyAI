# frontend.py — audio frontend, VAD, inhibition, feature extraction, F0

import numpy as np
from scipy.signal import get_window
from config import SR, F_MAX, TARGET_FREQS, USE_MEL_FRONTEND, MEL_BINS, MEL_FMIN, MEL_FMAX, WIN_MS, HOP_MS
from config import GUI_FRAME_MS
from utils import clamp

# windows for 25 ms / 10 ms features
WIN = get_window('hann', int(SR*WIN_MS/1000.0), fftbins=True)

def frame_rms(frame): return float(np.sqrt(np.mean(frame**2) + 1e-12))

def vad_trim(y, sr=SR, win_ms=20, hop_ms=10, pad_ms=40):
    win = max(1, int(sr * win_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    frames = [y[i:i+win] for i in range(0, len(y)-win+1, hop)]
    if not frames: return y, {"onset":0,"offset":len(y),"note":"no_frames"}
    rms = np.array([frame_rms(f) for f in frames])
    def thr(v,k): med=np.median(v); mad=np.median(np.abs(v-med))+1e-9; return med+k*mad
    thr_rms=max(0.002,thr(rms,2.5))
    run=0; onset_idx=0
    for i,v in enumerate(rms):
        run = run+1 if v>thr_rms else 0
        if run>=3: onset_idx=max(0,i-2); break
    run=0; offset_idx=len(frames)-1
    for j in range(len(frames)-1,-1,-1):
        run = run+1 if rms[j]>thr_rms else 0
        if run>=3: offset_idx=min(len(frames)-1,j+2); break
    pad = int(sr*pad_ms/1000.0)
    onset = max(0, onset_idx*hop - pad)
    offset= min(len(y), offset_idx*hop + win + pad)
    if offset<=onset: return y, {"onset":0,"offset":len(y),"note":"fallback"}
    return y[onset:offset], {"onset":onset,"offset":offset,"note":"ok"}

def vad_best_segment(y, sr=SR,
                     win_ms=25, hop_ms=10,
                     min_speech_ms=150, merge_gap_ms=120,
                     prepad_ms=60, postpad_ms=80):
    """Pick best single speech island with padding."""
    win = max(1, int(sr * win_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if len(y) < win:
        pad = np.zeros(win, dtype=y.dtype); pad[:len(y)] = y; y = pad
    frames = [y[i:i+win] for i in range(0, len(y)-win+1, hop)]
    if not frames:
        return y, {"onset":0, "offset":len(y), "note":"no_frames"}

    rms = np.array([frame_rms(f) for f in frames])
    db  = 20.0 * np.log10(rms + 1e-9)
    noise = np.percentile(db, 10)
    thr = noise + max(6.0, 2.5 * float(np.std(db)))

    mask = db > thr
    segs = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif (not m) and (start is not None):
            end = i; segs.append([start, end]); start = None
    if start is not None:
        segs.append([start, len(mask)])
    if not segs:
        return y, {"onset":0, "offset":len(y), "note":"no_speech"}

    merged = []
    gap_frames = int(round(merge_gap_ms / hop_ms))
    cur = segs[0]
    for s, e in segs[1:]:
        if s - cur[1] <= gap_frames:
            cur[1] = e
        else:
            merged.append(cur); cur = [s, e]
    merged.append(cur)

    min_frames = int(round(min_speech_ms / hop_ms))
    merged = [se for se in merged if (se[1] - se[0]) >= min_frames] or [max(segs, key=lambda x: x[1]-x[0])]

    def seg_score(se):
        s, e = se; return float(rms[s:e].sum() * (e - s))
    best = max(merged, key=seg_score); s, e = best

    pre = int(sr * prepad_ms / 1000.0); post = int(sr * postpad_ms / 1000.0)
    samp_on = max(0, s * hop - pre)
    samp_off = min(len(y), e * hop + win + post)
    y_cut = y[samp_on:samp_off]
    return y_cut, {"onset": int(samp_on), "offset": int(samp_off), "note": "ok_best"}

# -------- Dev 0–990 Hz features (100 bins) --------
def dev_spec_100(frame):
    n_fft = int(4096)
    frame = frame * get_window('hann', len(frame), fftbins=True)
    spec = np.fft.rfft(frame, n=n_fft)
    mags = np.abs(spec)
    freqs = np.fft.rfftfreq(n_fft, d=1.0/SR)
    valid = freqs <= F_MAX
    return np.interp(TARGET_FREQS, freqs[valid], mags[valid])

# -------- Mel filterbank (no librosa) --------
def hz_to_mel(f): return 2595.0*np.log10(1.0+f/700.0)
def mel_to_hz(m): return 700.0*(10**(m/2595.0)-1.0)

def mel_filterbank(n_fft, sr, n_mels, fmin, fmax):
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0/sr)
    m_min, m_max = hz_to_mel(fmin), hz_to_mel(fmax)
    m_points = np.linspace(m_min, m_max, n_mels+2)
    hz_points = mel_to_hz(m_points)
    bins = np.floor((n_fft+1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, len(fft_freqs)))
    for i in range(1, n_mels+1):
        l, c, r = bins[i-1], bins[i], bins[i+1]
        if c == l: c += 1
        if r == c: r += 1
        fb[i-1, l:c] = (np.arange(l, c)-l) / max(1, (c-l))
        fb[i-1, c:r] = (r - np.arange(c, r)) / max(1, (r-c))
    fb /= fb.sum(axis=1, keepdims=True) + 1e-12
    return fb

_MEL_FB = None
_MEL_FFT = 2048

def extract_features(y, inhibit=True):
    """
    Returns (T, N) features normalized per-utterance (not per-frame max).
    If USE_MEL_FRONTEND: N=MEL_BINS; else N=100 (0..990 Hz dev path).
    """
    hop = int(SR*HOP_MS/1000.0)
    win = int(SR*WIN_MS/1000.0)
    if len(y) < win:
        pad = np.zeros(win, dtype=y.dtype); pad[:len(y)] = y; y = pad

    frames = [y[i:i+win] for i in range(0, len(y)-win+1, hop)]
    if not frames:
        return np.zeros((1, 100 if not USE_MEL_FRONTEND else MEL_BINS), float)

    if USE_MEL_FRONTEND:
        global _MEL_FB
        if _MEL_FB is None:
            _MEL_FB = mel_filterbank(_MEL_FFT, SR, MEL_BINS, MEL_FMIN, MEL_FMAX)
        S = []
        for f in frames:
            spec = np.abs(np.fft.rfft(f*WIN, n=_MEL_FFT))
            mel = _MEL_FB @ spec
            S.append(mel)
        X = np.vstack(S)
    else:
        X = np.vstack([dev_spec_100(f) for f in frames])

    # per-utterance loudness normalize
    rms = np.sqrt((X**2).mean()) + 1e-9
    X = X / rms

    # lateral inhibition
    if inhibit:
        k = 5
        ker = np.ones(k)/k
        X_ma = np.apply_along_axis(lambda v: np.convolve(v, ker, mode='same'), 1, X)
        X = np.maximum(0.0, X - X_ma)

    mx = X.max(axis=1,keepdims=True); mx[mx<1e-12]=1.0
    X = X / mx
    return X

def estimate_f0_autocorr(y, sr=SR, fmin=80.0, fmax=400.0):
    """Simple robust F0 via autocorrelation peak. Returns float Hz or None."""
    if len(y) < int(sr*0.05):
        return None
    x = y.astype(np.float64)
    x -= np.mean(x)
    if np.max(np.abs(x)) < 1e-6:
        return None
    corr = np.correlate(x, x, mode='full')
    corr = corr[len(x)-1:]
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    if max_lag >= len(corr):
        max_lag = len(corr) - 1
    corr[:min_lag] = 0.0
    if max_lag > 0:
        corr[max_lag+1:] = 0.0
    k = np.argmax(corr[min_lag:max_lag+1]) + min_lag
    if corr[k] <= 0:
        return None
    return float(sr / k)

def stream_block_size():
    """GUI stream block aligned to GUI_FRAME_MS."""
    return int(SR * (GUI_FRAME_MS/1000.0))
