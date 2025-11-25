# synthesis.py — voice growth, warping, additive + harmonic synth (per-label F0)

import numpy as np
from scipy.io import wavfile
import sounddevice as sd
from config import SR, TARGET_FREQS, SYNTH_GAIN
from utils import clamp, next_indexed_name

def default_voice_profile():
    return {
        "age_years": 6.0, "sex": "Male", "auto_age": True,
        "custom": {"f0_scale":1.0, "formant_scale":1.0, "tilt_db_per_dec":0.0}
    }

def voice_params_from_age_sex(age_years: float, sex: str):
    a = clamp(age_years, 0.0, 20.0)
    if sex == "Male":
        f0 = np.interp(a, [0,10,13,20], [1.8,1.2,0.9,0.7])
        formant = np.interp(a, [0,10,13,20], [1.35,1.15,0.95,0.85])
        tilt = np.interp(a, [0,20], [-2.0,+3.0])
    elif sex == "Female":
        f0 = np.interp(a, [0,10,13,20], [1.8,1.25,1.05,0.95])
        formant = np.interp(a, [0,10,13,20], [1.35,1.15,1.05,0.95])
        tilt = np.interp(a, [0,20], [-2.0,+1.5])
    else:
        f0, formant, tilt = 1.0, 1.0, 0.0
    return float(f0), float(formant), float(tilt)

def warp_freq_axis_row(row_amps: np.ndarray, scale: float):
    freqs = TARGET_FREQS.astype(float)
    f_warp = freqs / max(1e-9, scale)
    return np.interp(freqs, f_warp, row_amps, left=0.0, right=0.0)

def apply_voice_warp(P: np.ndarray, f0_scale: float, formant_scale: float, tilt_db_per_dec: float):
    T,N=P.shape
    P1 = np.zeros_like(P)
    for t in range(T):
        P1[t] = warp_freq_axis_row(P[t], f0_scale)
    formant_eff = max(0.5, min(1.7, formant_scale))
    P2 = np.zeros_like(P1)
    for t in range(T):
        P2[t] = warp_freq_axis_row(P1[t], formant_eff**0.5)
    f = TARGET_FREQS.copy().astype(float); f[0]=1.0
    tilt_w = 10 ** (-tilt_db_per_dec * np.log10(f) / 20.0)
    tilt_w /= tilt_w.max()
    P3 = P2 * tilt_w.reshape(1,-1)
    mx=P3.max(axis=1,keepdims=True); mx[mx<1e-12]=1.0
    return P3/mx

def expand_to_duration(P_canon: np.ndarray, dur_s: float, frame_s: float=0.1):
    Tt=max(1, int(round(dur_s/frame_s))); Ts=P_canon.shape[0]
    t_src=np.linspace(0,1,Ts); t_dst=np.linspace(0,1,Tt)
    X=np.zeros((Tt,P_canon.shape[1]),float)
    for n in range(P_canon.shape[1]): X[:,n]=np.interp(t_dst,t_src,P_canon[:,n])
    mx=X.max(axis=1,keepdims=True); mx[mx<1e-12]=1.0
    return X/mx

def synthesize_additive(P: np.ndarray, speak_model: dict):
    gamma=float(speak_model.get("gamma",1.0))
    tilt=float(speak_model.get("tilt",0.0))
    noise=float(speak_model.get("noise",0.0))
    smooth=float(speak_model.get("smooth",0.0))
    attack_ms=int(speak_model.get("attack_ms",10)); release_ms=int(speak_model.get("release_ms",20))
    gain=float(speak_model.get("gain",SYNTH_GAIN))
    frame_s = 0.1
    samples_per_frame=int(SR*frame_s)
    attack=max(1,int(SR*attack_ms/1000.0)); release=max(1,int(SR*release_ms/1000.0))
    sustain=max(1, samples_per_frame-attack-release)
    env=np.concatenate([np.linspace(0,1,attack), np.ones(sustain), np.linspace(1,0,release)])[:samples_per_frame]
    f=TARGET_FREQS.copy().astype(float); f[0]=1.0
    tilt_w=10 ** (-tilt * np.log10(f) / 20.0); tilt_w/=tilt_w.max()
    N=P.shape[1]
    out=np.zeros(samples_per_frame*P.shape[0], float)
    phases=np.zeros(N,float); omega=2*np.pi*TARGET_FREQS
    prev_a=np.zeros(N,float)
    for t in range(P.shape[0]):
        a = np.power(P[t], max(1e-3,gamma)) * tilt_w
        if smooth>0: a=(1.0-smooth)*a + smooth*prev_a
        prev_a=a.copy()
        n=np.arange(samples_per_frame); tt=n/SR
        angles = tt.reshape(-1,1)*omega.reshape(1,-1) + phases.reshape(1,-1)
        frame = (np.sin(angles) * a.reshape(1,-1)).sum(axis=1)
        if noise>0: frame += noise*np.random.randn(len(frame))*0.05
        frame *= env
        start=t*samples_per_frame; out[start:start+samples_per_frame]=frame
        phases = (phases + omega*frame_s) % (2*np.pi)
    peak=np.max(np.abs(out))
    if peak>1e-9: out=(out/peak)*gain
    return out.astype(np.float32)

def synthesize_harmonic_from_proto(P: np.ndarray, speak_model: dict, frame_s: float=0.1):
    """Harmonic additive synth driven by per-label F0."""
    f0 = float(speak_model.get("f0_hz", 0.0))
    if not (60.0 <= f0 <= 500.0):
        return synthesize_additive(P, speak_model)

    gamma=float(speak_model.get("gamma",1.0))
    tilt=float(speak_model.get("tilt",0.0))
    noise=float(speak_model.get("noise",0.0))
    smooth=float(speak_model.get("smooth",0.0))
    attack_ms=int(speak_model.get("attack_ms",10)); release_ms=int(speak_model.get("release_ms",20))
    gain=float(speak_model.get("gain",SYNTH_GAIN))

    samples_per_frame=int(SR*frame_s)
    attack=max(1,int(SR*attack_ms/1000.0)); release=max(1,int(SR*release_ms/1000.0))
    sustain=max(1, samples_per_frame-attack-release)
    env_win=np.concatenate([np.linspace(0,1,attack), np.ones(sustain), np.linspace(1,0,release)])[:samples_per_frame]

    max_h = max(1, int(1000.0 // f0))
    harm_freqs = np.arange(1, max_h+1, dtype=float) * f0

    f = TARGET_FREQS.copy().astype(float); f[0]=1.0
    tilt_w_full = 10 ** (-tilt * np.log10(f) / 20.0)
    tilt_w_full /= tilt_w_full.max()

    out=np.zeros(samples_per_frame*P.shape[0], float)
    phases=np.zeros(max_h, float)
    prev_amp=np.zeros(max_h, float)

    for t in range(P.shape[0]):
        a_bins = np.power(P[t], max(1e-3, gamma)) * tilt_w_full
        a_harm = np.interp(harm_freqs, f, a_bins, left=0.0, right=0.0)
        if smooth>0:
            a_harm = (1.0 - smooth)*a_harm + smooth*prev_amp
        prev_amp = a_harm.copy()

        n = np.arange(samples_per_frame)
        tt = n / SR
        angles = tt.reshape(-1,1) * (2*np.pi*harm_freqs.reshape(1,-1)) + phases.reshape(1,-1)
        frame = (np.sin(angles) * a_harm.reshape(1,-1)).sum(axis=1)
        if noise>0: frame += noise*np.random.randn(len(frame))*0.03
        frame *= env_win

        start=t*samples_per_frame
        out[start:start+samples_per_frame] = frame
        phases = (phases + 2*np.pi*harm_freqs*frame_s) % (2*np.pi)

    peak=np.max(np.abs(out))
    if peak>1e-9: out=(out/peak)*gain
    return out.astype(np.float32)

def speak_and_save(label, P_canon, avg_dur, speak_model, vp, play=True):
    if vp.get("sex","Male") == "Custom":
        f0_scale = float(vp["custom"].get("f0_scale",1.0))
        formant_scale = float(vp["custom"].get("formant_scale",1.0))
        tilt = float(vp["custom"].get("tilt_db_per_dec",0.0))
    else:
        f0_scale, formant_scale, tilt = voice_params_from_age_sex(vp.get("age_years",6.0), vp.get("sex","Male"))

    X = expand_to_duration(P_canon, max(0.6, float(avg_dur)), frame_s=0.1)
    Xv = apply_voice_warp(X, f0_scale, formant_scale, tilt)

    # Prefer harmonic synth if label has F0
    if "f0_hz" in speak_model and 60.0 <= float(speak_model["f0_hz"]) <= 500.0:
        y = synthesize_harmonic_from_proto(Xv, speak_model, frame_s=0.1)
    else:
        y = synthesize_additive(Xv, speak_model)

    wav_name = next_indexed_name("speak_", label, "wav")
    wavfile.write(wav_name, SR, (y*32767).astype(np.int16))
    if play:
        try: sd.play(y, SR); sd.wait()
        except Exception: pass
    return wav_name, dict(f0=f0_scale, formant=formant_scale, tilt=tilt)
