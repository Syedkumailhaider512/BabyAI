# multisensory_workspace.py — fixed JSON/NumPy persistence + robust I/O
import os, io, time, math, json
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

# --- Import user's cores ---
import sys
sys.path.append(os.getcwd())  # allow local imports in PyCharm run dir
import vision_babyai as V
import audition_babyai as A

SR = getattr(A, "SR", 16000)

def _safe_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).ravel()
    n = float(np.linalg.norm(x) + eps)
    return (x / n).astype(np.float32)

def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.dot(_safe_norm(a, eps), _safe_norm(b, eps)))

def now_ts() -> float:
    return time.time()

class MultisensoryWorkspace:
    """
    Cross-modal association brain that wraps Vision + Audition cores.
    - Hebbian A (vision_dim × audio_dim) with dopamine
    - EMA prototypes per label (vision/audio)
    - Time decay
    - Clean persistence: JSON for metadata + .npy for A matrix
    """

    def __init__(
        self,
        vision_dim: int = 512,
        audio_dim: int = 100,
        eta: float = 0.6,
        lam: float = 0.01,
        beta: float = 0.2,
        tau_A_seconds: float = 3600.0,
        tau_proto_seconds: float = 1800.0,
        tau_vis_match: float = 0.60,
        tau_aud_match: float = 0.60,
        mem_prefix: str = "./mem/multisensory_mem",
    ):
        self.vision_dim = int(vision_dim)
        self.audio_dim  = int(audio_dim)
        self.eta, self.lam, self.beta = float(eta), float(lam), float(beta)
        self.tau_A_seconds = float(tau_A_seconds)
        self.tau_proto_seconds = float(tau_proto_seconds)
        self.tau_vis_match = float(tau_vis_match)
        self.tau_aud_match = float(tau_aud_match)

        rng = np.random.default_rng(42)
        self.A = rng.normal(0.0, 0.01, size=(self.vision_dim, self.audio_dim)).astype(np.float32)

        self.vision_proto: Dict[str, np.ndarray] = {}
        self.audio_proto:  Dict[str, np.ndarray] = {}

        self.last_decay_ts = now_ts()

        self.vbrain = V.VisionBrain(path=os.path.join(os.path.dirname(mem_prefix), "vision_brain.json"))
        self.mem_prefix = mem_prefix
        os.makedirs(os.path.dirname(self.mem_prefix) or ".", exist_ok=True)

    # ---------- Persistence ----------
    def save(self):
        meta = dict(
            vision_dim=self.vision_dim,
            audio_dim=self.audio_dim,
            eta=self.eta, lam=self.lam, beta=self.beta,
            tau_A_seconds=self.tau_A_seconds,
            tau_proto_seconds=self.tau_proto_seconds,
            tau_vis_match=self.tau_vis_match,
            tau_aud_match=self.tau_aud_match,
            last_decay_ts=self.last_decay_ts,
        )
        # Convert prototypes to lists for JSON
        vis = {k: v.tolist() for k, v in self.vision_proto.items()}
        aud = {k: v.tolist() for k, v in self.audio_proto.items()}

        # Save JSON (metadata + prototypes)
        with open(self.mem_prefix + ".json", "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "vision_proto": vis, "audio_proto": aud}, f)

        # Save A separately as .npy (fast, compact)
        np.save(self.mem_prefix + "_A.npy", self.A)

    @staticmethod
    def load(mem_prefix: str = "./mem/multisensory_mem") -> "MultisensoryWorkspace":
        with open(mem_prefix + ".json", "r", encoding="utf-8") as f:
            pack = json.load(f)
        m = pack["meta"]
        ws = MultisensoryWorkspace(
            vision_dim=m["vision_dim"],
            audio_dim=m["audio_dim"],
            eta=m["eta"], lam=m["lam"], beta=m["beta"],
            tau_A_seconds=m["tau_A_seconds"],
            tau_proto_seconds=m["tau_proto_seconds"],
            tau_vis_match=m["tau_vis_match"],
            tau_aud_match=m["tau_aud_match"],
            mem_prefix=mem_prefix,
        )
        # Load A (or reinit if missing)
        A_path = mem_prefix + "_A.npy"
        if os.path.exists(A_path):
            ws.A = np.load(A_path).astype(np.float32)
        ws.vision_proto = {k: np.array(v, dtype=np.float32) for k, v in pack.get("vision_proto", {}).items()}
        ws.audio_proto  = {k: np.array(v, dtype=np.float32) for k, v in pack.get("audio_proto", {}).items()}
        ws.last_decay_ts = float(m.get("last_decay_ts", now_ts()))
        return ws

    # ---------- Decay ----------
    def _time_decay(self):
        now = now_ts()
        dt = max(0.0, now - self.last_decay_ts)
        if dt <= 0.0:
            return
        if self.tau_A_seconds > 0:
            self.A *= math.exp(-dt / self.tau_A_seconds)
        if self.tau_proto_seconds > 0:
            f = math.exp(-dt / self.tau_proto_seconds)
            for d in (self.vision_proto, self.audio_proto):
                for k in list(d.keys()):
                    d[k] = (d[k] * f).astype(np.float32)
        self.last_decay_ts = now

    # ---------- Vision ----------
    def encode_vision(self, img_bgr: np.ndarray,
                      ret_params: Optional[V.RetinaParams] = None,
                      v1_params: Optional[V.V1Params] = None) -> np.ndarray:
        ret_params = ret_params or V.RetinaParams()
        v1_params  = v1_params  or V.V1Params()
        feat = V.v1_feature_from_bgr(img_bgr, ret_params, v1_params, it_params=V.ITParams(use_it=True), out_dim=self.vision_dim)
        return _safe_norm(feat)

    def learn_vision(self, img_bgr: np.ndarray, label: str,
                     auto_dopamine: bool = True,
                     it_params: Optional[V.ITParams] = None) -> Dict[str, Any]:
        it_params = it_params or V.ITParams(use_it=True)
        dopa_used, dopa_info = self.vbrain.record_image(
            img_bgr, label,
            ret_params=V.RetinaParams(),
            v1_params=V.V1Params(),
            auto_dopamine=bool(auto_dopamine),
            it_params=it_params,
            store_dir=None
        )
        v_feat = self.encode_vision(img_bgr)
        proto = self.vision_proto.get(label)
        self.vision_proto[label] = _safe_norm(v_feat if proto is None else (1.0 - self.beta) * proto + self.beta * v_feat)
        return {"dopamine": float(dopa_used), "dopamine_info": dopa_info, "v_feat": v_feat}

    # ---------- Audition ----------
    def encode_audio(self, y: np.ndarray, sr: int = SR) -> Tuple[np.ndarray, Dict[str, Any]]:
        y_cut, vad_meta = A.vad_best_segment(y, sr=sr)
        X = A.extract_features(y_cut)  # (T, N)
        cur = A.AuditoryCuriosity(A.CuriosityConfig(frame_rate=1000.0 / A.GUI_FRAME_MS))
        series = []
        t = 0.0
        for row in X:
            series.append(cur.process_frame(row, t_sec=t))
            t += (A.GUI_FRAME_MS / 1000.0)
        csum = A.offline_summary(series, A.CURIOSITY_THR)
        a = _safe_norm(X.mean(axis=0))
        return a, {"series": series, "summary": csum, "vad": vad_meta, "frames": X}

    def learn_audio(self, y: np.ndarray, label: str, sr: int = SR) -> Dict[str, Any]:
        y_cut, _ = A.vad_best_segment(y, sr=sr)
        X = A.extract_features(y_cut)
        dur_s = max(0.1, float(len(y_cut) / sr))
        cur = A.AuditoryCuriosity(A.CuriosityConfig(frame_rate=1000.0 / A.GUI_FRAME_MS))
        series = []
        t = 0.0
        for row in X:
            series.append(cur.process_frame(row, t_sec=t))
            t += (A.GUI_FRAME_MS / 1000.0)
        csum = A.offline_summary(series, A.CURIOSITY_THR)
        f0 = A.estimate_f0_autocorr(y_cut, sr=sr)
        brain = A.load_brain()
        entry, _, _, info = A.update_label(
            brain, label, X_raw=X, duration_s=dur_s, curiosity=csum,
            N_feat=X.shape[1], canon_frames=A.CANON_FRAMES,
            MAIN_SPIKE_LOW=A.MAIN_SPIKE_LOW, f0_hz=f0
        )
        A.save_brain(brain)
        a = _safe_norm(X.mean(axis=0))
        proto = self.audio_proto.get(label)
        self.audio_proto[label] = _safe_norm(a if proto is None else (1.0 - self.beta) * proto + self.beta * a)
        return {"a_feat": a, "f0": f0, "curiosity": csum, "entry_info": info}

    # ---------- Cross-modal ----------
    def _hebbian(self, v: np.ndarray, a: np.ndarray, dopamine: float):
        self.A *= (1.0 - self.lam)
        self.A += (self.eta * float(dopamine)) * np.outer(_safe_norm(v), _safe_norm(a))

    def learn_pair(self, img_bgr: np.ndarray, y: np.ndarray, label: str, sr: int = SR,
                   override_da: Optional[float] = None) -> Dict[str, Any]:
        self._time_decay()
        vpack = self.learn_vision(img_bgr, label, auto_dopamine=(override_da is None))
        apack = self.learn_audio(y, label, sr=sr)
        DA = float(vpack["dopamine"]) if override_da is None else float(override_da)
        self._hebbian(vpack["v_feat"], apack["a_feat"], DA)
        self.save()
        return {"dopamine": DA, "vision": vpack, "audio": apack}

    # ---------- Predict / Speak ----------
    def predict_from_image(self, img_bgr: np.ndarray) -> Tuple[Optional[str], float, Dict[str, float]]:
        self._time_decay()
        v = self.encode_vision(img_bgr)
        a_hat = self.A.T @ v
        scores: Dict[str, float] = {}
        best, best_s = None, -1.0
        for lab, proto in self.audio_proto.items():
            s = cosine(a_hat, proto)
            scores[lab] = float(s)
            if s > best_s:
                best, best_s = lab, float(s)
        if best is None or best_s < self.tau_aud_match:
            return None, float(best_s), scores
        return best, float(best_s), scores

    def speak_label(self, label: str, avg_dur: float = 1.0, play: bool = False) -> Optional[str]:
        brain = A.load_brain()
        entry = brain.get(A.sanitize_label(label), None)
        if not entry or not entry.get("prototypes"):
            return None
        P = np.array(entry["prototypes"][0]["P"], dtype=float)
        vp = A.default_voice_profile()
        wav_name, _ = A.speak_and_save(label, P_canon=P, avg_dur=float(entry.get("avg_duration") or 1.0),
                                       speak_model=entry.get("speak_model", {}), vp=vp, play=play)
        return wav_name
