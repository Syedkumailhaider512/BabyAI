# config.py — global settings and toggles

import numpy as np

# I/O and data
SR = 16_000
MASTER_CSV = "family_dataset.csv"
BRAIN_JSON = "brain_family.json"

# GUI timing
GUI_FRAME_MS = 100   # live curiosity refresh (doesn't force feature hop)
MAX_RECORD_SECONDS = 30  # safety cap for a single recording

# Front-end toggles
USE_MEL_FRONTEND = False     # True = 64 mel bins 50–8000 Hz (evals); False = 0–990 Hz (dev demo)
MEL_BINS = 64
MEL_FMIN = 50.0
MEL_FMAX = 8000.0

# Dev front-end (frequency bins 0..990 Hz)
F_STEP = 10
F_MAX = 990
TARGET_FREQS = np.arange(0, F_MAX + 1, F_STEP)  # N=100
N_FEAT_DEV = len(TARGET_FREQS)

# Canonical feature framing for learning (independent of GUI refresh)
WIN_MS = 25     # analysis window (features)
HOP_MS = 10     # analysis hop (features)

# Canonical prototype frames (DTW aligns here)
CANON_FRAMES = 20  # ~2 s @ 100 ms legacy; OK for demos

# Neuron/spike
MAIN_SPIKE_LOW = 0.85

# Learning rates
ETA_HEBB   = 1.0
A_PLUS     = 1.0
A_MINUS    = 0.5
DELTA_T    = 0.1
TAU_PLUS   = 0.050
TAU_MINUS  = 0.050
MIX_STDP   = 0.3

ETA_DELTA  = 0.6
L2_DECAY   = 0.01           # per-update weight shrink
EMA_ALPHA  = 0.3            # prototype EMA
EDGE_PRUNE = 0.05           # store edges >= this

# Time-based forgetting (truth in advertising)
LAMBDA_FORGET = 5e-6        # per-second; ~10% per ~6 hours; tune in experiments
PRUNE_EPS     = 1e-3        # prune edges below this after decay/update

# Eligibility traces (dopamine-style credit assignment)
ELIG_TAU_S    = 1.0         # seconds
ELIG_SCALE    = 1.0

# Curiosity & safety
LOUDNESS_MAX_DB = 85.0
CURIOSITY_THR   = 0.4
DOPAMINE_THR    = 0.6

# Voice growth defaults
SYNTH_GAIN = 0.9

# Prototypes per label (online clustering)
MAX_PROTOS_PER_LABEL = 3

# Visualization
FPS = 20
DURATION_VIS = 4
LAYOUT = "helix"
MAX_EDGES_DRAW = 900
