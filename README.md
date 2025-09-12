# BabyAI 👶🧠
*A Biologically Inspired Digital Infant Learning Framework*

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Research--Prototype-orange)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

---

## 📖 Overview
**BabyAI** explores how an artificial agent can learn **like a human infant from scratch**. Instead of relying on curated datasets, BabyAI learns from raw **sensory experience** using biologically inspired mechanisms:

- **Hebbian Learning** (“neurons that fire together, wire together”)
- **Spike-Timing Dependent Plasticity (STDP)**
- **Dopamine-Modulated Plasticity** (reward/punishment shaping)
- **Curiosity-Based Replay** and exploratory guessing
- **Dynamic Memory Graphs** with reinforcement & decay
- **Global Workspace** style selection for “conscious” output

BabyAI is **multi-sensory** (vision + audition + tactile ready) and **persistent** (memories survive restarts). It’s a live research project forming the basis of an upcoming paper.

---

## ✨ Key Features
- 👁 **Vision Pipeline**: Retina DoG (ON/OFF) → LGN normalization → V1 Gabor bank (orientation × scale) → optional IT invariance (pooling over rotation/scale/shift)
- 👂 **Audition Pipeline**: STFT spectrogram → spike encoding → Hebbian + STDP learning; temporal alignment via cross-correlation
- 🧠 **Persistent Brain (`brain.json`)**:
  - Per-label prototypes (EMA), counts, updated_at
  - `W_sparse` (sparse synaptic graph after Hebb/STDP/delta updates)
  - Dopamine history, learning params, decay metadata
  - Optional `history` of prototype evolution
- 🗣 **Speech Synthesis**: Reconstructs audio from learned patterns; **growth model** simulates vocal fold changes with “age”
- 🔗 **Cross-Modal Links**: Associate sound patterns with visual features (toward a unified workspace)
- 🎭 **Emotion & Personality**:
  - **Temporary emotion** (on-the-spot valence/arousal)
  - **Permanent personality** (slowly evolving traits influencing learning)
- 🧭 **Curiosity Engine**: Exploratory guesses when recognition confidence is low; dopamine modulates reinforcement
- 📊 **Visualizations**: 3D neuron graphs, co-firing networks, voice helix, spike heatmaps, animated GIFs

---

## 🧬 System Architecture

