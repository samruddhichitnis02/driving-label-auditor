#!/usr/bin/env bash
# Reproduces the whole project. Needs Python 3.10+ and an NVIDIA GPU. Run from the repo root (Git Bash on Windows works).
set -euo pipefail
cd "$(dirname "$0")"

# 1. environment (virtualenv inside the repo)
[ -d .venv ] || python -m venv .venv
PY=".venv/Scripts/python.exe"; [ -f "$PY" ] || PY=".venv/bin/python"
$PY -m pip install -q --upgrade pip
# CUDA 12.8 build (needed for RTX 50-series; use another index-url for other GPUs)
$PY -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install -q -r requirements.txt

# BDD100K labels (public HF copy). Images are downloaded by stage 1.
mkdir -p data/raw/bdd_hf
[ -f data/raw/bdd_hf/samples.json ] || curl -sL -o data/raw/bdd_hf/samples.json https://huggingface.co/datasets/dgural/bdd100k/resolve/main/samples.json

$PY src/s1_prep.py          # 1 data prep + stats
$PY src/s2_noise.py         # 2 noise injection + noise_manifest.csv
$PY src/s3_crossfit.py      # 3 2-fold cross-fitted detector (~45 min on RTX 5090)
$PY src/s4_score.py         # 4 label scores
$PY src/s5_eval.py          # 5 evaluate the auditor (v2, percentile scores)
$PY src/s5_eval.py v1       # 5a same, for the v1 raw sub-scores (comparison)
$PY src/s6_downstream.py    # 6 clean vs noisy vs cleaned detectors (~2 h)
$PY src/s7_audit.py         # 7 audit of original labels -> review queue
$PY src/s8_report.py        # 8 report.md + DEVIATIONS.md
$PY src/s9_visuals.py       # 8b demo GIF + all charts + README.md
$PY verify_claims.py        # 9 verify every claim
