"""Shared YOLO training settings so every experiment uses identical hyper-parameters."""
import shutil, json
from pathlib import Path
from common import *

MODEL = "yolo11s.pt"
HP = dict(imgsz=640, epochs=30, batch=32, workers=8, seed=SEED, deterministic=True,
          patience=100, amp=True, plots=False, exist_ok=True, verbose=False, cache=False)

def train_yolo(name, data_yaml, epochs=None):
    from ultralytics import YOLO
    hp = dict(HP); 
    if epochs: hp["epochs"] = epochs
    m = YOLO(MODEL)
    m.train(data=str(data_yaml), project=str(WORK / "runs"), name=name, device=0, **hp)
    rd = WORK / "runs" / name
    (RES / "logs").mkdir(exist_ok=True)
    shutil.copy(rd / "results.csv", RES / "logs" / f"{name}_epochs.csv")   # real per-epoch loss + val mAP
    return rd / "weights" / "last.pt"
