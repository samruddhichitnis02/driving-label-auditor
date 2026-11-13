"""Shared paths, class list and YOLO-format helpers."""
import os, csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw" / "bdd_hf"
IMG_ALL = DATA / "images_all"          # all 10k jpgs (flat)
LAB = DATA / "labels"                  # label sets: clean/, noisy/ (each has train/ val/)
RES = ROOT / "results"
FIG = ROOT / "figures"
WORK = ROOT / "work"                   # per-experiment dataset dirs + yolo runs (gitignored)
CLASSES = ["car", "truck", "bus", "person", "rider", "bike", "motor",
           "traffic light", "traffic sign", "train"]
C2I = {c: i for i, c in enumerate(CLASSES)}
SEED = 0
# BDD100K raw names -> our class names (other vehicle/trailer/other person are ignored)
ALIAS = {"pedestrian": "person", "bicycle": "bike", "motorcycle": "motor"}
# confusable pairs used for class-flip noise (and by the scorer's "class mismatch")
CONFUSABLE = {"car": ["truck"], "truck": ["car", "bus"], "bus": ["truck"],
              "bike": ["motor"], "motor": ["bike"], "person": ["rider"],
              "rider": ["person"], "traffic light": ["traffic sign"],
              "traffic sign": ["traffic light"], "train": ["bus"]}
for d in (RES, FIG, WORK, LAB):
    d.mkdir(parents=True, exist_ok=True)

def read_labels(path):
    """-> array (n,5): cls, cx, cy, w, h (normalised)."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return np.zeros((0, 5))
    a = np.loadtxt(p, ndmin=2)
    return a.reshape(-1, 5)

def write_labels(path, arr):
    with open(path, "w") as f:
        for r in arr:
            f.write(f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n")

def load_split():
    sp = {}
    with open(DATA / "split.csv") as f:
        for r in csv.DictReader(f):
            sp[r["image_id"]] = r["split"]
    return sp

def xywh2xyxy(a):
    cx, cy, w, h = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)

def iou_matrix(a, b):
    """a (n,4) b (m,4) xyxy -> (n,m)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)

def make_dataset(name, train_ids, val_ids, train_label_dir, val_label_dir):
    """Build WORK/name/{images,labels}/{train,val} using hard links for images; return data.yaml path."""
    import yaml
    base = WORK / name
    for sp, ids, ld in (("train", train_ids, train_label_dir), ("val", val_ids, val_label_dir)):
        (base / "images" / sp).mkdir(parents=True, exist_ok=True)
        (base / "labels" / sp).mkdir(parents=True, exist_ok=True)
        for i in ids:
            dst = base / "images" / sp / f"{i}.jpg"
            if not dst.exists():
                try: os.link(IMG_ALL / f"{i}.jpg", dst)
                except OSError:
                    import shutil; shutil.copy(IMG_ALL / f"{i}.jpg", dst)
            lp = Path(ld) / f"{i}.txt"
            out = base / "labels" / sp / f"{i}.txt"
            if lp.exists(): out.write_bytes(lp.read_bytes())
            else: out.write_text("")
    y = {"path": str(base), "train": "images/train", "val": "images/val",
         "names": {i: c for i, c in enumerate(CLASSES)}}
    p = base / "data.yaml"; p.write_text(yaml.safe_dump(y))
    return p
