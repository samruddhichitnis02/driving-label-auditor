"""Stage 1: download BDD100K 10K images + labels, convert to YOLO, fixed split, dataset stats."""
import json, csv, random, shutil
from collections import Counter
from common import *

def main():
    from huggingface_hub import snapshot_download
    IMG_ALL.mkdir(parents=True, exist_ok=True)
    if len(list(IMG_ALL.glob("*.jpg"))) < 10000:
        tmp = DATA / "raw" / "bdd_hf_images"
        snapshot_download("dgural/bdd100k", repo_type="dataset", local_dir=str(tmp),
                          allow_patterns=["data/*.jpg", "samples.json"], max_workers=16)
        for p in (tmp / "data").glob("*.jpg"):
            shutil.move(str(p), IMG_ALL / p.name)
    samples = json.load(open(RAW / "samples.json"))["samples"]
    ids, rows = [], []
    for s in samples:
        iid = Path(s["filepath"]).stem
        W, H = s["metadata"]["width"], s["metadata"]["height"]
        dets = (s.get("detections") or {}).get("detections", [])
        arr = []
        for d in dets:
            lab = ALIAS.get(d["label"], d["label"])
            if lab not in C2I: continue
            x, y, w, h = d["bounding_box"]            # normalised top-left x,y,w,h
            if w <= 0 or h <= 0: continue
            arr.append([C2I[lab], x + w / 2, y + h / 2, w, h])
        ids.append(iid); rows.append((iid, np.array(arr).reshape(-1, 5), W, H))
    rng = random.Random(SEED); order = sorted(ids); rng.shuffle(order)
    n_val = int(0.2 * len(order)); val = set(order[:n_val])
    with open(DATA / "split.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image_id", "split"])
        for i in sorted(ids): w.writerow([i, "val" if i in val else "train"])
    for sp in ("train", "val"): (LAB / "clean" / sp).mkdir(parents=True, exist_ok=True)
    stats = {"train": Counter(), "val": Counter()}; nimg = Counter(); nbox = Counter()
    for iid, arr, W, H in rows:
        sp = "val" if iid in val else "train"
        write_labels(LAB / "clean" / sp / f"{iid}.txt", arr)
        nimg[sp] += 1
        for c in arr[:, 0].astype(int): stats[sp][CLASSES[c]] += 1; nbox[sp] += 1
    out = {"source": "HuggingFace dgural/bdd100k (BDD100K 10K images with original detection boxes)",
           "n_images": dict(nimg), "n_boxes": dict(nbox),
           "boxes_per_class": {sp: {c: stats[sp][c] for c in CLASSES} for sp in stats}}
    json.dump(out, open(RES / "dataset_stats.json", "w"), indent=2)
    with open(RES / "dataset_stats.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["class", "train_boxes", "val_boxes"])
        for c in CLASSES: w.writerow([c, stats["train"][c], stats["val"][c]])
    print(json.dumps(out, indent=2))

if __name__ == "__main__": main()
