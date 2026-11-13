"""Stage 3: 2-fold cross-fitting. Train on noisy labels of half A -> predict half B, and reverse."""
import csv, sys
from common import *
from train_utils import *

def main():
    from ultralytics import YOLO
    split = load_split()
    train_ids = sorted(i for i, s in split.items() if s == "train")
    val_ids = sorted(i for i, s in split.items() if s == "val")
    rng = np.random.default_rng(SEED + 1); perm = rng.permutation(len(train_ids))
    fold = {train_ids[k]: int(n % 2) for n, k in enumerate(perm)}
    with open(DATA / "folds.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image_id", "fold"]); [w.writerow([i, fold[i]]) for i in train_ids]
    rows = []
    for k in (0, 1):                                   # model k trains on fold k, predicts fold 1-k
        tr = [i for i in train_ids if fold[i] == k]; ho = [i for i in train_ids if fold[i] != k]
        yaml_p = make_dataset(f"cf{k}", tr, val_ids, LAB / "noisy" / "train", LAB / "clean" / "val")
        wts = train_yolo(f"crossfit_model{k}", yaml_p)
        model = YOLO(str(wts))
        paths = [str(IMG_ALL / f"{i}.jpg") for i in ho]
        for s in range(0, len(paths), 64):
            for r in model.predict(paths[s:s + 64], imgsz=640, conf=0.05, iou=0.6, max_det=300,
                                   device=0, verbose=False, stream=False):
                iid = Path(r.path).stem; b = r.boxes
                for c, cf, xywhn in zip(b.cls.cpu().numpy(), b.conf.cpu().numpy(), b.xywhn.cpu().numpy()):
                    rows.append([iid, int(c), float(cf), *map(float, xywhn)])
        print(f"fold {k}: predicted {len(ho)} held-out images", flush=True)
    import pandas as pd
    pd.DataFrame(rows, columns=["image_id", "cls", "conf", "cx", "cy", "w", "h"]).to_csv(RES / "crossfit_predictions.csv", index=False)
    print("predictions:", len(rows))

if __name__ == "__main__": main()
