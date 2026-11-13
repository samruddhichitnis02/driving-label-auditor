"""Stage 6: three detectors, identical settings+seed, evaluated on the CLEAN val set.
 (a) clean labels  (b) noisy labels  (c) noisy labels after fixing/removing the top-flagged labels.
Cleaning rule for (c) (uses NO ground truth): take the top-K ranked rows, K = 15% of training boxes (an assumed
noise budget). Label rows flagged 'class_mismatch' with a suggested class are RELABELLED; other flagged label
rows are REMOVED. Flagged 'missing label' candidates are not added (no label to fix)."""
import json, shutil, sys
import pandas as pd
from common import *
from train_utils import *

def build_cleaned():
    sc = pd.read_csv(RES / "label_scores.csv")
    n_boxes = int((sc.kind == "label").sum()); K = int(0.15 * n_boxes)
    top = sc.sort_values("suspicion", ascending=False, kind="stable").head(K)
    top = top[top.kind == "label"]
    fix = {(r.image_id, r.label_idx): r for r in top.itertuples()}
    out = LAB / "cleaned" / "train"
    if out.parent.exists(): shutil.rmtree(out.parent)
    out.mkdir(parents=True); (LAB / "cleaned" / "val").mkdir()
    n_rel = n_rm = 0
    for p in (LAB / "noisy" / "train").glob("*.txt"):
        a = read_labels(p); rows = []
        for j, b in enumerate(a):
            f = fix.get((p.stem, j))
            if f is None: rows.append(b)
            elif f.suspected_type == "class_mismatch" and isinstance(f.suggested_class, str) and f.suggested_class in C2I:
                b = b.copy(); b[0] = C2I[f.suggested_class]; rows.append(b); n_rel += 1
            else: n_rm += 1
        write_labels(out / p.name, np.array(rows).reshape(-1, 5))
    info = dict(K_top_rows=K, label_rows_flagged=len(top), relabelled=n_rel, removed=n_rm)
    json.dump(info, open(RES / "cleaning_summary.json", "w"), indent=2); print(info)

def main():
    from ultralytics import YOLO
    split = load_split()
    tr = sorted(i for i, s in split.items() if s == "train"); va = sorted(i for i, s in split.items() if s == "val")
    build_cleaned()
    sets = {"a_clean": LAB / "clean" / "train", "b_noisy": LAB / "noisy" / "train", "c_cleaned": LAB / "cleaned" / "train"}
    res = {}
    for name, ld in sets.items():
        run = f"downstream_{name}"
        y = make_dataset(run, tr, va, ld, LAB / "clean" / "val")
        wts = train_yolo(run, y)
        m = YOLO(str(wts)).val(data=str(y), imgsz=640, device=0, verbose=False, plots=False, split="val")
        res[name] = dict(mAP50=float(m.box.map50), mAP50_95=float(m.box.map),
                         per_class_mAP50={CLASSES[int(c)]: float(v) for c, v in zip(m.box.ap_class_index, m.box.ap50)})
        json.dump(res, open(RES / "downstream_results.json", "w"), indent=2)
        print(name, res[name]["mAP50"], res[name]["mAP50_95"], flush=True)

if __name__ == "__main__": main()
