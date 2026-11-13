"""Stage 2: inject controlled label noise into a COPY of the training labels (val never touched).
~15% of training boxes corrupted, split evenly over: class_flip, box_jitter, dropped, spurious."""
import csv, shutil
from common import *

NOISE_RATE = 0.15
def L(a): return [round(float(x), 6) for x in a]
TYPES = ["class_flip", "box_jitter", "dropped", "spurious"]

def jitter(box, rng):
    cx, cy, w, h = box
    for _ in range(20):
        ncx = cx + rng.choice([-1, 1]) * rng.uniform(0.15, 0.40) * w
        ncy = cy + rng.choice([-1, 1]) * rng.uniform(0.15, 0.40) * h
        sw = rng.choice([rng.uniform(0.6, 0.8), rng.uniform(1.25, 1.6)])
        sh = rng.choice([rng.uniform(0.6, 0.8), rng.uniform(1.25, 1.6)])
        nb = np.array([ncx, ncy, w * sw, h * sh])
        x = xywh2xyxy(nb); x = np.clip(x, 0, 1)
        nb = np.array([(x[0]+x[2])/2, (x[1]+x[3])/2, x[2]-x[0], x[3]-x[1]])
        if nb[2] > 0.002 and nb[3] > 0.002:
            iou = iou_matrix(xywh2xyxy(box[None]), xywh2xyxy(nb[None]))[0, 0]
            if iou < 0.7: return nb, iou
    return nb, iou

def main():
    rng = np.random.default_rng(SEED); import random; prng = random.Random(SEED)
    split = load_split()
    train_ids = sorted(i for i, s in split.items() if s == "train")
    gt = {i: read_labels(LAB / "clean" / "train" / f"{i}.txt") for i in train_ids}
    index = [(i, j) for i in train_ids for j in range(len(gt[i]))]
    N = len(index); n_noise = int(round(NOISE_RATE * N)); per = n_noise // 4
    perm = rng.permutation(N)
    assign = {}                       # (img, j) -> type
    for t_i, t in enumerate(TYPES[:3]):          # spurious are added, not taken from existing boxes
        for k in perm[t_i * per:(t_i + 1) * per]: assign[index[k]] = t
    n_spur = n_noise - 3 * per
    # size/class pool for spurious boxes
    allb = np.concatenate([g for g in gt.values() if len(g)])
    manifest, noisy = [], {}
    for i in train_ids:
        rows = []
        for j, b in enumerate(gt[i]):
            t = assign.get((i, j))
            c, box = int(b[0]), b[1:5]
            if t is None: rows.append(b); continue
            if t == "dropped":
                manifest.append(dict(image_id=i, noise_type=t, orig_idx=j, orig_class=CLASSES[c], new_class="",
                                     orig_box=L(box), new_box=""))
                continue
            if t == "class_flip":
                nc = C2I[prng.choice(CONFUSABLE[CLASSES[c]])]
                rows.append(np.r_[nc, box]); new_box = box
                manifest.append(dict(image_id=i, noise_type=t, orig_idx=j, orig_class=CLASSES[c], new_class=CLASSES[nc],
                                     orig_box=L(box), new_box=L(box)))
            elif t == "box_jitter":
                nb, iou = jitter(box, prng); rows.append(np.r_[c, nb])
                manifest.append(dict(image_id=i, noise_type=t, orig_idx=j, orig_class=CLASSES[c], new_class=CLASSES[c],
                                     orig_box=L(box), new_box=L(nb)))
        noisy[i] = rows
    # spurious boxes on background: sample class+size from real boxes, location with IoU<0.02 vs every GT box
    spur_imgs = rng.choice(train_ids, n_spur, replace=True)
    for i in spur_imgs:
        ref = allb[rng.integers(len(allb))]
        w, h = ref[3], ref[4]
        gx = xywh2xyxy(gt[i][:, 1:5]) if len(gt[i]) else np.zeros((0, 4))
        for _ in range(50):
            cx = rng.uniform(w / 2, 1 - w / 2); cy = rng.uniform(h / 2, 1 - h / 2)
            nb = np.array([cx, cy, w, h])
            if len(gx) == 0 or iou_matrix(xywh2xyxy(nb[None]), gx).max() < 0.02: break
        else: continue
        noisy[i].append(np.r_[ref[0], nb])
        manifest.append(dict(image_id=i, noise_type="spurious", orig_idx=-1, orig_class="", new_class=CLASSES[int(ref[0])],
                             orig_box="", new_box=L(nb)))
    # write noisy label set: train = noisy copy, val = hard copy of clean
    out = LAB / "noisy"
    if out.exists(): shutil.rmtree(out)
    (out / "train").mkdir(parents=True); (out / "val").mkdir(parents=True)
    # record the line index of each box in the noisy file (jitter/flip keep order; spurious are appended last)
    line_of = {}
    for i in train_ids:
        rows = noisy[i]
        write_labels(out / "train" / f"{i}.txt", np.array(rows).reshape(-1, 5))
    for p in (LAB / "clean" / "val").glob("*.txt"): shutil.copy(p, out / "val" / p.name)
    # attach noisy line index: re-derive by scanning in write order
    counters = {}
    kept = {}
    for m in manifest: pass
    # build noisy_idx: for flip/jitter the row position = number of non-dropped rows before it in that image
    dropped = {(m["image_id"], m["orig_idx"]) for m in manifest if m["noise_type"] == "dropped"}
    pos = {}
    for i in train_ids:
        k = 0
        for j in range(len(gt[i])):
            if (i, j) in dropped: continue
            pos[(i, j)] = k; k += 1
        counters[i] = k
    for m in manifest:
        if m["noise_type"] in ("class_flip", "box_jitter"): m["noisy_idx"] = pos[(m["image_id"], m["orig_idx"])]
        elif m["noise_type"] == "spurious":
            m["noisy_idx"] = counters[m["image_id"]]; counters[m["image_id"]] += 1
        else: m["noisy_idx"] = -1
    flds = ["image_id", "noise_type", "orig_idx", "noisy_idx", "orig_class", "new_class", "orig_box", "new_box"]
    with open(RES / "noise_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, flds); w.writeheader(); w.writerows(manifest)
    from collections import Counter
    cnt = Counter(m["noise_type"] for m in manifest)
    print(f"train boxes={N}  corrupted={len(manifest)} ({len(manifest)/N:.1%})  by type={dict(cnt)}")
    json_out = {"train_boxes": N, "corrupted": len(manifest), "rate": len(manifest) / N, "by_type": dict(cnt)}
    import json; json.dump(json_out, open(RES / "noise_summary.json", "w"), indent=2)

if __name__ == "__main__": main()
