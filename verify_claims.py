"""Stage 9: independently re-derive the claims in README.md / report.md from raw artifacts and check them.
Run:  python verify_claims.py   (exit code 1 if any check fails)"""
import ast, json, re, sys
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parent
RES, DATA, LAB = ROOT / "results", ROOT / "data", ROOT / "data" / "labels"
checks = []
def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail)); print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))

def nlines(p):
    t = Path(p).read_text().strip(); return len(t.splitlines()) if t else 0

# ---- data / split
split = pd.read_csv(DATA / "split.csv"); tr = set(split[split.split == "train"].image_id); va = set(split[split.split == "val"].image_id)
check("split is 8000 train / 2000 val, disjoint", len(tr) == 8000 and len(va) == 2000 and not tr & va)
n_tr = sum(nlines(LAB / "clean" / "train" / f"{i}.txt") for i in tr); n_va = sum(nlines(LAB / "clean" / "val" / f"{i}.txt") for i in va)
claims = json.load(open(RES / "claims.json"))
check("train/val box counts match claims", n_tr == claims["train_boxes"] and n_va == claims["val_boxes"], f"{n_tr}/{n_va}")

# ---- val never corrupted
same = all((LAB / "clean" / "val" / f"{i}.txt").read_bytes() == (LAB / "noisy" / "val" / f"{i}.txt").read_bytes() for i in va)
check("val labels identical in clean and noisy sets (val never corrupted)", same)
man = pd.read_csv(RES / "noise_manifest.csv")
check("manifest only references training images", set(man.image_id) <= tr)

# ---- noise rate and type split
rate = len(man) / n_tr * 100
check("corrupted ~15% of training boxes", abs(rate - 15.0) < 0.1 and len(man) == claims["corrupted"], f"{rate:.2f}% ({len(man)})")
vc = man.noise_type.value_counts()
check("four noise types each within 1% of an equal share", set(vc.index) == {"class_flip", "box_jitter", "dropped", "spurious"} and (vc.max() - vc.min()) / vc.mean() < 0.01, str(vc.to_dict()))
# noisy label file line counts = clean - dropped + spurious
n_noisy = sum(nlines(LAB / "noisy" / "train" / f"{i}.txt") for i in tr)
check("noisy label count = clean - dropped + spurious", n_noisy == n_tr - vc["dropped"] + vc["spurious"], f"{n_noisy}")
# spot-check the manifest against the actual files: class flips really differ, jitter really moved
rng = np.random.default_rng(0); ok_flip = ok_jit = 0; samp = man[man.noise_type == "class_flip"].sample(200, random_state=0)
for r in samp.itertuples():
    ln = (LAB / "noisy" / "train" / f"{r.image_id}.txt").read_text().splitlines()[r.noisy_idx].split()
    cl = (LAB / "clean" / "train" / f"{r.image_id}.txt").read_text().splitlines()[r.orig_idx].split()
    ok_flip += ln[0] != cl[0] and ln[1:] == cl[1:]
check("200 sampled class-flip manifest rows match the files", ok_flip == 200, f"{ok_flip}/200")
samp = man[man.noise_type == "box_jitter"].sample(200, random_state=0)
for r in samp.itertuples():
    ln = (LAB / "noisy" / "train" / f"{r.image_id}.txt").read_text().splitlines()[r.noisy_idx].split()
    cl = (LAB / "clean" / "train" / f"{r.image_id}.txt").read_text().splitlines()[r.orig_idx].split()
    ok_jit += ln[0] == cl[0] and ln[1:] != cl[1:]
check("200 sampled box-jitter manifest rows match the files", ok_jit == 200, f"{ok_jit}/200")

# ---- cross-fitting: folds are disjoint halves, predictions only on training images
folds = pd.read_csv(DATA / "folds.csv"); preds = pd.read_csv(RES / "crossfit_predictions.csv", usecols=["image_id"])
check("folds partition the training set into two halves", set(folds.image_id) == tr and abs((folds.fold == 0).sum() - (folds.fold == 1).sum()) <= 1)
check("cross-fit predictions exist only for training images (none for val)", set(preds.image_id) <= tr)
for k in (0, 1):
    d = pd.read_csv(RES / "logs" / f"crossfit_model{k}_epochs.csv").rename(columns=str.strip)
    check(f"cross-fit model {k}: 30 logged epochs with real loss and val mAP", len(d) == 30 and d["train/box_loss"].notna().all() and d["metrics/mAP50(B)"].notna().all())

# ---- evaluation recomputed from scratch
from sklearn.metrics import average_precision_score
sc = pd.read_csv(RES / "label_scores_with_truth.csv")
ap = average_precision_score(sc.is_noise, sc.suspicion)
check("average precision re-computed matches README claim", abs(ap - claims["ap_v2"]) < 0.0015, f"{ap:.3f}")
# independent precision@k from the ranked list
sc = sc.sort_values("suspicion", ascending=False, kind="stable"); ev = json.load(open(RES / "audit_eval.json"))
for r in ev["precision_recall_at_k"]:
    p = sc.head(r["k"]).is_noise.mean(); check(f"precision@{r['k']} recomputed", abs(p - r["precision"]) < 1e-3, f"{p:.3f}")
# the truth labels in that file are right: every positive label row is in the manifest
lab = sc[sc.kind == "label"]; pos = lab[lab.is_noise]
key = set(zip(man[man.noise_type != "dropped"].image_id, man[man.noise_type != "dropped"].noisy_idx))
check("every positive label row is a manifest entry", all((r.image_id, r.label_idx) in key for r in pos.itertuples()))
check("auditor beats random ranking", ap > claims["random_precision"] * 1.5, f"AP {ap:.3f} vs random {claims['random_precision']}")

# ---- downstream numbers and the honesty statement
dn = json.load(open(RES / "downstream_results.json"))
for k, s in (("a_clean", "a"), ("b_noisy", "b"), ("c_cleaned", "c")):
    check(f"downstream {k} mAP50/mAP50-95 equal claims", round(dn[k]["mAP50"], 3) == claims[f"mAP50_{s}"] and round(dn[k]["mAP50_95"], 3) == claims[f"mAP5095_{s}"])
readme = (ROOT / "README.md").read_text(encoding="utf-8"); report = (ROOT / "report.md").read_text(encoding="utf-8")
c_beats_b = dn["c_cleaned"]["mAP50"] > dn["b_noisy"]["mAP50"]
check("README states honestly whether (c) beats (b)", ("did **not** help" in readme) == (not c_beats_b) and ("did **not** help" in report) == (not c_beats_b), f"c beats b: {c_beats_b}")
for k in ("mAP50_a", "mAP50_b", "mAP50_c"):
    check(f"README table contains {k} = {claims[k]:.3f}", f"{claims[k]:.3f}" in readme and f"{claims[k]:.3f}" in report)
import yaml
args = [yaml.safe_load(open(ROOT / "work" / "runs" / f"downstream_{n}" / "args.yaml")) for n in ("a_clean", "b_noisy", "c_cleaned")] if (ROOT / "work" / "runs").exists() else None
if args:
    drop = {"data", "name", "save_dir"}; base = {k: v for k, v in args[0].items() if k not in drop}
    check("downstream run args.yaml identical (epochs, seed, imgsz, batch, model...)", all({k: v for k, v in a.items() if k not in drop} == base for a in args[1:]) and base["seed"] == 0)
else:
    print("SKIP downstream args.yaml check (work/ runs not present)")

# ---- real-label audit
q = pd.read_csv(RES / "real_audit" / "review_queue.csv")
check("review queue has exactly 100 boxes", len(q) == 100)
check("every queue row is labelled 'flagged, unverified' and has an empty human verdict", (q.status == "flagged, unverified").all() and q.human_verdict.isna().all())
check("100 crop images exist", len(list((RES / "real_audit" / "crops").glob("*.jpg"))) == 100)
html = (RES / "real_audit" / "review.html").read_text(encoding="utf-8")
check("review.html shows 'flagged, unverified' and makes no 'confirmed error' claim", html.count("flagged, unverified") >= 100 and "confirmed error" not in html.lower())
check("README says real-audit boxes are not claimed as errors", "None of these is claimed to be a real error" in readme)
# every required deliverable exists
for f in ["README.md", "report.md", "DEVIATIONS.md", "requirements.txt", "run_all.sh", "figures/pr_curves.png", "figures/map_comparison.png",
          "figures/example_flagged_crops.jpg", "results/noise_manifest.csv", "results/label_scores.csv", "results/real_audit/review_queue.csv"]:
    check(f"deliverable exists: {f}", (ROOT / f).exists())

bad = [c for c in checks if not c[1]]
print(f"\n{len(checks) - len(bad)}/{len(checks)} checks passed")
sys.exit(1 if bad else 0)
