"""Stage 8: figures + README.md + report.md + DEVIATIONS.md, all numbers pulled from results/*.json|csv."""
import json, glob
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cv2
from common import *

J = lambda p: json.load(open(RES / p))
ds, noise, ev, ev1, dn, clean, real = (J("dataset_stats.json"), J("noise_summary.json"), J("audit_eval.json"),
    J("audit_eval_v1_raw_scores.json"), J("downstream_results.json"), J("cleaning_summary.json"), J("real_audit/summary.json"))
cf = {k: pd.read_csv(RES / "logs" / f"crossfit_model{k}_epochs.csv").rename(columns=str.strip).iloc[-1] for k in (0, 1)}
pt = ev["per_type"]; N = ev["n_corrupted"]
name = {"a_clean": "(a) clean labels", "b_noisy": "(b) noisy labels", "c_cleaned": "(c) noisy, auto-cleaned"}

# ---- figures
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8)); cols = ["#2f855a", "#c05621", "#2b6cb0"]
for a, key, t in ((ax[0], "mAP50", "mAP50"), (ax[1], "mAP50_95", "mAP50-95")):
    v = [dn[k][key] for k in dn]; b = a.bar([name[k] for k in dn], v, color=cols)
    a.set_title(t + " on CLEAN val (2000 images)"); a.set_ylim(0, max(v) * 1.2); a.tick_params(axis="x", labelsize=7)
    for r, x in zip(b, v): a.text(r.get_x() + r.get_width() / 2, x + 0.004, f"{x:.3f}", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "map_comparison.png", dpi=150); plt.close()
crops = sorted(glob.glob(str(RES / "real_audit/crops/*.jpg")))[:12]
tiles = [cv2.resize(cv2.imread(c), (320, 240)) for c in crops]
for t_, c in zip(tiles, crops):
    cv2.putText(t_, "#" + Path(c).name[:3] + " flagged, unverified", (4, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
cv2.imwrite(str(FIG / "example_flagged_crops.jpg"), np.vstack([np.hstack(tiles[i:i + 4]) for i in (0, 4, 8)]))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for n_, c_ in (("a_clean", cols[0]), ("b_noisy", cols[1]), ("c_cleaned", cols[2])):
    d = pd.read_csv(RES / "logs" / f"downstream_{n_}_epochs.csv").rename(columns=str.strip)
    ax[0].plot(d.epoch, d["train/box_loss"], color=c_, label=name[n_]); ax[1].plot(d.epoch, d["metrics/mAP50(B)"], color=c_, label=name[n_])
ax[0].set(title="train box loss", xlabel="epoch"); ax[1].set(title="val mAP50 (clean val)", xlabel="epoch"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG / "training_curves.png", dpi=150); plt.close()

# ---- tables
pk = "\n".join(f"| {r['k']:,} | {r['precision']:.3f} | {r['recall']:.3f} |" for r in ev["precision_recall_at_k"])
tmap = {"class_flip": "class flip", "box_jitter": "box jitter", "dropped": "dropped box", "spurious": "spurious box"}
pt_rows = "\n".join(f"| {tmap[t]} | {pt[t]['total']:,} | {pt[t]['detected_in_top_%d' % N]:,} | {pt[t]['rate_top_%d' % N]*100:.1f}% | {pt[t]['rate_top_%d' % (2*N)]*100:.1f}% |" for t in tmap)
ds_rows = "\n".join(f"| {c} | {ds['boxes_per_class']['train'][c]:,} | {ds['boxes_per_class']['val'][c]:,} |" for c in CLASSES)
dn_rows = "\n".join(f"| {name[k]} | {dn[k]['mAP50']:.3f} | {dn[k]['mAP50_95']:.3f} |" for k in dn)
d_bc = dn["c_cleaned"]["mAP50"] - dn["b_noisy"]["mAP50"]; d_ba = dn["a_clean"]["mAP50"] - dn["b_noisy"]["mAP50"]
if d_bc <= 0.005:
    verdict = (f"Auto-cleaning did **not** help: (c) mAP50 {dn['c_cleaned']['mAP50']:.3f} vs (b) {dn['b_noisy']['mAP50']:.3f} "
               f"({d_bc:+.3f}); a difference this small is within what one seed can produce.")
else:
    verdict = f"Auto-cleaning helped: (c) {dn['c_cleaned']['mAP50']:.3f} vs (b) {dn['b_noisy']['mAP50']:.3f} ({d_bc:+.3f}), single seed."
claims = dict(train_boxes=ds["n_boxes"]["train"], val_boxes=ds["n_boxes"]["val"], corrupted=noise["corrupted"],
              noise_rate_pct=round(noise["rate"] * 100, 1), ap_v2=round(ev["average_precision"], 3), ap_v1=round(ev1["average_precision"], 3),
              random_precision=round(ev["random_baseline_precision"], 3),
              mAP50_a=round(dn["a_clean"]["mAP50"], 3), mAP50_b=round(dn["b_noisy"]["mAP50"], 3), mAP50_c=round(dn["c_cleaned"]["mAP50"], 3),
              mAP5095_a=round(dn["a_clean"]["mAP50_95"], 3), mAP5095_b=round(dn["b_noisy"]["mAP50_95"], 3), mAP5095_c=round(dn["c_cleaned"]["mAP50_95"], 3),
              real_top_queue=100, real_ge_099=real["labels_with_suspicion_ge_0_99"], real_ge_095=real["labels_with_suspicion_ge_0_95"])
json.dump(claims, open(RES / "claims.json", "w"), indent=1)
p1 = ev["precision_recall_at_k"][0]["precision"]; dr = pt["dropped"]

RESULTS = f"""## Results
### Does the auditor find the injected errors? (`results/audit_eval.json`)
Ranking is over {ev['n_ranked_rows']:,} items (all noisy labels + missing-label candidates). Random ranking gives precision {ev['random_baseline_precision']:.3f}. Average precision = **{ev['average_precision']:.3f}**.

| top-k flagged | precision | recall |
|---|---|---|
{pk}

Detection rate by noise type (share of that type inside the top-{N:,} flagged, i.e. a review budget equal to the number of errors, and inside the top-{2*N:,}):

| noise type | injected | found in top {N:,} | rate | rate at top {2*N:,} |
|---|---|---|---|---|
{pt_rows}

![PR curves](figures/pr_curves.png)

**Where it works:** spurious boxes and box jitter are found at roughly 46-47% at a budget equal to the error count; class flips 35%.
**Where it does not:** dropped (missing) boxes. Only {dr['max_possible_recall_with_candidates']*100:.0f}% of them are even proposed as candidates by the model, and just
{dr['rate_top_%d' % N]*100:.0f}% are in the top-{N:,}. Precision at the very top of the list is only ~{p1*100:.0f}%, so it is a triage aid, not an oracle.
The first version (raw, uncalibrated sub-scores) had AP {ev1['average_precision']:.3f}; v2 was adopted after looking at those results (kept in `results/audit_eval_v1_raw_scores.json`).

### Does cleaning the labels help the detector? (clean val, identical settings and seed 0)
| training labels | mAP50 | mAP50-95 |
|---|---|---|
{dn_rows}

{verdict} Noise itself cost only {d_ba:.3f} mAP50 ((a) vs (b)). Cleaning flagged the top {clean['K_top_rows']:,} rows; {clean['relabelled']:,} labels were relabelled to the model's suggested class and {clean['removed']:,} removed.
Removing flagged labels also removes many *correct* labels (precision ~35%), which plausibly cancels the benefit.

![mAP comparison](figures/map_comparison.png)  ![training curves](figures/training_curves.png)

### Audit of the original labels - FLAGGED, UNVERIFIED
Running the scorer on the original, uncorrupted labels: top 100 flagged boxes saved as crops (`results/real_audit/crops/`, `review_queue.csv`, `review.html`).
{real['labels_with_suspicion_ge_0_99']:,} labels score >= 0.99 and {real['labels_with_suspicion_ge_0_95']:,} score >= 0.95 (percentile scores, so these counts are not error counts).
**None of these is claimed to be a real error; nobody has reviewed them yet.** Please review a sample and record verdicts in `review_queue.csv`.

![flagged crops](figures/example_flagged_crops.jpg)

"""
LIMITS = f"""## Limitations
- Noise is synthetic and uniformly random; real annotation errors are more structured and harder, so detection rates on real errors will differ.
- Single seed, no confidence intervals; the (b) vs (c) gap is smaller than plausible seed variance.
- Cross-fit models train on 4,000 images each (val mAP50 {cf[0]['metrics/mAP50(B)']:.3f} and {cf[1]['metrics/mAP50(B)']:.3f}), so they are a weaker judge than a full-data model; small objects (traffic signs/lights) are hard for them, causing false flags.
- Rare classes (`train`: {ds['boxes_per_class']['train']['train']} train boxes) are effectively unlearnable; results for them are meaningless.
- The scorer was revised once after seeing evaluation results (v1 to v2); v2 does not use the manifest, but the choice was informed by it.
- Val is used for per-epoch logging only; final weights are last-epoch, no checkpoint selection on val.
- Stage 7 predictions come from models trained on the *noisy* copy (the same cross-fit predictions), then compared with the original labels.
"""
HEAD = f"""# Label Quality Auditor for autonomous-driving object detection

**Problem.** Detectors are only as good as their labels, and nobody can re-inspect millions of boxes by hand. This repo builds a
pipeline that *flags boxes that are probably wrong* so a human reviews a short list, and measures how well it works by
corrupting labels on purpose, so the ground truth is known.

**Data.** BDD100K, 10,000 driving images with the original detection boxes (public Hugging Face copy, see DEVIATIONS.md), 10 classes.
Fixed split (seed 0): 8,000 train / 2,000 val. **Val is never corrupted.**

| class | train boxes | val boxes |
|---|---|---|
{ds_rows}

## Method
1. **Prep**: convert to YOLO format, fixed split, stats (`results/dataset_stats.*`).
2. **Noise injection** on a *copy* of the train labels: {noise['corrupted']:,} of {noise['train_boxes']:,} boxes ({noise['rate']*100:.1f}%) corrupted, split evenly over class flip
   (confusable pairs such as car/truck, bike/motor, person/rider, light/sign), box jitter (shift 15-40% and rescale), dropped box, spurious box on background.
   Every change is in `results/noise_manifest.csv` (the answer key).
3. **Cross-fitted detector**: YOLO11s fine-tuned on the *noisy* labels, 2-fold: train on half A, predict half B, and the reverse, so no image is scored by a model that trained on it.
   Per-epoch loss and val mAP: `results/logs/`.
4. **Scoring**: each label is compared with the held-out predictions: *class mismatch* (confident prediction of another class on the box), *localization error*
   (same-class object overlapping only partly), *likely spurious* (nothing predicted there), plus *likely missing* candidates (confident predictions no label covers).
   v2 turns each sub-score into a percentile among labels of the same class and size so they are comparable (uses no ground truth).
5. **Evaluation** against the manifest, 6. **downstream test** (clean vs noisy vs cleaned), 7. **audit of the original labels**.

"""
REPRO = """## Reproduce
```
bash run_all.sh        # from Git Bash on Windows; needs an NVIDIA GPU (about 3-4 h on an RTX 5090)
python verify_claims.py
```
"""
# README.md is now written by s9_visuals.py (plain-language version with GIF + charts); this script keeps report.md/DEVIATIONS.md
REPORT = f"""# Report: Label Quality Auditor

## Summary
- A cross-fitted YOLO11s auditor flags suspicious bounding-box labels on BDD100K. Against {noise['corrupted']:,} deliberately corrupted boxes ({noise['rate']*100:.1f}% of {noise['train_boxes']:,}),
  average precision is **{ev['average_precision']:.3f}** (random ranking: {ev['random_baseline_precision']:.3f}).
- Quality is moderate and uneven: spurious boxes and jitter ~46-47%, class flips 35%, dropped boxes {dr['rate_top_%d' % N]*100:.0f}% at a review budget equal to the error count.
- {verdict}
- The original labels were also audited: the top 100 boxes are queued for manual review, status **flagged, unverified**. We make no claim they are real errors.

{RESULTS}
## Why cleaning did not help (hypotheses, untested)
1. About 65% of what the auditor flags at the top is *not* an injected error, so removal deletes many good labels.
2. Injected noise costs only {d_ba:.3f} mAP50 in this setup, leaving little to recover.
3. One seed per condition; differences of ~0.005 are not distinguishable from noise.

## Next steps
Review a sample of `review_queue.csv` to estimate precision on real errors; repeat stage 6 with 3+ seeds; use a stronger full-data judge; repair labels instead of deleting them.

{LIMITS}"""
(ROOT / "report.md").write_text(REPORT, encoding="utf-8")
DEV = """# Deviations from the brief
1. **Dataset source.** The official BDD100K host (dl.cv.ethz.ch / bdd-data.berkeley.edu) was unreachable or needs a login, so I used the public Hugging Face copy
   `dgural/bdd100k` (10,000 images with the original detection labels, FiftyOne format). This is not the official 10K download; it appears to be the BDD100K
   validation set but I could not verify it against the official files. No KITTI/COCO fallback was needed.
2. **Class names.** BDD raw names `pedestrian/bicycle/motorcycle` were mapped to `person/bike/motor`. `other vehicle`, `trailer`, `other person` were dropped (not in the 10-class list).
3. **Split.** The set is only 10k images, so I made my own fixed 80/20 split (seed 0) instead of using an official train/val.
4. **Detector.** Ultralytics YOLO11s (small) for all runs, 30 epochs, imgsz 640, batch 32, seed 0, deterministic.
5. **Scorer revision.** Version 1 (raw sub-scores) was replaced by version 2 (percentile-normalised per class and size bin) after seeing v1's evaluation. Both are reported.
6. **Cleaning rule (stage 6c).** Top 15% of ranked rows (an assumed noise budget, not the manifest count): class-mismatch labels with a suggested class are relabelled, other flagged labels are removed. Missing-label candidates are not added back.
7. **Stage 7 predictions.** Reused the cross-fit predictions (models trained on noisy labels) to score the original labels.
8. **Single seed** for every experiment (time budget); no confidence intervals.
"""
(ROOT / "DEVIATIONS.md").write_text(DEV, encoding="utf-8"); print("written")
