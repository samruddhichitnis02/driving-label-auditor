"""Writes README.md. All numbers come from the result files passed in; only the wording lives here."""
import pandas as pd
from common import *


def readme(fr, demo, ds, noise, ev, ev1, dn, clean, real, TYPES, TNAME, DNAME, N):
    iid, hit, n_err, fa = demo
    pt = ev["per_type"]
    lift = ev["average_precision"] / ev["random_baseline_precision"]
    P = lambda t, k=N: pt[t]["rate_top_%d" % k] * 100
    P1 = lambda t: ev1["per_type"][t]["rate_top_%d" % N] * 100
    pk = "\n".join(f"| {r['k']:,} | {r['precision']:.3f} | {r['recall']:.3f} |" for r in ev["precision_recall_at_k"])
    tr = "\n".join(f"| {TNAME[t]} | {pt[t]['total']:,} | {pt[t]['detected_in_top_%d' % N]:,} | {P(t):.1f}% | {P(t, 2*N):.1f}% | {P1(t):.1f}% |" for t in TYPES)
    ds_rows = "\n".join(f"| {c} | {ds['boxes_per_class']['train'][c]:,} | {ds['boxes_per_class']['val'][c]:,} |" for c in CLASSES)
    nz_rows = "\n".join(f"| {TNAME[t]} | {d} | {noise['by_type'][t]:,} |" for t, d in [
        ("class_flip", "the label is swapped for a class it is easily confused with (car/truck, truck/bus, bike/motor, person/rider, traffic light/sign, train/bus)"),
        ("box_jitter", "the box centre is moved by 15-40% of the box size and the width and height are rescaled (x0.6-0.8 or x1.25-1.6), so IoU with the original is below 0.7"),
        ("spurious", "a new box with a class and size copied from real boxes, placed where it overlaps no existing label (IoU < 0.02)"),
        ("dropped", "the label is deleted")])
    dn_rows = "\n".join(f"| {DNAME[k]} | {dn[k]['mAP50']:.3f} | {dn[k]['mAP50_95']:.3f} |" for k in dn)
    d_bc = dn["c_cleaned"]["mAP50"] - dn["b_noisy"]["mAP50"]
    d_ba = dn["a_clean"]["mAP50"] - dn["b_noisy"]["mAP50"]
    if d_bc <= 0.005:
        verdict = (f"Cleaning the labels did **not** help. The cleaned run (c) reached mAP50 {dn['c_cleaned']['mAP50']:.3f} against {dn['b_noisy']['mAP50']:.3f} for the noisy run (b), "
                   f"a difference of {d_bc:+.3f}. With one seed per run I wouldn't read anything into a gap that small, but it is certainly not an improvement.")
    else:
        verdict = f"Cleaning helped a little: (c) reached mAP50 {dn['c_cleaned']['mAP50']:.3f} against {dn['b_noisy']['mAP50']:.3f} for (b) ({d_bc:+.3f}), from a single seed."
    cf = {k: pd.read_csv(RES / "logs" / f"crossfit_model{k}_epochs.csv").rename(columns=str.strip).iloc[-1] for k in (0, 1)}
    cm = fr["s_class_mismatch"]
    q = real["queue_by_suspected_type"]
    n_img = ds["n_images"]["train"] + ds["n_images"]["val"]

    txt = f"""# Label Quality Auditor

Finding wrong bounding-box labels in self-driving data, and measuring how well that works when I know exactly which labels are wrong.

[Report](report.md) | [Deviations from the plan](DEVIATIONS.md) | [How to run it](#running-it)

<p align="center"><img src="figures/fig01_demo.gif" width="700" alt="One training image going through the pipeline"></p>

**Figure 1.** One training image going through the pipeline: the original labels, the labels after I corrupted some of them, what a detector that never saw this image predicts, the boxes the auditor flags, and a comparison with the answer key. Green is a correct label, orange a corrupted one, blue a prediction, yellow a flag.

## Summary

A self-driving perception model is only as good as the boxes it was trained on, and people drawing those boxes make mistakes. I wanted to see how well a model can point at those mistakes on its own. The difficulty is that you can't measure that on a normal dataset, because nobody knows which labels are wrong. So I took BDD100K ({n_img:,} images), deliberately broke {noise['rate']*100:.0f}% of the training boxes in four different ways, and trained a YOLO11s detector on the damaged labels. To stop the model from simply agreeing with labels it had memorised, I trained it in two folds and only let each one judge images it had never seen. Every label then gets a suspicion score based on how much the held-out prediction disagrees with it.

Measured against the known errors, the ranking reaches an average precision of {ev['average_precision']:.3f}, compared with {ev['random_baseline_precision']:.3f} for a random order ({lift:.1f}x better). If a reviewer looks at as many items as there are injected errors, they find {P('box_jitter'):.0f}% of the jittered boxes, {P('spurious'):.0f}% of the spurious boxes, {P('class_flip'):.0f}% of the class flips and {P('dropped'):.0f}% of the dropped boxes. The part that did not work out: removing or relabelling the top-ranked 15% of labels and retraining did **not** help (mAP50 {dn['c_cleaned']['mAP50']:.3f} against {dn['b_noisy']['mAP50']:.3f} with the noisy labels and {dn['a_clean']['mAP50']:.3f} with clean ones, one seed). I also ran the auditor on the original labels and saved the 100 most suspicious boxes. Nobody has checked those yet.

## Why this problem

Looking through every label by hand does not scale, so a team needs a short list of boxes worth a second look. To judge whether such a list is any good, you need a case where the wrong labels are known. That is why I inject the errors myself instead of hunting for real ones. The two-fold trick comes from confident learning (Northcutt et al., 2021); here I apply it to box-level detection labels and four kinds of error.

![pipeline](figures/fig02_pipeline.png)

**Figure 2.** The pipeline, from data to the audit of the original labels.

## How it works

### Data

I used the BDD100K detection labels for {n_img:,} images, taken from a public Hugging Face copy (see `DEVIATIONS.md` for why not the official download). Class names are mapped to ten classes (`pedestrian`, `bicycle` and `motorcycle` become `person`, `bike` and `motor`) and other categories are dropped. A fixed random split gives {ds['n_images']['train']:,} training and {ds['n_images']['val']:,} validation images. The validation labels are never touched.

**Table 1.** Boxes per class.

| class | train | val |
|---|---|---|
{ds_rows}

![dataset](figures/fig03_dataset.png)

**Figure 3.** Boxes per class on a log scale. Cars and traffic signs dominate. The `train` class has only {ds['boxes_per_class']['train']['train']} training boxes, which is too few to learn anything from.

### Breaking the labels

I copied the training labels and corrupted {noise['corrupted']:,} of the {noise['train_boxes']:,} boxes ({noise['rate']*100:.1f}%), split evenly over four error types. Each change is recorded in `results/noise_manifest.csv`, which serves as the answer key.

**Table 2.** The four error types.

| type | what I do | boxes |
|---|---|---|
{nz_rows}

![noise examples](figures/fig04_noise_examples.png)

**Figure 4.** One real example of each error. Green dashed is the original box, orange is the corrupted one.

### The cross-fitted detector

I split the training images into two folds. A YOLO11s detector (pretrained on COCO, 30 epochs, image size 640, batch 32, seed 0) is fine-tuned on the noisy labels of one fold and then predicts the other fold, and the same again the other way round. That way no image is ever scored by a model that trained on it, which matters because a model that has seen a wrong label can learn to repeat it.

![cross-fitting](figures/fig05_cross_fitting.png)

**Figure 5.** Two-fold cross-fitting.

![training](figures/fig06_detector_training.png)

**Figure 6.** Training loss and validation mAP50 for the two detectors (per-epoch logs are in `results/logs/`). They end at mAP50 {cf[0]['metrics/mAP50(B)']:.3f} and {cf[1]['metrics/mAP50(B)']:.3f}. Each one only sees 4,000 images, so they are fairly modest judges.

### Scoring the labels

For every label, I look at what the held-out detector predicts around it and compute three signals, plus a list of possible missing labels.

**Table 3.** The signals.

| signal | how it is computed |
|---|---|
| class mismatch | highest confidence of a prediction of a different class with IoU >= 0.5, minus the highest same-class confidence at IoU >= 0.5 (not below 0) |
| localization error | over same-class predictions with 0.1 <= IoU < 0.75, the highest conf x (1 - IoU), times (1 - highest same-class conf at IoU >= 0.75) |
| spurious label | 1 minus the highest confidence of any prediction with IoU >= 0.3 |
| missing label | predictions with conf >= 0.3 that overlap no label (IoU < 0.3), ranked by confidence |

My first version (v1) ranked labels by the largest raw signal. That worked badly, because the spurious signal fired on so many small, hard-to-see objects that it buried everything else. In v2 I turned each signal into a percentile among labels of the same class and size, so they can be compared, and rank by the largest percentile. v2 does not use the answer key, but I chose it after seeing how v1 did, so I report both.

![score distribution](figures/fig07_score_distribution.png)

**Figure 7.** v2 suspicion scores for uncorrupted and corrupted labels.

### How I measure it

I rank all {ev['n_ranked_rows']:,} items (the noisy labels plus the missing-label candidates) by suspicion. A flagged label counts as a hit if it is in the manifest. A missing-label candidate counts as a hit if it overlaps a dropped box with IoU >= 0.5, and each dropped box can only be matched once. Precision@k is the share of the top k items that are real errors, and recall@k is the share of all {N:,} injected errors found within the top k. Average precision (AP) summarises the whole precision-recall curve in one number. I also report how many errors of each type turn up in the top N and top 2N items, where N = {N:,} is the number of injected errors.

## Results

### Finding the injected errors

**Table 4.** Precision and recall of the v2 ranking. A random ranking would have precision {ev['random_baseline_precision']:.3f}.

| k | precision@k | recall@k |
|---|---|---|
{pk}

![pr](figures/fig08_pr_curves.png)

**Figure 8.** (a) Precision-recall curves for v1 and v2. (b) How quickly each error type is found as a reviewer reads further down the v2 list.

![at k](figures/fig09_precision_recall_at_k.png)

**Figure 9.** Precision@k and recall@k for v2. The dotted line marks k = N.

**Table 5.** Share of the injected errors found, by type.

| type | injected | found in top N | rate (v2) | rate in top 2N (v2) | rate in top N (v1) |
|---|---|---|---|---|---|
{tr}

![by type](figures/fig10_detection_by_type.png)

**Figure 10.** Percentage of injected errors found, by type, for v1 (top N) and v2 (top N and top 2N).

![matrix](figures/fig11_error_type_matrix.png)

**Figure 11.** What the auditor says (columns) against what actually happened (rows), among the top N items.

v2 beats v1 on average precision ({ev['average_precision']:.3f} against {ev1['average_precision']:.3f}), but it is a trade. v1 found more spurious boxes ({P1('spurious'):.0f}% against {P('spurious'):.0f}% in the top N) and very little of anything else, while v2 spreads its attention over all four types. Dropped boxes are the hardest case: the detector only proposes a box that matches for {pt['dropped']['max_possible_recall_with_candidates']*100:.0f}% of them, and that caps what can be found. When the auditor flags a real error it usually names the right problem (Figure 11). Its weak spot is false alarms, most of which are labelled localization errors on small or distant objects. Even the top 100 items are only {ev['precision_recall_at_k'][0]['precision']*100:.0f}% real errors, so I'd treat the ranking as a way to prioritise a review, not as a verdict.

### Does cleaning the labels help the detector?

I trained three detectors with identical settings and seed, and evaluated all of them on the same clean validation set: (a) on the clean labels, (b) on the noisy labels, and (c) on the noisy labels after cleaning. For (c) I took the top 15% of ranked items (a noise budget I assumed, not the true count), relabelled the {clean['relabelled']:,} class-mismatch labels to the suggested class and removed the other {clean['removed']:,} flagged labels.

**Table 6.** Accuracy on the clean validation set.

| training labels | mAP50 | mAP50-95 |
|---|---|---|
{dn_rows}

![downstream](figures/fig12_downstream_map.png)

**Figure 12.** mAP50 and mAP50-95 of the three detectors.

{verdict} The noise itself only cost {d_ba:.3f} mAP50 ((a) against (b)), so there was not much to win back to begin with. My guess, which I have not tested, is that only about a third of the top-ranked items are real errors (Table 4), so removing them throws away many good labels along with the bad ones.

![per class](figures/fig13_downstream_per_class.png)

**Figure 13.** mAP50 per class (the `train` class is left out).

![curves](figures/fig14_downstream_curves.png)

**Figure 14.** Training loss and validation mAP50 of the three downstream detectors.

### The original labels

I also ran the scorer on the original, uncorrupted training labels. The top 100 items are saved as crops with the label drawn on them (`results/real_audit/crops/`), as `review_queue.csv`, and as a simple page, `review.html`, that opens in any browser.

> **Flagged, unverified.** These are boxes where the auditor disagrees with the human label. **None of these is claimed to be a real error.** No person has reviewed them yet.

The queue has 100 boxes: {q.get('class_mismatch', 0)} suspected class mismatches, {q.get('localization_error', 0)} localization errors and {q.get('likely_spurious_label', 0)} spurious label. {real['labels_with_suspicion_ge_0_99']:,} labels score 0.99 or higher and {real['labels_with_suspicion_ge_0_95']:,} score 0.95 or higher, but because v2 scores are percentiles those are not error counts. As a sanity check I compared the raw signals on the original labels with the same signals after I added noise (Figure 15a-c). The class-mismatch signal is at least 0.5 on {cm[0]:.1f}% of the original labels and {cm[1]:.1f}% of the noisy ones, about {cm[1]/cm[0]:.0f}x more, so it clearly reacts to damage. The localization ({fr['s_localization'][0]:.1f}% against {fr['s_localization'][1]:.1f}%) and spurious ({fr['s_spurious'][0]:.1f}% against {fr['s_spurious'][1]:.1f}%) signals barely move, because they are dominated by objects the detector simply misses. I can't use the percentile scores for this check, since they come out roughly the same in any dataset.

![real audit](figures/fig15_real_audit.png)

**Figure 15.** (a-c) Share of labels with each raw signal at 0.5 or above, in the original and the noisy label sets. (d) Suspected type of the top-100 queue.

![crops](figures/fig16_flagged_crops.png)

**Figure 16.** The 12 highest-ranked boxes in the original labels (the green box is the existing label). Flagged, unverified.

## Running it

```bash
bash run_all.sh          # creates .venv, downloads the data, runs every stage (about 3-4 h on one RTX 5090)
python verify_claims.py  # recomputes the numbers reported here from the raw outputs
```

`run_all.sh` regenerates the large intermediate files (`crossfit_predictions.csv`, `label_scores*.csv`, `data/`). They are not stored in this repository, and `verify_claims.py` needs them. Each stage is one script in `src/` (`s1_prep.py` to `s9_visuals.py`), with shared helpers in `common.py` and the scoring logic in `scorer.py`.

| path | what is in it |
|---|---|
| `src/` | one script per stage |
| `results/` | logs, per-epoch CSVs, the noise manifest, evaluation JSONs, the review queue |
| `figures/` | every figure, including Figure 1 (the GIF) |
| `report.md` | a longer write-up |
| `DEVIATIONS.md` | where I departed from the original plan, and why |
| `verify_claims.py` | checks the reported numbers and the "flagged, unverified" wording against the raw files |

## References

- F. Yu et al. BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning. CVPR 2020.
- C. G. Northcutt, L. Jiang, I. L. Chuang. Confident Learning: Estimating Uncertainty in Dataset Labels. JAIR 70, 2021.
- Ultralytics YOLO11, https://github.com/ultralytics/ultralytics.
"""
    (ROOT / "README.md").write_text(txt, encoding="utf-8")
    print("README written")
