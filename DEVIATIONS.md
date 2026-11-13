# Deviations from the brief
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
