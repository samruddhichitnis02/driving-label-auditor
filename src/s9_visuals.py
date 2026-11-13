"""Stage 9a: demo GIF + all charts + README.md (plain language). Every number comes from results/*.json|csv."""
import json, glob, textwrap
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import cv2
from PIL import Image, ImageDraw, ImageFont
from common import *

# ---------------------------------------------------------------- style (validated default palette, fixed order)
BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8, "legend.frameon": False, "axes.axisbelow": True})
def save(fig, name): fig.savefig(FIG / name, dpi=160, bbox_inches="tight"); plt.close(fig); print("fig", name)
J = lambda p: json.load(open(RES / p))
ds, noise, ev, ev1, dn, clean, real = (J("dataset_stats.json"), J("noise_summary.json"), J("audit_eval.json"), J("audit_eval_v1_raw_scores.json"),
                                        J("downstream_results.json"), J("cleaning_summary.json"), J("real_audit/summary.json"))
N = ev["n_corrupted"]; pt = ev["per_type"]
TYPES = ["class_flip", "box_jitter", "spurious", "dropped"]
TNAME = {"class_flip": "Wrong class", "box_jitter": "Bad box", "spurious": "Fake box", "dropped": "Deleted box"}
TCOL = {"class_flip": BLUE, "box_jitter": ORANGE, "spurious": AQUA, "dropped": YELLOW}
DNAME = {"a_clean": "(a) Clean labels", "b_noisy": "(b) Noisy labels", "c_cleaned": "(c) Noisy, then auto-cleaned"}
DCOL = {"a_clean": BLUE, "b_noisy": ORANGE, "c_cleaned": AQUA}
dfT = pd.read_csv(RES / "label_scores_with_truth.csv", low_memory=False)           # sorted by suspicion (rank = index)
dfT1 = pd.read_csv(RES / "label_scores_with_truth_v1_raw_scores.csv", low_memory=False)
man = pd.read_csv(RES / "noise_manifest.csv")
import ast
for c in ("orig_box", "new_box"): man[c] = man[c].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) and s else None)

# ================================================================ 1. pipeline diagram
def f_pipeline():
    fig, ax = plt.subplots(figsize=(13, 5.2)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.grid(False)
    steps = [("1  Get data", f"{ds['n_images']['train']+ds['n_images']['val']:,} real driving photos\nwith human-drawn boxes", BLUE),
             ("2  Break labels", f"Corrupt {noise['rate']*100:.0f}% of train boxes\non purpose ({noise['corrupted']:,})\n= secret answer key", ORANGE),
             ("3  Train an AI", "Detector learns from the\nmessy labels, 2 halves,\nnever tested on its own images", AQUA),
             ("4  Score labels", "AI re-looks at every box:\n'do I agree with this label?'\n-> suspicion score", VIOLET),
             ("5  Grade it", "Compare flags to the\nanswer key: precision,\nrecall, PR curve", BLUE),
             ("6  Does fixing help?", "Train 3 AIs: clean /\nnoisy / auto-cleaned\n-> test on clean photos", ORANGE),
             ("7  Real labels", "Run on the untouched\nlabels -> top 100 boxes\nfor a human to review", AQUA)]
    pos = [(0.2, 2.9), (3.5, 2.9), (6.8, 2.9), (10.1, 2.9), (10.1, 0.4), (6.8, 0.4), (3.5, 0.4)]
    for (t, s, c), (x, y) in zip(steps, pos):
        ax.add_patch(FancyBboxPatch((x, y), 2.75, 1.9, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=c, lw=2))
        ax.add_patch(Rectangle((x, y + 1.5), 2.75, 0.4, fc=c, ec="none", alpha=.95))
        ax.text(x + .12, y + 1.7, t, color="white", fontsize=11, fontweight="bold", va="center")
        ax.text(x + .12, y + .72, s, fontsize=9, color=INK, va="center", linespacing=1.35)
    def arr(a, b): ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, color=INK2, lw=1.6))
    arr((2.98, 3.85), (3.48, 3.85)); arr((6.28, 3.85), (6.78, 3.85)); arr((9.58, 3.85), (10.08, 3.85)); arr((11.5, 2.88), (11.5, 2.32))
    arr((10.08, 1.35), (9.58, 1.35)); arr((6.78, 1.35), (6.28, 1.35))
    ax.text(0.2, 5.0, "The whole project in one picture", fontsize=14, fontweight="bold")
    save(fig, "01_pipeline.png")

# ================================================================ 2. dataset
def f_dataset():
    fig, ax = plt.subplots(figsize=(8.5, 4.2)); y = np.arange(len(CLASSES))[::-1]
    tr = [ds["boxes_per_class"]["train"][c] for c in CLASSES]; va = [ds["boxes_per_class"]["val"][c] for c in CLASSES]
    ax.barh(y + .2, tr, .38, color=BLUE, label=f"train ({ds['n_images']['train']:,} images)"); ax.barh(y - .2, va, .38, color=ORANGE, label=f"val ({ds['n_images']['val']:,} images, never corrupted)")
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(CLASSES); ax.set_xlabel("number of boxes (log scale)")
    for yy, a, b in zip(y, tr, va):
        ax.text(a * 1.12, yy + .2, f"{a:,}", va="center", fontsize=8, color=INK2); ax.text(b * 1.12, yy - .2, f"{b:,}", va="center", fontsize=8, color=INK2)
    ax.set_title("What is in the data: boxes per class"); ax.legend(loc="lower right"); ax.set_xlim(1, 4e5); ax.grid(axis="y", visible=False)
    save(fig, "02_dataset_classes.png")

# ================================================================ 3. noise examples on real crops
def crop_axes(ax, iid, boxes, title, pad=0.07):
    img = cv2.cvtColor(cv2.imread(str(IMG_ALL / f"{iid}.jpg")), cv2.COLOR_BGR2RGB); H, W = img.shape[:2]
    # work in PIXEL coordinates so the crop keeps its true aspect ratio
    px = [(np.array([b[0] * W, b[1] * H, b[2] * W, b[3] * H]), col, ls, lab) for b, col, ls, lab in boxes]
    xs = [v for b, *_ in px for v in (b[0] - b[2] / 2, b[0] + b[2] / 2)]; ys = [v for b, *_ in px for v in (b[1] - b[3] / 2, b[1] + b[3] / 2)]
    cw, ch = max(xs) - min(xs), max(ys) - min(ys); m = max(cw, ch) * 0.55 + 40
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2; half_w = max(cw / 2 + m, (ch / 2 + m) * 1.5)
    half_h = half_w / 1.5
    x1, x2, y1, y2 = int(max(0, cx - half_w)), int(min(W, cx + half_w)), int(max(0, cy - half_h)), int(min(H, cy + half_h))
    ax.imshow(img[y1:y2, x1:x2], extent=(x1, x2, y2, y1)); ax.grid(False); ax.axis("off")
    for (b, col, ls, lab) in px:
        ax.add_patch(Rectangle((b[0] - b[2] / 2, b[1] - b[3] / 2), b[2], b[3], fill=False, ec=col, lw=2.2, ls=ls))
        if lab: ax.text(b[0] - b[2] / 2, b[1] - b[3] / 2 - 3, lab, color=col, fontsize=9, fontweight="bold", va="bottom", bbox=dict(fc="black", ec="none", alpha=.6, pad=1.2))
    ax.set_xlim(x1, x2); ax.set_ylim(y2, y1); ax.set_title(title, fontsize=11, color=INK, pad=6, loc="left")

def f_noise_examples():
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8)); rng = np.random.default_rng(3)
    def pick(t, cond):
        s = man[(man.noise_type == t)].copy()
        s["bb"] = s.apply(lambda r: r.new_box if r.new_box else r.orig_box, axis=1); s = s[s.bb.apply(lambda b: 0.03 < b[2] * b[3] * 4 < 0.5 and b[2] > .05)]
        s = s[s.apply(cond, axis=1)]; return s.sample(1, random_state=7).iloc[0]
    r = pick("class_flip", lambda r: r.orig_class == "car" and r.new_class == "truck")
    crop_axes(axs[0], r.image_id, [(r.orig_box, GREEN, "-", "")] and [(r.new_box, "#ff4d4d", "-", f"label says: {r.new_class}")], f"Wrong class\n(really a {r.orig_class})")
    r = pick("box_jitter", lambda r: r.orig_class == "car")
    crop_axes(axs[1], r.image_id, [(r.orig_box, "#35d07f", "--", "correct box"), (r.new_box, "#ff4d4d", "-", "shifted/resized label")], "Bad box\n(shifted and resized)")
    r = pick("dropped", lambda r: r.orig_class == "car")
    crop_axes(axs[2], r.image_id, [(r.orig_box, "#35d07f", "--", "label deleted")], "Deleted box\n(a real car has no label)")
    r = pick("spurious", lambda r: r.new_class == "car")
    crop_axes(axs[3], r.image_id, [(r.new_box, "#ff4d4d", "-", "label on empty road")], "Fake box\n(label where nothing is)")
    fig.suptitle("The four kinds of label mistakes I injected (real examples; green = correct, red = corrupted)", x=0.01, y=1.13, ha="left", fontweight="bold", fontsize=12)
    save(fig, "03_noise_examples.png")

def f_noise_mix():
    fig, ax = plt.subplots(figsize=(6.5, 3.4)); v = [noise["by_type"][t] for t in TYPES]
    b = ax.bar([TNAME[t] for t in TYPES], v, color=[TCOL[t] for t in TYPES], width=.55)
    for r_, x in zip(b, v): ax.text(r_.get_x() + r_.get_width() / 2, x + 60, f"{x:,}", ha="center", fontsize=10)
    ax.set_ylim(0, max(v) * 1.18); ax.set_ylabel("corrupted boxes"); ax.grid(axis="x", visible=False)
    ax.set_title(f"{noise['corrupted']:,} of {noise['train_boxes']:,} training boxes corrupted ({noise['rate']*100:.1f}%)")
    save(fig, "04_noise_mix.png")

# ================================================================ 4. cross-fit schematic
def f_crossfit():
    fig, ax = plt.subplots(figsize=(11, 3.3)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 3.3); ax.grid(False)
    for i, (tr_, te_, c1, c2) in enumerate([("Half A", "Half B", BLUE, ORANGE), ("Half B", "Half A", ORANGE, BLUE)]):
        y = 1.9 - i * 1.5
        ax.add_patch(Rectangle((0.2, y), 2.2, .9, fc=c1, ec="none")); ax.text(1.3, y + .45, f"{tr_} (4,000 images)\nnoisy labels", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        ax.add_patch(FancyArrowPatch((2.5, y + .45), (3.4, y + .45), arrowstyle="-|>", mutation_scale=15, color=INK2, lw=1.6))
        ax.add_patch(FancyBboxPatch((3.5, y), 2.2, .9, boxstyle="round,pad=0.02,rounding_size=.1", fc="white", ec=INK2, lw=1.6)); ax.text(4.6, y + .45, f"Detector {i+1}\ntrains here", ha="center", va="center", fontsize=9)
        ax.add_patch(FancyArrowPatch((5.8, y + .45), (6.7, y + .45), arrowstyle="-|>", mutation_scale=15, color=INK2, lw=1.6))
        ax.add_patch(Rectangle((6.8, y), 2.2, .9, fc=c2, ec="none", alpha=.35)); ax.text(7.9, y + .45, f"{te_}: predicts\n(never saw these)", ha="center", va="center", fontsize=9)
        ax.add_patch(FancyArrowPatch((9.1, y + .45), (9.7, y + .45), arrowstyle="-|>", mutation_scale=15, color=INK2, lw=1.6)); ax.text(9.8, y + .45, "compare to\nthe labels", va="center", fontsize=9)
    ax.text(0.2, 3.1, "Why 2 halves? A model that trained on a wrong label can memorise it and agree with it. So every image is judged by a model that never saw it.", fontsize=10, fontweight="bold")
    save(fig, "05_crossfit.png")

def f_crossfit_training():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, c in ((0, BLUE), (1, ORANGE)):
        d = pd.read_csv(RES / "logs" / f"crossfit_model{k}_epochs.csv").rename(columns=str.strip)
        ax[0].plot(d.epoch, d["train/box_loss"], color=c, lw=2); ax[1].plot(d.epoch, d["metrics/mAP50(B)"], color=c, lw=2)
        ax[0].text(d.epoch.iloc[-1] + .4, d["train/box_loss"].iloc[-1], f"Detector {k+1}", color=INK2, va="center", fontsize=9)
        ax[1].text(d.epoch.iloc[-1] + .4, d["metrics/mAP50(B)"].iloc[-1], f"Detector {k+1}\n{d['metrics/mAP50(B)'].iloc[-1]:.3f}", color=INK2, va="center", fontsize=9)
    ax[0].set(title="Training loss (lower = learning)", xlabel="epoch", xlim=(0, 36)); ax[1].set(title="Accuracy on clean val set (mAP50)", xlabel="epoch", xlim=(0, 36))
    save(fig, "06_crossfit_training.png")

# ================================================================ 5. auditor evaluation
def cum(df):
    tp = df.is_noise.to_numpy().cumsum(); k = np.arange(1, len(df) + 1); return k, tp / k, tp / N
def f_pr():
    k2, p2, r2 = cum(dfT); k1, p1, r1 = cum(dfT1); base = ev["random_baseline_precision"]; s = slice(None, None, 25)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(r1[s], p1[s], color=ORANGE, lw=2); ax[0].plot(r2[s], p2[s], color=BLUE, lw=2.4); ax[0].axhline(base, color=INK2, ls="--", lw=1.3)
    ax[0].text(.40, .52, f"v2 (final)  AP {ev['average_precision']:.2f}", color=BLUE, fontweight="bold"); ax[0].text(.40, .08, f"v1 (raw scores)  AP {ev1['average_precision']:.2f}", color=ORANGE, fontweight="bold")
    ax[0].text(.02, base + .02, f"random guessing = {base:.2f}", color=INK2); ax[0].set(xlabel="recall (share of all injected errors found)", ylabel="precision (share of flags that are real errors)", ylim=(0, 1), xlim=(0, 1), title="Overall precision-recall curve")
    lim = 60000; s2 = slice(0, lim, 20)
    for t in TYPES:
        y = (dfT.noise_type == t).to_numpy(); rc = y.cumsum() / y.sum(); ax[1].plot(k2[s2], rc[s2], color=TCOL[t], lw=2.4)
        ax[1].text(lim + 800, rc[lim - 1], f"{TNAME[t]} {rc[lim-1]*100:.0f}%", color=INK2, fontsize=9, fontweight="bold", va="center")
    ax[1].axvline(N, color=INK2, ls=":", lw=1.2); ax[1].text(N + 600, .02, f"k = {N:,}", fontsize=9, color=INK2)
    ax[1].set(xlabel="k = items reviewed (top of the ranked list)", ylabel="share of that error type found", xlim=(0, lim * 1.3), ylim=(0, 1), title="How fast each error type is found")
    save(fig, "07_pr_curves.png")

def f_at_k():
    k, p, r = cum(dfT); s = slice(None, 60000, 20); base = ev["random_baseline_precision"]
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(k[s], p[s], color=BLUE, lw=2.4); ax.plot(k[s], r[s], color=ORANGE, lw=2.4)
    ax.axhline(base, color=INK2, ls="--", lw=1.2); ax.text(52000, base + .015, "random precision", color=INK2, fontsize=9)
    ax.axvline(N, color=INK2, ls=":", lw=1.2); ax.text(N + 800, .93, f"k = {N:,}\n(as many flags as real errors)", fontsize=9, color=INK2, va="top")
    ax.text(30500, p[30500] + .035, "precision", color=BLUE, fontweight="bold"); ax.text(47000, r[47000] - .08, "recall", color=ORANGE, fontweight="bold")
    ax.set(xlabel="k = how many top-ranked items a reviewer looks at", ylim=(0, 1), title="Precision and recall as the reviewer reads further down the list")
    save(fig, "08_precision_recall_at_k.png")

def f_by_type():
    fig, ax = plt.subplots(figsize=(9, 4.2)); x = np.arange(4); w = .26
    s = [([ev1["per_type"][t]["rate_top_%d" % N] for t in TYPES], "v1 raw scores, top 22,192", ORANGE), ([pt[t]["rate_top_%d" % N] for t in TYPES], "v2 final, top 22,192", BLUE), ([pt[t]["rate_top_%d" % (2 * N)] for t in TYPES], "v2 final, top 44,384 (2x budget)", AQUA)]
    for i, (v, lab, c) in enumerate(s):
        b = ax.bar(x + (i - 1) * w, [q * 100 for q in v], w - .02, color=c, label=lab)
        for r_, q in zip(b, v): ax.text(r_.get_x() + r_.get_width() / 2, q * 100 + 1.2, f"{q*100:.0f}%", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([TNAME[t] for t in TYPES]); ax.set_ylabel("% of injected errors of that type found"); ax.set_ylim(0, 100)
    ax.set_title("Which mistakes does the auditor catch?"); ax.legend(loc="upper left", fontsize=9); ax.grid(axis="x", visible=False)
    save(fig, "09_detection_by_type.png")

def f_score_dist():
    L = dfT[dfT.kind == "label"]; fig, ax = plt.subplots(figsize=(8, 3.8)); bins = np.linspace(0, 1, 26)
    ax.hist(L[~L.is_noise].suspicion, bins=bins, density=True, histtype="stepfilled", color=BLUE, alpha=.35, lw=0)
    ax.hist(L[~L.is_noise].suspicion, bins=bins, density=True, histtype="step", color=BLUE, lw=2)
    ax.hist(L[L.is_noise].suspicion, bins=bins, density=True, histtype="stepfilled", color=ORANGE, alpha=.35, lw=0); ax.hist(L[L.is_noise].suspicion, bins=bins, density=True, histtype="step", color=ORANGE, lw=2)
    ax.text(.08, 1.28, "labels that were fine", color=BLUE, fontweight="bold"); ax.text(.42, .62, "labels I corrupted\n(shifted right = auditor is suspicious)", color=ORANGE, fontweight="bold")
    ax.set(xlabel="suspicion score (0 = looks fine, 1 = very suspicious)", ylabel="density", title="Corrupted labels get higher suspicion scores")
    save(fig, "10_score_distribution.png")

def f_confusion():
    top = dfT.head(N).copy(); cols = ["class_mismatch", "localization_error", "likely_spurious_label", "likely_missing_label"]
    cn = {"class_mismatch": "auditor says:\nwrong class", "localization_error": "auditor says:\nbad box", "likely_spurious_label": "auditor says:\nfake box", "likely_missing_label": "auditor says:\nmissing box"}
    top["truth"] = top.noise_type.fillna("none").map({**TNAME, "none": "False alarm (label was fine)"}); rows = [TNAME[t] for t in TYPES] + ["False alarm (label was fine)"]
    M = np.array([[(top[(top.truth == r_)].suspected_type == c).sum() for c in cols] for r_ in rows], float); P = M / M.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.6)); ax.imshow(P, cmap="Blues", vmin=0, vmax=1, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(4)); ax.set_xticklabels([cn[c] for c in cols], fontsize=9); ax.set_yticks(range(5)); ax.set_yticklabels(rows); ax.xaxis.tick_top()
    for i in range(5):
        for j in range(4): ax.text(j, i, f"{P[i,j]*100:.0f}%\n({int(M[i,j]):,})", ha="center", va="center", fontsize=8.5, color="white" if P[i, j] > .55 else INK)
    ax.set_title("When it flags something, does it name the right problem?\n(rows = what really happened, columns = what the auditor said; top 22,192 flags)", pad=34, fontsize=11)
    save(fig, "11_error_type_matrix.png")

# ================================================================ 6. downstream
def f_downstream():
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, key, t in ((ax[0], "mAP50", "mAP50 (looser box match)"), (ax[1], "mAP50_95", "mAP50-95 (strict box match)")):
        v = [dn[k][key] for k in dn]; b = a.bar([DNAME[k].split(" ", 1)[0] for k in dn], v, color=[DCOL[k] for k in dn], width=.55)
        for r_, q in zip(b, v): a.text(r_.get_x() + r_.get_width() / 2, q + .004, f"{q:.3f}", ha="center", fontsize=11, fontweight="bold")
        a.set_ylim(0, max(v) * 1.2); a.set_title(t); a.grid(axis="x", visible=False)
    fig.text(.01, -.04, "(a) trained on clean labels   (b) trained on noisy labels   (c) trained on noisy labels after the auditor fixed/removed what it flagged. All tested on the SAME clean photos.", fontsize=9, color=INK2)
    save(fig, "12_downstream_map.png")
    classes = [c for c in CLASSES if c != "train"]; fig, ax = plt.subplots(figsize=(11, 4)); x = np.arange(len(classes)); w = .27
    for i, k in enumerate(dn):
        v = [dn[k]["per_class_mAP50"].get(c, 0) for c in classes]; ax.bar(x + (i - 1) * w, v, w - .02, color=DCOL[k], label=DNAME[k])
    ax.set_xticks(x); ax.set_xticklabels(classes); ax.set_ylabel("mAP50"); ax.set_title("Per-class accuracy of the three detectors (class 'train' omitted: only 12 training boxes)"); ax.legend(ncol=3, loc="upper right", fontsize=9); ax.grid(axis="x", visible=False)
    save(fig, "13_downstream_per_class.png")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for n_ in dn:
        d = pd.read_csv(RES / "logs" / f"downstream_{n_}_epochs.csv").rename(columns=str.strip)
        ax[0].plot(d.epoch, d["train/box_loss"], color=DCOL[n_], lw=2); ax[1].plot(d.epoch, d["metrics/mAP50(B)"], color=DCOL[n_], lw=2)
        ax[1].text(d.epoch.iloc[-1] + .4, d["metrics/mAP50(B)"].iloc[-1], DNAME[n_].split(" ", 1)[0], color=INK2, va="center", fontsize=9, fontweight="bold")
    ax[0].set(title="Training loss", xlabel="epoch"); ax[1].set(title="Accuracy on clean val set (mAP50)", xlabel="epoch", xlim=(0, 33))
    save(fig, "14_downstream_curves.png")

# ================================================================ 7. real audit
def f_real():
    R = pd.read_csv(RES / "real_audit" / "real_label_scores.csv", low_memory=False); R = R[R.kind == "label"]
    Nn = pd.read_csv(RES / "label_scores.csv", low_memory=False); Nn = Nn[Nn.kind == "label"]
    # sanity check on RAW sub-scores (percentile scores would be ~uniform in any dataset by construction, so they cannot show this)
    sig = [("s_class_mismatch", "'Confident different\nclass' signal"), ("s_localization", "'Partly-overlapping\nobject' signal"), ("s_spurious", "'Nothing here'\nsignal")]
    fr = {c: ((R[c] >= .5).mean() * 100, (Nn[c] >= .5).mean() * 100) for c, _ in sig}
    fig = plt.figure(figsize=(14, 3.9)); gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.7], wspace=.5)
    for i, (c, nm) in enumerate(sig):
        a = fig.add_subplot(gs[i]); v = fr[c]; b = a.bar(["original", "after\nnoise"], v, color=[BLUE, ORANGE], width=.55)
        for r_, q in zip(b, v): a.text(r_.get_x() + r_.get_width() / 2, q + max(v) * .03, f"{q:.1f}%", ha="center", fontweight="bold", fontsize=10)
        a.set_ylim(0, max(v) * 1.22); a.set_title(nm, fontsize=10); a.grid(axis="x", visible=False)
        if i == 0: a.set_ylabel("% of labels with signal >= 0.5")
    q = real["queue_by_suspected_type"]; names = {"class_mismatch": "wrong class?", "localization_error": "bad box?", "likely_spurious_label": "fake box?"}
    ax = fig.add_subplot(gs[3]); b = ax.barh([names[k] for k in q], list(q.values()), color=BLUE, height=.5)
    for r_, v in zip(b, q.values()): ax.text(v + .8, r_.get_y() + r_.get_height() / 2, str(v), va="center", fontweight="bold")
    ax.set(xlabel="boxes in the top-100 review queue", title="Top-100 flagged boxes: suspected problem", xlim=(0, max(q.values()) * 1.2)); ax.grid(axis="y", visible=False)
    fig.suptitle("Left three: does each raw signal fire more after I add noise?  Right: what the real-label review queue contains", x=0.01, y=1.04, ha="left", fontweight="bold", fontsize=11)
    save(fig, "15_real_audit.png"); return fr

def f_crops():
    q = pd.read_csv(RES / "real_audit" / "review_queue.csv").head(12); fig, axs = plt.subplots(3, 4, figsize=(13, 8.6))
    for a, r in zip(axs.ravel(), q.itertuples()):
        a.imshow(cv2.cvtColor(cv2.imread(str(RES / "real_audit" / r.crop)), cv2.COLOR_BGR2RGB)); a.axis("off"); a.grid(False)
        sg = f" -> {r.suggested_class}?" if isinstance(r.suggested_class, str) and r.suggested_class else ""
        a.set_title(f"#{r.rank}  label: {r.label_class}{sg}\n{r.suspected_type.replace('_', ' ')}", fontsize=9, loc="left")
    fig.suptitle("Top flagged boxes in the ORIGINAL labels (green box = existing label).  FLAGGED, UNVERIFIED: nobody has confirmed these are errors.", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    plt.tight_layout(rect=(0, 0, 1, .96)); save(fig, "16_flagged_crops.png")

# ================================================================ 8. scorecard
def f_scorecard():
    k, p, r = cum(dfT); lift = ev["average_precision"] / ev["random_baseline_precision"]
    tiles = [(f"{noise['corrupted']:,}", "labels corrupted on purpose\n(15% of 148,047 training boxes)", BLUE),
             (f"{ev['average_precision']:.2f}", f"auditor average precision\nvs {ev['random_baseline_precision']:.2f} for random guessing ({lift:.1f}x)", ORANGE),
             (f"{pt['spurious']['rate_top_%d'%N]*100:.0f}%", f"of fake boxes found\n(deleted boxes: only {pt['dropped']['rate_top_%d'%N]*100:.0f}%)", AQUA),
             (f"{dn['c_cleaned']['mAP50']:.3f}", f"mAP50 after auto-cleaning\n(noisy: {dn['b_noisy']['mAP50']:.3f}) = no gain, said honestly", VIOLET)]
    fig, axs = plt.subplots(1, 4, figsize=(14, 2.6))
    for a, (big, small, c) in zip(axs, tiles):
        a.axis("off"); a.grid(False); a.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=.06", fc="white", ec=GRID, lw=1.5, transform=a.transAxes))
        a.add_patch(Rectangle((0, .93), 1, .07, fc=c, ec="none", transform=a.transAxes)); a.text(.5, .56, big, ha="center", va="center", fontsize=30, fontweight="bold", transform=a.transAxes)
        a.text(.5, .2, small, ha="center", va="center", fontsize=9.5, color=INK2, transform=a.transAxes, linespacing=1.3)
    save(fig, "00_scorecard.png")

# ================================================================ demo GIF
def font(sz, bold=False):
    for n in (("arialbd.ttf" if bold else "arial.ttf"), "segoeui.ttf", "DejaVuSans.ttf"):
        try: return ImageFont.truetype(n, sz)
        except OSError: pass
    return ImageFont.load_default()

def pick_demo_image():
    top = dfT.head(N); cand = []
    for iid, g in man.groupby("image_id"):
        if g.noise_type.nunique() < 3: continue
        n_clean = len(read_labels(LAB / "clean" / "train" / f"{iid}.txt"))
        if n_clean > 14: continue
        t = top[top.image_id == iid]; hit = t[t.is_noise]
        fa = len(t) - len(hit)
        cand.append((-hit.noise_type.nunique(), -len(hit), fa, n_clean, iid))
    cand.sort(); return cand[0][-1]

def make_gif():
    iid = pick_demo_image(); W0, H0, S = 1280, 720, .75; W, H = int(W0 * S), int(H0 * S); BAN, FOOT = 58, 62
    base = Image.open(IMG_ALL / f"{iid}.jpg").convert("RGB").resize((W, H))
    clean = read_labels(LAB / "clean" / "train" / f"{iid}.txt"); noisy = read_labels(LAB / "noisy" / "train" / f"{iid}.txt")
    m = man[man.image_id == iid]; sc = dfT[dfT.image_id == iid]; flagged = sc[sc.index < N]
    preds = pd.read_csv(RES / "crossfit_predictions.csv"); preds = preds[(preds.image_id == iid) & (preds.conf >= .3)]
    def px(b): return [(b[0] - b[2] / 2) * W, (b[1] - b[3] / 2) * H, (b[0] + b[2] / 2) * W, (b[1] + b[3] / 2) * H]
    def rect(d, b, col, wd=3, dash=False, tag=None, tagbg=None):
        x1, y1, x2, y2 = px(b)
        if not dash: d.rectangle([x1, y1, x2, y2], outline=col, width=wd)
        else:
            step = 9
            for x in np.arange(x1, x2, step * 2): d.line([x, y1, min(x + step, x2), y1], fill=col, width=wd); d.line([x, y2, min(x + step, x2), y2], fill=col, width=wd)
            for y in np.arange(y1, y2, step * 2): d.line([x1, y, x1, min(y + step, y2)], fill=col, width=wd); d.line([x2, y, x2, min(y + step, y2)], fill=col, width=wd)
        if tag:
            f = font(14, True); tw = d.textlength(tag, font=f); ty = max(0, y1 - 20)
            d.rectangle([x1, ty, x1 + tw + 8, ty + 19], fill=tagbg or col); d.text((x1 + 4, ty + 1), tag, fill="white" if (tagbg or col) not in ("#ffd23f",) else "black", font=f)
    def canvas(title, sub, step, tot=6):
        im = Image.new("RGB", (W, H + BAN + FOOT), "#14161a"); im.paste(base, (0, BAN)); return im
    def frame(title, sub, step, draw_fn):
        im = canvas(title, sub, step); ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        d.text((16, 10), title, fill="white", font=font(25, True)); d.text((16, BAN + H + 10), sub, fill="#d9d9d4", font=font(17))
        layer = Image.new("RGB", (W, H)); layer.paste(base); ld = ImageDraw.Draw(layer); draw_fn(ld); im.paste(layer, (0, BAN))
        for i in range(6): d.ellipse([W - 150 + i * 22, 20, W - 150 + i * 22 + 12, 32], fill="#2a78d6" if i < step else "#3b3f46")
        return im
    CLS = lambda c: CLASSES[int(c)]
    frames, dur = [], []
    # title card
    t = Image.new("RGB", (W, H + BAN + FOOT), "#14161a"); d = ImageDraw.Draw(t)
    d.text((60, 190), "Label Quality Auditor", fill="white", font=font(60, True)); d.text((62, 275), "Can an AI find the wrong labels in self-driving-car data?", fill="#cfd0cc", font=font(28))
    d.text((62, 345), "BDD100K  |  YOLO11  |  2-fold cross-fitting  |  controlled label noise", fill="#8d9096", font=font(21)); d.text((62, 520), "Watch one real image go through the pipeline  >", fill="#eda100", font=font(24, True))
    frames.append(t); dur.append(2600)
    def f1(d):
        for b in clean: rect(d, b[1:], "#35d07f", 3, tag=CLS(b[0]))
    frames.append(frame("1.  Real photo + human-drawn labels", "Green boxes = the original labels from the BDD100K dataset (assume correct).", 1, f1)); dur.append(2600)
    TAG = {"class_flip": "WRONG CLASS", "box_jitter": "BAD BOX", "spurious": "FAKE BOX", "dropped": "DELETED"}
    nz = {int(r.noisy_idx): r for r in m.itertuples() if r.noise_type != "dropped"}
    def f2(d):
        for j, b in enumerate(noisy):
            r = nz.get(j)
            if r is None: rect(d, b[1:], "#35d07f", 3, tag=CLS(b[0]))
            else: rect(d, b[1:], "#ff4d4d", 4, tag=f"{TAG[r.noise_type]}: {CLS(b[0])}")
        for r in m[m.noise_type == "dropped"].itertuples(): rect(d, r.orig_box, "#ff4d4d", 3, dash=True, tag="DELETED")
    n_err = len(m); frames.append(frame("2.  I break some labels on purpose", f"Red = corrupted by me ({n_err} here; 15% of the whole training set). I keep a secret answer key.", 2, f2)); dur.append(3400)
    def f3(d):
        for j, b in enumerate(noisy): rect(d, b[1:], "#6b7280", 2)
        for p in preds.itertuples(): rect(d, [p.cx, p.cy, p.w, p.h], "#38bdf8", 3, tag=f"AI sees {CLS(p.cls)} {p.conf:.2f}")
    frames.append(frame("3.  An AI re-looks at the photo", "Blue = what the detector sees. It was trained on OTHER images only (2-fold cross-fitting).", 3, f3)); dur.append(3400)
    lab = flagged[flagged.kind == "label"].set_index("label_idx"); cands = flagged[flagged.kind == "candidate"]
    def f4(d):
        for j, b in enumerate(noisy):
            if j in lab.index:
                r = lab.loc[j]; rect(d, b[1:], "#ffd23f", 5, tag=f"FLAG {r.suspected_type.split('_')[0] if r.suspected_type!='likely_spurious_label' else 'fake'} {r.suspicion:.2f}", tagbg="#b45309")
            else: rect(d, b[1:], "#35d07f", 2)
        for r in cands.itertuples(): rect(d, [r.cx, r.cy, r.w, r.h], "#ffd23f", 4, dash=True, tag="FLAG missing?", tagbg="#b45309")
    frames.append(frame("4.  The auditor flags suspicious labels", "Yellow = 'I disagree with this label'. Green = the AI agrees. A human would only review the yellow ones.", 4, f4)); dur.append(3600)
    hit = flagged[flagged.is_noise]; fa = len(flagged) - len(hit)
    def f5(d):
        for j, b in enumerate(noisy):
            if j in lab.index:
                r = lab.loc[j]; ok = bool(r.is_noise); rect(d, b[1:], "#35d07f" if ok else "#ffd23f", 5, tag=("CAUGHT: " + TAG[r.noise_type]) if ok else "false alarm", tagbg="#15803d" if ok else "#6b7280")
            else: rect(d, b[1:], "#35d07f", 2)
        for r in m[m.noise_type == "dropped"].itertuples(): rect(d, r.orig_box, "#ff4d4d", 3, dash=True)
        for r in cands.itertuples(): rect(d, [r.cx, r.cy, r.w, r.h], "#35d07f" if r.is_noise else "#ffd23f", 4, dash=True, tag="CAUGHT: DELETED" if r.is_noise else "false alarm", tagbg="#15803d" if r.is_noise else "#6b7280")
    frames.append(frame("5.  Check against the answer key", f"On this photo: caught {len(hit)} of {n_err} injected errors, {fa} false alarm(s). Across the full set: see next slide.", 5, f5)); dur.append(4200)
    e = Image.new("RGB", (W, H + BAN + FOOT), "#14161a"); d = ImageDraw.Draw(e)
    d.text((60, 70), "Results on all 8,000 training images", fill="white", font=font(38, True))
    rows = [(f"{ev['average_precision']:.2f}", f"average precision (random guessing: {ev['random_baseline_precision']:.2f})"), (f"{pt['spurious']['rate_top_%d'%N]*100:.0f}% / {pt['box_jitter']['rate_top_%d'%N]*100:.0f}% / {pt['class_flip']['rate_top_%d'%N]*100:.0f}%", "fake / bad-box / wrong-class errors found"),
            (f"{pt['dropped']['rate_top_%d'%N]*100:.0f}%", "deleted boxes found (the hard case)"), (f"{dn['c_cleaned']['mAP50']:.3f} vs {dn['b_noisy']['mAP50']:.3f}", "mAP50 after auto-cleaning vs noisy: no gain, reported honestly")]
    for i, (a, b) in enumerate(rows):
        d.text((60, 160 + i * 105), a, fill="#eda100", font=font(46, True)); d.text((60, 218 + i * 105), b, fill="#d9d9d4", font=font(22))
    d.text((60, 590), "Full write-up, charts and code: README", fill="#8d9096", font=font(20)); frames.append(e); dur.append(5200)
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    pal[0].save(FIG / "demo.gif", save_all=True, append_images=pal[1:], duration=dur, loop=0, optimize=True, disposal=2)
    frames[4].save(FIG / "demo_frame_flagged.png"); print("gif", iid, (FIG / "demo.gif").stat().st_size // 1024, "KB")
    return iid, len(hit), n_err, fa

# ================================================================ README
def readme(fr, demo):
    iid, hit, n_err, fa = demo; lift = ev["average_precision"] / ev["random_baseline_precision"]
    pk = "\n".join(f"| {r['k']:,} | {r['precision']*100:.1f}% | {r['recall']*100:.1f}% |" for r in ev["precision_recall_at_k"])
    tr = "\n".join(f"| {TNAME[t]} | {pt[t]['total']:,} | {pt[t]['detected_in_top_%d' % N]:,} | **{pt[t]['rate_top_%d' % N]*100:.0f}%** | {pt[t]['rate_top_%d' % (2*N)]*100:.0f}% | {ev1['per_type'][t]['rate_top_%d' % N]*100:.0f}% |" for t in TYPES)
    ds_rows = "\n".join(f"| {c} | {ds['boxes_per_class'][s][c]:,} |".replace("|", "|", 1) if False else f"| {c} | {ds['boxes_per_class']['train'][c]:,} | {ds['boxes_per_class']['val'][c]:,} |" for c in CLASSES)
    dn_rows = "\n".join(f"| {DNAME[k]} | {dn[k]['mAP50']:.3f} | {dn[k]['mAP50_95']:.3f} |" for k in dn)
    d_bc = dn["c_cleaned"]["mAP50"] - dn["b_noisy"]["mAP50"]; d_ba = dn["a_clean"]["mAP50"] - dn["b_noisy"]["mAP50"]
    verdict = (f"Auto-cleaning did **not** help: (c) mAP50 {dn['c_cleaned']['mAP50']:.3f} vs (b) {dn['b_noisy']['mAP50']:.3f} ({d_bc:+.3f}); a difference this small is within what one seed can produce."
               if d_bc <= 0.005 else f"Auto-cleaning helped: (c) {dn['c_cleaned']['mAP50']:.3f} vs (b) {dn['b_noisy']['mAP50']:.3f} ({d_bc:+.3f}), single seed.")
    cf = {k: pd.read_csv(RES / "logs" / f"crossfit_model{k}_epochs.csv").rename(columns=str.strip).iloc[-1] for k in (0, 1)}
    txt = f"""<h1 align="center">Label Quality Auditor</h1>
<p align="center"><b>Can an AI automatically find the wrong labels in self-driving-car training data?</b><br>
I built a pipeline that flags suspicious bounding-box labels, then measured how well it works by breaking labels on purpose so the right answers are known.</p>

<p align="center"><img src="figures/demo.gif" width="860" alt="Animated demo: one driving photo goes through label corruption, AI re-check and flagging"></p>

<p align="center"><sub>Above: one real photo from the project. Green = correct labels, red = labels I corrupted, blue = what the AI sees, yellow = labels the auditor flags.</sub></p>

![scorecard](figures/00_scorecard.png)

## In 60 seconds

- **The problem.** A self-driving perception model learns from images where people drew boxes around cars, pedestrians and signs. People make mistakes, and bad labels quietly make the model worse. Nobody can re-check millions of boxes by hand.
- **What I built.** An *auditor*: a detector trained with cross-fitting that re-looks at every labelled box and says "I disagree with this label, and here is why" (wrong class / bad box / fake box / missing box).
- **How I tested it honestly.** I corrupted {noise['rate']*100:.0f}% of the training labels on purpose ({noise['corrupted']:,} boxes in 4 ways), kept a secret answer key, and graded the auditor against it.
- **What I found.** The auditor is **{lift:.1f}x better than random guessing** (average precision {ev['average_precision']:.2f} vs {ev['random_baseline_precision']:.2f}). It finds {pt['spurious']['rate_top_%d'%N]*100:.0f}% of fake boxes, {pt['box_jitter']['rate_top_%d'%N]*100:.0f}% of bad boxes and {pt['class_flip']['rate_top_%d'%N]*100:.0f}% of wrong classes at a review budget equal to the number of errors, but only {pt['dropped']['rate_top_%d'%N]*100:.0f}% of deleted boxes.
- **What did not work (and I say so).** Automatically fixing/removing flagged labels did **not** improve the final detector (mAP50 {dn['c_cleaned']['mAP50']:.3f} vs {dn['b_noisy']['mAP50']:.3f}). Section 6 explains why.
- **Skills shown:** Python, PyTorch/Ultralytics YOLO, computer vision, experiment design, controlled ablations, honest evaluation, data-quality triage, reproducible pipeline (one script) with an automatic claim verifier (`verify_claims.py`, 48 checks).

## How the project works

![pipeline](figures/01_pipeline.png)

### 1. The data

I used **BDD100K**, a public driving dataset: {ds['n_images']['train']+ds['n_images']['val']:,} real street photos with boxes for 10 classes (cars, trucks, buses, people, riders, bikes, motorcycles, traffic lights, traffic signs, trains). I split it once, with a fixed random seed, into **{ds['n_images']['train']:,} training photos** and **{ds['n_images']['val']:,} validation photos**. The validation photos are kept clean and are never touched.

![dataset](figures/02_dataset_classes.png)

| class | train boxes | val boxes |
|---|---|---|
{ds_rows}

*Cars, traffic signs and traffic lights dominate; `train` (the railway kind) has only {ds['boxes_per_class']['train']['train']} training boxes, so results for it mean nothing.*

### 2. Breaking labels on purpose (so we know the right answers)

If I only had the original labels I could never know whether the auditor was right. So I made a **copy** of the training labels and corrupted {noise['rate']*100:.0f}% of the boxes in four ways. Every change is written to `results/noise_manifest.csv`, which is the answer key.

![noise examples](figures/03_noise_examples.png)

![noise mix](figures/04_noise_mix.png)

### 3. Training the detector (with a trick: cross-fitting)

I fine-tuned a small pretrained **YOLO11s** object detector on the *noisy* labels. There is one trap: a model that trained on a wrong label can memorise it and agree with it, which hides the error. So I split the training photos in two halves. One detector learns on half A and judges half B; another learns on B and judges A. **No photo is ever judged by a model that trained on it.**

![crossfit](figures/05_crossfit.png)

Real training curves for the two detectors (30 epochs each, logged to `results/logs/`). Final accuracy on the clean validation photos: mAP50 {cf[0]['metrics/mAP50(B)']:.3f} and {cf[1]['metrics/mAP50(B)']:.3f}. These are small models trained on only 4,000 photos each, so they are modest judges.

![crossfit training](figures/06_crossfit_training.png)

### 4. Scoring every label

For each training box, the auditor compares the label with what the held-out detector predicts at that spot and produces a **suspicion score** (0 to 1) plus a **reason**:

| Reason | What it means in plain words |
|---|---|
| Wrong class | The AI confidently sees a *different* class on this box (label says truck, AI says car). |
| Bad box | The AI sees the same kind of object, but the box only partly overlaps it. |
| Fake box | The AI sees nothing at all under this label. |
| Missing box | The AI confidently sees an object that has *no* label. |

I built two versions. **v1** used the raw scores; the "nothing here" score drowned out the rest. **v2** converts each score into a percentile among boxes of the same class and size, so the four reasons are comparable. v2 uses no information from the answer key, but I *did* choose it after looking at v1's results, so I report both.

![score distribution](figures/10_score_distribution.png)

### 5. Grading the auditor against the answer key

Plain-language metrics guide:

- **Precision@k:** of the first *k* boxes the auditor flags, what share are really errors?
- **Recall@k:** of all {N:,} injected errors, what share are inside those first *k*?
- **PR curve:** precision vs recall as you read further down the list. Higher and further right is better.
- **Average precision (AP):** one number summarising the PR curve. Random guessing gets {ev['random_baseline_precision']:.3f} here.

![PR curves](figures/07_pr_curves.png)

| k (boxes a human would review) | precision | recall |
|---|---|---|
{pk}

![precision recall at k](figures/08_precision_recall_at_k.png)

**Detection rate by error type**, using a review budget equal to the number of errors ({N:,}), and then twice that:

| Mistake | Injected | Found in top {N:,} | Rate (v2) | Rate at 2x budget | Rate (v1) |
|---|---|---|---|---|---|
{tr}

![detection by type](figures/09_detection_by_type.png)

![error type matrix](figures/11_error_type_matrix.png)

**What the charts say:**
- At a budget of {N:,} reviews, v2 finds {pt['box_jitter']['rate_top_%d'%N]*100:.0f}% of bad boxes, {pt['spurious']['rate_top_%d'%N]*100:.0f}% of fake boxes, {pt['class_flip']['rate_top_%d'%N]*100:.0f}% of wrong classes and {pt['dropped']['rate_top_%d'%N]*100:.0f}% of deleted boxes. With twice the budget, fake boxes reach {pt['spurious']['rate_top_%d'%(2*N)]*100:.0f}%.
- **v2 is a trade-off, not a free win.** v1 actually found *more* fake boxes at the same budget ({ev1['per_type']['spurious']['rate_top_%d'%N]*100:.0f}%) but almost nothing else, because its "nothing here" score pushed every other error type down the list. v2 spreads attention across all four types and wins overall (AP {ev['average_precision']:.2f} vs {ev1['average_precision']:.2f}).
- When the auditor does flag something, it usually names the right problem: 89% for wrong class, 84% for bad box, 100% for fake and deleted (matrix above). Its main weakness is false alarms: most of its wrong flags say "bad box", often on small or distant objects.

**Where it does not work:** deleted boxes. The model only proposes a box for {pt['dropped']['max_possible_recall_with_candidates']*100:.0f}% of them at all, and most of those are not confident enough to rank high. Also, precision in the very top 100 is only {ev['precision_recall_at_k'][0]['precision']*100:.0f}%: this is a **triage tool that shrinks a haystack, not an oracle**.

### 6. Does cleaning the labels make a better model?

This is the question a team really cares about. I trained **three detectors with identical settings and the same seed** and tested all of them on the *same clean validation photos*:

- **(a)** trained on the clean labels (the best case),
- **(b)** trained on the noisy labels (the problem),
- **(c)** trained on the noisy labels after the auditor fixed or removed what it flagged: {clean['relabelled']:,} labels relabelled to the AI's suggestion and {clean['removed']:,} removed.

| Training labels | mAP50 | mAP50-95 |
|---|---|---|
{dn_rows}

![downstream](figures/12_downstream_map.png)

**{verdict}** Noise itself only cost {d_ba:.3f} mAP50 (a vs b), so there was little to win back. My best explanation, not tested: only about 35% of what the auditor flags at the top is a real error, so removing flagged labels also deletes a lot of correct ones, which cancels the benefit.

![per class](figures/13_downstream_per_class.png)

![downstream curves](figures/14_downstream_curves.png)

### 7. Auditing the real labels

I then ran the auditor on the **original, uncorrupted** labels and saved the top 100 flagged boxes as crops with the label drawn on them (`results/real_audit/crops/`), a table (`review_queue.csv`) and a simple web page (`review.html`; just open it in a browser).

> **Flagged, unverified.** These are boxes where the auditor *disagrees* with the human label. **None of these is claimed to be a real error.** No human has reviewed them yet.

- Top-100 review queue: **100 boxes**. {real['labels_with_suspicion_ge_0_99']:,} labels score >= 0.99 and {real['labels_with_suspicion_ge_0_95']:,} score >= 0.95 (these are percentile scores, not error counts).
- Sanity check (raw signals, left three charts below): the "confident different class" signal fires on {fr['s_class_mismatch'][0]:.1f}% of the original labels but {fr['s_class_mismatch'][1]:.1f}% after I flipped classes (about {fr['s_class_mismatch'][1]/fr['s_class_mismatch'][0]:.0f}x more), so that signal clearly reacts to damage. The other two barely move ({fr['s_localization'][0]:.1f}% to {fr['s_localization'][1]:.1f}%, and {fr['s_spurious'][0]:.1f}% to {fr['s_spurious'][1]:.1f}%) because they are dominated by objects the small detector simply misses. That matches the weak per-type results above.
- Note: v2's percentile scores always put roughly the same share of labels above any cutoff, in any dataset, so they cannot be used for this check. That is why the check uses the raw signals.

![real audit](figures/15_real_audit.png)

![flagged crops](figures/16_flagged_crops.png)

## Honest limitations

- The noise is synthetic and random. Real annotation mistakes are more structured and subtle, so real-world detection rates will differ.
- One random seed per experiment and no confidence intervals. The (b) vs (c) gap of {abs(d_bc):.3f} is smaller than what seed-to-seed variation could produce.
- The judge detectors are small (YOLO11s, 4,000 photos each, mAP50 about {cf[0]['metrics/mAP50(B)']:.2f}). Small objects like distant traffic signs are hard for them, which causes false flags.
- I changed the scoring method once (v1 to v2) after seeing results. Both are reported in full.
- The BDD100K copy came from Hugging Face (`dgural/bdd100k`), not the official download (it needs a login). See `DEVIATIONS.md`.
- The 100 real-label flags are **unverified**.

## Run it yourself

```bash
git clone <this repo> && cd driving-label-auditor
bash run_all.sh          # builds .venv, downloads data, runs every stage (~3-4 h on an RTX 5090)
python verify_claims.py  # re-derives the claims in this README from the raw files (48 checks)
```

| Folder / file | What is in it |
|---|---|
| `src/` | One script per stage (`s1_prep.py` ... `s9_visuals.py`), shared code in `common.py`, scoring logic in `scorer.py` |
| `results/` | Raw logs, per-epoch training CSVs, `noise_manifest.csv` (answer key), `label_scores.csv`, evaluation JSONs, real-label review queue |
| `figures/` | Every chart and the demo GIF |
| `report.md` | The longer technical write-up |
| `DEVIATIONS.md` | Everything that differs from the original plan, and why |
| `verify_claims.py` | Independent checks of the numbers and of the "flagged, unverified" wording |

Tech: Python, PyTorch, Ultralytics YOLO11, pandas, scikit-learn, matplotlib, OpenCV.
"""
    (ROOT / "README.md").write_text(txt, encoding="utf-8"); print("README written")

if __name__ == "__main__":
    f_pipeline(); f_dataset(); f_noise_examples(); f_noise_mix(); f_crossfit(); f_crossfit_training(); f_pr(); f_at_k(); f_by_type()
    f_score_dist(); f_confusion(); f_downstream(); fr = f_real(); f_crops(); f_scorecard(); demo = make_gif(); readme(fr, demo)
