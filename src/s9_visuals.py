"""Stage 9: publication-style figures, the qualitative demo GIF, and README.md.
Every number is read from results/*.json|csv; figure captions live in the README, not inside the images."""
import json, glob, ast
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import cv2
from PIL import Image, ImageDraw, ImageFont
from common import *

# ------------------------------------------------------------------ style: white ground, Okabe-Ito (colour-blind safe)
INK, INK2, GRID = "#1a1a1a", "#555555", "#d4d4d4"
BLUE, ORANGE, GREEN, GOLD, PURPLE, SKY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": INK2, "axes.linewidth": .8,
                     "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "normal", "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .5,
                     "grid.linestyle": ":", "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 1.8})
def save(fig, name): fig.savefig(FIG / name, dpi=200, bbox_inches="tight"); plt.close(fig); print("fig", name)
J = lambda p: json.load(open(RES / p))
ds, noise, ev, ev1, dn, clean, real = (J("dataset_stats.json"), J("noise_summary.json"), J("audit_eval.json"), J("audit_eval_v1_raw_scores.json"),
                                        J("downstream_results.json"), J("cleaning_summary.json"), J("real_audit/summary.json"))
N = ev["n_corrupted"]; pt = ev["per_type"]
TYPES = ["class_flip", "box_jitter", "spurious", "dropped"]
TNAME = {"class_flip": "Class flip", "box_jitter": "Box jitter", "spurious": "Spurious box", "dropped": "Dropped box"}
TCOL = {"class_flip": BLUE, "box_jitter": ORANGE, "spurious": GREEN, "dropped": GOLD}
DNAME = {"a_clean": "(a) Clean", "b_noisy": "(b) Noisy", "c_cleaned": "(c) Noisy + auto-cleaned"}
DCOL = {"a_clean": BLUE, "b_noisy": ORANGE, "c_cleaned": GREEN}
dfT = pd.read_csv(RES / "label_scores_with_truth.csv", low_memory=False)           # sorted by suspicion (rank = row index)
dfT1 = pd.read_csv(RES / "label_scores_with_truth_v1_raw_scores.csv", low_memory=False)
man = pd.read_csv(RES / "noise_manifest.csv")
for c in ("orig_box", "new_box"): man[c] = man[c].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) and s else None)

# ------------------------------------------------------------------ Fig 2: pipeline (neutral block diagram)
def f_pipeline():
    fig, ax = plt.subplots(figsize=(10.5, 3.9)); ax.axis("off"); ax.set_xlim(0, 10.5); ax.set_ylim(0, 3.9); ax.grid(False)
    steps = [("1. Data", "BDD100K, 10,000 images\nfixed 80/20 split"), ("2. Noise injection", "15% of train boxes corrupted\n-> ground-truth manifest"),
             ("3. Cross-fitted detector", "YOLO11s, 2 folds\nheld-out predictions"), ("4. Label scoring", "disagreement between label\nand held-out prediction"),
             ("5. Evaluation", "precision@k, recall@k,\nPR curve vs. manifest"), ("6. Downstream test", "detectors trained on clean /\nnoisy / cleaned labels"),
             ("7. Original-label audit", "top-100 review queue\n(unverified)")]
    pos = [(0.1, 2.1), (2.75, 2.1), (5.4, 2.1), (8.05, 2.1), (8.05, 0.2), (5.4, 0.2), (2.75, 0.2)]
    for (t, s), (x, y) in zip(steps, pos):
        ax.add_patch(Rectangle((x, y), 2.35, 1.5, fc="white", ec=INK2, lw=1)); ax.add_patch(Rectangle((x, y + 1.1), 2.35, .4, fc="#efefef", ec=INK2, lw=1))
        ax.text(x + .1, y + 1.3, t, fontsize=9, fontweight="bold", va="center"); ax.text(x + .1, y + .5, s, fontsize=8.3, va="center", color=INK, linespacing=1.4)
    def arr(a, b): ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=INK2, lw=1.1))
    arr((2.47, 2.85), (2.73, 2.85)); arr((5.12, 2.85), (5.38, 2.85)); arr((7.77, 2.85), (8.03, 2.85)); arr((9.22, 2.08), (9.22, 1.72))
    arr((8.03, .95), (7.77, .95)); arr((5.38, .95), (5.12, .95))
    save(fig, "fig02_pipeline.png")

# ------------------------------------------------------------------ Fig 3: class counts
def f_dataset():
    fig, ax = plt.subplots(figsize=(6.6, 3.3)); y = np.arange(len(CLASSES))[::-1]
    tr = [ds["boxes_per_class"]["train"][c] for c in CLASSES]; va = [ds["boxes_per_class"]["val"][c] for c in CLASSES]
    ax.barh(y + .19, tr, .36, color=BLUE, label=f"train ({ds['n_images']['train']:,} images)"); ax.barh(y - .19, va, .36, color=SKY, label=f"val ({ds['n_images']['val']:,} images)")
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(CLASSES); ax.set_xlabel("boxes (log scale)"); ax.set_xlim(1, 6e5); ax.grid(axis="y", visible=False)
    for yy, a, b in zip(y, tr, va):
        ax.text(a * 1.15, yy + .19, f"{a:,}", va="center", fontsize=7.5, color=INK2); ax.text(b * 1.15, yy - .19, f"{b:,}", va="center", fontsize=7.5, color=INK2)
    ax.legend(loc="lower right"); save(fig, "fig03_dataset.png")

# ------------------------------------------------------------------ Fig 4: noise examples (real crops)
def crop_axes(ax, iid, boxes, title):
    img = cv2.cvtColor(cv2.imread(str(IMG_ALL / f"{iid}.jpg")), cv2.COLOR_BGR2RGB); H, W = img.shape[:2]
    px = [(np.array([b[0] * W, b[1] * H, b[2] * W, b[3] * H]), col, ls, lab) for b, col, ls, lab in boxes]
    xs = [v for b, *_ in px for v in (b[0] - b[2] / 2, b[0] + b[2] / 2)]; ys = [v for b, *_ in px for v in (b[1] - b[3] / 2, b[1] + b[3] / 2)]
    cw, ch = max(xs) - min(xs), max(ys) - min(ys); m = max(cw, ch) * .55 + 40; cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    hw = max(cw / 2 + m, (ch / 2 + m) * 1.5); hh = hw / 1.5
    x1, x2, y1, y2 = int(max(0, cx - hw)), int(min(W, cx + hw)), int(max(0, cy - hh)), int(min(H, cy + hh))
    ax.imshow(img[y1:y2, x1:x2], extent=(x1, x2, y2, y1)); ax.grid(False); ax.axis("off")
    for (b, col, ls, lab) in px:
        ax.add_patch(Rectangle((b[0] - b[2] / 2, b[1] - b[3] / 2), b[2], b[3], fill=False, ec=col, lw=1.8, ls=ls))
        if lab: ax.text(b[0] - b[2] / 2, b[1] - b[3] / 2 - 3, lab, color="white", fontsize=7.5, va="bottom", bbox=dict(fc=col, ec="none", pad=1.2, alpha=.9))
    ax.set_xlim(x1, x2); ax.set_ylim(y2, y1); ax.set_title(title, fontsize=9, pad=4)

def f_noise_examples():
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.9))
    def pick(t, cond):
        s = man[man.noise_type == t].copy(); s["bb"] = s.apply(lambda r: r.new_box if r.new_box else r.orig_box, axis=1)
        s = s[s.bb.apply(lambda b: 0.03 < b[2] * b[3] * 4 < 0.5 and b[2] > .05)]; return s[s.apply(cond, axis=1)].sample(1, random_state=7).iloc[0]
    OK, BAD = "#2e8b57", ORANGE
    r = pick("class_flip", lambda r: r.orig_class == "car" and r.new_class == "truck")
    crop_axes(axs[0], r.image_id, [(r.new_box, BAD, "-", f"label: {r.new_class}")], f"(a) Class flip (original: {r.orig_class})")
    r = pick("box_jitter", lambda r: r.orig_class == "car")
    crop_axes(axs[1], r.image_id, [(r.orig_box, OK, "--", "original"), (r.new_box, BAD, "-", "jittered")], "(b) Box jitter")
    r = pick("dropped", lambda r: r.orig_class == "car")
    crop_axes(axs[2], r.image_id, [(r.orig_box, OK, "--", "label removed")], "(c) Dropped box")
    r = pick("spurious", lambda r: r.new_class == "car")
    crop_axes(axs[3], r.image_id, [(r.new_box, BAD, "-", "label: car")], "(d) Spurious box")
    plt.tight_layout(); save(fig, "fig04_noise_examples.png")

# ------------------------------------------------------------------ Fig 5: cross-fitting schematic
def f_crossfit():
    fig, ax = plt.subplots(figsize=(8.6, 2.2)); ax.axis("off"); ax.set_xlim(0, 8.6); ax.set_ylim(0, 2.2); ax.grid(False)
    for i, (tr_, te_) in enumerate([("A", "B"), ("B", "A")]):
        y = 1.25 - i * 1.15
        ax.add_patch(Rectangle((.1, y), 1.9, .8, fc="#e8f1f8", ec=INK2, lw=1)); ax.text(1.05, y + .4, f"train fold {tr_}\n(noisy labels)", ha="center", va="center", fontsize=8.3)
        ax.add_patch(FancyArrowPatch((2.05, y + .4), (2.75, y + .4), arrowstyle="-|>", mutation_scale=11, color=INK2, lw=1.1))
        ax.add_patch(Rectangle((2.8, y), 1.7, .8, fc="white", ec=INK2, lw=1)); ax.text(3.65, y + .4, f"detector {i+1}", ha="center", va="center", fontsize=8.3)
        ax.add_patch(FancyArrowPatch((4.55, y + .4), (5.25, y + .4), arrowstyle="-|>", mutation_scale=11, color=INK2, lw=1.1))
        ax.add_patch(Rectangle((5.3, y), 1.7, .8, fc="#f3f3f3", ec=INK2, lw=1)); ax.text(6.15, y + .4, f"predict fold {te_}\n(held out)", ha="center", va="center", fontsize=8.3)
        ax.add_patch(FancyArrowPatch((7.05, y + .4), (7.65, y + .4), arrowstyle="-|>", mutation_scale=11, color=INK2, lw=1.1)); ax.text(7.7, y + .4, "score labels\nin fold " + te_, va="center", fontsize=8.3)
    save(fig, "fig05_cross_fitting.png")

def f_crossfit_training():
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.8))
    for k, c in ((0, BLUE), (1, ORANGE)):
        d = pd.read_csv(RES / "logs" / f"crossfit_model{k}_epochs.csv").rename(columns=str.strip)
        ax[0].plot(d.epoch, d["train/box_loss"], color=c); ax[1].plot(d.epoch, d["metrics/mAP50(B)"], color=c)
        ax[0].text(d.epoch.iloc[-1] + .5, d["train/box_loss"].iloc[-1], f"detector {k+1}", color=c, va="center", fontsize=8)
        ax[1].text(d.epoch.iloc[-1] + .5, d["metrics/mAP50(B)"].iloc[-1], f"detector {k+1}", color=c, va="center", fontsize=8)
    ax[0].set(title="(a) Training box loss", xlabel="epoch", xlim=(0, 38)); ax[1].set(title="(b) Validation mAP50 (clean val)", xlabel="epoch", xlim=(0, 38))
    plt.tight_layout(); save(fig, "fig06_detector_training.png")

# ------------------------------------------------------------------ Fig 7-11: auditor evaluation
def cum(df):
    tp = df.is_noise.to_numpy().cumsum(); k = np.arange(1, len(df) + 1); return k, tp / k, tp / N

def f_score_dist():
    L = dfT[dfT.kind == "label"]; fig, ax = plt.subplots(figsize=(5.2, 3)); bins = np.linspace(0, 1, 26)
    for sel, c, lab in ((~L.is_noise, BLUE, "uncorrupted labels"), (L.is_noise, ORANGE, "corrupted labels")):
        ax.hist(L[sel].suspicion, bins=bins, density=True, histtype="stepfilled", color=c, alpha=.25, lw=0); ax.hist(L[sel].suspicion, bins=bins, density=True, histtype="step", color=c, lw=1.6, label=lab)
    ax.set(xlabel="suspicion score (v2)", ylabel="density"); ax.legend(loc="upper left"); save(fig, "fig07_score_distribution.png")

def f_pr():
    k2, p2, r2 = cum(dfT); k1, p1, r1 = cum(dfT1); base = ev["random_baseline_precision"]; s = slice(None, None, 25)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(r1[s], p1[s], color=ORANGE, label=f"v1, raw signals (AP = {ev1['average_precision']:.3f})"); ax[0].plot(r2[s], p2[s], color=BLUE, label=f"v2, percentile (AP = {ev['average_precision']:.3f})")
    ax[0].axhline(base, color=INK2, ls="--", lw=1, label=f"random ({base:.3f})"); ax[0].legend(loc="upper right")
    ax[0].set(xlabel="recall", ylabel="precision", ylim=(0, 1), xlim=(0, 1), title="(a) Precision-recall, all error types")
    lim = 60000; s2 = slice(0, lim, 20)
    for t in TYPES:
        y = (dfT.noise_type == t).to_numpy(); rc = y.cumsum() / y.sum(); ax[1].plot(k2[s2], rc[s2], color=TCOL[t])
        ax[1].text(lim + 800, rc[lim - 1], TNAME[t], color=TCOL[t], fontsize=8, va="center")
    ax[1].axvline(N, color=INK2, ls=":", lw=1); ax[1].text(N + 700, .03, f"k = {N:,}", fontsize=8, color=INK2)
    ax[1].set(xlabel="k (items reviewed)", ylabel="recall of error type", xlim=(0, lim * 1.28), ylim=(0, 1), title="(b) Recall by error type (v2)")
    plt.tight_layout(); save(fig, "fig08_pr_curves.png")

def f_at_k():
    k, p, r = cum(dfT); s = slice(None, 60000, 20); base = ev["random_baseline_precision"]
    fig, ax = plt.subplots(figsize=(5.6, 3.1)); ax.plot(k[s], p[s], color=BLUE); ax.plot(k[s], r[s], color=ORANGE)
    ax.axhline(base, color=INK2, ls="--", lw=1); ax.axvline(N, color=INK2, ls=":", lw=1)
    ax.text(51000, base + .015, "random precision", fontsize=8, color=INK2); ax.text(N + 800, .93, f"k = {N:,}", fontsize=8, color=INK2)
    ax.text(32000, p[32000] + .035, "precision@k", color=BLUE, fontsize=8.5); ax.text(44000, r[44000] - .075, "recall@k", color=ORANGE, fontsize=8.5)
    ax.set(xlabel="k (top-ranked items reviewed)", ylim=(0, 1)); save(fig, "fig09_precision_recall_at_k.png")

def f_by_type():
    fig, ax = plt.subplots(figsize=(6.4, 3.2)); x = np.arange(4); w = .26
    S = [([ev1["per_type"][t]["rate_top_%d" % N] for t in TYPES], "v1, k = N", "#b9b9b9"), ([pt[t]["rate_top_%d" % N] for t in TYPES], "v2, k = N", BLUE), ([pt[t]["rate_top_%d" % (2 * N)] for t in TYPES], "v2, k = 2N", SKY)]
    for i, (v, lab, c) in enumerate(S):
        b = ax.bar(x + (i - 1) * w, [q * 100 for q in v], w - .02, color=c, label=lab)
        for r_, q in zip(b, v): ax.text(r_.get_x() + r_.get_width() / 2, q * 100 + 1.2, f"{q*100:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([TNAME[t] for t in TYPES]); ax.set_ylabel("injected errors recovered (%)"); ax.set_ylim(0, 100); ax.grid(axis="x", visible=False); ax.legend(loc="upper right", ncol=3)
    save(fig, "fig10_detection_by_type.png")

def f_confusion():
    top = dfT.head(N).copy(); cols = ["class_mismatch", "localization_error", "likely_spurious_label", "likely_missing_label"]
    cn = {"class_mismatch": "class\nmismatch", "localization_error": "localization\nerror", "likely_spurious_label": "spurious\nlabel", "likely_missing_label": "missing\nlabel"}
    top["truth"] = top.noise_type.fillna("none").map({**TNAME, "none": "No injected error"}); rows = [TNAME[t] for t in TYPES] + ["No injected error"]
    M = np.array([[(top[top.truth == r_].suspected_type == c).sum() for c in cols] for r_ in rows], float); P = M / M.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.4)); ax.imshow(P, cmap="Blues", vmin=0, vmax=1, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(4)); ax.set_xticklabels([cn[c] for c in cols], fontsize=8.5); ax.set_yticks(range(5)); ax.set_yticklabels(rows); ax.xaxis.tick_top()
    ax.set_xlabel("suspected error type (auditor output)", labelpad=8); ax.xaxis.set_label_position("top"); ax.set_ylabel("ground truth")
    for i in range(5):
        for j in range(4): ax.text(j, i, f"{P[i,j]*100:.0f}%\n({int(M[i,j]):,})", ha="center", va="center", fontsize=7.8, color="white" if P[i, j] > .55 else INK)
    for sp in ax.spines.values(): sp.set_visible(False)
    save(fig, "fig11_error_type_matrix.png")

# ------------------------------------------------------------------ Fig 12-14: downstream
def f_downstream():
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3))
    for a, key, t in ((ax[0], "mAP50", "(a) mAP50"), (ax[1], "mAP50_95", "(b) mAP50-95")):
        v = [dn[k][key] for k in dn]; b = a.bar(["(a) clean", "(b) noisy", "(c) cleaned"], v, color=[DCOL[k] for k in dn], width=.55)
        for r_, q in zip(b, v): a.text(r_.get_x() + r_.get_width() / 2, q + .004, f"{q:.3f}", ha="center", fontsize=8.5)
        a.set_ylim(0, max(v) * 1.18); a.set_title(t); a.grid(axis="x", visible=False)
    plt.tight_layout(); save(fig, "fig12_downstream_map.png")
    classes = [c for c in CLASSES if c != "train"]; fig, ax = plt.subplots(figsize=(8.4, 3)); x = np.arange(len(classes)); w = .27
    for i, k in enumerate(dn):
        ax.bar(x + (i - 1) * w, [dn[k]["per_class_mAP50"].get(c, 0) for c in classes], w - .02, color=DCOL[k], label=DNAME[k])
    ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=8); ax.set_ylabel("mAP50"); ax.legend(ncol=3, loc="upper right"); ax.grid(axis="x", visible=False); ax.set_ylim(0, .85)
    save(fig, "fig13_downstream_per_class.png")
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.8))
    for n_ in dn:
        d = pd.read_csv(RES / "logs" / f"downstream_{n_}_epochs.csv").rename(columns=str.strip)
        ax[0].plot(d.epoch, d["train/box_loss"], color=DCOL[n_]); ax[1].plot(d.epoch, d["metrics/mAP50(B)"], color=DCOL[n_])
        ax[1].text(d.epoch.iloc[-1] + .5, d["metrics/mAP50(B)"].iloc[-1], n_[0], color=DCOL[n_], va="center", fontsize=8.5)
    ax[0].set(title="(a) Training box loss", xlabel="epoch"); ax[1].set(title="(b) Validation mAP50 (clean val)", xlabel="epoch", xlim=(0, 33))
    plt.tight_layout(); save(fig, "fig14_downstream_curves.png")

# ------------------------------------------------------------------ Fig 15-16: original-label audit
def f_real():
    R = pd.read_csv(RES / "real_audit" / "real_label_scores.csv", low_memory=False); R = R[R.kind == "label"]
    Nn = pd.read_csv(RES / "label_scores.csv", low_memory=False); Nn = Nn[Nn.kind == "label"]
    sig = [("s_class_mismatch", "(a) class-mismatch signal"), ("s_localization", "(b) localization signal"), ("s_spurious", "(c) spurious signal")]
    fr = {c: ((R[c] >= .5).mean() * 100, (Nn[c] >= .5).mean() * 100) for c, _ in sig}
    fig = plt.figure(figsize=(11.2, 2.9)); gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.4], wspace=.62)
    for i, (c, nm) in enumerate(sig):
        a = fig.add_subplot(gs[i]); v = fr[c]; b = a.bar(["original", "noisy"], v, color=[BLUE, ORANGE], width=.55)
        for r_, q in zip(b, v): a.text(r_.get_x() + r_.get_width() / 2, q + max(v) * .03, f"{q:.1f}", ha="center", fontsize=8.5)
        a.set_ylim(0, max(v) * 1.2); a.set_title(nm, fontsize=9); a.grid(axis="x", visible=False)
        if i == 0: a.set_ylabel("labels with signal >= 0.5 (%)")
    q = real["queue_by_suspected_type"]; names = {"class_mismatch": "class", "localization_error": "localization", "likely_spurious_label": "spurious"}
    ax = fig.add_subplot(gs[3]); b = ax.barh([names[k] for k in q][::-1], list(q.values())[::-1], color=BLUE, height=.5)
    for r_, v in zip(b, list(q.values())[::-1]): ax.text(v + .8, r_.get_y() + r_.get_height() / 2, str(v), va="center", fontsize=8.5)
    ax.set(xlabel="boxes in top-100 queue", title="(d) Review queue by suspected type", xlim=(0, max(q.values()) * 1.2)); ax.grid(axis="y", visible=False)
    save(fig, "fig15_real_audit.png"); return fr

def f_crops():
    q = pd.read_csv(RES / "real_audit" / "review_queue.csv").head(12); fig, axs = plt.subplots(3, 4, figsize=(9.6, 6.6))
    for a, r in zip(axs.ravel(), q.itertuples()):
        a.imshow(cv2.cvtColor(cv2.imread(str(RES / "real_audit" / r.crop)), cv2.COLOR_BGR2RGB)); a.axis("off"); a.grid(False)
        sg = f" -> {r.suggested_class}" if isinstance(r.suggested_class, str) and r.suggested_class else ""
        a.set_title(f"#{r.rank}  {r.label_class}{sg}\n{r.suspected_type.replace('_', ' ')}  ({r.suspicion:.2f})", fontsize=7.8, loc="left")
    plt.tight_layout(); save(fig, "fig16_flagged_crops.png")

# ------------------------------------------------------------------ Fig 1: qualitative GIF (understated)
def font(sz, bold=False):
    for n in (("arialbd.ttf" if bold else "arial.ttf"), "segoeui.ttf", "DejaVuSans.ttf"):
        try: return ImageFont.truetype(n, sz)
        except OSError: pass
    return ImageFont.load_default()

def pick_demo_image():
    top = dfT.head(N); cand = []
    for iid, g in man.groupby("image_id"):
        if g.noise_type.nunique() < 3 or len(read_labels(LAB / "clean" / "train" / f"{iid}.txt")) > 14: continue
        t = top[top.image_id == iid]; hit = t[t.is_noise]; cand.append((-hit.noise_type.nunique(), -len(hit), len(t) - len(hit), iid))
    cand.sort(); return cand[0][-1]

def make_gif():
    iid = pick_demo_image(); W0, H0, S = 1280, 720, .7; W, H = int(W0 * S), int(H0 * S); BAN, FOOT = 40, 40
    base = Image.open(IMG_ALL / f"{iid}.jpg").convert("RGB").resize((W, H))
    clean_l = read_labels(LAB / "clean" / "train" / f"{iid}.txt"); noisy = read_labels(LAB / "noisy" / "train" / f"{iid}.txt")
    m = man[man.image_id == iid]; sc = dfT[dfT.image_id == iid]; flagged = sc[sc.index < N]
    preds = pd.read_csv(RES / "crossfit_predictions.csv"); preds = preds[(preds.image_id == iid) & (preds.conf >= .3)]
    col = dict(ok="#00b386", bad="#e8590c", pred="#56b4e9", flag="#f2b01e", gray="#9aa0a6")
    def px(b): return [(b[0] - b[2] / 2) * W, (b[1] - b[3] / 2) * H, (b[0] + b[2] / 2) * W, (b[1] + b[3] / 2) * H]
    def rect(d, b, c, wd=2, dash=False, tag=None):
        x1, y1, x2, y2 = px(b)
        if not dash: d.rectangle([x1, y1, x2, y2], outline=c, width=wd)
        else:
            st = 8
            for x in np.arange(x1, x2, st * 2): d.line([x, y1, min(x + st, x2), y1], fill=c, width=wd); d.line([x, y2, min(x + st, x2), y2], fill=c, width=wd)
            for y in np.arange(y1, y2, st * 2): d.line([x1, y, x1, min(y + st, y2)], fill=c, width=wd); d.line([x2, y, x2, min(y + st, y2)], fill=c, width=wd)
        if tag:
            f = font(12); tw = d.textlength(tag, font=f); ty = max(0, y1 - 17); d.rectangle([x1, ty, x1 + tw + 7, ty + 16], fill=c); d.text((x1 + 3, ty + 1), tag, fill="black" if c == col["flag"] else "white", font=f)
    def frame(top_txt, foot_txt, fn):
        im = Image.new("RGB", (W, H + BAN + FOOT), "#202225"); layer = Image.new("RGB", (W, H)); layer.paste(base); fn(ImageDraw.Draw(layer)); im.paste(layer, (0, BAN))
        d = ImageDraw.Draw(im); d.text((12, 10), top_txt, fill="white", font=font(17, True)); d.text((12, BAN + H + 11), foot_txt, fill="#c9cbd0", font=font(14)); return im
    CLS = lambda c: CLASSES[int(c)]; frames, dur = [], []
    def f1(d):
        for b in clean_l: rect(d, b[1:], col["ok"], 2, tag=CLS(b[0]))
    frames.append(frame("Original annotations", "Boxes and classes as released with BDD100K.", f1)); dur.append(2400)
    TAG = {"class_flip": "class flip", "box_jitter": "box jitter", "spurious": "spurious", "dropped": "dropped"}
    nz = {int(r.noisy_idx): r for r in m.itertuples() if r.noise_type != "dropped"}
    def f2(d):
        for j, b in enumerate(noisy):
            r = nz.get(j); rect(d, b[1:], col["ok"] if r is None else col["bad"], 2 if r is None else 3, tag=CLS(b[0]) if r is None else f"{TAG[r.noise_type]}: {CLS(b[0])}")
        for r in m[m.noise_type == "dropped"].itertuples(): rect(d, r.orig_box, col["bad"], 2, dash=True, tag="dropped")
    frames.append(frame("Injected label noise", "Orange: corrupted by the noise injector (dashed: removed label). Training labels only; the val set is never corrupted.", f2)); dur.append(3000)
    def f3(d):
        for b in noisy: rect(d, b[1:], col["gray"], 1)
        for p in preds.itertuples(): rect(d, [p.cx, p.cy, p.w, p.h], col["pred"], 2, tag=f"{CLS(p.cls)} {p.conf:.2f}")
    frames.append(frame("Held-out detector predictions", "Blue: predictions (conf >= 0.3) of a detector that did not train on this image. Grey: noisy labels.", f3)); dur.append(3000)
    lab = flagged[flagged.kind == "label"].set_index("label_idx"); cands = flagged[flagged.kind == "candidate"]
    short = {"class_mismatch": "class", "localization_error": "loc.", "likely_spurious_label": "spurious"}
    def f4(d):
        for j, b in enumerate(noisy):
            if j in lab.index: r = lab.loc[j]; rect(d, b[1:], col["flag"], 3, tag=f"{short[r.suspected_type]} {r.suspicion:.2f}")
            else: rect(d, b[1:], col["ok"], 1)
        for r in cands.itertuples(): rect(d, [r.cx, r.cy, r.w, r.h], col["flag"], 3, dash=True, tag="missing?")
    frames.append(frame("Auditor output", "Yellow: labels the auditor flags (top-ranked, k = number of injected errors) with suspected type and score.", f4)); dur.append(3200)
    hit = flagged[flagged.is_noise]; fa = len(flagged) - len(hit)
    def f5(d):
        for j, b in enumerate(noisy):
            if j in lab.index: r = lab.loc[j]; ok = bool(r.is_noise); rect(d, b[1:], col["ok"] if ok else col["gray"], 3, tag=("TP: " + TAG[r.noise_type]) if ok else "FP")
            else: rect(d, b[1:], col["ok"], 1)
        for r in m[m.noise_type == "dropped"].itertuples(): rect(d, r.orig_box, col["bad"], 2, dash=True)
        for r in cands.itertuples(): rect(d, [r.cx, r.cy, r.w, r.h], col["ok"] if r.is_noise else col["gray"], 3, dash=True, tag="TP: dropped" if r.is_noise else "FP")
    frames.append(frame("Comparison with ground truth", f"This image: {len(hit)} of {len(m)} injected errors flagged, {fa} false positive(s). Aggregate results: see tables.", f5)); dur.append(4000)
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    pal[0].save(FIG / "fig01_demo.gif", save_all=True, append_images=pal[1:], duration=dur, loop=0, optimize=True, disposal=2)
    print("gif", iid, (FIG / "fig01_demo.gif").stat().st_size // 1024, "KB"); return iid, len(hit), len(m), fa

if __name__ == "__main__":
    for old in list(FIG.glob("*")): old.unlink()         # figures/ contains only what this script produces
    f_pipeline(); f_dataset(); f_noise_examples(); f_crossfit(); f_crossfit_training(); f_score_dist(); f_pr(); f_at_k(); f_by_type()
    f_confusion(); f_downstream(); fr = f_real(); f_crops(); demo = make_gif()
    from readme_gen import readme
    readme(fr, demo, ds, noise, ev, ev1, dn, clean, real, TYPES, TNAME, DNAME, N)
