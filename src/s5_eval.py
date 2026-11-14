"""Stage 5: evaluate the auditor against noise_manifest.csv (precision@k, recall@k, PR curve, per-type detection)."""
import json, ast, sys
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

TYPE_MAP = {"class_flip": "class_mismatch", "box_jitter": "localization_error",
            "spurious": "likely_spurious_label", "dropped": "likely_missing_label"}

def load_truth():
    m = pd.read_csv(RES / "noise_manifest.csv")
    for c in ("orig_box", "new_box"): m[c] = m[c].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) and s else None)
    return m

def label_rows(df, m, scores_col="suspicion"):
    """Attach is_noise / noise_type to each ranked row. Candidate rows match unmatched dropped boxes (IoU>=0.5)."""
    lab_truth = {(r.image_id, r.noisy_idx): r.noise_type for r in m.itertuples() if r.noise_type != "dropped"}
    dropped = {}
    for r in m[m.noise_type == "dropped"].itertuples(): dropped.setdefault(r.image_id, []).append([r.orig_box, False])
    df = df.sort_values(scores_col, ascending=False, kind="stable").reset_index(drop=True)
    ntype = [None] * len(df)
    for n, r in enumerate(df.itertuples()):
        if r.kind == "label":
            ntype[n] = lab_truth.get((r.image_id, r.label_idx))
        else:
            for item in dropped.get(r.image_id, []):
                if not item[1]:
                    iou = iou_matrix(xywh2xyxy(np.array([[r.cx, r.cy, r.w, r.h]])), xywh2xyxy(np.array([item[0]])))[0, 0]
                    if iou >= 0.5: item[1] = True; ntype[n] = "dropped"; break
    df["noise_type"] = ntype; df["is_noise"] = df.noise_type.notna()
    return df

def main():
    v1 = len(sys.argv) > 1 and sys.argv[1] == "v1"          # `s5_eval.py v1` evaluates the raw-score version
    suf = "_v1_raw_scores" if v1 else ""
    m = load_truth(); n_pos = len(m)
    sc = pd.read_csv(RES / ("label_scores_v1_raw.csv" if v1 else "label_scores.csv"))
    df = label_rows(sc, m)
    tp = df.is_noise.to_numpy().cumsum(); k = np.arange(1, len(df) + 1)
    prec, rec = tp / k, tp / n_pos
    ks = [100, 500, 1000, 5000, 10000, n_pos, 30000, 50000]
    table = [dict(k=int(x), precision=float(prec[x-1]), recall=float(rec[x-1])) for x in ks if x <= len(df)]
    base = n_pos / (len(sc[sc.kind == "label"]) + (m.noise_type == "dropped").sum())
    # per-type detection: share of each noise type inside the top-K flagged, K = number of corrupted boxes
    pert = {}
    for t, st in TYPE_MAP.items():
        tot = (m.noise_type == t).sum()
        for K in (n_pos, 2 * n_pos):
            top = df.iloc[:K]; hit = top[top.noise_type == t]
            pert.setdefault(t, {})[f"detected_in_top_{K}"] = int(len(hit))
            pert[t][f"rate_top_{K}"] = float(len(hit) / tot)
            pert[t][f"type_attribution_correct_top_{K}"] = float((hit.suspected_type == st).mean()) if len(hit) else None
        pert[t]["total"] = int(tot)
        # type-specific ranking quality: AUC of the relevant sub-score among label rows (dropped: candidate confidence)
    from sklearn.metrics import roc_auc_score, average_precision_score
    labs = df[df.kind == "label"]
    for t, col in (("class_flip", "s_class_mismatch"), ("box_jitter", "s_localization"), ("spurious", "s_spurious")):
        y = (labs.noise_type == t).to_numpy(); pert[t]["auc_vs_clean_labels"] = float(roc_auc_score(y, labs[col]))
        pert[t]["avg_precision_own_subscore"] = float(average_precision_score(y, labs[col]))
    pert["dropped"]["max_possible_recall_with_candidates"] = float((df.noise_type == "dropped").sum() / (m.noise_type == "dropped").sum())
    out = dict(n_corrupted=int(n_pos), n_ranked_rows=int(len(df)), random_baseline_precision=float(base),
               precision_recall_at_k=table, per_type=pert,
               average_precision=float(average_precision_score(df.is_noise, df.suspicion)))
    json.dump(out, open(RES / f"audit_eval{suf}.json", "w"), indent=2)
    df.to_csv(RES / f"label_scores_with_truth{suf}.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(rec[::20], prec[::20], lw=2, color="#2b6cb0", label=f"auditor (AP={out['average_precision']:.2f})")
    ax[0].axhline(base, ls="--", color="gray", label=f"random ({base:.2f})")
    ax[0].set(xlabel="recall", ylabel="precision", title="Overall PR curve (all noise types)", ylim=(0, 1.02)); ax[0].legend(); ax[0].grid(alpha=.3)
    colors = {"class_flip": "#c05621", "box_jitter": "#2f855a", "spurious": "#805ad5", "dropped": "#b83280"}
    for t in TYPE_MAP:
        y = (df.noise_type == t).to_numpy(); tpt = y.cumsum(); tot = y.sum()
        ax[1].plot((tpt / tot)[::20], (tpt / k)[::20], color=colors[t], label=t)
    ax[1].set(xlabel="recall of that type", ylabel="precision (hits of that type / flagged)", title="PR by noise type (same global ranking)"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(WORK / f"pr_curves{suf}_quicklook.png", dpi=100)   # quick-look only; publication figure is made by s9_visuals.py
    print(json.dumps(out, indent=1))

if __name__ == "__main__": main()
