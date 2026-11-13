"""Label scoring logic (reused for noisy labels in Stage 4 and original labels in Stage 7).
For each label box we compare against cross-fitted predictions and compute four sub-scores in [0,1]:
  class_mismatch : a confident prediction of a DIFFERENT class sits on this box, and the label's class is not supported
  localization   : the model sees a same-class object that overlaps the label only partly (IoU 0.1-0.75)
  spurious       : no prediction of any class overlaps the label (model sees nothing there)
Plus 'likely_missing' candidates: confident predictions that no label covers.
suspicion = max sub-score; suspected_type = argmax."""
import numpy as np, pandas as pd
from common import *

MISS_CONF = 0.30          # min confidence for an unmatched prediction to become a missing-label candidate

def score_labels(label_dir, preds, image_ids, id_to_split=None):
    """label_dir has {id}.txt files. preds: DataFrame of cross-fitted predictions. -> DataFrame of rows."""
    pg = {k: g for k, g in preds.groupby("image_id")}
    out = []
    for iid in image_ids:
        lab = read_labels(Path(label_dir) / f"{iid}.txt")
        g = pg.get(iid)
        P = g[["cls", "conf", "cx", "cy", "w", "h"]].to_numpy() if g is not None else np.zeros((0, 6))
        pb = xywh2xyxy(P[:, 2:6]) if len(P) else np.zeros((0, 4))
        lb = xywh2xyxy(lab[:, 1:5]) if len(lab) else np.zeros((0, 4))
        iou = iou_matrix(lb, pb)                                  # (labels, preds)
        for j in range(len(lab)):
            c = int(lab[j, 0])
            if len(P):
                io = iou[j]; same = P[:, 0] == c
                conf_same_ok = P[same & (io >= 0.5), 1].max(initial=0.0)
                conf_oth = P[~same & (io >= 0.5), 1]
                conf_other = conf_oth.max(initial=0.0)
                oth_cls = int(P[~same & (io >= 0.5)][conf_oth.argmax(), 0]) if len(conf_oth) else -1
                good = P[same & (io >= 0.75), 1].max(initial=0.0)
                near = same & (io >= 0.1) & (io < 0.75)
                loc = (P[near, 1] * (1 - io[near])).max(initial=0.0) * (1 - good)
                anyc = P[io >= 0.3, 1].max(initial=0.0)
            else:
                conf_same_ok = conf_other = good = loc = anyc = 0.0; oth_cls = -1
            s_cls = max(0.0, conf_other - conf_same_ok)
            s_loc = float(loc)
            s_sp = 1.0 - anyc
            subs = {"class_mismatch": s_cls, "localization_error": s_loc, "likely_spurious_label": s_sp}
            t = max(subs, key=subs.get)
            out.append(dict(image_id=iid, kind="label", label_idx=j, label_class=CLASSES[c],
                            cx=lab[j, 1], cy=lab[j, 2], w=lab[j, 3], h=lab[j, 4],
                            s_class_mismatch=s_cls, s_localization=s_loc, s_spurious=s_sp,
                            suspicion=subs[t], suspected_type=t,
                            suggested_class=CLASSES[oth_cls] if (t == "class_mismatch" and oth_cls >= 0) else ""))
        # missing-label candidates: confident predictions not covered by any label
        for k in range(len(P)):
            if P[k, 1] < MISS_CONF: continue
            cov = iou[:, k].max(initial=0.0) if len(lab) else 0.0
            if cov >= 0.3: continue
            out.append(dict(image_id=iid, kind="candidate", label_idx=-1, label_class=CLASSES[int(P[k, 0])],
                            cx=P[k, 2], cy=P[k, 3], w=P[k, 4], h=P[k, 5],
                            s_class_mismatch=0.0, s_localization=0.0, s_spurious=0.0,
                            suspicion=float(P[k, 1]), suspected_type="likely_missing_label", suggested_class=""))
    return pd.DataFrame(out)

def normalize_scores(df):
    """v2 (unsupervised): convert each sub-score to its percentile rank among labels of the same class and
    size bin, so the three sub-scores are comparable and 'small hard-to-see object' labels stop dominating.
    Uses NO ground-truth noise information. Candidates keep their confidence as score."""
    lab = df.kind == "label"; L = df[lab].copy()
    L["area"] = L.w * L.h
    L["sbin"] = L.groupby("label_class")["area"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False, duplicates="drop"))
    for raw, nm in (("s_class_mismatch", "p_cls"), ("s_localization", "p_loc"), ("s_spurious", "p_sp")):
        L[nm] = L.groupby(["label_class", "sbin"])[raw].rank(pct=True, method="average")
    # a sub-score of exactly 0 carries no evidence
    L.loc[L.s_class_mismatch <= 0.05, "p_cls"] = 0.0
    L.loc[L.s_localization <= 0.05, "p_loc"] = 0.0
    P = L[["p_cls", "p_loc", "p_sp"]].to_numpy(); names = np.array(["class_mismatch", "localization_error", "likely_spurious_label"])
    L["suspicion_raw"] = L.suspicion; L["suspicion"] = P.max(1); L["suspected_type"] = names[P.argmax(1)]
    L.loc[L.suspected_type != "class_mismatch", "suggested_class"] = ""
    df = df.copy()
    for c in ("suspicion_raw", "suspicion", "suspected_type", "suggested_class", "p_cls", "p_loc", "p_sp"):
        df.loc[lab, c] = L[c]
    df.loc[~lab, "suspicion_raw"] = df.loc[~lab, "suspicion"]
    # candidates (missing labels): confidence in [0.3,1] mapped to a comparable scale (percentile among candidates)
    cand = ~lab
    df.loc[cand, "suspicion"] = df.loc[cand, "suspicion"].rank(pct=True)
    return df
