"""Stage 7: run the scorer on the ORIGINAL (uncorrupted) training labels. Output = 'flagged, unverified' review set.
Nothing here is claimed to be a real error until a human checks it."""
import html, cv2
import pandas as pd
from common import *
from scorer import score_labels, normalize_scores

OUT = RES / "real_audit"; CROPS = OUT / "crops"

def main():
    CROPS.mkdir(parents=True, exist_ok=True)
    split = load_split(); ids = sorted(i for i, s in split.items() if s == "train")
    preds = pd.read_csv(RES / "crossfit_predictions.csv")
    df = normalize_scores(score_labels(LAB / "clean" / "train", preds, ids))
    df.to_csv(OUT / "real_label_scores.csv", index=False)
    labs = df[df.kind == "label"].sort_values("suspicion", ascending=False, kind="stable")
    top = labs.head(100).reset_index(drop=True)
    rows = []
    for r in top.itertuples():
        img = cv2.imread(str(IMG_ALL / f"{r.image_id}.jpg")); H, W = img.shape[:2]
        x1, y1, x2, y2 = int((r.cx - r.w/2) * W), int((r.cy - r.h/2) * H), int((r.cx + r.w/2) * W), int((r.cy + r.h/2) * H)
        m = int(max(x2 - x1, y2 - y1) * 0.6) + 25
        cx1, cy1, cx2, cy2 = max(0, x1 - m), max(0, y1 - m), min(W, x2 + m), min(H, y2 + m)
        crop = img[cy1:cy2, cx1:cx2].copy()
        cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 255, 0), 2)
        cv2.putText(crop, r.label_class, (x1 - cx1, max(12, y1 - cy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        scale = max(1, 220 // max(1, min(crop.shape[:2])))
        if scale > 1: crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        fn = f"{r.Index+1:03d}_{r.image_id}_{r.label_idx}.jpg"; cv2.imwrite(str(CROPS / fn), crop)
        rows.append(dict(rank=r.Index + 1, image_id=r.image_id, label_idx=r.label_idx, label_class=r.label_class,
                         suspected_type=r.suspected_type, suggested_class=r.suggested_class if isinstance(r.suggested_class, str) else "",
                         suspicion=round(r.suspicion, 4), status="flagged, unverified", crop=f"crops/{fn}",
                         human_verdict="", human_notes=""))
    q = pd.DataFrame(rows); q.to_csv(OUT / "review_queue.csv", index=False)
    cards = "\n".join(
        f'<div class="c"><img src="{r.crop}" loading="lazy"><div><b>#{r.rank}</b> label: <b>{html.escape(r.label_class)}</b><br>'
        f'suspected: {html.escape(r.suspected_type)}{(" → "+html.escape(r.suggested_class)) if r.suggested_class else ""}<br>'
        f'score {r.suspicion}<br><span class="u">flagged, unverified</span><br><small>{r.image_id} #{r.label_idx}</small></div></div>'
        for r in q.itertuples())
    (OUT / "review.html").write_text(f"""<!doctype html><meta charset=utf-8><title>Label review queue (flagged, unverified)</title>
<style>body{{font-family:system-ui;margin:20px;background:#fafafa}}.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
.c{{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px;font-size:13px}}.c img{{max-width:100%;display:block;margin-bottom:6px}}
.u{{color:#b45309;font-weight:600}}</style><h2>Top 100 flagged labels - FLAGGED, UNVERIFIED</h2>
<p>Green box = the existing label. These are only boxes where the auditor disagrees with the label. None is confirmed as an error: review manually and record verdicts in review_queue.csv.</p>
<div class="g">{cards}</div>""", encoding="utf-8")
    summ = dict(n_labels_scored=int(len(labs)), top_flagged_in_queue=100,
                labels_with_suspicion_ge_0_99=int((labs.suspicion >= 0.99).sum()),
                labels_with_suspicion_ge_0_95=int((labs.suspicion >= 0.95).sum()),
                missing_label_candidates_conf_ge_0_3=int((df.kind == "candidate").sum()),
                queue_by_suspected_type=q.suspected_type.value_counts().to_dict(),
                status="flagged, unverified - not claimed to be real errors")
    json.dump(summ, open(OUT / "summary.json", "w"), indent=2); print(json.dumps(summ, indent=1))

import json
if __name__ == "__main__": main()
