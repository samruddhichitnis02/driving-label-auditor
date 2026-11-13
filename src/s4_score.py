"""Stage 4: score every noisy training label against cross-fitted predictions -> results/label_scores.csv"""
from common import *
from scorer import score_labels, normalize_scores
import pandas as pd

def main():
    split = load_split(); ids = sorted(i for i, s in split.items() if s == "train")
    preds = pd.read_csv(RES / "crossfit_predictions.csv")
    df = score_labels(LAB / "noisy" / "train", preds, ids)
    df.to_csv(RES / "label_scores_v1_raw.csv", index=False)   # v1: raw sub-scores, kept for comparison
    df = normalize_scores(df)
    df.to_csv(RES / "label_scores.csv", index=False)
    print(df.kind.value_counts().to_dict(), df.groupby("suspected_type").size().to_dict())

if __name__ == "__main__": main()
