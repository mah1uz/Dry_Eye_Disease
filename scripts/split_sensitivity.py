"""Split sensitivity: repeat the held-out comparison on 10 other random splits at 80/20 and at 70/30.

Usage: python scripts/split_sensitivity.py Dry_Eye_Dataset.csv results/split_sensitivity.csv
It re-uses the cleaning code and model settings at the top of scripts/extra_checks.py.
"""
import sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd

csv, outfile = sys.argv[1], Path(sys.argv[2])
head = (Path(__file__).resolve().parent / "extra_checks.py").read_text().split("# ---------------- 1. reproduction")[0]
sys.argv = ["extra_checks.py", csv, tempfile.mkdtemp()]   # the head writes nothing we need
exec(head)                                                # gives X, y, MODELS, CFG, clone, roc_auc_score
from sklearn.model_selection import train_test_split

rows = []
for frac, name in [(0.20, "80/20"), (0.30, "70/30")]:
    for seed in range(10):
        a, b, c, d_ = train_test_split(X, y, test_size=frac, random_state=seed, stratify=y)
        for m, est in MODELS.items():
            r = {"split": name, "seed": seed, "model": m}
            for cfg, f in CFG.items():
                r[cfg] = roc_auc_score(d_, clone(est).fit(a[f], c).predict_proba(b[f])[:, 1])
            rows.append(r)
R = pd.DataFrame(rows)
outfile.parent.mkdir(parents=True, exist_ok=True)
R.to_csv(outfile, index=False)
R["drop"] = R["All Features"] - R["No Ocular Symptoms"]
R["gap"] = R["All Features"] - R["Ocular Symptoms Only"]
print(R.groupby(["split", "model"])[["All Features", "No Ocular Symptoms", "Ocular Symptoms Only", "drop", "gap"]].agg(["mean", "std"]).round(4).T)
