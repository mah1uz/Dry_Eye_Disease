"""Figure: stable SHAP importance vs. AUC gain from each non-ocular feature.

Inputs (both already in results/):
  results/shap_stability_folds.csv  mean |SHAP| of the full XGBoost model, 15 validation-fold refits
  results/add_one_feature.csv       gain in 5x5 CV ROC-AUC when one feature is added to the 3 ocular symptoms (LR)
Outputs:
  figures/shap_vs_gain.pdf
  results/shap_vs_gain_summary.json   (Spearman rho and ranges quoted in the paper)

Usage: python scripts/make_shap_vs_gain.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
shap_df = pd.read_csv(ROOT / "results" / "shap_stability_folds.csv")
gain = pd.read_csv(ROOT / "results" / "add_one_feature.csv").set_index("feature")

m = shap_df.groupby("Feature")["MeanAbsSHAP"].agg(["mean", "std"])
d = gain.join(m, how="inner")
assert len(d) == 23, "expected the 23 non-ocular features"

rho, p = spearmanr(d["mean"], d["mean_delta"])
summary = {
    "n_features": int(len(d)),
    "spearman_rho": float(rho), "spearman_p": float(p),
    "gain_min": float(d.mean_delta.min()), "gain_max": float(d.mean_delta.max()),
    "shap_min": float(d["mean"].min()), "shap_max": float(d["mean"].max()),
    "min_p_holm": float(d.p_holm.min()),
    "n_gain_above_0.005": int((d.mean_delta > 0.005).sum()),
}
(ROOT / "results" / "shap_vs_gain_summary.json").write_text(json.dumps(summary, indent=1))

short = {"Average screen time": "Screen time", "Smart device before bed": "Device at bed",
         "Systolic BP": "Systolic BP", "Alcohol consumption": "Alcohol", "Physical activity": "Activity"}
label = {"Systolic BP": (-4, -10), "Average screen time": (-10, 6), "Smart device before bed": (-12, 6), "Alcohol consumption": (-6, 6)}

plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.5, 2.3))
ax.axhline(0, color="#555", lw=0.8)
ax.errorbar(d["mean"], d["mean_delta"], xerr=d["std"], fmt="o", ms=3.6, color="#1f4e8c",
            ecolor="#9db4d3", elinewidth=0.7, capsize=0, mew=0)
for f, off in label.items():
    x, y = d.loc[f, "mean"], d.loc[f, "mean_delta"]
    ax.annotate(short.get(f, f), (x, y), xytext=off, textcoords="offset points", fontsize=6)
ax.set_ylim(-0.008, 0.008)
ax.set_xlabel("Mean |SHAP| (15 refits, bar = SD)", fontsize=7)
ax.set_ylabel(r"$\Delta$ROC-AUC when added to" "\nthe three symptoms", fontsize=7)
ax.tick_params(labelsize=6)
ax.grid(axis="y", color="#e5e5e5", lw=0.5)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.3)
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "shap_vs_gain.pdf", bbox_inches="tight")
print(json.dumps(summary, indent=1))
