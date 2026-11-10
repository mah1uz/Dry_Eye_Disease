"""
Reproduces the notebook pipeline (same cleaning, seed 42, 80/20 stratified split,
same model settings) and runs the extra checks used in the paper:

  1. Reproduction check of the held-out ROC-AUC values reported in the notebook.
  2. Label-permutation test (Ojala & Garriga, 2010) for the No-Ocular feature set.
  3. Ocular-symptom AUC ceiling: DED rate in each of the 8 symptom patterns.
  4. Figures with readable fonts: held-out ROC, SHAP stability, calibration.

Usage:  python extra_checks.py Dry_Eye_Dataset.csv OUTPUT_DIR
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.base import clone
from xgboost import XGBClassifier

SEED = 42
csv, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
figdir = out / "figures"; figdir.mkdir(exist_ok=True)
log = {}

# ---------------- preprocessing identical to the notebook ----------------
df = pd.read_csv(csv)
bp = df["Blood pressure"].astype(str).str.strip().str.extract(r"^(\d{2,3})/(\d{2,3})$").astype(int)
df["Systolic BP"], df["Diastolic BP"] = bp[0], bp[1]
df = df.drop(columns=["Blood pressure"])
yn = ["Sleep disorder","Wake up during night","Feel sleepy during day","Caffeine consumption",
      "Alcohol consumption","Smoking","Medical issue","Ongoing medication","Smart device before bed",
      "Blue-light filter","Discomfort Eye-strain","Redness in eye","Itchiness/Irritation in eye"]
for c in yn: df[c] = df[c].astype(str).str.strip().map({"N":0,"Y":1})
df["Gender_Male"] = (df["Gender"].astype(str).str.strip()=="M").astype(int); df = df.drop(columns=["Gender"])
df["Dry_Eye"] = df["Dry Eye Disease"].astype(str).str.strip().map({"N":0,"Y":1}); df = df.drop(columns=["Dry Eye Disease"])
df = df.loc[~(df["Systolic BP"] <= df["Diastolic BP"])].reset_index(drop=True)
log["n_rows"] = len(df)

G = {"Demographic":["Gender_Male","Age","Height","Weight"],
     "Sleep_Health":["Sleep duration","Sleep quality","Stress level","Systolic BP","Diastolic BP","Heart rate",
                     "Sleep disorder","Wake up during night","Feel sleepy during day","Medical issue","Ongoing medication"],
     "Lifestyle":["Daily steps","Physical activity","Caffeine consumption","Alcohol consumption","Smoking"],
     "Digital_Exposure":["Smart device before bed","Average screen time","Blue-light filter"],
     "Ocular_Symptoms":["Discomfort Eye-strain","Redness in eye","Itchiness/Irritation in eye"]}
ALL = sum(G.values(), []); OC = G["Ocular_Symptoms"]; NO = [f for f in ALL if f not in OC]
CFG = {"All Features":ALL, "No Ocular Symptoms":NO, "Ocular Symptoms Only":OC}

X = df.drop(columns=["Dry_Eye"]); y = df["Dry_Eye"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)

lr = Pipeline([("scaler",StandardScaler()),("classifier",LogisticRegression(max_iter=2000,class_weight="balanced",random_state=SEED))])
xgb = XGBClassifier(n_estimators=250,learning_rate=0.05,max_depth=4,min_child_weight=1,subsample=0.85,colsample_bytree=0.85,
                    objective="binary:logistic",eval_metric="logloss",tree_method="hist",random_state=SEED,n_jobs=-1,verbosity=0)
MODELS = {"Logistic Regression":lr, "XGBoost":xgb}
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ---------------- 1. reproduction of held-out ROC-AUC ----------------
NB = {("Logistic Regression","All Features"):0.6113,("Logistic Regression","No Ocular Symptoms"):0.5117,
      ("Logistic Regression","Ocular Symptoms Only"):0.6069,("XGBoost","All Features"):0.6016,
      ("XGBoost","No Ocular Symptoms"):0.5045,("XGBoost","Ocular Symptoms Only"):0.6038}
probs = {}; log["reproduction"] = {}
for m, est in MODELS.items():
    for c, feats in CFG.items():
        f = clone(est).fit(Xtr[feats], ytr); p = f.predict_proba(Xte[feats])[:,1]; probs[(m,c)] = p
        a = roc_auc_score(yte, p); log["reproduction"][f"{m} | {c}"] = {"reproduced":round(a,4),"notebook":NB[(m,c)]}
print(json.dumps(log["reproduction"], indent=1))

# ---------------- 2. permutation test (No-Ocular, 5-fold CV on training) ----------------
rng = np.random.default_rng(SEED)
N_PERM = {"Logistic Regression":1000, "XGBoost":200}
log["permutation"] = {}
for m, est in MODELS.items():
    real = cross_val_score(clone(est), Xtr[NO], ytr, cv=cv5, scoring="roc_auc").mean()
    null = []
    t = time.time()
    for i in range(N_PERM[m]):
        yp = pd.Series(rng.permutation(ytr.values), index=ytr.index)
        null.append(cross_val_score(clone(est), Xtr[NO], yp, cv=cv5, scoring="roc_auc").mean())
    null = np.array(null); p = (1 + np.sum(null >= real)) / (1 + len(null))
    log["permutation"][m] = {"n_perm":N_PERM[m],"real_auc":real,"null_mean":null.mean(),"null_sd":null.std(),
                             "null_q95":float(np.quantile(null,0.95)),"p_value":p,"seconds":time.time()-t}
    print(m, log["permutation"][m])

# ---------------- 3. ocular-symptom pattern table and AUC ceiling ----------------
pat = df[OC].astype(str).agg("".join, axis=1)
tab = df.groupby(pat)["Dry_Eye"].agg(["size","mean"]).rename(columns={"size":"n","mean":"ded_rate"})
tab["n_symptoms"] = [s.count("1") for s in tab.index]
tab = tab.sort_values(["n_symptoms","ded_rate"])
tab.to_csv(out/"ocular_pattern_table.csv")
print(tab)
# ceiling: score = pattern DED rate estimated on TRAIN, evaluated on TEST (no leakage)
ptr = Xtr[OC].astype(str).agg("".join, axis=1); pte = Xte[OC].astype(str).agg("".join, axis=1)
rate = ytr.groupby(ptr).mean()
ceiling_test = roc_auc_score(yte, pte.map(rate))
ceiling_insample = roc_auc_score(df["Dry_Eye"], pat.map(df.groupby(pat)["Dry_Eye"].mean()))
# number of symptoms as a score
nsym_test = roc_auc_score(yte, Xte[OC].sum(axis=1))
log["ceiling"] = {"pattern_rate_auc_test":ceiling_test,"pattern_rate_auc_full_insample":ceiling_insample,
                  "n_symptoms_auc_test":nsym_test,
                  "pattern_table":tab.reset_index().rename(columns={"index":"pattern"}).to_dict(orient="records")}
print(log["ceiling"])

# ---------------- 4a. ROC figure ----------------
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
COL = {"All Features":"#2a78d6","No Ocular Symptoms":"#eb6834","Ocular Symptoms Only":"#1baf7a"}
LAB = {"All Features":"All (26)","No Ocular Symptoms":"No ocular (23)","Ocular Symptoms Only":"Ocular only (3)"}
fig, ax = plt.subplots(1,2,figsize=(3.5,1.95),sharey=True)
for k,(m,title) in enumerate([("Logistic Regression","(a) LR"),("XGBoost","(b) XGBoost")]):
    for c in CFG:
        fpr,tpr,_ = roc_curve(yte, probs[(m,c)])
        ax[k].plot(fpr,tpr,lw=1.3,color=COL[c],label=LAB[c],
                   ls="--" if c=="Ocular Symptoms Only" else "-")
    ax[k].plot([0,1],[0,1],ls=":",lw=0.8,color="#777")
    ax[k].set_title(title,fontsize=7.5); ax[k].set_xlabel("False positive rate",fontsize=7)
    ax[k].legend(frameon=False,fontsize=5.6,loc="lower right",handlelength=1.6)
    ax[k].tick_params(labelsize=6)
ax[0].set_ylabel("True positive rate",fontsize=7)
fig.tight_layout(pad=0.3); fig.savefig(figdir/"roc_test.pdf",bbox_inches="tight"); fig.savefig(out/"roc_test.png",dpi=150,bbox_inches="tight")

# ---------------- 4b. SHAP stability figure (15 validation-fold refits) ----------------
import shap
rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
Xall = Xtr[ALL]; rows = []
for s,(tri,vai) in enumerate(rs.split(Xall, ytr), start=1):
    fm = clone(xgb).fit(Xall.iloc[tri], ytr.iloc[tri])
    sv = shap.TreeExplainer(fm)(Xall.iloc[vai]).values
    imp = np.abs(sv).mean(axis=0)
    for f,v in zip(ALL, imp): rows.append({"Split":s,"Feature":f,"MeanAbsSHAP":v})
st = pd.DataFrame(rows); st.to_csv(out/"shap_stability_folds.csv",index=False)
order = st.groupby("Feature")["MeanAbsSHAP"].mean().sort_values(ascending=False).index[:10].tolist()
short = {"Discomfort Eye-strain":"Eye strain","Redness in eye":"Redness","Itchiness/Irritation in eye":"Itchiness",
         "Average screen time":"Screen time"}
fig, ax = plt.subplots(figsize=(3.5,1.9))
data = [st.loc[st.Feature==f,"MeanAbsSHAP"].values for f in order]
bp_ = ax.boxplot(data, widths=0.55, patch_artist=True, medianprops=dict(color="black",lw=1), flierprops=dict(markersize=2.5))
for i,b in enumerate(bp_["boxes"]): b.set_facecolor("#2a78d6" if order[i] in OC else "#b7d3f6"); b.set_linewidth(0.6)
ax.set_xticks(range(1,11),[short.get(f,f) for f in order],rotation=40,ha="right",fontsize=6)
ax.set_ylabel("Mean |SHAP|",fontsize=7); ax.tick_params(axis="y",labelsize=6)
fig.tight_layout(pad=0.3); fig.savefig(figdir/"shap_stability.pdf",bbox_inches="tight"); fig.savefig(out/"shap_stability.png",dpi=150,bbox_inches="tight")
log["shap_stability_mean"] = st.groupby("Feature")["MeanAbsSHAP"].mean().sort_values(ascending=False).round(4).to_dict()

# ---------------- 4c. calibration figure (out-of-fold, training set) ----------------
fig, ax = plt.subplots(figsize=(3.0,2.1))
ax.plot([0,1],[0,1],ls=":",lw=0.8,color="#777",label="Perfect")
log["calibration_oof_mean"] = {}
for m,col,mk in [("Logistic Regression","#2a78d6","o"),("XGBoost","#eb6834","s")]:
    p = cross_val_predict(clone(MODELS[m]), Xtr[ALL], ytr, cv=cv5, method="predict_proba")[:,1]
    fo, mp = calibration_curve(ytr, p, n_bins=10, strategy="quantile")
    ax.plot(mp, fo, marker=mk, ms=3.5, lw=1.3, color=col, label=m)
    log["calibration_oof_mean"][m] = float(p.mean())
ax.set_xlabel("Mean predicted probability",fontsize=7); ax.set_ylabel("Observed DED rate",fontsize=7)
ax.set_xlim(0.2,0.85); ax.set_ylim(0.2,0.85); ax.tick_params(labelsize=6); ax.legend(frameon=False,fontsize=6)
fig.tight_layout(pad=0.3); fig.savefig(figdir/"calibration.pdf",bbox_inches="tight"); fig.savefig(out/"calibration.png",dpi=150,bbox_inches="tight")

json.dump(log, open(out/"extra_checks.json","w"), indent=1, default=float)
print("DONE")
