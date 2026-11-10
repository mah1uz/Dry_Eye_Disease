"""Add-one-feature test: Ocular-only baseline vs Ocular + one non-ocular feature.
Same cleaning, seed and 80/20 split as the notebook. 5x5 repeated CV on the training set (25 paired splits),
Nadeau-Bengio corrected resampled t-test, Holm correction. Also: held-out AUC of Ocular vs Ocular+Screen time (saved next to the csv as *_test_auc.json).
Usage: python add_one_feature.py Dry_Eye_Dataset.csv results/add_one_feature.csv"""
import sys, json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import t as tdist
SEED=42
df=pd.read_csv(sys.argv[1])
bp=df["Blood pressure"].astype(str).str.strip().str.extract(r"^(\d{2,3})/(\d{2,3})$").astype(int)
df["Systolic BP"],df["Diastolic BP"]=bp[0],bp[1]; df=df.drop(columns=["Blood pressure"])
yn=["Sleep disorder","Wake up during night","Feel sleepy during day","Caffeine consumption","Alcohol consumption","Smoking","Medical issue","Ongoing medication","Smart device before bed","Blue-light filter","Discomfort Eye-strain","Redness in eye","Itchiness/Irritation in eye"]
for c in yn: df[c]=df[c].astype(str).str.strip().map({"N":0,"Y":1})
df["Gender_Male"]=(df["Gender"].astype(str).str.strip()=="M").astype(int); df=df.drop(columns=["Gender"])
df["Dry_Eye"]=df["Dry Eye Disease"].astype(str).str.strip().map({"N":0,"Y":1}); df=df.drop(columns=["Dry Eye Disease"])
df=df.loc[~(df["Systolic BP"]<=df["Diastolic BP"])].reset_index(drop=True)
OC=["Discomfort Eye-strain","Redness in eye","Itchiness/Irritation in eye"]
NO=[c for c in df.columns if c not in OC+["Dry_Eye"]]
X=df.drop(columns=["Dry_Eye"]); y=df["Dry_Eye"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=SEED,stratify=y)
mk=lambda: Pipeline([("s",StandardScaler()),("c",LogisticRegression(max_iter=2000,class_weight="balanced",random_state=SEED))])
cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=SEED)
splits=list(cv.split(Xtr,ytr))
def cvauc(cols):
    return np.array([roc_auc_score(ytr.iloc[v],mk().fit(Xtr.iloc[t][cols],ytr.iloc[t]).predict_proba(Xtr.iloc[v][cols])[:,1]) for t,v in splits])
base=cvauc(OC); rows=[]
for f in NO:
    a=cvauc(OC+[f]); d=a-base
    k=len(d); var=d.var(ddof=1)*(1/k+0.25)   # Nadeau-Bengio corrected resampled t (n_test/n_train = 1/4)
    p=2*tdist.sf(abs(d.mean()/np.sqrt(var)),k-1)
    rows.append(dict(feature=f,mean_delta=d.mean(),ci_lo=np.percentile(d,2.5),ci_hi=np.percentile(d,97.5),p=p))
r=pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
m=len(r); r["p_holm"]=np.minimum(1,np.maximum.accumulate((m-np.arange(m))*r.p.values))
r.to_csv(sys.argv[2],index=False); print(r.round(4).head(8).to_string())
print("baseline cv auc",base.mean().round(4))
test_auc={}
for name,cols in [("ocular",OC),("ocular+screen",OC+["Average screen time"]),("ocular+all non-ocular",OC+NO)]:
    m_=mk().fit(Xtr[cols],ytr); test_auc[name]=round(roc_auc_score(yte,m_.predict_proba(Xte[cols])[:,1]),4); print(name,"test auc",test_auc[name])
# save the held-out numbers quoted in the paper, so that they can be checked
json.dump({"baseline_cv_auc":float(base.mean()),"held_out_auc":test_auc},open(sys.argv[2].replace(".csv","_test_auc.json"),"w"),indent=1)
