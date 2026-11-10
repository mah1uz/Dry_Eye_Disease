import sys, json, numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.formula.api import logit
from scipy import stats
d=pd.read_csv(sys.argv[1])
d.columns=[c.strip() for c in d.columns]
y=(d['Dry Eye Disease']=='Y').astype(int)
bp=d['Blood pressure'].str.split('/',expand=True).astype(float)
X=pd.DataFrame({'y':y,'screen':d['Average screen time'],'age':d['Age'],
 'male':(d['Gender']=='M').astype(int),'sleep':d['Sleep duration'],'sbp':bp[0],'dbp':bp[1],
 'bmi':d['Weight']/(d['Height']/100)**2})
for k,c in {'sd':'Sleep disorder','wake':'Wake up during night','sleepy':'Feel sleepy during day','caf':'Caffeine consumption','alc':'Alcohol consumption','smk':'Smoking','med':'Medical issue','meds':'Ongoing medication','dev':'Smart device before bed','strain':'Discomfort Eye-strain','red':'Redness in eye','itch':'Itchiness/Irritation in eye'}.items():
    X[k]=(d[c]=='Y').astype(int)
out={}
out['n']=len(X); out['ded']=int(y.sum()); out['ded_pct']=round(100*y.mean(),1)
out['female_ded']=int(((d.Gender=='F')&(y==1)).sum()); out['strain_yes_ded']=int(((X.strain==1)&(y==1)).sum())
out['screen_mean']=[round(X.screen.mean(),2),round(X.screen.std(),2),X.screen.min(),X.screen.max()]
noc='age male sleep sbp dbp bmi sd wake sleepy caf alc smk med meds dev'.split()
oc=['strain','red','itch']
def fit(f,extra=''):
    m=logit('y ~ '+f,X).fit(disp=0); return m
def row(m,v='screen'):
    b=m.params[v]; lo,hi=m.conf_int().loc[v]; return [round(float(np.exp(b)),3),round(float(np.exp(lo)),3),round(float(np.exp(hi)),3),float(m.pvalues[v])]
out['M1']=row(fit('screen'))
out['M2']=row(fit('screen+age+male'))
out['M3_with_ocular']=row(fit('screen+'+'+'.join(noc+oc)))
out['M3_no_ocular_terms']=row(fit('screen+'+'+'.join(noc)))
# ocular only effect
m=fit('screen+'+'+'.join(noc+oc)); out['ocular_ORs']={v:round(float(np.exp(m.params[v])),2) for v in oc}
# tertiles
X['tert']=pd.qcut(X.screen,3,labels=['low','mid','high'])
mt=logit('y ~ C(tert, Treatment("low")) + age+male',X).fit(disp=0)
out['tertile_M2']={k:[round(float(np.exp(mt.params[k])),3),float(mt.pvalues[k])] for k in mt.params.index if 'tert' in k}
# within ocular strata: any symptom vs none
X['anyoc']=(X[oc].sum(axis=1)>0).astype(int)
for g,s in X.groupby('anyoc'):
    r=row(logit('y ~ screen',s).fit(disp=0)); out[f'screen_within_anyoc={g}']=[len(s)]+r
# screen vs ocular symptoms (is screen time associated with the symptoms?)
for v in oc:
    r=row(logit(f'{v} ~ screen',X).fit(disp=0)); out[f'screen_vs_{v}']=r
# sex interaction
mi=logit('y ~ screen*male + age',X).fit(disp=0); out['sex_interaction_p']=float(mi.pvalues['screen:male'])
out['screen_by_sex']={g:row(logit('y ~ screen+age',s).fit(disp=0)) for g,s in X.groupby('male')}
# nonlinearity: spline (natural cubic, df=4) LR test, M3 adj
f3='+'.join(noc+oc)
m0=logit('y ~ screen+'+f3,X).fit(disp=0); m1=logit('y ~ cr(screen,df=4)+'+f3,X).fit(disp=0)
lr=2*(m1.llf-m0.llf); out['spline_LR_p_adj']=float(stats.chi2.sf(lr,m1.df_model-m0.df_model))
ms0=logit('y ~ screen',X).fit(disp=0); ms1=logit('y ~ cr(screen,df=4)',X).fit(disp=0)
out['spline_LR_p_unadj']=float(stats.chi2.sf(2*(ms1.llf-ms0.llf),ms1.df_model-ms0.df_model))
# AUC of screen alone & added value
from sklearn.metrics import roc_auc_score
out['auc_screen_alone']=round(roc_auc_score(y,X.screen),4)
# permutation null for the OR p-value: shuffle screen 2000 times
rng=np.random.default_rng(42); Xs=X.copy(); ps=[]
obs=m.params.get('screen')
b0=logit('y ~ screen+'+f3,X).fit(disp=0).params['screen']
cnt=0;N=500
for i in range(N):
    Xs['screen']=rng.permutation(X.screen.values)
    bb=logit('y ~ screen+'+f3,Xs).fit(disp=0).params['screen']
    cnt+= abs(bb)>=abs(b0)
out['perm_p_M3']=(cnt+1)/(N+1)
# binary-column check of balance: DED rate by screen decile
X['dec']=pd.qcut(X.screen,10,duplicates='drop')
out['ded_rate_by_decile']=[round(v,3) for v in X.groupby('dec',observed=True).y.mean()]
json.dump(out,open(sys.argv[2],'w'),indent=1,default=str)
print(json.dumps(out,indent=1,default=str))
