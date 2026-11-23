import numpy as np, pandas as pd, time
from sklearn.model_selection import train_test_split, RepeatedKFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import wilcoxon, friedmanchisquare
from common import *
MODELS=list(PSO_PARAMS)
def ev(m,Xa,ya,Xb,yb):
    m.fit(Xa,np.log1p(ya)); p=np.expm1(m.predict(Xb)); return r2_score(yb,p),mean_squared_error(yb,p)**.5,mean_absolute_error(yb,p),p

X,y,year,district,raw=load()
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)

# ---- E2a repeated 5x5 CV default vs PSO (training partition, real-scale R2)
rkf=RepeatedKFold(n_splits=5,n_repeats=5,random_state=7)
folds=list(rkf.split(Xtr))
rec=[]
for n in MODELS:
    for tuned in [False,True]:
        for i,(a,b) in enumerate(folds):
            r2,rm,ma,_=ev(make_model(n,tuned),Xtr.iloc[a],ytr.iloc[a],Xtr.iloc[b],ytr.iloc[b])
            rec.append(dict(model=n,setting='PSO' if tuned else 'Default',fold=i,R2=r2,RMSE=rm,MAE=ma))
    print('cv',n,flush=True)
cv=pd.DataFrame(rec); cv.to_csv('e2_cv_folds.csv',index=False)
summ=[]
for n in MODELS:
    d=cv[(cv.model==n)&(cv.setting=='Default')].R2.values; p=cv[(cv.model==n)&(cv.setting=='PSO')].R2.values
    try: st=wilcoxon(p,d); pv=st.pvalue
    except Exception: pv=np.nan
    summ.append(dict(model=n,def_mean=d.mean(),def_sd=d.std(ddof=1),pso_mean=p.mean(),pso_sd=p.std(ddof=1),delta=(p-d).mean(),wins=int((p>d).sum()),p=pv))
s=pd.DataFrame(summ); print(s.round(4).to_string()); s.to_csv('e2_cv_summary.csv',index=False)
# Friedman across the 5 tree ensembles (PSO)
ens=['RF','GBM','XGBoost','LightGBM','CatBoost']
mat=np.array([cv[(cv.model==n)&(cv.setting=='PSO')].R2.values for n in ens])
print('Friedman ensembles PSO',friedmanchisquare(*mat))
for n in ens:
    if n!='CatBoost': print('CatBoost vs',n,wilcoxon(mat[ens.index('CatBoost')],mat[ens.index(n)]).pvalue, (mat[ens.index('CatBoost')]-mat[ens.index(n)]).mean())
# ---- bootstrap CI test
rng=np.random.default_rng(42); boot=[]
for n in MODELS:
    _,_,_,p=ev(make_model(n,True),Xtr,ytr,Xte,yte)
    bs=[r2_score(yte.values[ix],p[ix]) for ix in (rng.integers(0,len(yte),len(yte)) for _ in range(1000))]
    boot.append(dict(model=n,R2=r2_score(yte,p),lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5)))
b=pd.DataFrame(boot); print(b.round(4).to_string()); b.to_csv('e2_boot.csv',index=False)
