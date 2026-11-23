import numpy as np, pandas as pd, time
from sklearn.model_selection import train_test_split, KFold, cross_val_score, LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from common import *
TOP=['LR','KNN','RF','GBM','XGBoost','LightGBM','CatBoost']
def ev(m,Xa,ya,Xb,yb):
    m.fit(Xa,np.log1p(ya)); p=np.expm1(np.clip(m.predict(Xb),0,np.log1p(2*ya.max()))); return r2_score(yb,p),mean_squared_error(yb,p)**.5,mean_absolute_error(yb,p),p
X,y,year,district,raw=load()
HEAD=[c for c in X.columns if c.startswith('total_ekor')]
FS={'F1 full (35)':list(X.columns),
    'F2 no derived manure (31)':[c for c in X.columns if c not in DERIVED],
    'F3 no derived, no headcounts (27)':[c for c in X.columns if c not in DERIVED+HEAD]}
print({k:len(v) for k,v in FS.items()}); print('HEAD',HEAD)
idx=np.arange(len(y)); itr,ite=train_test_split(idx,test_size=0.2,random_state=42)
rows=[]
# ---- E3 feature-set ablation (random split)
for fs,cols in FS.items():
    for n in TOP:
        r2,rm,ma,_=ev(make_model(n),X.iloc[itr][cols],y.iloc[itr],X.iloc[ite][cols],y.iloc[ite])
        rows.append(dict(exp='ablation_random',fs=fs,model=n,R2=r2,RMSE=rm,MAE=ma))
print('ablation done',flush=True)
# ---- E4 temporal hold-out 2021-22 -> 2023 (caps & scaler fit on train years only)
mask=year<2023
Xt,yt,_,_,_=load(fit_mask=mask)
for fs,cols in FS.items():
    for n in TOP:
        r2,rm,ma,_=ev(make_model(n),Xt[mask][cols],yt[mask],Xt[~mask][cols],yt[~mask])
        rows.append(dict(exp='temporal_2023',fs=fs,model=n,R2=r2,RMSE=rm,MAE=ma))
print('temporal done',flush=True)
# ---- E5 leave-one-district-out
logo=LeaveOneGroupOut()
for fs,cols in list(FS.items())[:2]:
    for n in ['RF','XGBoost','LightGBM','CatBoost']:
        pred=np.zeros(len(y)); per=[]
        for a,b in logo.split(X,y,district):
            r2,_,_,p=ev(make_model(n),X.iloc[a][cols],y.iloc[a],X.iloc[b][cols],y.iloc[b]); pred[b]=p; per.append(r2)
        rows.append(dict(exp='LODO',fs=fs,model=n,R2=r2_score(y,pred),RMSE=mean_squared_error(y,pred)**.5,MAE=mean_absolute_error(y,pred),R2_median_district=np.median(per)))
print('lodo done',flush=True)
# ---- E6 uncapped target
Xu,yu,_,_,_=load(cap_target=False)
for n in ['RF','GBM','XGBoost','LightGBM','CatBoost']:
    r2,rm,ma,_=ev(make_model(n),Xu.iloc[itr],yu.iloc[itr],Xu.iloc[ite],yu.iloc[ite])
    rows.append(dict(exp='uncapped_target',fs='F1 full (35)',model=n,R2=r2,RMSE=rm,MAE=ma))
r=pd.DataFrame(rows); r.to_csv('e3_robust.csv',index=False); print(r.round(4).to_string())
