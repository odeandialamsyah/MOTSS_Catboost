"""Re-run the notebook PSO (10 particles, 20 iterations, w=0.7, c1=c2=1.5, 5-fold, real-scale R2)
on the variant without IQR capping (features and target in original scale)."""
import numpy as np, pandas as pd, time, json, sys
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import common
from common import *
SPACE={"GBM":{"n_estimators":(50,300,int),"learning_rate":(0.01,0.2,float),"max_depth":(2,10,int)},
 "XGBoost":{"n_estimators":(50,300,int),"learning_rate":(0.01,0.2,float),"max_depth":(3,12,int),"subsample":(0.5,1.0,float)},
 "LightGBM":{"n_estimators":(50,300,int),"learning_rate":(0.01,0.2,float),"num_leaves":(31,255,int),"max_depth":(-1,15,int)},
 "CatBoost":{"depth":(4,10,int),"learning_rate":(0.01,0.2,float),"n_estimators":(100,500,int)}}
def to_params(pos,space,name):
    p={}
    for v,(k,(lo,hi,t)) in zip(pos,space.items()):
        v=int(round(v)) if t is int else float(v); v=max(lo,min(hi,v))
        if name=="LightGBM" and k=="max_depth" and v<=0: v=-1
        p[k]=v
    return p
def scorer(est,X,ylog): return r2_score(np.expm1(ylog),np.expm1(est.predict(X)))
def pso(name,X,ylog,n_particles=10,n_iter=20,w=.7,c1=1.5,c2=1.5,seed=42):
    space=SPACE[name]; np.random.seed(seed); lo=np.array([v[0] for v in space.values()],float); hi=np.array([v[1] for v in space.values()],float)
    pos=lo+(hi-lo)*np.random.rand(n_particles,len(space)); vel=np.zeros_like(pos)
    pb=pos.copy(); pbs=np.full(n_particles,-np.inf); gb=pos[0].copy(); gbs=-np.inf; kf=KFold(5,shuffle=True,random_state=42); hist=[]
    for it in range(n_iter):
        for i in range(n_particles):
            m=base_models()[name]; m.set_params(**to_params(pos[i],space,name))
            s=np.nanmean(cross_val_score(m,X,ylog,cv=kf,scoring=scorer))
            if s>pbs[i]: pbs[i]=s; pb[i]=pos[i].copy()
            if s>gbs: gbs=s; gb=pos[i].copy()
        hist.append(gbs)
        r1,r2=np.random.rand(),np.random.rand()
        vel=w*vel+c1*r1*(pb-pos)+c2*r2*(gb-pos); pos=np.clip(pos+vel,lo,hi)
    return to_params(gb,space,name),gbs,hist
common.iqr_limits=lambda s:(None,None)
X,y,year,district,raw=common.load()
idx=np.arange(len(y)); itr,ite=train_test_split(idx,test_size=0.2,random_state=42)
Xtr,Xte,ytr,yte=X.iloc[itr],X.iloc[ite],y.iloc[itr],y.iloc[ite]
names=sys.argv[1:] or ["CatBoost","XGBoost","LightGBM","GBM"]
for n in names:
    t=time.time(); bp,cvs,hist=pso(n,Xtr,np.log1p(ytr)); el=time.time()-t
    rows=[]
    for tag,m in [('Default',base_models()[n]),('PSO',base_models()[n].set_params(**bp))]:
        m.fit(Xtr,np.log1p(ytr)); p=np.expm1(m.predict(Xte))
        rows.append(dict(model=n,setting=tag,R2=r2_score(yte,p),RMSE=mean_squared_error(yte,p)**.5,MAE=mean_absolute_error(yte,p)))
    res=dict(model=n,best_params=bp,cv_r2=cvs,history=hist,time_s=el,test=rows)
    json.dump(res,open(f'e6_pso_nocap_{n}.json','w'),indent=1,default=float); print(json.dumps(res,default=float),flush=True)
