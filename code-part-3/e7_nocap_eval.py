import numpy as np, pandas as pd, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import common
from common import *
common.iqr_limits=lambda s:(None,None)
X,y,_,_,_=common.load(); idx=np.arange(len(y)); itr,ite=train_test_split(idx,test_size=0.2,random_state=42)
yt=y.iloc[ite].values; hot=yt>189.6366; rng=np.random.default_rng(42); rows=[]; hist={}
for n in ['GBM','XGBoost','CatBoost','LightGBM']:
    d=json.load(open(f'e6_pso_nocap_{n}.json')); hist[n]=d['history']
    m=base_models()[n].set_params(**d['best_params']); m.fit(X.iloc[itr],np.log1p(y.iloc[itr])); p=np.expm1(m.predict(X.iloc[ite]))
    bs=[r2_score(yt[i],p[i]) for i in (rng.integers(0,len(yt),len(yt)) for _ in range(1000))]
    rows.append(dict(model=n,params=d['best_params'],cv=d['cv_r2'],R2=r2_score(yt,p),lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5),RMSE=mean_squared_error(yt,p)**.5,MAE=mean_absolute_error(yt,p),
        MAPE_hot=np.mean(np.abs(p[hot]-yt[hot])/yt[hot])*100, R2_hot=r2_score(yt[hot],p[hot]), MAE_bulk=mean_absolute_error(yt[~hot],p[~hot]),time=d['time_s']))
r=pd.DataFrame(rows); pd.set_option('display.width',250); print(r.round(4).to_string()); r.to_csv('e7_nocap_eval.csv',index=False)
# capped-protocol CatBoost on hotspots (vs raw)
Xc,yc,_,_,raw=load(); m=make_model('CatBoost'); m.fit(Xc.iloc[itr],np.log1p(yc.iloc[itr])); p=np.expm1(m.predict(Xc.iloc[ite])); yr=raw.Potential_biogas_day.values[ite]
print('capped CatBoost hot MAPE',np.mean(np.abs(p[hot]-yr[hot])/yr[hot])*100,'max pred',p.max(),'n hot',hot.sum())
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif'],'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(3.4,2.5))
for n,c in zip(hist,['#2a6f97','#e07a2f','#6a994e','#8d5a97']): ax.plot(range(1,21),hist[n],marker='o',ms=3,label=n,color=c)
ax.set_xlabel('PSO iteration'); ax.set_ylabel('Best 5-fold real-scale $R^2$'); ax.legend(frameon=False,fontsize=7); ax.set_xticks([1,5,10,15,20])
plt.tight_layout(); plt.savefig('figures/FigS3_pso_convergence.png',dpi=300)
