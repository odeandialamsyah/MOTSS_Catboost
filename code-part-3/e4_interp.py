import numpy as np, pandas as pd, shap
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, make_scorer
from scipy.stats import kendalltau, spearmanr
from common import *
X,y,year,district,raw=load()
idx=np.arange(len(y)); itr,ite=train_test_split(idx,test_size=0.2,random_state=42)
sc=make_scorer(lambda yt,pl: r2_score(yt,np.expm1(pl)))
out={}
for tag,cols in [('F1',list(X.columns)),('F2',[c for c in X.columns if c not in DERIVED])]:
    Xtr,Xte,ytr,yte=X.iloc[itr][cols],X.iloc[ite][cols],y.iloc[itr],y.iloc[ite]
    m=make_model('CatBoost'); m.fit(Xtr,np.log1p(ytr))
    sv=shap.TreeExplainer(m).shap_values(Xte)
    shap_imp=np.abs(sv).mean(0)
    mdi=m.get_feature_importance()
    pi=permutation_importance(m,Xte,yte,n_repeats=10,random_state=42,scoring=sc)
    df=pd.DataFrame({'feature':cols,'SHAP':shap_imp,'MDI':mdi,'PI':pi.importances_mean,'PI_sd':pi.importances_std})
    df=df.sort_values('SHAP',ascending=False); df.to_csv(f'e4_importance_{tag}.csv',index=False)
    nz=df[(df.SHAP>0)|(df.MDI>0)|(df.PI.abs()>1e-6)]
    print(tag, 'nonzero',len(nz)); print(df.head(12).round(4).to_string())
    for a,b in [('SHAP','MDI'),('SHAP','PI'),('MDI','PI')]:
        t=kendalltau(nz[a],nz[b]); s=spearmanr(nz[a],nz[b]); print(f'  {a}-{b}: tau={t.statistic:.3f} p={t.pvalue:.2g}; rho={s.statistic:.3f}')
    top=lambda c:set(df.sort_values(c,ascending=False).feature.head(5))
    print('  top5 overlap SHAP/MDI/PI:', top('SHAP')&top('MDI')&top('PI'))
    np.save(f'shap_{tag}.npy',sv); Xte.to_pickle(f'Xte_{tag}.pkl')
