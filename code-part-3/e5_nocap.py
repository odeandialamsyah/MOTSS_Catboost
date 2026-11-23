import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import common
from common import *
# variant: no IQR capping at all (features and target in original scale)
orig=common.iqr_limits
common.iqr_limits=lambda s:(None,None)
Xn,yn,year,district,raw=common.load()
common.iqr_limits=orig
Xc,yc,_,_,_=common.load()   # notebook: everything capped
idx=np.arange(len(yn)); itr,ite=train_test_split(idx,test_size=0.2,random_state=42)
yraw=raw['Potential_biogas_day'].values
rows=[]
for n in ['RF','GBM','XGBoost','LightGBM','CatBoost']:
    for tag,X,ytrain in [('no capping',Xn,yn),('capped (notebook)',Xc,yc)]:
        m=make_model(n); m.fit(X.iloc[itr],np.log1p(ytrain.iloc[itr])); p=np.expm1(m.predict(X.iloc[ite]))
        for evtag,yt in [('vs raw target',yraw[ite]),('vs capped target',yc.values[ite])]:
            rows.append(dict(variant=tag,eval=evtag,model=n,R2=r2_score(yt,p),RMSE=mean_squared_error(yt,p)**.5,MAE=mean_absolute_error(yt,p)))
r=pd.DataFrame(rows); r.to_csv('e5_nocap.csv',index=False); print(r.round(4).to_string())
print('n test extreme', (yraw[ite]>189.64).sum(), len(ite))
