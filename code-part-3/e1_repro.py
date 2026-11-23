import numpy as np, pandas as pd, time, json
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from common import *
X,y,year,district,raw=load()
print(X.shape, y.max())
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
rows=[]
for tuned in [False,True]:
    for n in PSO_PARAMS:
        m=make_model(n,tuned); t=time.time(); m.fit(Xtr,np.log1p(ytr)); p=np.expm1(m.predict(Xte))
        rows.append(dict(setting='PSO' if tuned else 'Default',model=n,R2=r2_score(yte,p),RMSE=mean_squared_error(yte,p)**.5,MAE=mean_absolute_error(yte,p),sec=time.time()-t))
r=pd.DataFrame(rows); print(r.round(4).to_string()); r.to_csv('e1_repro.csv',index=False)
