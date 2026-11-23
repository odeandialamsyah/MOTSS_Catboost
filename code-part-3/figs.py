import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, shap
from sklearn.model_selection import train_test_split
from common import *
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
C1,C2,C3='#2a6f97','#e07a2f','#6a994e'
raw=pd.read_excel(DATA); y=raw.Potential_biogas_day
# Fig2 target distribution
fig,ax=plt.subplots(1,3,figsize=(7.2,2.05))
ax[0].hist(y,bins=40,color=C1); ax[0].axvline(189.64,ls='--',c=C2,lw=1); ax[0].set_xlabel('Daily biogas potential (m$^3$/day)'); ax[0].set_ylabel('Frequency'); ax[0].set_title('(a) Raw target, skew = 4.77')
ax[0].text(260,330,'IQR cap\n(189.6)',color=C2,fontsize=7)
ax[1].hist(np.log1p(y),bins=40,color=C1); ax[1].set_xlabel('log(1 + potential)'); ax[1].set_title('(b) log1p-transformed target')
g=raw.groupby('year').Potential_biogas_day
ax[2].bar(np.arange(3)-0.18,g.mean(),0.36,color=C1,label='Mean'); ax[2].bar(np.arange(3)+0.18,g.median(),0.36,color=C2,label='Median')
ax[2].set_xticks(range(3),['2021','2022','2023']); ax[2].set_ylabel('m$^3$/day'); ax[2].set_title('(c) Target by year'); ax[2].legend(frameon=False,fontsize=7)
plt.tight_layout(); plt.savefig('figures/Fig2_target.png',dpi=300); plt.close()
# Fig3 repeated CV default vs PSO
cv=pd.read_csv('e2_cv_folds.csv'); ms=['Lasso','ElasticNet','KNN','CART','RF','GBM','XGBoost','LightGBM','CatBoost']
fig,ax=plt.subplots(figsize=(7.2,2.0)); pos=np.arange(len(ms))
for off,st,c in [(-0.18,'Default',C1),(0.18,'PSO',C2)]:
    data=[cv[(cv.model==m)&(cv.setting==st)].R2.values for m in ms]
    bp=ax.boxplot(data,positions=pos+off,widths=0.32,patch_artist=True,showfliers=False,medianprops=dict(color='k'))
    for b in bp['boxes']: b.set_facecolor(c)
ax.set_xticks(pos,ms); ax.set_ylabel('Real-scale $R^2$ (25 folds)'); ax.set_ylim(-0.45,1.02)
ax.legend([plt.Rectangle((0,0),1,1,fc=C1),plt.Rectangle((0,0),1,1,fc=C2)],['Default','PSO-tuned'],frameon=False,loc='lower right')
plt.tight_layout(); plt.savefig('figures/Fig3_repeated_cv.png',dpi=300); plt.close()
# Fig4 actual vs predicted
X,yy,year,district,_=load(); idx=np.arange(len(yy)); itr,ite=train_test_split(idx,test_size=0.2,random_state=42)
m=make_model('CatBoost'); m.fit(X.iloc[itr],np.log1p(yy.iloc[itr])); p=np.expm1(m.predict(X.iloc[ite])); t=yy.iloc[ite].values
sl,ic=np.polyfit(t,p,1)
fig,ax=plt.subplots(figsize=(3.4,3.0)); ax.scatter(t,p,s=10,alpha=.7,color=C1,label='Test samples (n = 155)')
mx=max(t.max(),p.max()); ax.plot([0,mx],[0,mx],'k--',lw=1,label='Ideal y = x'); xs=np.linspace(0,mx,50); ax.plot(xs,sl*xs+ic,color=C2,lw=1.2,label=f'Fit y = {sl:.3f}x + {ic:.3f}')
ax.set_xlabel('Actual (capped) potential, m$^3$/day'); ax.set_ylabel('Predicted potential, m$^3$/day'); ax.legend(frameon=False,fontsize=7); plt.tight_layout(); plt.savefig('figures/FigS1_actual_vs_predicted.png',dpi=300); plt.close()
print('fit',sl,ic)
# Fig5 SHAP beeswarm F1
sv=np.load('shap_F1.npy'); Xte=pd.read_pickle('Xte_F1.pkl'); NN={'total_ekor_ayam_pedaging':'Broiler headcount','total_ekor_sapi_perah':'Dairy cattle headcount','total_ekor_sapi_daging':'Beef cattle headcount','total_ekor_ayam_petelur':'Layer headcount','total_manure_potential_day_chicken_kg':'Chicken manure (kg/day)','total_manure_potential_year_chicken_ton':'Chicken manure (t/yr)','total_manure_potential_day_cattle_kg':'Cattle manure (kg/day)','total_manure_potential_year_cattle_ton':'Cattle manure (t/yr)','district_kabupaten_purwakarta':'Regency: Purwakarta','usd_idr_avg':'USD/IDR rate','humidity':'Humidity','average_daily_rainfall':'Avg. daily rainfall'}; Xte=Xte.rename(columns=NN)
shap.summary_plot(sv,Xte,max_display=12,show=False,plot_size=(5.2,3.8)); plt.tight_layout(); plt.savefig('figures/FigS2_shap_summary.png',dpi=300); plt.close()
# Fig6 importance agreement F1/F2
NICE={'total_ekor_ayam_pedaging':'Broiler headcount','total_ekor_sapi_perah':'Dairy cattle headcount','total_ekor_sapi_daging':'Beef cattle headcount','total_ekor_ayam_petelur':'Layer headcount','total_manure_potential_day_chicken_kg':'Chicken manure (kg/day)','total_manure_potential_year_chicken_ton':'Chicken manure (t/yr)','total_manure_potential_day_cattle_kg':'Cattle manure (kg/day)','total_manure_potential_year_cattle_ton':'Cattle manure (t/yr)','average_temperature_per_district':'Avg. temperature','non_industrial_feed':'Non-industrial feed','average_daily_rainfall':'Avg. daily rainfall','humidity':'Humidity'}
fig,axs=plt.subplots(1,2,figsize=(7.2,2.45))
for ax,tag,ttl in [(axs[0],'F1','(a) F1: full feature set'),(axs[1],'F2','(b) F2: derived manure removed')]:
    d=pd.read_csv(f'e4_importance_{tag}.csv').head(8).iloc[::-1]
    for k in ['SHAP','MDI','PI']: d[k+'n']=d[k].clip(lower=0)/d[k].clip(lower=0).max()
    yy2=np.arange(len(d))
    for off,k,c in [(-0.27,'SHAP',C1),(0,'MDI',C2),(0.27,'PI',C3)]: ax.barh(yy2+off,d[k+'n'],0.26,color=c,label=k)
    ax.set_yticks(yy2,[NICE.get(f,f.replace('_',' ')) for f in d.feature]); ax.set_xlabel('Normalized importance'); ax.set_title(ttl,fontsize=9)
axs[1].legend(frameon=False,fontsize=7,loc='lower right'); plt.tight_layout(); plt.savefig('figures/Fig5_importance_F1_F2.png',dpi=300); plt.close()
# Fig7 robustness
r=pd.read_csv('e3_robust.csv'); mods=['RF','GBM','XGBoost','LightGBM','CatBoost']
fig,axs=plt.subplots(1,3,figsize=(7.2,2.25),sharey=True)
for ax,exp,ttl in [(axs[0],'ablation_random','(a) Random 80/20 split'),(axs[1],'temporal_2023','(b) Temporal: 2021-22 to 2023'),(axs[2],'LODO','(c) Leave-one-regency-out')]:
    fss=r[r.exp==exp].fs.unique(); w=0.8/len(fss)
    for j,(fs,c) in enumerate(zip(fss,[C1,C2,C3])):
        mm=[m for m in mods if not (exp=='LODO' and m=='GBM')]
        v=[r[(r.exp==exp)&(r.fs==fs)&(r.model==m)].R2.values[0] for m in mm]
        ax.bar(np.arange(len(mm))+(j-(len(fss)-1)/2)*w,np.clip(v,-0.2,None),w,color=c,label=fs.split(' (')[0])
    ax.set_xticks(range(len(mm)),mm,rotation=35,fontsize=7.5); ax.set_title(ttl,fontsize=9); ax.axhline(0,c='k',lw=.6); ax.set_ylim(-0.2,1)
axs[0].set_ylabel('Real-scale $R^2$'); h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,frameon=False,fontsize=7.5,loc='upper center',ncol=3)
plt.tight_layout(rect=(0,0,1,0.88)); plt.savefig('figures/Fig4_robustness.png',dpi=300); plt.close()
