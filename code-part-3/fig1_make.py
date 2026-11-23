import os; os.makedirs("figures", exist_ok=True)
# Hand-laid SVG framework diagram: three phases, orthogonal connectors only.
W,H=1860,1010
PH=[(20,'#1f4e79','#e8f0f8','PHASE A  |  MOTSS target &amp; feature curation'),
    (660,'#7a4a12','#fbf1e4','PHASE B  |  Prediction &amp; PSO tuning'),
    (1300,'#2f5d32','#eaf4ea','PHASE C  |  Audit &amp; interpretation')]
PW=540
s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Liberation Serif, Times New Roman, serif">',
 '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker>',
 '<marker id="b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#1f4e79"/></marker></defs>',
 f'<rect width="{W}" height="{H}" fill="white"/>']
for x,c,bg,t in PH:
    s.append(f'<rect x="{x}" y="20" width="{PW}" height="{H-40}" rx="14" fill="{bg}" stroke="{c}" stroke-width="2"/>')
    s.append(f'<rect x="{x}" y="20" width="{PW}" height="54" rx="14" fill="{c}"/><rect x="{x}" y="50" width="{PW}" height="24" fill="{c}"/>')
    s.append(f'<text x="{x+PW/2}" y="55" text-anchor="middle" font-size="25" font-weight="bold" fill="white">{t}</text>')
BW,BH=460,142
import re
fmt=lambda t: re.sub(r'_\[([^\]]+)\]', r'<tspan baseline-shift="sub" font-size="14">\1</tspan>', t)
def box(cx,y,title,lines,c,h=BH,strong=False):
    x=cx-BW/2
    s.append(f'<rect x="{x}" y="{y}" width="{BW}" height="{h}" rx="8" fill="white" stroke="{c}" stroke-width="{3 if strong else 1.8}"/>')
    s.append(f'<text x="{cx}" y="{y+34}" text-anchor="middle" font-size="23" font-weight="bold" fill="{c}">{title}</text>')
    for i,l in enumerate(lines): s.append(f'<text x="{cx}" y="{y+66+i*27}" text-anchor="middle" font-size="20" fill="#222">{fmt(l)}</text>')
    return (cx,y,h)
def vline(cx,y1,y2,col='#333',m='a',dash=''):
    s.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2-2}" stroke="{col}" stroke-width="2.4" marker-end="url(#{m})" {dash}/>')
Y=[100,325,550,775]
cA,cB,cC=20+PW/2,660+PW/2,1300+PW/2
CA,CB,CC='#1f4e79','#7a4a12','#2f5d32'
a=[box(cA,Y[0],'Raw multi-source panel',['BPS headcounts, IPCC factors, ERA5','774 records, 9 regencies, 2021-2023','30 columns'],CA),
   box(cA,Y[1],'MOTSS scoring',['19 candidates x 11 regressors = 209 runs','MOTSS = (S_[ind] + S_[perf] + S_[weak] + S_[Δ]) / 4','four-tier recommendation'],CA,strong=True),
   box(cA,Y[2],'MOTSS decisions',['Y_[1] = daily biogas potential (fixed)','deterministic aggregates flagged','(linear R² = 1.000, MOTSS &lt; 0.40)'],CA,strong=True),
   box(cA,Y[3],'Predictor redundancy screen',['R²(x_[j] | other predictors) ≥ 0.999','feature sets F1 (35), F2 (31), F3 (27)'],CA,h=118)]
b=[box(cB,Y[0],'Preprocessing',['one-hot regency, IQR capping,','min-max scaling, ln(1 + y) target','80:20 split, seed 42'],CB),
   box(cB,Y[1],'Stage 1: baseline screening',['11 regressors (linear, KNN, CART,','RF, GBM, XGBoost, LightGBM, CatBoost)','K-fold CV, K ∈ {5, ..., 10}'],CB),
   box(cB,Y[2],'Stage 2: PSO tuning',['10 particles x 20 iterations','ω = 0.7, c_[1] = c_[2] = 1.5','fitness: 5-fold real-scale R²'],CB),
   box(cB,Y[3],'Model selection',['highest PSO cross-validated R²','(test set never used for selection)'],CB,h=118)]
c=[box(cC,Y[0],'Stage 3: statistical evaluation',['bootstrap 95% CI of test R²','5 x 5 repeated CV','Wilcoxon and Friedman tests'],CC),
   box(cC,Y[1],'Stage 4: leakage-aware audit',['F1 / F2 / F3 ablation','temporal 2021-22 → 2023, leave-one-regency-out','capping audit + PSO re-tuning'],CC),
   box(cC,Y[2],'Stage 5: interpretability',['SHAP, MDI, permutation importance','Kendall τ agreement'],CC),
   box(cC,Y[3],'Decision support',['routine update (capped model)','hotspot ranking (uncapped model)'],CC,h=118)]
for col,cx,C in [(a,cA,'#333'),(b,cB,'#333'),(c,cC,'#333')]:
    for i in range(3): vline(cx,col[i][1]+col[i][2],col[i+1][1])
# phase links through gutters (gutter A|B centre 600, B|C centre 1240)
def link(src,dst,gx):
    sx,sy,sh=src; dx,dy,dh=dst
    s.append(f'<path d="M{sx+BW/2},{sy+sh/2} H{gx} V{dy+dh/2} H{dx-BW/2-2}" fill="none" stroke="#333" stroke-width="2.4" marker-end="url(#a)"/>')
link(a[3],b[0],588); link(b[3],c[0],1228)
yy=a[2][1]+a[2][2]/2; yb=H-48; ys=c[1][1]+c[1][2]/2
s.append(f'<path d="M{cA+BW/2},{yy} H612 V{yb} H1252 V{ys} H{cC-BW/2-2}" fill="none" stroke="#1f4e79" stroke-width="2.4" stroke-dasharray="10 6" marker-end="url(#b)"/>')
s.append(f'<text x="{cB}" y="{yb-12}" text-anchor="middle" font-size="19" font-style="italic" fill="#1f4e79">MOTSS flags define the leakage-controlled feature set F2</text>')
s.append('</svg>'); open('figures/Fig1_framework.svg','w').write('\n'.join(s))
