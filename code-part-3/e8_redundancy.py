"""Table II: predictor-side redundancy screen R^2(x_j | other predictors), Eq. (7)."""
import pandas as pd
from sklearn.linear_model import LinearRegression
from common import DATA
d = pd.read_excel(DATA)
def screen(cols):
    X = d[cols].astype(float); rows = []
    for c in cols:
        o = [k for k in cols if k != c]
        rows.append((c, LinearRegression().fit(X[o], X[c]).score(X[o], X[c]), abs(X[c].corr(d.Potential_biogas_day))))
    return pd.DataFrame(rows, columns=['predictor', 'R2_given_others', 'abs_rho_Y1']).sort_values('R2_given_others', ascending=False)
num = [c for c in d.columns if c not in ['year', 'district', 'subdistrict', 'Potential_biogas_day'] and d[c].nunique() > 1]
full = screen(num); full.to_csv('e8_redundancy.csv', index=False); print(full.round(5).to_string())
f2 = screen([c for c in num if 'manure_potential' not in c]); f2.to_csv('e8_redundancy_F2.csv', index=False)
print('\nWithout derived manure aggregates (F2):'); print(f2.head(6).round(4).to_string())
