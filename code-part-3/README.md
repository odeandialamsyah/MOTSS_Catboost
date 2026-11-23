# code-part-3 — Statistical, robustness and interpretability analyses

Supporting code for the article *"MOTSS-Guided PSO-Tuned Ensembles for Leakage-Aware Prediction of Livestock Biogas Potential in West Java"* (submitted to JOIV).

- `code-part-1/kandidat_target_y.ipynb` — MOTSS target selection (Table I).
- `code-part-2/MOTSS_PSO_Y1.ipynb` — baseline screening and PSO tuning of the primary target Y1 (Table III).
- `code-part-3/` (this folder) — the additional analyses reported in the article.

`common.py` reproduces the preprocessing of `MOTSS_PSO_Y1.ipynb` (one-hot regency, IQR capping, min–max scaling, log1p target, 80:20 split with seed 42) and uses the PSO-optimal hyperparameters reported by that notebook. The dataset is read from `../dataset/dataset_pipeline_penelitian_utama.xlsx`.

## Setup

```bash
pip install -r requirements.txt
cd code-part-3
```

## Scripts and the article items they produce

| Script | Output | Article item |
|---|---|---|
| `e8_redundancy.py` | `e8_redundancy.csv`, `e8_redundancy_F2.csv` | Table II (predictor redundancy screen, Eq. 7) |
| `e1_repro.py` | `e1_repro.csv` | Table III (re-run of default vs PSO models) |
| `e2_validation.py` | `e2_cv_folds.csv`, `e2_cv_summary.csv`, `e2_boot.csv` | Table III bootstrap CIs, Table IV, Fig. 3 |
| `e3_robust.py` | `e3_robust.csv` | Table V, Fig. 4 (F1–F3, temporal 2021–22 → 2023, LORO) |
| `e5_nocap.py` | `e5_nocap.csv` | Table VI (capping audit) |
| `e6_pso_nocap.py` | `e6_pso_nocap_<model>.json` | Table VI (PSO re-tuning without capping; ~20–40 min) |
| `e7_nocap_eval.py` | `e7_nocap_eval.csv`, `figures/FigS3_pso_convergence.png` | Table VI (bootstrap CI, hotspot MAPE) |
| `e4_interp.py` | `e4_importance_F1.csv`, `e4_importance_F2.csv` | Fig. 5 (SHAP, MDI, PI, Kendall τ) |
| `figs.py` | `figures/Fig2_*.png` … `figures/Fig5_*.png` | Fig. 2–5 (run after e2, e3 and e4) |
| `fig1_make.py` | `figures/Fig1_framework.svg` | Fig. 1 |

Recommended order: `e8 → e1 → e2 → e3 → e4 → e5 → e6 → e7 → figs → fig1_make`.

Results may differ in the third decimal across library versions and hardware.
