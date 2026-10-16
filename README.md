---
title: ChurnGuard
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# ChurnGuard: Customer Churn Prediction & Retention Tool

Predicts which telecom customers are likely to churn, **why**, and which ones are worth an offer.
Built on imbalanced data with a cost-based decision threshold and per-customer SHAP explanations.

- **Live demo:** `<ADD HUGGING FACE SPACE URL>`
- **Demo video (2-3 min):** `<ADD LOOM / YOUTUBE URL>`
- **Code:** this repository

## 1. Problem
Subscription companies lose revenue when customers leave silently. A retention team cannot call everyone, so it needs
(1) a ranked list of at-risk customers, (2) the reason for each, and (3) a rule for how many to contact so that the offers
cost less than the revenue they save.

## 2. Dataset
IBM / Kaggle **Telco Customer Churn**: 7,043 customers, 21 columns, **26.5% churn** (imbalanced).
Cleaning: `TotalCharges` is stored as text and has 11 blank values (all new customers with `tenure = 0`); they are
filled with 0 inside the pipeline. `customerID` is dropped.
There is **no support-calls column** in this dataset (only `TechSupport` yes/no), so support calls could not be used.

## 3. Approach
1. **EDA** of churn drivers (contract, tenure, internet service, payment method, charges).
2. **Stratified 80/20 split.** The test set stays locked until the very end.
3. **Pipelines only:** `ColumnTransformer` (impute, scale, one-hot) + model in one scikit-learn `Pipeline`, so cross-validation
   re-learns every step per fold (no data leakage).
4. **Baseline + 3 models** (Logistic Regression, Random Forest, XGBoost) compared with **stratified 5-fold CV on PR-AUC**.
5. **Imbalance:** class weights vs SMOTE. SMOTE only ran **inside the CV loop** (imbalanced-learn `Pipeline`, training folds only).
6. **Tuning:** randomized search on PR-AUC, checked again on fresh folds.
7. **Calibration check** (sigmoid / isotonic, Brier score).
8. **Cost-based threshold** from out-of-fold predictions on the training data.
9. **SHAP** global and per-customer explanations, **final test evaluation (once)**, business impact report.

### Why not accuracy?
A model that always predicts "no churn" is **73.5% accurate** and catches **0** churners. Accuracy rewards ignoring the
minority class. The main metric is **PR-AUC (average precision)**, which is not inflated by the large "stay" class (the
no-skill baseline is the churn rate, 0.265). Recall and precision are reported at the chosen threshold, because they map
directly to business cost: recall is churners we reach, precision is offers that are not wasted.

## 4. Results

### Cross-validation (training set, stratified 5-fold, mean ± std)
Recall and precision are at the default threshold 0.5.

| Model | Imbalance handling | PR-AUC | ROC-AUC | Recall | Precision |
|---|---|---|---|---|---|
| Dummy | baseline | 0.265 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| LogReg | none | 0.662 ± 0.019 | 0.846 ± 0.013 | 0.544 ± 0.041 | 0.654 ± 0.027 |
| LogReg | smote | 0.658 ± 0.023 | 0.845 ± 0.013 | 0.789 ± 0.040 | 0.526 ± 0.018 |
| LogReg | weights | 0.660 ± 0.020 | 0.846 ± 0.012 | 0.801 ± 0.038 | 0.517 ± 0.019 |
| RandomForest | none | 0.655 ± 0.021 | 0.845 ± 0.009 | 0.496 ± 0.021 | 0.673 ± 0.028 |
| RandomForest | smote | 0.634 ± 0.027 | 0.840 ± 0.009 | 0.572 ± 0.021 | 0.621 ± 0.024 |
| RandomForest | weights | 0.654 ± 0.020 | 0.845 ± 0.008 | 0.731 ± 0.029 | 0.561 ± 0.010 |
| XGBoost | none | 0.664 ± 0.018 | 0.844 ± 0.010 | 0.522 ± 0.027 | 0.660 ± 0.029 |
| XGBoost | smote | 0.657 ± 0.024 | 0.844 ± 0.010 | 0.561 ± 0.021 | 0.637 ± 0.019 |
| XGBoost | weights | 0.663 ± 0.020 | 0.844 ± 0.010 | 0.777 ± 0.026 | 0.532 ± 0.014 |

- The three models are **statistically tied** on PR-AUC (about 0.655 to 0.664, std about 0.02). There is no clear winner.
- **Class weights** did not improve PR-AUC. They only moved the operating point (XGBoost recall 0.52 to 0.78, precision 0.66 to 0.53).
- **SMOTE** did not help either (equal or lower PR-AUC; clearly worst for Random Forest).
- Tie-break rule (written before the run): among tied models choose XGBoost (SHAP TreeExplainer, project brief), no resampling.
- Tuning: tuned XGBoost scored 0.668 vs 0.660 untuned on fresh folds (std about 0.027). That gain is **within noise**.
- **Calibration:** raw XGBoost probabilities were already calibrated (Brier 0.1334 raw, 0.1346 sigmoid, 0.1333 isotonic), so no
  calibration is applied. Isotonic produced step-shaped probabilities with many exact 0 and 1 values.

### Final model on the held-out test set (used once)
Model: XGBoost pipeline, no resampling, no calibration. Threshold **0.40**.

| Metric | At threshold 0.40 | At default 0.50 |
|---|---|---|
| PR-AUC | 0.658 | 0.658 |
| ROC-AUC | 0.847 | 0.847 |
| Precision | 0.598 | 0.676 |
| Recall | 0.652 | 0.524 |
| F1 | 0.624 | 0.590 |
| Brier score | 0.135 | 0.135 |

Test PR-AUC (0.658) matches the CV PR-AUC (0.664), so there is no sign of leakage or overfitting.

### Threshold decision (documented with numbers)
Business assumptions (**assumptions, not facts from the data**): customer value = `MonthlyCharges x 12`, one retention offer
costs **100**, **30%** of real churners stay after an offer.

Total cost = revenue lost from missed churners + offer costs + revenue still lost when an offer fails.

| Policy | Total cost, training (out-of-fold) | Total cost, test set |
|---|---|---|
| Target nobody | 1,342,991 | 326,579 |
| Target everyone | 1,503,494 | 369,505 |
| Default threshold 0.50 | 1,233,736 | 301,512 |
| **Cost-optimal threshold 0.40** | **1,223,686** | **300,936** |

- Targeting everyone costs **more** than targeting nobody, so a model is needed.
- Moving from 0.50 to 0.40 helps only a little: the cost curve is flat around its minimum.
- **The threshold depends heavily on the assumptions.** The best threshold ranged from 0.12 to 0.90 when offer cost (50 to 200)
  and offer success (20% to 50%) were varied (notebook section 18).
- Risk levels in the app: Low < 0.40, Medium 0.40 to 0.70, High >= 0.70.

### Business impact (test set, 1,409 customers, 374 real churners, revenue at risk about 326,600)

| Strategy | Targeted | Churners caught | Precision | Offer cost | Revenue saved | Net benefit | Net benefit if random |
|---|---|---|---|---|---|---|---|
| Top 10% by risk | 140 | 107 | 0.76 | 14,000 | 30,678 | 16,678 | -4,265 |
| Top 20% by risk | 281 | 190 | 0.68 | 28,100 | 52,515 | 24,415 | -8,561 |
| Top 30% by risk | 422 | 248 | 0.59 | 42,200 | 67,540 | 25,340 | -12,857 |
| Cost-optimal threshold (0.4) | 408 | 244 | 0.60 | 40,800 | 66,442 | 25,642 | -12,430 |

Targeting the top 20% by risk gives a net benefit of about 24,400. Picking the same number of customers at random would lose about 8,600.

### What drives churn (SHAP)
Global SHAP ranks a **month-to-month contract** first, then **tenure**, **no online security**, **monthly charges**,
**fiber optic internet**, **no tech support** and **electronic check payment**. This agrees with the EDA
(churn: 42.7% month-to-month vs 2.8% on a two-year contract; 52.9% in the first 6 months vs 9.5% after 48 months).
Every prediction in the app shows its **top 3 reasons**.

## 5. App
Gradio app with three tabs:
1. **Single customer:** enter details, get probability, risk level, decision, top 3 reasons and a suggested offer.
2. **Upload CSV:** the same for every row, an expected-impact table and a downloadable result file.
3. **Model & business report:** metrics, assumptions and the validated business impact table.

Screenshots (add yours to `reports/figures/`):

![EDA - contract](reports/figures/02_contract.png)
![SHAP summary](reports/figures/12_shap_summary.png)
![Cost vs threshold](reports/figures/10_threshold_cost.png)
![App - single customer](reports/figures/app_single.png)
![App - CSV upload](reports/figures/app_csv.png)

## 6. How to run
```bash
git clone <this repo> && cd churnguard
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                          # opens http://127.0.0.1:7860
pytest -q                                              # 9 tests
python -m src.train --out-dir models_retrained         # optional: retrain from the raw CSV
```
Library versions are pinned because the saved model (`joblib`) must be loaded with the same scikit-learn / xgboost versions
it was trained with (1.6.1, 3.4.1).

### Troubleshooting: model loading and library versions
A pickled scikit-learn / XGBoost model only works with (almost) the same library versions it was saved with. With other
versions it can crash with errors such as `input stream corrupted` or `Can't get attribute ...`, or, worse, load and return
**slightly different probabilities without any error**. `src/model_store.py` therefore compares the loaded model with reference
predictions stored in `models/artifacts.json` and, if they do not match or loading fails, **retrains the same pipeline from
`data/WA_Fn-UseC_-Telco-Customer-Churn.csv` at startup** (a few seconds). The "Model & business report" tab shows which case happened.
The most reliable setup is still a fresh virtual environment with `pip install -r requirements.txt`.

## 7. Repository structure
```
app.py                       Gradio app (entry point for Hugging Face Spaces)
requirements.txt             pinned versions
src/
  config.py                  paths + business assumptions (read from models/artifacts.json)
  pipeline.py                column groups, ColumnTransformer, final XGBoost pipeline
  data.py                    loading, cleaning, input validation
  business.py                cost function, threshold search, risk levels, expected revenue saved
  explain.py                 SHAP top-3 reasons per customer
  predict.py                 probability + risk level + decision + reasons
  model_store.py             safe model loading: version check + automatic retrain fallback
  train.py                   retrain the final model from the raw CSV
models/                      churn_model.joblib, churn_explain_pipeline.joblib, artifacts.json
data/                        raw Telco CSV, sample_customers.csv
notebooks/                   EDA, modelling, threshold and SHAP notebook
reports/                     cv_results.csv, business_impact.csv, figures/
tests/                       pytest tests
```

## 8. Limitations
- **Business numbers are assumptions.** Customer value, offer cost and offer success rate are not in the data. The best threshold
  and all money figures change a lot when they change. The offer-success rate in particular should come from a real A/B test.
- **Modest signal.** PR-AUC is about 0.66 and the three models are tied; better modelling is unlikely to help much without new features.
- **Missed churners look loyal.** At threshold 0.40, 130 of 374 test churners were missed. They have long tenure (28 months on average)
  and nothing in the data warns about them. There is no support-call, usage or complaint data.
- **One company, one snapshot.** No time dimension: we cannot check how the model behaves on later data (drift).
- **Predictions are not causes.** SHAP explains the model, not why customers really leave. Offer suggestions are simple rules, not a tested uplift model.
- **Probabilities were not recalibrated** because the raw model was already well calibrated on this data. They should be checked again on new data.
- Single train/test split plus CV on 7k rows: expect some variance (CV std about 0.02 PR-AUC).
- Uploaded CSVs are scored up to 5,000 rows in the demo.

## 9. Resume bullet
Developed a customer churn prediction tool (XGBoost, SHAP, Gradio) with cost-based threshold tuning for imbalanced data and per-customer explanations for retention teams.
