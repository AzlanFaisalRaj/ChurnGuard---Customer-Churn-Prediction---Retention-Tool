"""Retrain the final model from the raw CSV and save new artifacts (reproduces the notebook).

Usage:  python -m src.train --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --out-dir models_retrained
Steps: clean -> stratified split -> out-of-fold probabilities (5-fold CV, training data only) -> cost-based threshold
       -> fit on the full training set -> evaluate ONCE on the test set -> save.
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from . import config as cfg
from .business import best_threshold, customer_value, total_cost
from .data import load_clean
from .pipeline import FEATURES, build_pipeline


def split_data(df):
    X, y = df[FEATURES], df[cfg.TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=cfg.RANDOM_STATE)


def fit_final_model(data_path):
    """Same split and same pipeline as the notebook; used by main() and by the startup fallback in model_store.py."""
    df = load_clean(data_path)
    X_tr, X_te, y_tr, y_te = split_data(df)
    return build_pipeline().fit(X_tr, y_tr)


def main(data_path, out_dir):
    df = load_clean(data_path)
    X_tr, X_te, y_tr, y_te = split_data(df)
    pipe = build_pipeline()

    cv = StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_STATE)
    oof = cross_val_predict(clone(pipe), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    value_tr, value_te = customer_value(X_tr["MonthlyCharges"]), customer_value(X_te["MonthlyCharges"])
    thr, _, _ = best_threshold(y_tr, oof, value_tr)
    print(f"Cost-optimal threshold (out-of-fold, training data): {thr}")

    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= thr).astype(int)
    metrics = {"threshold": thr, "pr_auc": average_precision_score(y_te, proba), "roc_auc": roc_auc_score(y_te, proba),
               "precision": precision_score(y_te, pred), "recall": recall_score(y_te, pred), "f1": f1_score(y_te, pred),
               "brier": brier_score_loss(y_te, proba)}
    print({k: round(float(v), 4) for k, v in metrics.items()})
    print("Test cost: nobody =", round(total_cost(y_te, proba, 1.01, value_te)),
          "| model =", round(total_cost(y_te, proba, thr, value_te)))

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(out_dir, "churn_model.joblib"))
    joblib.dump(pipe, os.path.join(out_dir, "churn_explain_pipeline.joblib"))   # no calibration was needed, same object
    with open(os.path.join(out_dir, "artifacts_retrained.json"), "w") as f:
        json.dump({"threshold": thr, "test_metrics": {k: float(v) for k, v in metrics.items()},
                   "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
                                "pandas": pd.__version__, "numpy": np.__version__}}, f, indent=2)
    print("Saved to", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(cfg.DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"))
    ap.add_argument("--out-dir", default=str(cfg.ROOT / "models_retrained"))
    a = ap.parse_args()
    main(a.data, a.out_dir)
