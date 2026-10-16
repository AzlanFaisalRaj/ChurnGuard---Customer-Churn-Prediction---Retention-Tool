"""Prediction for one customer or a whole table: probability, risk level, decision and top 3 reasons."""
from functools import lru_cache

import joblib
import pandas as pd

from . import config as cfg
from .business import customer_value, risk_level, suggest_offer
from .data import prepare_input
from .explain import explain_dataframe
from .pipeline import FEATURES


@lru_cache(maxsize=1)
def get_model():
    return joblib.load(cfg.MODEL_PATH)


def predict_proba(df_clean):
    return get_model().predict_proba(df_clean[FEATURES])[:, 1]


def predict_dataframe(df_raw, top_k=3):
    """Returns (result_df, clean_df, reasons, warnings). result_df has one row per input row, same order."""
    clean, warnings = prepare_input(df_raw)
    proba = predict_proba(clean)
    reasons = explain_dataframe(clean, top_k, proba)
    value = customer_value(clean["MonthlyCharges"])

    res = pd.DataFrame(index=clean.index)
    if "customerID" in df_raw.columns:
        res["customerID"] = df_raw.loc[clean.index, "customerID"].values
    res["churn_probability"] = proba.round(4)
    res["risk_level"] = [risk_level(p) for p in proba]
    res["send_offer"] = ["Yes" if p >= cfg.THRESHOLD else "No" for p in proba]
    res["expected_revenue_at_risk"] = (proba * value).round(0)
    for i in range(top_k):
        res[f"reason_{i + 1}"] = [f"{r[i]['arrow']} {r[i]['text']}" for r in reasons]
    res["suggested_offer"] = [suggest_offer(clean.loc[i]) if p >= cfg.THRESHOLD else "-"
                              for i, p in zip(clean.index, proba)]
    return res, clean, reasons, warnings
