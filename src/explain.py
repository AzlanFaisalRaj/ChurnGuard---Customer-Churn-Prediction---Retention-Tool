"""Per-customer explanations with SHAP: the top reasons behind each churn probability."""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import shap

from . import config as cfg
from .pipeline import FEATURES


@lru_cache(maxsize=1)
def _load():
    pipe = joblib.load(cfg.EXPLAIN_PATH)
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    explainer = shap.TreeExplainer(clf)
    to_raw = {f: max([c for c in FEATURES if f == c or f.startswith(c + "_")], key=len) for f in names}
    return prep, explainer, names, to_raw


def format_value(col, val):
    if pd.isna(val):
        return "not available"
    if col == "tenure":
        v = int(val)
        return f"{v} month" + ("" if v == 1 else "s")
    if col in ("MonthlyCharges", "TotalCharges"):
        return f"${float(val):,.2f}"
    if col == "SeniorCitizen":
        return "Yes" if int(val) == 1 else "No"
    return str(val)


def shap_by_column(df_clean):
    """SHAP value per ORIGINAL column (one-hot columns are summed back together). Positive = raises churn risk."""
    prep, explainer, names, to_raw = _load()
    Xt = pd.DataFrame(prep.transform(df_clean[FEATURES]), columns=names, index=df_clean.index)
    sv = explainer.shap_values(Xt)
    return pd.DataFrame(sv, columns=names, index=df_clean.index).T.groupby(to_raw).sum().T


def explain_dataframe(df_clean, top_k=3, proba=None):
    """For every row return exactly `top_k` reasons (list of dicts).
    Customers at or above the offer threshold: the strongest factors that RAISE the risk first (fill up with the
    strongest protective factors if fewer exist). Low-risk customers: the strongest factors in either direction,
    because that explains the low score better than tiny risk-raising factors."""
    by_col = shap_by_column(df_clean)
    out = []
    for n, (idx, row) in enumerate(by_col.iterrows()):
        low = proba is not None and proba[n] < cfg.THRESHOLD
        pos = row[row > 0].sort_values(ascending=False)
        chosen = [] if low else list(pos.index[:top_k])
        if len(chosen) < top_k:
            rest = row.drop(chosen)
            chosen += list(rest.abs().sort_values(ascending=False).index[: top_k - len(chosen)])
        reasons = []
        for c in chosen:
            raised = row[c] > 0
            reasons.append({
                "feature": c,
                "text": f"{cfg.LABELS.get(c, c)}: {format_value(c, df_clean.loc[idx, c])}",
                "impact": round(float(row[c]), 3),
                "direction": "raises risk" if raised else "lowers risk",
                "arrow": "↑" if raised else "↓",
            })
        out.append(reasons)
    return out
