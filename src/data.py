"""Loading, cleaning and validating raw customer data (training file and uploaded CSVs)."""
import pandas as pd

from . import config as cfg
from .pipeline import FEATURES, CATEGORICAL


def load_clean(path):
    """Training file: drop the ID, fix TotalCharges, encode the target (1 = churned)."""
    df = pd.read_csv(path)
    df = df.drop(columns="customerID", errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")    # 11 blanks -> NaN
    df[cfg.TARGET] = (df[cfg.TARGET] == "Yes").astype(int)
    return df


def prepare_input(df_raw):
    """Validate and clean raw customer rows for prediction.
    Returns (clean_df with the 19 feature columns, list of warning strings). Raises ValueError for fatal problems."""
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))
    if len(df) == 0:
        raise ValueError("The file has no rows.")

    warnings = []
    for c in CATEGORICAL:
        df[c] = df[c].astype(str).str.strip()

    # SeniorCitizen: accept 0/1 or Yes/No
    sc = df["SeniorCitizen"].astype(str).str.strip().str.lower().map(
        {"1": 1, "0": 0, "1.0": 1, "0.0": 0, "yes": 1, "no": 0, "true": 1, "false": 0})
    if sc.isna().any():
        warnings.append(f"{int(sc.isna().sum())} row(s) have an invalid SeniorCitizen value (treated as 0).")
    df["SeniorCitizen"] = sc.fillna(0).astype(int)

    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")            # Kaggle file stores TotalCharges as text
    bad = df[["tenure", "MonthlyCharges"]].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} row(s) have a missing or non-numeric tenure / MonthlyCharges.")
    # blank TotalCharges: a new customer (tenure 0) has paid nothing; otherwise estimate tenure x monthly
    est = df["tenure"] * df["MonthlyCharges"]
    n_est = int((df["TotalCharges"].isna() & (df["tenure"] > 0)).sum())
    df["TotalCharges"] = df["TotalCharges"].where(df["TotalCharges"].notna() | (df["tenure"] == 0), est)
    if n_est:
        warnings.append(f"{n_est} row(s) had no TotalCharges; estimated as tenure x MonthlyCharges.")

    allowed = cfg.ARTIFACTS.get("categorical_values", {})
    for c in CATEGORICAL:
        if c in allowed:
            unknown = ~df[c].isin(allowed[c])
            if unknown.any():
                warnings.append(f"{int(unknown.sum())} row(s) have an unseen value in '{c}' "
                                f"(allowed: {', '.join(allowed[c])}). The model ignores it.")
    return df[FEATURES], warnings
