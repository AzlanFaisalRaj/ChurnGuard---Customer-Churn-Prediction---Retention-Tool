"""Business logic: cost of decisions, threshold search, risk levels, expected revenue saved."""
import numpy as np
import pandas as pd

from . import config as cfg


def customer_value(monthly_charges):
    return np.asarray(monthly_charges, dtype=float) * cfg.VALUE_MONTHS


def total_cost(y_true, proba, threshold, value, offer_cost=None, success=None):
    """Total cost if we send an offer to everyone with proba >= threshold.
    missed churners lose the full value; every offer costs money; offers to churners fail with prob (1 - success)."""
    offer_cost = cfg.OFFER_COST if offer_cost is None else offer_cost
    success = cfg.OFFER_SUCCESS if success is None else success
    y = np.asarray(y_true); v = np.asarray(value); flag = np.asarray(proba) >= threshold
    return (v * y)[~flag].sum() + offer_cost * flag.sum() + ((1 - success) * v * y)[flag].sum()


def best_threshold(y_true, proba, value, grid=None):
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2) if grid is None else grid
    costs = np.array([total_cost(y_true, proba, t, value) for t in grid])
    return float(grid[costs.argmin()]), grid, costs


def risk_level(p):
    if p >= cfg.HIGH_FROM:
        return "High"
    if p >= cfg.THRESHOLD:
        return "Medium"
    return "Low"


def suggest_offer(row):
    """Rule-based uplift idea (not a tested uplift model)."""
    if row["Contract"] == "Month-to-month":
        return "Contract upgrade offer (discount for a 1-year plan)"
    if row["PaymentMethod"] == "Electronic check":
        return "Autopay discount (switch from electronic check)"
    if row["InternetService"] == "Fiber optic" and row["TechSupport"] == "No":
        return "Free tech-support bundle"
    return "General loyalty discount"


def expected_impact(proba, value, pcts=(10, 20, 30)):
    """Expected revenue saved for an uploaded customer list (no real labels, so we use the probabilities)."""
    proba = np.asarray(proba, dtype=float); value = np.asarray(value, dtype=float)
    order = np.argsort(-proba); n = len(proba)
    rows = []

    def row(label, idx):
        k = len(idx)
        saved = cfg.OFFER_SUCCESS * float((proba[idx] * value[idx]).sum())
        offers = cfg.OFFER_COST * k
        return {"strategy": label, "customers_targeted": k,
                "expected_churners_caught": round(float(proba[idx].sum()), 1),
                "offer_cost": round(offers), "expected_revenue_saved": round(saved),
                "expected_net_benefit": round(saved - offers)}

    for pct in pcts:
        rows.append(row(f"Top {pct}% by risk", order[:max(1, round(n * pct / 100))]))
    rows.append(row(f"All customers with probability >= {cfg.THRESHOLD:.2f}", np.where(proba >= cfg.THRESHOLD)[0]))
    return pd.DataFrame(rows)


def load_test_impact():
    """Validated results on the held-out test set (real churn labels), written by the notebook."""
    if cfg.BUSINESS_IMPACT_CSV.exists():
        return pd.read_csv(cfg.BUSINESS_IMPACT_CSV)
    return pd.DataFrame()
