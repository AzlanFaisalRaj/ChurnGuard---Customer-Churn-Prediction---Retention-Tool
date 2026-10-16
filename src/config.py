"""Paths and business settings. Values in models/artifacts.json (written by the notebook) override the defaults."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data"

MODEL_PATH = MODELS_DIR / "churn_model.joblib"
EXPLAIN_PATH = MODELS_DIR / "churn_explain_pipeline.joblib"
ARTIFACTS_PATH = MODELS_DIR / "artifacts.json"
SAMPLE_CSV = DATA_DIR / "sample_customers.csv"
BUSINESS_IMPACT_CSV = REPORTS_DIR / "business_impact.csv"

RANDOM_STATE = 42
TARGET = "Churn"

# Business assumptions (NOT facts from the data - see README "Limitations")
VALUE_MONTHS = 12        # customer value = MonthlyCharges x 12
OFFER_COST = 100         # cost of one retention offer
OFFER_SUCCESS = 0.30     # share of real churners who stay after an offer
THRESHOLD = 0.40         # cost-optimal decision threshold
HIGH_FROM = 0.70         # High risk starts here (Low < THRESHOLD <= Medium < HIGH_FROM <= High)

LABELS = {
    "tenure": "Tenure", "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges",
    "SeniorCitizen": "Senior citizen", "Contract": "Contract type", "InternetService": "Internet service",
    "PaymentMethod": "Payment method", "TechSupport": "Tech support", "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup", "DeviceProtection": "Device protection", "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies", "PaperlessBilling": "Paperless billing",
    "MultipleLines": "Multiple lines", "PhoneService": "Phone service", "Partner": "Has partner",
    "Dependents": "Has dependents", "gender": "Gender",
}


def load_artifacts():
    if ARTIFACTS_PATH.exists():
        with open(ARTIFACTS_PATH) as f:
            return json.load(f)
    return {}


ARTIFACTS = load_artifacts()
if ARTIFACTS:
    THRESHOLD = ARTIFACTS.get("threshold", THRESHOLD)
    HIGH_FROM = ARTIFACTS.get("risk_levels", {}).get("high_from", HIGH_FROM)
    _c = ARTIFACTS.get("costs", {})
    VALUE_MONTHS = _c.get("value_months", VALUE_MONTHS)
    OFFER_COST = _c.get("offer_cost", OFFER_COST)
    OFFER_SUCCESS = _c.get("offer_success", OFFER_SUCCESS)
    LABELS = ARTIFACTS.get("labels", LABELS)
