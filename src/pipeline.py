"""Column groups, preprocessing and the final model definition (same as the notebook)."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY = ["SeniorCitizen"]
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
FEATURES = NUMERIC + BINARY + CATEGORICAL          # 19 raw input columns

# Hyperparameters found by RandomizedSearchCV (PR-AUC, stratified 5-fold) in the notebook
XGB_PARAMS = dict(
    n_estimators=154, learning_rate=0.03272290672827732, max_depth=4, min_child_weight=1,
    subsample=0.9083868719818244, colsample_bytree=0.7361074625809747, reg_lambda=3.3063859878474813,
    eval_metric="logloss", random_state=42, n_jobs=1,
)


def make_preprocessor(scale=False):
    """Everything that is learned from data lives INSIDE this transformer, so cross-validation cannot leak."""
    num_steps = [("imputer", SimpleImputer(strategy="constant", fill_value=0))]   # blank TotalCharges = new customer
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), NUMERIC),
            ("bin", "passthrough", BINARY),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline():
    return Pipeline([("prep", make_preprocessor(scale=False)), ("clf", XGBClassifier(**XGB_PARAMS))])
