"""Safe model loading.

A pickled scikit-learn / XGBoost model only loads correctly with (almost) the same library versions it was saved with.
With other versions it either crashes (e.g. 'input stream corrupted', "Can't get attribute ...") or, worse, loads and
silently returns DIFFERENT probabilities. So we do two things:
  1. check the loaded model against reference predictions stored in models/artifacts.json;
  2. if loading fails or the check fails, retrain the same model from the raw CSV with the installed versions
     (takes a few seconds) and cache it in models_runtime/.
"""
import warnings
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from . import config as cfg

RUNTIME_DIR = cfg.ROOT / "models_runtime"
RAW_CSV = cfg.DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
STATUS = "model not loaded yet"


def _fingerprint_ok(model):
    ref = cfg.ARTIFACTS.get("reference_predictions")
    if not ref or not cfg.SAMPLE_CSV.exists():
        return True
    from .data import prepare_input
    clean, _ = prepare_input(pd.read_csv(cfg.SAMPLE_CSV).head(len(ref)))
    return bool(np.allclose(model.predict_proba(clean)[:, 1], ref, atol=2e-3))


def try_load_saved(path):
    """Returns (model or None, reason if None)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(path)
        if not _fingerprint_ok(model):
            return None, "predictions differ from the reference (library version mismatch)"
        return model, ""
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:160]}"


def retrain_for_this_environment():
    from .train import fit_final_model
    if not RAW_CSV.exists():
        raise RuntimeError(f"The saved model cannot be used with the installed library versions and {RAW_CSV} "
                           "is missing, so it cannot be retrained. Install the pinned versions from requirements.txt.")
    cache = RUNTIME_DIR / f"churn_model_sk{sklearn.__version__}_xgb{xgboost.__version__}.joblib"
    if cache.exists():
        try:
            return joblib.load(cache)
        except Exception:
            pass
    model = fit_final_model(RAW_CSV)
    try:
        RUNTIME_DIR.mkdir(exist_ok=True)
        joblib.dump(model, cache)
    except Exception:
        pass
    return model


@lru_cache(maxsize=1)
def get_model():
    global STATUS
    model, reason = try_load_saved(cfg.MODEL_PATH)
    if model is not None:
        STATUS = "Loaded the saved model (models/churn_model.joblib)."
        return model
    try:
        model = retrain_for_this_environment()
    except Exception as e:
        raise RuntimeError(
            f"Could not load the saved model ({reason}) and retraining also failed ({type(e).__name__}: {str(e)[:200]}). "
            "Install the exact library versions with:  pip install -r requirements.txt") from e
    STATUS = ("The saved model does not match the installed library versions "
              f"(scikit-learn {sklearn.__version__}, xgboost {xgboost.__version__}): {reason}. "
              "The same pipeline was retrained from the raw CSV at startup instead.")
    print("[ChurnGuard]", STATUS)
    return model
