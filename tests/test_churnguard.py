"""Run with: pytest -q   (from the repo root)"""
import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.business import best_threshold, risk_level, total_cost
from src.data import prepare_input
from src.pipeline import FEATURES
from src.predict import predict_dataframe


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(cfg.SAMPLE_CSV)


def test_predictions_are_valid(sample):
    res, _, reasons, _ = predict_dataframe(sample)
    assert len(res) == len(sample)
    assert res["churn_probability"].between(0, 1).all()
    assert set(res["risk_level"]) <= {"Low", "Medium", "High"}


def test_every_prediction_has_three_reasons(sample):
    _, _, reasons, _ = predict_dataframe(sample)
    assert all(len(r) == 3 for r in reasons)


def test_missing_column_raises(sample):
    with pytest.raises(ValueError):
        prepare_input(sample.drop(columns=["Contract"]))


def test_blank_total_charges_text_is_handled(sample):
    raw = sample.copy()
    raw["TotalCharges"] = raw["TotalCharges"].astype(object)
    raw.loc[0, "TotalCharges"] = " "                    # the Kaggle file has blanks like this
    raw.loc[0, "tenure"] = 0
    res, _, _, _ = predict_dataframe(raw)
    assert np.isfinite(res["churn_probability"]).all()


def test_risk_levels_follow_threshold():
    assert risk_level(cfg.THRESHOLD - 0.01) == "Low"
    assert risk_level(cfg.THRESHOLD) == "Medium"
    assert risk_level(cfg.HIGH_FROM) == "High"


def test_cost_logic():
    y = np.array([1, 0, 1, 0]); p = np.array([0.9, 0.8, 0.1, 0.05]); v = np.array([1000.0] * 4)
    assert total_cost(y, p, 1.01, v) == 2000.0                       # target nobody: lose both churners
    thr, _, _ = best_threshold(y, p, v)
    assert 0.0 < thr < 1.0


def test_feature_list_matches_artifacts():
    a = cfg.ARTIFACTS["features"]
    assert FEATURES == a["numeric"] + a["binary"] + a["categorical"]


def test_corrupted_model_file_is_detected(tmp_path):
    from src.model_store import try_load_saved
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"this is not a model")
    model, reason = try_load_saved(bad)
    assert model is None and reason


def test_model_matches_reference_predictions(sample):
    """Whether the saved model loaded or was retrained, predictions must be close to the notebook's."""
    ref = cfg.ARTIFACTS["reference_predictions"]
    res, _, _, _ = predict_dataframe(sample.head(len(ref)))
    assert np.allclose(res["churn_probability"].values, ref, atol=0.02)
