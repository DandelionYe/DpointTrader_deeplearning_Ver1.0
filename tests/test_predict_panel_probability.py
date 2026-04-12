import numpy as np
import pandas as pd

from panel_trainer import predict_panel


class DummyProbModel:
    def predict_proba(self, X):
        n = len(X)
        p1 = np.linspace(0.1, 0.9, n, dtype=np.float32)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


class DummyDecisionModel:
    def decision_function(self, X):
        return np.arange(len(X), dtype=np.float32)


class DummyPredictModel:
    def predict(self, X):
        return np.ones(len(X), dtype=np.float32)


def _sample_X():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "ticker": ["AAA", "BBB", "AAA"],
            "f1": [1.0, 2.0, 3.0],
            "f2": [10.0, 20.0, 30.0],
        }
    )


def test_predict_panel_with_predict_proba_returns_probability_columns():
    X = _sample_X()
    out = predict_panel(DummyProbModel(), X, return_proba=True)

    assert list(out.columns) == ["date", "ticker", "score", "prediction", "raw_output", "proba_up", "proba", "probability_available"]
    assert out["probability_available"].eq(True).all()
    assert np.allclose(out["score"].to_numpy(), out["proba"].to_numpy(), equal_nan=False)
    assert np.allclose(out["proba_up"].to_numpy(), out["proba"].to_numpy(), equal_nan=False)
    assert set(out["prediction"].unique()).issubset({0.0, 1.0})
    assert out["proba"].notna().all()


def test_predict_panel_without_predict_proba_returns_nan_proba():
    X = _sample_X()
    out = predict_panel(DummyDecisionModel(), X, return_proba=True)

    assert list(out.columns) == ["date", "ticker", "score", "prediction", "raw_output", "proba_up", "proba", "probability_available"]
    assert out["probability_available"].eq(False).all()
    assert out["proba"].isna().all()
    assert out["proba_up"].isna().all()
    assert out["score"].notna().all()


def test_predict_panel_return_proba_false_returns_score_only():
    X = _sample_X()
    out = predict_panel(DummyPredictModel(), X, return_proba=False)

    assert list(out.columns) == ["date", "ticker", "score"]
    assert "proba" not in out.columns
    assert "probability_available" not in out.columns
