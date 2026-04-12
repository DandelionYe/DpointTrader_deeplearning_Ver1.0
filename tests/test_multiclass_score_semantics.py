import numpy as np
import pandas as pd

from panel_trainer import predict_panel
from tasks import multiclass_probabilities_to_score


class DummyMulticlassProbModel:
    _task_type = "multiclass_classification"

    def predict_proba(self, X):
        del X
        return np.asarray(
            [
                [0.70, 0.20, 0.10],
                [0.20, 0.35, 0.45],
                [0.05, 0.15, 0.80],
            ],
            dtype=np.float32,
        )


def test_multiclass_probability_score_maps_to_directional_range():
    probs = np.asarray(
        [
            [0.80, 0.15, 0.05],
            [0.10, 0.25, 0.65],
        ],
        dtype=np.float32,
    )
    scores = multiclass_probabilities_to_score(probs)

    assert scores.shape == (2,)
    assert np.all(scores >= -1.0)
    assert np.all(scores <= 1.0)
    assert scores[0] < scores[1]


def test_predict_panel_multiclass_uses_expected_score_and_bullish_probability():
    X = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="B"),
            "ticker": ["A", "B", "C"],
            "feature_1": [0.1, 0.2, 0.3],
        }
    )
    model = DummyMulticlassProbModel()

    out = predict_panel(model, X, date_col="date", ticker_col="ticker")
    expected_probs = model.predict_proba(None)

    assert out["probability_available"].eq(True).all()
    assert np.allclose(
        out["score"].to_numpy(), multiclass_probabilities_to_score(expected_probs), atol=1e-6
    )
    assert np.allclose(out["proba_up"].to_numpy(), expected_probs[:, -1], atol=1e-6)
    assert np.array_equal(out["prediction"].to_numpy(), np.argmax(expected_probs, axis=1))
    assert np.allclose(out["raw_output"].to_numpy(), np.max(expected_probs, axis=1), atol=1e-6)
