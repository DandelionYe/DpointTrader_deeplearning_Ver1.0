import numpy as np
import pandas as pd
import pytest

from models import TORCH_AVAILABLE
from panel_trainer import predict_panel, train_panel_model


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_sequence_prediction_shape_by_ticker():
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    tickers = ["A", "B"]
    rows = []
    for ticker_idx, ticker in enumerate(tickers):
        close = 10.0 + ticker_idx
        for day_idx, date in enumerate(dates):
            close = close * (1.0 + 0.001 + 0.0003 * np.cos(day_idx / 3.0))
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "feature1": np.sin(day_idx / 4.0) + ticker_idx,
                    "feature2": np.cos(day_idx / 5.0),
                    "feature3": close / 10.0,
                }
            )
    X = pd.DataFrame(rows)
    y = pd.Series(((X["feature1"] + X["feature2"]) > 0).astype(float), index=X.index)

    config = {
        "model_type": "lstm",
        "device": "cpu",
        "model_params": {
            "hidden_dim": 16,
            "num_layers": 1,
            "dropout_rate": 0.1,
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "epochs": 1,
            "batch_size": 16,
            "seq_len": 20,
            "bidirectional": False,
        },
    }

    model, _ = train_panel_model(
        X,
        y,
        config,
        date_col="date",
        ticker_col="ticker",
        seed=42,
    )
    pred = predict_panel(model, X, date_col="date", ticker_col="ticker")

    expected_rows = sum(max(0, len(dates) - 20 + 1) for _ in tickers)
    assert len(pred) == expected_rows
    assert list(pred.columns) == [
        "date",
        "ticker",
        "score",
        "prediction",
        "raw_output",
        "proba_up",
        "proba",
        "probability_available",
    ]
    assert pred["probability_available"].eq(True).all()
    assert pred["score"].between(0.0, 1.0).all()
    assert pred["proba"].between(0.0, 1.0).all()
