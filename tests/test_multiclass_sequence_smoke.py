import numpy as np
import pandas as pd
import pytest

from models import TORCH_AVAILABLE
from panel_trainer import (
    align_scores_with_labels,
    evaluate_scores_df,
    predict_panel,
    train_panel_model,
)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_torch_sequence_multiclass_pipeline_smoke():
    dates = pd.date_range("2024-01-01", periods=36, freq="B")
    rows = []
    labels = []
    for ticker_idx, ticker in enumerate(["A", "B"]):
        for day_idx, date in enumerate(dates):
            signal = np.sin(day_idx / 3.0) + (ticker_idx * 0.15)
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "feature1": signal,
                    "feature2": np.cos(day_idx / 5.0),
                    "feature3": np.sin(day_idx / 7.0 + ticker_idx),
                }
            )
            if signal > 0.35:
                labels.append(2)
            elif signal < -0.35:
                labels.append(0)
            else:
                labels.append(1)

    X = pd.DataFrame(rows)
    y = pd.Series(labels, index=X.index, dtype=int)
    config = {
        "task_type": "multiclass_classification",
        "n_classes": 3,
        "model_type": "lstm",
        "device": "cpu",
        "model_params": {
            "hidden_dim": 16,
            "num_layers": 1,
            "dropout_rate": 0.1,
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "epochs": 1,
            "batch_size": 8,
            "seq_len": 6,
            "bidirectional": False,
            "early_stop_ratio": 0.2,
            "early_stop_min_dates": 5,
            "early_stop_min_rows": 10,
        },
    }

    model, _ = train_panel_model(X, y, config, date_col="date", ticker_col="ticker", seed=42)
    pred = predict_panel(model, X, date_col="date", ticker_col="ticker")

    expected_rows = sum(
        max(0, len(dates) - config["model_params"]["seq_len"] + 1) for _ in ["A", "B"]
    )
    assert len(pred) == expected_rows
    assert pred["probability_available"].eq(True).all()
    assert pred["score"].between(-1.0, 1.0).all()
    assert pred["proba_up"].between(0.0, 1.0).all()
    assert set(pred["prediction"].unique()).issubset({0, 1, 2})

    scores_df = align_scores_with_labels(
        pred, X, y, config=config, date_col="date", ticker_col="ticker"
    )
    metrics = evaluate_scores_df(scores_df, date_col="date", ticker_col="ticker", config=config)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "rank_ic_mean" in metrics
