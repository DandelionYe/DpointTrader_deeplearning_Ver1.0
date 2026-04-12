import numpy as np
import pandas as pd

from feature_dpoint import build_features_and_labels_panel
from panel_trainer import evaluate_scores_df, predict_panel, train_panel_model
from tasks import LabelSpec, multiclass_probabilities_to_score


def _sample_panel() -> pd.DataFrame:
    rng = np.random.RandomState(7)
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    rows = []
    for ticker_idx, ticker in enumerate(["A", "B", "C"]):
        close = 10.0 + ticker_idx
        for date in dates:
            close = close * (1.0 + rng.normal(0.0008, 0.015))
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open_qfq": close * 0.99,
                    "high_qfq": close * 1.01,
                    "low_qfq": close * 0.98,
                    "close_qfq": close,
                    "volume": 1_000_000 + rng.randint(0, 1000),
                }
            )
    return pd.DataFrame(rows)


def test_multiclass_pipeline_smoke():
    panel_df = _sample_panel()
    feature_config = {
        "basket_name": "multiclass_smoke",
        "windows": [5, 10],
        "use_momentum": True,
        "use_volatility": True,
        "use_volume": True,
        "use_candle": True,
        "use_ta_indicators": False,
    }
    X, y, _ = build_features_and_labels_panel(
        panel_df,
        feature_config,
        date_col="date",
        ticker_col="ticker",
        label_mode="multiclass_3",
        include_cross_section=True,
        label_spec=LabelSpec(
            task_type="multiclass_classification",
            label_mode="multiclass_3",
            horizon_days=1,
            n_classes=3,
        ),
        label_horizon_days=1,
    )
    config = {
        "task_type": "multiclass_classification",
        "model_type": "xgb",
        "n_classes": 3,
        "label_mode": "multiclass_3",
        "label_horizon_days": 1,
        "device": "cpu",
        "model_params": {
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "n_jobs": 1,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "verbosity": 0,
        },
    }

    model, _ = train_panel_model(X, y, config, date_col="date", ticker_col="ticker", seed=42)
    pred_df = predict_panel(model, X, date_col="date", ticker_col="ticker")

    assert "score" in pred_df.columns
    assert "prediction" in pred_df.columns
    assert "raw_output" in pred_df.columns
    assert "proba_up" in pred_df.columns
    assert pred_df["probability_available"].eq(True).all()
    assert pred_df["score"].between(-1.0, 1.0).all()
    assert pred_df["proba_up"].between(0.0, 1.0).all()
    assert set(np.unique(pred_df["prediction"].to_numpy())).issubset({0, 1, 2})

    feature_names = list(getattr(model, "_feature_names", []))
    proba_matrix = model.predict_proba(X[feature_names].to_numpy())
    expected_scores = multiclass_probabilities_to_score(proba_matrix)
    assert np.allclose(pred_df["score"].to_numpy(), expected_scores, atol=1e-6)
    assert np.allclose(pred_df["proba_up"].to_numpy(), proba_matrix[:, -1], atol=1e-6)

    scores_df = pred_df.copy()
    scores_df["label"] = y.to_numpy()
    metrics = evaluate_scores_df(scores_df, date_col="date", ticker_col="ticker", config=config)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "rank_ic_mean" in metrics
