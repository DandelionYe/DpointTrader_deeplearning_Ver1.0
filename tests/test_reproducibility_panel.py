import random

import numpy as np
import pandas as pd
import pytest

from splitters import walkforward_splits_by_date
from utils import set_global_seed


@pytest.fixture
def sample_panel():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=160, freq="B")
    tickers = ["A", "B", "C"]

    rows = []
    for ticker in tickers:
        for date in dates:
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "feature1": np.random.randn(),
                    "feature2": np.random.randn(),
                }
            )

    return pd.DataFrame(rows)


class TestReproducibilityPanel:
    def test_global_seed_reproducibility_numpy(self):
        seed = 12345
        set_global_seed(seed)
        result1 = np.random.randn(10)
        set_global_seed(seed)
        result2 = np.random.randn(10)
        np.testing.assert_array_equal(result1, result2)

    def test_global_seed_reproducibility_python_random(self):
        seed = 12345
        set_global_seed(seed)
        result1 = [random.random() for _ in range(10)]
        set_global_seed(seed)
        result2 = [random.random() for _ in range(10)]
        assert result1 == result2

    def test_train_with_walkforward_reproducibility_xgb(self, sample_panel):
        from panel_trainer import train_with_walkforward

        sample_panel["label"] = (sample_panel["feature1"] > 0).astype(int)
        X = sample_panel[["date", "ticker", "feature1", "feature2"]]
        y = sample_panel["label"]

        splits = walkforward_splits_by_date(
            X,
            date_col="date",
            ticker_col="ticker",
            n_folds=3,
            train_start_ratio=0.5,
            min_rows=60,
        )
        indexed_splits = [
            (
                X.index[X["date"].isin(train_dates)].tolist(),
                X.index[X["date"].isin(val_dates)].tolist(),
            )
            for train_dates, val_dates in splits
        ]

        config = {
            "model_type": "xgb",
            "device": "cpu",
            "model_params": {
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "n_jobs": 1,
                "tree_method": "hist",
                "eval_metric": "logloss",
                "verbosity": 0,
            },
        }

        result1 = train_with_walkforward(
            X,
            y,
            config,
            indexed_splits,
            date_col="date",
            ticker_col="ticker",
            seed=42,
        )
        result2 = train_with_walkforward(
            X,
            y,
            config,
            indexed_splits,
            date_col="date",
            ticker_col="ticker",
            seed=42,
        )

        assert result1.oof_scores.shape == result2.oof_scores.shape
        assert len(result1.oof_scores) == len(result2.oof_scores)
        assert result1.val_metrics == result2.val_metrics

        if not result1.oof_scores.empty:
            score1 = result1.oof_scores.iloc[0]["score"]
            score2 = result2.oof_scores.iloc[0]["score"]
            assert abs(score1 - score2) < 1e-6

    def test_backtest_reproducibility(self):
        from backtester_engine import backtest_from_scores
        from portfolio_builder import PortfolioConfig

        dates = pd.date_range("2020-01-01", periods=80, freq="B")
        tickers = ["A", "B", "C"]
        rows = []
        for ticker_idx, ticker in enumerate(tickers):
            base = 10.0 + ticker_idx
            for date_idx, date in enumerate(dates):
                close = base + date_idx * 0.05
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "open_qfq": close * 0.99,
                        "high_qfq": close * 1.01,
                        "low_qfq": close * 0.98,
                        "close_qfq": close,
                        "volume": 1_000_000 + date_idx,
                    }
                )
        panel_df = pd.DataFrame(rows)

        np.random.seed(42)
        scores_df = pd.DataFrame(
            {
                "date": panel_df["date"],
                "ticker": panel_df["ticker"],
                "score": np.random.randn(len(panel_df)),
            }
        )

        config = PortfolioConfig(
            top_k=3,
            weighting="equal",
            max_weight=0.4,
            cash_buffer=0.05,
            rebalance_freq="monthly",
        )

        result1 = backtest_from_scores(
            panel_df,
            scores_df,
            portfolio_config=config,
            initial_cash=100000.0,
        )
        result2 = backtest_from_scores(
            panel_df,
            scores_df,
            portfolio_config=config,
            initial_cash=100000.0,
        )

        pd.testing.assert_frame_equal(result1.equity_curve, result2.equity_curve)
        pd.testing.assert_frame_equal(result1.trades, result2.trades)

    def test_torch_seed_reproducibility(self):
        torch = pytest.importorskip("torch")

        set_global_seed(2024)
        tensor1 = torch.randn(16)
        set_global_seed(2024)
        tensor2 = torch.randn(16)

        assert torch.equal(tensor1, tensor2)
