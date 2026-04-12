import argparse

import numpy as np
import pandas as pd
import pytest

from search_engine import SearchResult, run_search, score_candidate
from search_space import sample_model_config
from splitters import walkforward_splits_by_date


@pytest.fixture
def sample_data():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
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

    df = pd.DataFrame(rows)
    df["label"] = (df["feature1"] > 0).astype(int)
    X = df[["date", "ticker", "feature1", "feature2"]]
    y = df["label"]

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
    return X, y, indexed_splits


@pytest.fixture
def mock_args():
    return argparse.Namespace(
        runs=3,
        seed=42,
        selection_metric="rank_ic_mean",
        model_type="xgb",
    )


class TestSearchEngine:
    def test_score_candidate_supports_minimization_metrics(self):
        metrics = {"rmse": 0.12, "mae": 0.08, "macro_f1": 0.66}
        assert score_candidate(metrics, "rmse") == pytest.approx(-0.12)
        assert score_candidate(metrics, "mae") == pytest.approx(-0.08)
        assert score_candidate(metrics, "macro_f1") == pytest.approx(0.66)

    def test_run_search_returns_search_result(self, sample_data, mock_args):
        X, y, indexed_splits = sample_data
        base_config = {
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
        result = run_search(
            X,
            y,
            args=mock_args,
            split_mode="wf",
            indexed_splits=indexed_splits,
            base_config=base_config,
            date_col="date",
            ticker_col="ticker",
        )
        assert isinstance(result, SearchResult)

    def test_candidates_count_equals_runs(self, sample_data, mock_args):
        X, y, indexed_splits = sample_data
        base_config = {
            "model_type": "xgb",
            "device": "cpu",
            "model_params": {
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.1,
                "n_jobs": 1,
                "tree_method": "hist",
                "eval_metric": "logloss",
                "verbosity": 0,
            },
        }
        result = run_search(
            X,
            y,
            args=mock_args,
            split_mode="wf",
            indexed_splits=indexed_splits,
            base_config=base_config,
            date_col="date",
            ticker_col="ticker",
        )
        assert len(result.candidates) == mock_args.runs

    def test_best_config_not_empty(self, sample_data, mock_args):
        X, y, indexed_splits = sample_data
        base_config = {
            "model_type": "xgb",
            "device": "cpu",
            "model_params": {
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.1,
                "n_jobs": 1,
                "tree_method": "hist",
                "eval_metric": "logloss",
                "verbosity": 0,
            },
        }
        result = run_search(
            X,
            y,
            args=mock_args,
            split_mode="wf",
            indexed_splits=indexed_splits,
            base_config=base_config,
            date_col="date",
            ticker_col="ticker",
        )
        assert result.best_config is not None
        assert "model_type" in result.best_config

    def test_best_metrics_has_rank_ic_mean(self, sample_data, mock_args):
        X, y, indexed_splits = sample_data
        base_config = {
            "model_type": "xgb",
            "device": "cpu",
            "model_params": {
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.1,
                "n_jobs": 1,
                "tree_method": "hist",
                "eval_metric": "logloss",
                "verbosity": 0,
            },
        }
        result = run_search(
            X,
            y,
            args=mock_args,
            split_mode="wf",
            indexed_splits=indexed_splits,
            base_config=base_config,
            date_col="date",
            ticker_col="ticker",
        )
        assert "rank_ic_mean" in result.best_metrics

    def test_reproducibility_with_same_seed(self, sample_data):
        X, y, indexed_splits = sample_data
        base_config = {
            "model_type": "xgb",
            "device": "cpu",
            "model_params": {
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.1,
                "n_jobs": 1,
                "tree_method": "hist",
                "eval_metric": "logloss",
                "verbosity": 0,
            },
        }
        args1 = argparse.Namespace(runs=2, seed=12345, selection_metric="rank_ic_mean", model_type="xgb")
        args2 = argparse.Namespace(runs=2, seed=12345, selection_metric="rank_ic_mean", model_type="xgb")

        result1 = run_search(
            X,
            y,
            args=args1,
            split_mode="wf",
            indexed_splits=indexed_splits,
            base_config=base_config,
            date_col="date",
            ticker_col="ticker",
        )
        result2 = run_search(
            X,
            y,
            args=args2,
            split_mode="wf",
            indexed_splits=indexed_splits,
            base_config=base_config,
            date_col="date",
            ticker_col="ticker",
        )

        assert result1.best_seed == result2.best_seed
        assert abs(result1.best_metrics["rank_ic_mean"] - result2.best_metrics["rank_ic_mean"]) < 1e-6

    def test_sequence_candidate_config_sampling(self):
        rng = np.random.RandomState(42)
        base_config = {
            "model_type": "lstm",
            "device": "cpu",
            "model_params": {
                "hidden_dim": 16,
                "num_layers": 1,
                "dropout_rate": 0.1,
                "learning_rate": 0.01,
                "weight_decay": 1e-4,
                "epochs": 2,
                "batch_size": 32,
                "seq_len": 10,
                "bidirectional": False,
            },
        }
        candidate = sample_model_config(
            model_type="lstm",
            rng=rng,
            base_config=base_config,
        )
        assert candidate["model_type"] == "lstm"
        assert candidate["model_params"]["seq_len"] == 10
        assert "hidden_dim" in candidate["model_params"]
