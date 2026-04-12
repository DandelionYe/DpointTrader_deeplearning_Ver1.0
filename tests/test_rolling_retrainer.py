import argparse
import json
import os
import shutil
import uuid

import numpy as np
import pandas as pd
import pytest

from rolling_retrainer import RollingConfig, RollingRetrainer


@pytest.fixture
def sample_panel():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=320, freq="B")
    tickers = ["A", "B", "C"]
    rows = []
    for ticker in tickers:
        for date in dates:
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open_qfq": 10.0 + np.random.randn(),
                    "high_qfq": 10.5 + np.random.randn(),
                    "low_qfq": 9.5 + np.random.randn(),
                    "close_qfq": 10.0 + np.random.randn(),
                    "volume": 1_000_000 + abs(np.random.randn()) * 1000,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def mock_args():
    return argparse.Namespace(
        label_mode="classification",
        label_horizon_days=1,
        include_cross_section=1,
        model_type="xgb",
        device="cpu",
        n_folds=3,
        train_start_ratio=0.5,
        split_min_rows=60,
        split_mode="wf",
        use_holdout=0,
        holdout_ratio=0.15,
        min_holdout_rows=60,
        embargo_days=5,
        n_outer_folds=3,
        n_inner_folds=2,
        selection_metric="rank_ic_mean",
        runs=1,
        seed=42,
        hidden_dim=64,
        hidden_dims="64,32",
        dropout_rate=0.1,
        learning_rate=0.01,
        weight_decay=1e-4,
        epochs=2,
        batch_size=2048,
        seq_len=20,
        num_layers=2,
        bidirectional=0,
        num_filters=64,
        kernel_sizes="2,3,5",
        d_model=64,
        nhead=4,
        dim_feedforward=128,
        xgb_n_estimators=20,
        xgb_max_depth=3,
        xgb_subsample=0.8,
        xgb_colsample_bytree=0.8,
        cpu_threads=1,
    )


@pytest.fixture
def local_tmpdir():
    path = os.path.join(".local", "tmp", "rolling", str(uuid.uuid4()))
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class TestRollingRetrainer:
    def test_monthly_generates_multiple_retrain_dates(self, sample_panel, local_tmpdir):
        config = RollingConfig(
            window_type="expanding", retrain_frequency="monthly", min_history_days=120
        )
        retrainer = RollingRetrainer(config=config, experiment_dir=local_tmpdir)
        retrain_dates = retrainer.iter_retrain_dates(sample_panel)
        assert len(retrain_dates) > 1

    def test_expanding_window_grows_over_time(self, sample_panel, local_tmpdir):
        config = RollingConfig(
            window_type="expanding", retrain_frequency="monthly", min_history_days=120
        )
        retrainer = RollingRetrainer(config=config, experiment_dir=local_tmpdir)
        retrain_dates = retrainer.iter_retrain_dates(sample_panel)
        if len(retrain_dates) >= 2:
            train_data_1 = retrainer.get_training_window(sample_panel, retrain_dates[0])
            train_data_2 = retrainer.get_training_window(sample_panel, retrain_dates[1])
            assert len(train_data_2) >= len(train_data_1)

    def test_rolling_window_stays_fixed(self, sample_panel, local_tmpdir):
        config = RollingConfig(
            window_type="rolling",
            rolling_window_length=252,
            retrain_frequency="monthly",
            min_history_days=120,
        )
        retrainer = RollingRetrainer(config=config, experiment_dir=local_tmpdir)
        retrain_dates = retrainer.iter_retrain_dates(sample_panel)
        if len(retrain_dates) >= 2:
            train_data_1 = retrainer.get_training_window(sample_panel, retrain_dates[0])
            train_data_2 = retrainer.get_training_window(sample_panel, retrain_dates[1])
            assert abs(len(train_data_1) - len(train_data_2)) < 50

    def test_each_snapshot_writes_manifest(self, sample_panel, mock_args, local_tmpdir):
        config = RollingConfig(
            window_type="expanding", retrain_frequency="monthly", min_history_days=120
        )
        retrainer = RollingRetrainer(config=config, experiment_dir=local_tmpdir)
        snapshots = retrainer.run(sample_panel, mock_args)
        if snapshots:
            snapshot_dir = os.path.join(local_tmpdir, "snapshots", snapshots[0].snapshot_id)
            assert os.path.exists(os.path.join(snapshot_dir, "manifest.json"))
            assert os.path.exists(os.path.join(snapshot_dir, "model.joblib"))
            assert os.path.exists(os.path.join(snapshot_dir, "scores.csv"))
            assert os.path.exists(os.path.join(snapshot_dir, "equity_curve.csv"))
            with open(os.path.join(snapshot_dir, "manifest.json"), "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            assert "contracts" in manifest
            assert manifest["contracts"]["training"]["label_horizon_days"] == 1
            assert manifest["contracts"]["data"]["data_hash"]

    def test_train_end_date_before_eval_date(self, sample_panel, mock_args, local_tmpdir):
        config = RollingConfig(
            window_type="expanding", retrain_frequency="monthly", min_history_days=120
        )
        retrainer = RollingRetrainer(config=config, experiment_dir=local_tmpdir)
        snapshots = retrainer.run(sample_panel, mock_args)
        for snapshot in snapshots:
            train_end = pd.Timestamp(snapshot.train_end_date)
            eval_start = pd.Timestamp(snapshot.metrics["evaluation_start_date"])
            assert train_end < eval_start
            assert pd.Timestamp(snapshot.metrics["train_label_end_date_max"]) <= train_end
            assert pd.Timestamp(snapshot.metrics["eval_label_end_date_max"]) >= eval_start

    def test_top_level_rolling_manifest_writes_contracts(
        self, sample_panel, mock_args, local_tmpdir
    ):
        from basket_loader import BasketMeta
        from main_basket import _run_rolling_retrain

        rolling_args = argparse.Namespace(**vars(mock_args))
        rolling_args.output_dir = local_tmpdir
        rolling_args.rolling_mode = "expanding"
        rolling_args.rolling_window_length = 252
        rolling_args.retrain_frequency = "monthly"
        rolling_args.min_history_days = 120
        rolling_args.basket_path = local_tmpdir
        rolling_args.data_root = local_tmpdir
        rolling_args.basket = "test_basket"

        basket_meta = BasketMeta(
            basket_name="test_basket",
            basket_path=local_tmpdir,
            n_tickers=int(sample_panel["ticker"].nunique()),
            tickers=sorted(sample_panel["ticker"].unique().tolist()),
            date_range=(str(sample_panel["date"].min()), str(sample_panel["date"].max())),
        )

        _run_rolling_retrain(sample_panel, rolling_args, basket_meta)

        exp_dirs = sorted(
            path
            for path in os.listdir(local_tmpdir)
            if path.startswith("exp_") and os.path.isdir(os.path.join(local_tmpdir, path))
        )
        assert exp_dirs
        manifest_path = os.path.join(local_tmpdir, exp_dirs[-1], "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)

        assert "contracts" in manifest
        assert manifest["contracts"]["training"]["label_horizon_days"] == 1
        assert manifest["contracts"]["data"]["data_hash"]
