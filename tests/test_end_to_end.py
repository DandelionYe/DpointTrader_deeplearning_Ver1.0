import glob
import json
import os
import shutil
import sys
import uuid

import numpy as np
import pandas as pd
import pytest

from main_basket import main


@pytest.fixture
def local_tmpdir():
    path = os.path.join(".local", "tmp", "end_to_end", str(uuid.uuid4()))
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_basket_csvs(basket_dir: str, *, n_days: int = 320, tickers=None) -> None:
    tickers = tickers or ["600001", "600002", "600003"]
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    rng = np.random.RandomState(42)
    os.makedirs(basket_dir, exist_ok=True)

    for ticker_idx, ticker in enumerate(tickers):
        base = 10.0 + ticker_idx
        drift = 0.0008 + 0.0002 * ticker_idx
        noise = rng.normal(0.0, 0.015, size=n_days)
        close = base * np.exp(np.cumsum(drift + noise))
        open_ = close * (1 + rng.uniform(-0.01, 0.01, size=n_days))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.0, 0.02, size=n_days))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.0, 0.02, size=n_days))
        volume = rng.uniform(1_000_000, 5_000_000, size=n_days)

        pd.DataFrame(
            {
                "Date": dates,
                "Open (CNY, qfq)": open_,
                "High (CNY, qfq)": high,
                "Low (CNY, qfq)": low,
                "Close (CNY, qfq)": close,
                "Volume (shares)": volume,
            }
        ).to_csv(os.path.join(basket_dir, f"{ticker}.csv"), index=False)


def _run_case(workdir, monkeypatch, *, case_name: str, extra_args):
    basket_dir = os.path.join(workdir, f"{case_name}_basket")
    output_dir = os.path.join(workdir, f"{case_name}_output")
    _write_basket_csvs(basket_dir)

    argv = [
        "main_basket.py",
        "--basket",
        case_name,
        "--basket_path",
        basket_dir,
        "--output_dir",
        output_dir,
        "--runs",
        "1",
        "--seed",
        "42",
        "--n_folds",
        "2",
        "--top_k",
        "2",
        "--max_weight",
        "0.5",
        "--cash_buffer",
        "0.05",
        "--rebalance_freq",
        "monthly",
        "--initial_cash",
        "100000",
        "--cpu_threads",
        "1",
        "--xgb_n_estimators",
        "20",
        "--xgb_max_depth",
        "3",
        "--learning_rate",
        "0.05",
        "--batch_size",
        "64",
        "--epochs",
        "2",
        "--hidden_dim",
        "16",
        "--hidden_dims",
        "16,8",
        "--split_min_rows",
        "30",
        "--min_holdout_rows",
        "30",
    ] + extra_args

    monkeypatch.setattr(sys, "argv", argv)
    main()

    exp_dirs = sorted(glob.glob(os.path.join(output_dir, "exp_*")))
    assert exp_dirs
    exp_dir = exp_dirs[-1]

    manifest_path = os.path.join(exp_dir, "manifest.json")
    report_path = os.path.join(exp_dir, "report.html")
    excel_path = os.path.join(exp_dir, "results.xlsx")
    scores_path = os.path.join(exp_dir, "artifacts", "scores.csv")
    equity_path = os.path.join(exp_dir, "artifacts", "equity_curve.csv")

    assert os.path.exists(manifest_path)
    assert os.path.exists(report_path)
    assert os.path.exists(excel_path)
    assert os.path.exists(scores_path)
    assert os.path.exists(equity_path)

    scores_df = pd.read_csv(scores_path)
    equity_df = pd.read_csv(equity_path)
    assert not scores_df.empty
    assert not equity_df.empty
    assert "trade_date" in scores_df.columns
    assert "signal_date" in scores_df.columns
    signal_dates = pd.to_datetime(scores_df["signal_date"])
    trade_dates = pd.to_datetime(scores_df["trade_date"])
    assert (trade_dates > signal_dates).all()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["metrics"]["evaluation_split"] in {"oof", "holdout"}
    assert "split_info" in manifest
    assert "orders_submitted" in manifest["metrics"]
    assert "orders_filled" in manifest["metrics"]
    assert "orders_rejected" in manifest["metrics"]
    return manifest


def test_end_to_end_xgb_wf(local_tmpdir, monkeypatch):
    manifest = _run_case(
        local_tmpdir,
        monkeypatch,
        case_name="xgb_wf",
        extra_args=[
            "--model_type",
            "xgb",
            "--split_mode",
            "wf",
            "--use_holdout",
            "0",
        ],
    )
    assert manifest["split_info"]["split_mode"] == "wf"
    assert manifest["metrics"]["evaluation_split"] == "oof"


def test_end_to_end_mlp_wf_embargo_holdout(local_tmpdir, monkeypatch):
    torch = pytest.importorskip("torch")
    assert torch is not None

    manifest = _run_case(
        local_tmpdir,
        monkeypatch,
        case_name="mlp_holdout",
        extra_args=[
            "--model_type",
            "mlp",
            "--device",
            "cpu",
            "--split_mode",
            "wf_embargo",
            "--embargo_days",
            "5",
            "--use_holdout",
            "1",
            "--holdout_ratio",
            "0.15",
        ],
    )
    assert manifest["split_info"]["split_mode"] == "wf_embargo"
    assert manifest["metrics"]["evaluation_split"] == "holdout"


def test_end_to_end_lstm_wf_seq10(local_tmpdir, monkeypatch):
    torch = pytest.importorskip("torch")
    assert torch is not None

    manifest = _run_case(
        local_tmpdir,
        monkeypatch,
        case_name="lstm_wf",
        extra_args=[
            "--model_type",
            "lstm",
            "--device",
            "cpu",
            "--split_mode",
            "wf",
            "--use_holdout",
            "0",
            "--seq_len",
            "10",
            "--num_layers",
            "1",
        ],
    )
    assert manifest["split_info"]["split_mode"] == "wf"
    assert manifest["metrics"]["evaluation_split"] == "oof"
