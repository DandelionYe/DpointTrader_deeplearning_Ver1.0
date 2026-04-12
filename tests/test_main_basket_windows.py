import argparse
import os
import shutil
import uuid

import pandas as pd
import pytest

from main_basket import build_model_config, create_continue_run_dir, normalize_mode_args, resolve_window_config
from search_space import build_base_model_config


@pytest.fixture
def sample_panel():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    rows = []
    for ticker in ["A", "B"]:
        for date in dates:
            rows.append({"date": date, "ticker": ticker})
    return pd.DataFrame(rows)


@pytest.fixture
def local_tmpdir():
    path = os.path.join(".local", "tmp", "main_basket_windows", str(uuid.uuid4()))
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_resolve_window_config_splits_research_and_report(sample_panel):
    args = argparse.Namespace(
        research_start_date="2024-01-03",
        research_end_date="2024-01-12",
        report_start_date="2024-01-05",
        report_end_date="2024-01-10",
        backtest_start_date=None,
        backtest_end_date=None,
    )
    config = resolve_window_config(sample_panel, args)

    assert config["research_start"] == pd.Timestamp("2024-01-03")
    assert config["research_end"] == pd.Timestamp("2024-01-12")
    assert config["report_start"] == pd.Timestamp("2024-01-05")
    assert config["report_end"] == pd.Timestamp("2024-01-10")


def test_resolve_window_config_rejects_report_outside_research(sample_panel):
    args = argparse.Namespace(
        research_start_date="2024-01-03",
        research_end_date="2024-01-10",
        report_start_date="2024-01-02",
        report_end_date="2024-01-10",
        backtest_start_date=None,
        backtest_end_date=None,
    )
    with pytest.raises(ValueError, match="Report window must stay within the research window"):
        resolve_window_config(sample_panel, args)


def test_normalize_mode_args_for_continue_defaults_latest():
    args = argparse.Namespace(
        mode="continue",
        continue_from=None,
        label_mode="binary_next_close_up",
        task_type=None,
        label_horizon_days=1,
        primary_metric="auto",
        selection_metric="auto",
    )
    normalized = normalize_mode_args(args)
    assert normalized.continue_from == "latest"
    assert normalized.primary_metric == "rank_ic_mean"
    assert normalized.selection_metric == "rank_ic_mean"


def test_normalize_mode_args_rejects_continue_from_in_first_mode():
    args = argparse.Namespace(
        mode="first",
        continue_from="exp_001",
        label_mode="binary_next_close_up",
        task_type=None,
        label_horizon_days=1,
        primary_metric="auto",
        selection_metric="auto",
    )
    with pytest.raises(ValueError, match="--continue_from can only be used when --mode continue"):
        normalize_mode_args(args)


def test_create_continue_run_dir_creates_incrementing_subdirs(local_tmpdir):
    base = os.path.join(local_tmpdir, "exp_001")
    os.makedirs(base, exist_ok=True)

    first = create_continue_run_dir(base)
    second = create_continue_run_dir(base)

    assert first.endswith("continue_run_001")
    assert second.endswith("continue_run_002")
    assert os.path.isdir(os.path.join(second, "models"))
    assert os.path.isdir(os.path.join(second, "artifacts"))


def test_build_model_config_carries_multiclass_n_classes_for_sequence_models():
    args = argparse.Namespace(
        model_type="lstm",
        label_mode="multiclass_3",
        task_type="multiclass_classification",
        label_horizon_days=1,
        primary_metric="macro_f1",
        device="cpu",
        hidden_dim=64,
        dropout_rate=0.1,
        learning_rate=0.01,
        weight_decay=1e-4,
        epochs=2,
        batch_size=32,
        predict_batch_size=0,
        auto_batch_tune=1,
        target_vram_util=0.88,
        train_target_vram_util=None,
        predict_target_vram_util=None,
        use_amp=0,
        use_tf32=0,
        seq_len=10,
        num_layers=1,
        bidirectional=0,
        num_filters=32,
        kernel_sizes="2,3",
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        hidden_dims="64,32",
        xgb_n_estimators=100,
        xgb_max_depth=4,
        xgb_subsample=0.9,
        xgb_colsample_bytree=0.9,
        seed=42,
        cpu_threads=2,
    )

    config = build_model_config(args)

    assert config["task_type"] == "multiclass_classification"
    assert config["n_classes"] == 3


def test_build_base_model_config_uses_multiclass_defaults_for_xgb():
    args = argparse.Namespace(
        model_type="xgb",
        task_type="multiclass_classification",
        label_mode="multiclass_3",
        label_horizon_days=1,
        primary_metric="macro_f1",
        device="cpu",
        hidden_dim=64,
        hidden_dims="64,32",
        dropout_rate=0.1,
        learning_rate=0.05,
        weight_decay=1e-4,
        epochs=2,
        batch_size=32,
        predict_batch_size=0,
        auto_batch_tune=1,
        target_vram_util=0.88,
        train_target_vram_util=None,
        predict_target_vram_util=None,
        use_amp=0,
        use_tf32=0,
        seq_len=10,
        num_layers=1,
        bidirectional=0,
        num_filters=32,
        kernel_sizes="2,3",
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        xgb_n_estimators=100,
        xgb_max_depth=4,
        xgb_subsample=0.9,
        xgb_colsample_bytree=0.9,
        cpu_threads=2,
    )

    config = build_base_model_config(args)

    assert config["n_classes"] == 3
    assert config["model_params"]["eval_metric"] == "mlogloss"


def test_normalize_mode_args_defaults_multiclass_selection_metric_to_primary_metric():
    args = argparse.Namespace(
        mode="first",
        continue_from=None,
        label_mode="multiclass_3",
        task_type=None,
        label_horizon_days=1,
        primary_metric="auto",
        selection_metric="auto",
    )

    normalized = normalize_mode_args(args)

    assert normalized.task_type == "multiclass_classification"
    assert normalized.primary_metric == "macro_f1"
    assert normalized.selection_metric == "macro_f1"
