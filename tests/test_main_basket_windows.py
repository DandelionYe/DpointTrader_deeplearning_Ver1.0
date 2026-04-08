import argparse
import os
import shutil
import uuid

import pandas as pd
import pytest

from main_basket import create_continue_run_dir, normalize_mode_args, resolve_window_config


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
    args = argparse.Namespace(mode="continue", continue_from=None)
    normalized = normalize_mode_args(args)
    assert normalized.continue_from == "latest"


def test_normalize_mode_args_rejects_continue_from_in_first_mode():
    args = argparse.Namespace(mode="first", continue_from="exp_001")
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
