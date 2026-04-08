import argparse

import pandas as pd

from main_basket import build_split_plan


def _sample_panel():
    dates = pd.date_range("2020-01-01", periods=180, freq="B")
    rows = []
    for ticker in ["A", "B", "C"]:
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "feature1": float(idx),
                }
            )
    X = pd.DataFrame(rows)
    y = pd.Series([0.0] * len(X), index=X.index)
    return X, y


def _min_gap_days(X: pd.DataFrame, train_idx, val_idx):
    train_max = pd.Timestamp(X.loc[train_idx, "date"].max())
    val_min = pd.Timestamp(X.loc[val_idx, "date"].min())
    return int((val_min - train_max).days)


def test_nested_inner_embargo_days_override():
    X, y = _sample_panel()
    args = argparse.Namespace(
        split_mode="nested_wf",
        train_start_ratio=0.5,
        split_min_rows=30,
        use_holdout=0,
        holdout_ratio=0.15,
        min_holdout_rows=30,
        holdout_gap_days=0,
        embargo_days=10,
        inner_embargo_days=3,
        n_folds=3,
        n_outer_folds=3,
        n_inner_folds=2,
    )

    plan = build_split_plan(X, y, args, date_col="date", ticker_col="ticker")
    split = plan["indexed_splits"][0]

    assert _min_gap_days(X, split["outer_train_idx"], split["outer_val_idx"]) >= 10
    inner_train_idx, inner_val_idx = split["inner_splits"][0]
    assert _min_gap_days(plan["search_X"], inner_train_idx, inner_val_idx) >= 3


def test_nested_inner_embargo_inherits_outer_when_not_set():
    X, y = _sample_panel()
    args = argparse.Namespace(
        split_mode="nested_wf",
        train_start_ratio=0.5,
        split_min_rows=30,
        use_holdout=0,
        holdout_ratio=0.15,
        min_holdout_rows=30,
        holdout_gap_days=0,
        embargo_days=7,
        inner_embargo_days=None,
        n_folds=3,
        n_outer_folds=3,
        n_inner_folds=2,
    )

    plan = build_split_plan(X, y, args, date_col="date", ticker_col="ticker")
    split = plan["indexed_splits"][0]
    inner_train_idx, inner_val_idx = split["inner_splits"][0]

    assert _min_gap_days(plan["search_X"], inner_train_idx, inner_val_idx) >= 7


def test_holdout_gap_defaults_to_embargo_when_larger():
    X, y = _sample_panel()
    args = argparse.Namespace(
        split_mode="wf",
        train_start_ratio=0.5,
        split_min_rows=30,
        use_holdout=1,
        holdout_ratio=0.15,
        min_holdout_rows=30,
        holdout_gap_days=0,
        embargo_days=5,
        inner_embargo_days=None,
        n_folds=3,
        n_outer_folds=3,
        n_inner_folds=2,
    )

    plan = build_split_plan(X, y, args, date_col="date", ticker_col="ticker")
    assert plan["split_summary"]["holdout_gap_days"] == 5
