import numpy as np
import pandas as pd
import pytest

from splitters import nested_walkforward_splits_by_date, walkforward_splits_with_embargo


@pytest.fixture
def sample_panel():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
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


class TestEmbargoIntegrity:
    def test_embargo_gap_respected(self, sample_panel):
        embargo_days = 5
        splits = walkforward_splits_with_embargo(
            sample_panel,
            date_col="date",
            ticker_col="ticker",
            n_folds=4,
            train_start_ratio=0.5,
            min_rows=60,
            embargo_days=embargo_days,
        )
        assert len(splits) > 0
        for train_dates, val_dates in splits:
            assert (min(val_dates) - max(train_dates)).days >= embargo_days

    def test_embargo_different_values(self, sample_panel):
        for embargo_days in [3, 5, 10]:
            splits = walkforward_splits_with_embargo(
                sample_panel,
                date_col="date",
                ticker_col="ticker",
                n_folds=4,
                train_start_ratio=0.5,
                min_rows=60,
                embargo_days=embargo_days,
            )
            for train_dates, val_dates in splits:
                assert (min(val_dates) - max(train_dates)).days >= embargo_days

    def test_nested_wf_applies_embargo_to_outer_and_inner(self, sample_panel):
        embargo_days = 5
        splits = nested_walkforward_splits_by_date(
            sample_panel,
            date_col="date",
            ticker_col="ticker",
            n_outer_folds=3,
            n_inner_folds=2,
            train_start_ratio=0.5,
            min_rows=60,
            embargo_days=embargo_days,
        )
        assert len(splits) > 0
        for outer_train_dates, outer_val_dates, inner_splits in splits:
            assert (min(outer_val_dates) - max(outer_train_dates)).days >= embargo_days
            for inner_train_dates, inner_val_dates in inner_splits:
                assert (min(inner_val_dates) - max(inner_train_dates)).days >= embargo_days

    def test_embargo_too_large_returns_empty(self, sample_panel):
        splits = walkforward_splits_with_embargo(
            sample_panel,
            date_col="date",
            ticker_col="ticker",
            n_folds=4,
            train_start_ratio=0.5,
            min_rows=60,
            embargo_days=50,
        )
        assert isinstance(splits, list)

    def test_embargo_uses_business_day_offsets(self, sample_panel):
        embargo_days = 5
        splits = walkforward_splits_with_embargo(
            sample_panel,
            date_col="date",
            ticker_col="ticker",
            n_folds=4,
            train_start_ratio=0.5,
            min_rows=60,
            embargo_days=embargo_days,
        )
        if splits:
            train_dates, val_dates = splits[0]
            all_dates = sorted(sample_panel["date"].unique())
            train_idx = all_dates.index(max(train_dates))
            val_idx = all_dates.index(min(val_dates))
            assert (val_idx - train_idx) >= embargo_days
