import numpy as np
import pandas as pd
import pytest

from feature_dpoint import build_features_and_labels_panel
from splitters import final_holdout_split_by_date, walkforward_splits_by_date, walkforward_splits_with_embargo


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


class TestNoLeakagePanel:
    def test_validation_dates_after_training_dates(self, sample_panel):
        splits = walkforward_splits_by_date(
            sample_panel,
            date_col="date",
            ticker_col="ticker",
            n_folds=4,
            train_start_ratio=0.5,
            min_rows=60,
        )
        assert len(splits) > 0
        for train_dates, val_dates in splits:
            assert max(train_dates) < min(val_dates)

    def test_no_overlap_between_validation_folds(self, sample_panel):
        splits = walkforward_splits_by_date(
            sample_panel,
            date_col="date",
            ticker_col="ticker",
            n_folds=4,
            train_start_ratio=0.5,
            min_rows=60,
        )
        val_sets = [set(val_dates) for _, val_dates in splits]
        for i in range(len(val_sets)):
            for j in range(i + 1, len(val_sets)):
                assert len(val_sets[i] & val_sets[j]) == 0

    def test_holdout_not_seen_in_search(self, sample_panel):
        search_df, holdout_df = final_holdout_split_by_date(
            sample_panel,
            date_col="date",
            ticker_col="ticker",
            holdout_ratio=0.15,
            min_holdout_rows=60,
            enforce_non_empty_search=True,
        )
        assert len(set(search_df.index) & set(holdout_df.index)) == 0
        assert len(set(search_df["date"].unique()) & set(holdout_df["date"].unique())) == 0

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

    def test_feature_truncation_consistency(self):
        dates = pd.date_range("2020-01-01", periods=120, freq="B")
        rows = []
        close = 10.0
        for date_idx, date in enumerate(dates):
            close = close * (1.0 + 0.001 + 0.0005 * np.sin(date_idx / 5.0))
            rows.append(
                {
                    "date": date,
                    "ticker": "A",
                    "open_qfq": close * 0.99,
                    "high_qfq": close * 1.01,
                    "low_qfq": close * 0.98,
                    "close_qfq": close,
                    "volume": 1_000_000 + date_idx * 100,
                }
            )
        panel_df = pd.DataFrame(rows)

        feature_config = {
            "basket_name": "test",
            "windows": [5, 10, 20],
            "use_momentum": True,
            "use_volatility": True,
            "use_volume": True,
            "use_candle": True,
            "use_ta_indicators": False,
        }

        cutoff_date = dates[90]
        compare_date = dates[89]
        full_X, _, _ = build_features_and_labels_panel(
            panel_df,
            feature_config,
            date_col="date",
            ticker_col="ticker",
            include_cross_section=False,
        )
        truncated_X, _, _ = build_features_and_labels_panel(
            panel_df[panel_df["date"] <= cutoff_date].copy(),
            feature_config,
            date_col="date",
            ticker_col="ticker",
            include_cross_section=False,
        )

        full_row = full_X[full_X["date"] == compare_date].sort_values(["date", "ticker"])
        truncated_row = truncated_X[truncated_X["date"] == compare_date].sort_values(["date", "ticker"])

        assert not full_row.empty
        assert not truncated_row.empty

        feature_cols = [col for col in full_X.columns if col not in ["date", "ticker"]]
        np.testing.assert_allclose(
            full_row[feature_cols].to_numpy(),
            truncated_row[feature_cols].to_numpy(),
            atol=1e-10,
            rtol=1e-10,
        )
