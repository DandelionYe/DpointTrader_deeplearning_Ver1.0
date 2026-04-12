"""
测试holdout隔离性
"""

import numpy as np
import pandas as pd
import pytest

from splitters import final_holdout_split_by_date


@pytest.fixture
def sample_panel():
    """创建样本panel数据"""
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


class TestHoldoutIsolation:
    """测试holdout隔离性"""

    def test_search_holdout_no_overlap(self, sample_panel):
        """测试search和holdout无重叠"""
        search_df, holdout_df = final_holdout_split_by_date(
            sample_panel,
            date_col="date",
            ticker_col="ticker",
            holdout_ratio=0.15,
            min_holdout_rows=60,
            enforce_non_empty_search=True,
        )

        # 检查索引无重叠
        search_indices = set(search_df.index)
        holdout_indices = set(holdout_df.index)
        assert len(search_indices & holdout_indices) == 0

    def test_holdout_dates_after_search(self, sample_panel):
        """测试holdout日期晚于search日期"""
        search_df, holdout_df = final_holdout_split_by_date(
            sample_panel,
            date_col="date",
            ticker_col="ticker",
            holdout_ratio=0.15,
            min_holdout_rows=60,
            enforce_non_empty_search=True,
        )

        search_max_date = search_df["date"].max()
        holdout_min_date = holdout_df["date"].min()

        assert search_max_date < holdout_min_date

    def test_search_and_holdout_cover_all_dates(self, sample_panel):
        """测试search和holdout覆盖所有日期"""
        search_df, holdout_df = final_holdout_split_by_date(
            sample_panel,
            date_col="date",
            ticker_col="ticker",
            holdout_ratio=0.15,
            min_holdout_rows=60,
            enforce_non_empty_search=True,
        )

        all_dates = set(sample_panel["date"].unique())
        search_dates = set(search_df["date"].unique())
        holdout_dates = set(holdout_df["date"].unique())

        assert search_dates | holdout_dates == all_dates
        assert len(search_dates & holdout_dates) == 0

    def test_holdout_ratio_approximately_correct(self, sample_panel):
        """测试holdout比例大致正确"""
        search_df, holdout_df = final_holdout_split_by_date(
            sample_panel,
            date_col="date",
            ticker_col="ticker",
            holdout_ratio=0.15,
            min_holdout_rows=60,
            enforce_non_empty_search=True,
        )

        total_rows = len(sample_panel)
        holdout_rows = len(holdout_df)
        actual_ratio = holdout_rows / total_rows

        # 允许一定误差
        assert abs(actual_ratio - 0.15) < 0.05

    def test_different_tickers_in_both_sets(self, sample_panel):
        """测试search和holdout都包含所有ticker"""
        search_df, holdout_df = final_holdout_split_by_date(
            sample_panel,
            date_col="date",
            ticker_col="ticker",
            holdout_ratio=0.15,
            min_holdout_rows=60,
            enforce_non_empty_search=True,
        )

        all_tickers = set(sample_panel["ticker"].unique())
        search_tickers = set(search_df["ticker"].unique())
        holdout_tickers = set(holdout_df["ticker"].unique())

        assert search_tickers == all_tickers
        assert holdout_tickers == all_tickers
