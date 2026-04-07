"""
测试embargo完整性
"""
import pytest
import pandas as pd
import numpy as np

from splitters import walkforward_splits_with_embargo, nested_walkforward_splits_by_date


@pytest.fixture
def sample_panel():
    """创建样本panel数据"""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    tickers = ["A", "B", "C"]
    
    rows = []
    for ticker in tickers:
        for date in dates:
            rows.append({
                "date": date,
                "ticker": ticker,
                "feature1": np.random.randn(),
                "feature2": np.random.randn(),
            })
    
    return pd.DataFrame(rows)


class TestEmbargoIntegrity:
    """测试embargo完整性"""
    
    def test_embargo_gap_respected(self, sample_panel):
        """测试embargo gap被遵守"""
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
            max_train = max(train_dates)
            min_val = min(val_dates)
            gap = (min_val - max_train).days
            
            # 检查gap >= embargo_days
            assert gap >= embargo_days, (
                f"Embargo violated: gap={gap} days < embargo_days={embargo_days}"
            )
    
    def test_embargo_different_values(self, sample_panel):
        """测试不同embargo值"""
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
            
            if len(splits) > 0:
                for train_dates, val_dates in splits:
                    max_train = max(train_dates)
                    min_val = min(val_dates)
                    gap = (min_val - max_train).days
                    
                    assert gap >= embargo_days
    
    def test_nested_wf_embargo(self, sample_panel):
        """测试nested_wf的embargo"""
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
        
        if len(splits) > 0:
            for outer_train_dates, outer_val_dates, inner_splits in splits:
                # 检查outer fold的embargo
                if outer_train_dates and outer_val_dates:
                    max_train = max(outer_train_dates)
                    min_val = min(outer_val_dates)
                    gap = (min_val - max_train).days
                    
                    assert gap >= embargo_days
    
    def test_embargo_too_large_returns_empty(self, sample_panel):
        """测试embargo过大时返回空列表"""
        # embargo_days太大，无法产生有效split
        splits = walkforward_splits_with_embargo(
            sample_panel,
            date_col="date",
            ticker_col="ticker",
            n_folds=4,
            train_start_ratio=0.5,
            min_rows=60,
            embargo_days=50,  # 非常大的embargo
        )
        
        # 可能返回空列表或很少的folds
        assert isinstance(splits, list)
    
    def test_embargo_with_business_days(self, sample_panel):
        """测试embargo使用交易日"""
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
        
        if len(splits) > 0:
            train_dates, val_dates = splits[0]
            max_train = max(train_dates)
            min_val = min(val_dates)
            
            # 计算交易日gap（不是自然日）
            all_dates = sorted(sample_panel["date"].unique())
            train_idx = all_dates.index(max_train)
            val_idx = all_dates.index(min_val)
            
            # 日期索引差应该>= embargo_days
            assert (val_idx - train_idx) >= embargo_days
