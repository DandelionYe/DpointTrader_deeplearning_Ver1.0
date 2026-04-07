"""
测试split_plan功能
"""
import pytest
import pandas as pd
import numpy as np
from typing import List, Tuple

from splitters import (
    build_date_splits,
    walkforward_splits_by_date,
    walkforward_splits_with_embargo,
    nested_walkforward_splits_by_date,
    final_holdout_split_by_date,
)


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


class TestBuildDateSplits:
    """测试build_date_splits函数"""
    
    def test_wf_mode(self, sample_panel):
        """测试wf模式"""
        splits = build_date_splits(
            sample_panel,
            split_mode="wf",
            date_col="date",
            ticker_col="ticker",
            n_folds=4,
            train_start_ratio=0.5,
            min_rows=60,
        )
        
        assert isinstance(splits, list)
        assert len(splits) > 0
        
        # 检查返回的是日期元组列表
        for train_dates, val_dates in splits:
            assert isinstance(train_dates, list)
            assert isinstance(val_dates, list)
            assert len(train_dates) > 0
            assert len(val_dates) > 0
    
    def test_wf_embargo_mode(self, sample_panel):
        """测试wf_embargo模式"""
        splits = build_date_splits(
            sample_panel,
            split_mode="wf_embargo",
            date_col="date",
            ticker_col="ticker",
            n_folds=4,
            train_start_ratio=0.5,
            min_rows=60,
            embargo_days=5,
        )
        
        assert isinstance(splits, list)
        assert len(splits) > 0
        
        # 检查embargo gap
        for train_dates, val_dates in splits:
            max_train = max(train_dates)
            min_val = min(val_dates)
            gap = (min_val - max_train).days
            assert gap >= 5  # embargo_days
    
    def test_nested_wf_mode(self, sample_panel):
        """测试nested_wf模式"""
        splits = build_date_splits(
            sample_panel,
            split_mode="nested_wf",
            date_col="date",
            ticker_col="ticker",
            n_outer_folds=3,
            n_inner_folds=2,
            train_start_ratio=0.5,
            min_rows=60,
            embargo_days=5,
        )
        
        assert isinstance(splits, list)
        assert len(splits) > 0
        
        # 检查返回的是嵌套结构
        for outer_train_dates, outer_val_dates, inner_splits in splits:
            assert isinstance(outer_train_dates, list)
            assert isinstance(outer_val_dates, list)
            assert isinstance(inner_splits, list)
            assert len(inner_splits) > 0
    
    def test_invalid_mode(self, sample_panel):
        """测试无效模式"""
        with pytest.raises(ValueError, match="Invalid split_mode"):
            build_date_splits(
                sample_panel,
                split_mode="invalid",
                date_col="date",
                ticker_col="ticker",
            )


class TestHoldoutSplit:
    """测试holdout切分"""
    
    def test_holdout_split(self, sample_panel):
        """测试holdout切分"""
        search_df, holdout_df = final_holdout_split_by_date(
            sample_panel,
            date_col="date",
            ticker_col="ticker",
            holdout_ratio=0.15,
            min_holdout_rows=60,
            enforce_non_empty_search=True,
        )
        
        assert not search_df.empty
        assert not holdout_df.empty
        
        # 检查search和holdout无重叠
        search_dates = set(search_df["date"].unique())
        holdout_dates = set(holdout_df["date"].unique())
        assert len(search_dates & holdout_dates) == 0
        
        # 检查holdout日期晚于search日期
        assert max(search_dates) < min(holdout_dates)
    
    def test_holdout_enforce_non_empty_search(self, sample_panel):
        """测试enforce_non_empty_search"""
        # 使用非常大的holdout_ratio会导致search为空
        with pytest.raises(ValueError, match="Search set is empty"):
            final_holdout_split_by_date(
                sample_panel,
                date_col="date",
                ticker_col="ticker",
                holdout_ratio=1.0,
                min_holdout_rows=60,
                enforce_non_empty_search=True,
            )


class TestSplitPlanIntegration:
    """测试split_plan集成"""
    
    def test_wf_returns_indexed_splits(self, sample_panel):
        """测试wf模式返回indexed_splits"""
        from main_basket import dates_to_indices, build_split_plan
        import argparse
        
        # 创建mock args
        args = argparse.Namespace(
            split_mode="wf",
            use_holdout=0,
            holdout_ratio=0.15,
            min_holdout_rows=60,
            embargo_days=5,
            n_folds=4,
            n_outer_folds=3,
            n_inner_folds=2,
            train_start_ratio=0.5,
            split_min_rows=60,
        )
        
        # 添加label
        sample_panel["label"] = np.random.randn(len(sample_panel))
        
        X = sample_panel[["date", "ticker", "feature1", "feature2"]]
        y = sample_panel["label"]
        
        split_plan = build_split_plan(X, y, args, date_col="date", ticker_col="ticker")
        
        assert "search_X" in split_plan
        assert "search_y" in split_plan
        assert "holdout_X" in split_plan
        assert "holdout_y" in split_plan
        assert "indexed_splits" in split_plan
        assert "split_summary" in split_plan
        
        assert split_plan["holdout_X"] is None
        assert split_plan["holdout_y"] is None
        assert len(split_plan["indexed_splits"]) > 0
    
    def test_holdout_isolation(self, sample_panel):
        """测试holdout隔离"""
        from main_basket import build_split_plan
        import argparse
        
        args = argparse.Namespace(
            split_mode="wf",
            use_holdout=1,
            holdout_ratio=0.15,
            min_holdout_rows=60,
            embargo_days=5,
            n_folds=4,
            n_outer_folds=3,
            n_inner_folds=2,
            train_start_ratio=0.5,
            split_min_rows=60,
        )
        
        sample_panel["label"] = np.random.randn(len(sample_panel))
        
        X = sample_panel[["date", "ticker", "feature1", "feature2"]]
        y = sample_panel["label"]
        
        split_plan = build_split_plan(X, y, args, date_col="date", ticker_col="ticker")
        
        search_X = split_plan["search_X"]
        holdout_X = split_plan["holdout_X"]
        
        # 检查search和holdout无重叠
        search_indices = set(search_X.index)
        holdout_indices = set(holdout_X.index)
        assert len(search_indices & holdout_indices) == 0
        
        # 检查holdout日期晚于search日期
        assert search_X["date"].max() < holdout_X["date"].min()
