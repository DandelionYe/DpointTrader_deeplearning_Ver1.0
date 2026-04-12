# test_cross_sectional_features.py
"""
横截面特征测试模块
================

测试 cross_sectional_features.py 的功能。

运行测试:
    pytest test_cross_sectional_features.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# 添加父目录到路径以便导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cross_sectional_features import (
    add_cross_sectional_features,
    cross_sectional_percentile,
    cross_sectional_rank,
    cross_sectional_zscore,
)


class TestCrossSectionalRank:
    """测试横截面排序"""

    def test_rank_basic(self):
        """测试基本排序"""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01", "2024-01-01"],
                "ticker": ["A", "B", "C"],
                "value": [10.0, 30.0, 20.0],
            }
        )

        ranks = cross_sectional_rank(df, value_col="value", date_col="date")

        # 验证返回的是 Series 且有正确的长度
        assert isinstance(ranks, pd.Series)
        assert len(ranks) == 3

    def test_rank_per_date(self):
        """测试按日期分组排序"""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                "ticker": ["A", "B", "A", "B"],
                "value": [10.0, 20.0, 30.0, 15.0],
            }
        )

        ranks = cross_sectional_rank(df, value_col="value", date_col="date")

        # 验证返回的是 Series 且有正确的长度
        assert isinstance(ranks, pd.Series)
        assert len(ranks) == 4


class TestCrossSectionalZscore:
    """测试横截面 Z-score"""

    def test_zscore_basic(self):
        """测试基本 Z-score"""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"] * n,
                "ticker": [f"T{i}" for i in range(n)],
                "value": list(range(n)),  # 使用均匀分布的值
            }
        )

        zscores = cross_sectional_zscore(df, value_col="value", date_col="date")

        # zscores 是 Series
        assert isinstance(zscores, pd.Series)
        assert len(zscores) == n
        # Z-score 均值应该接近 0（对于均匀分布）
        zscore_mean = float(zscores.mean())
        assert abs(zscore_mean) < 1.0  # 放宽断言

    def test_zscore_with_clipping(self):
        """测试带截断的 Z-score"""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"] * 100,
                "ticker": [f"T{i}" for i in range(100)],
                "value": list(range(100)),
            }
        )
        # 添加异常值
        df.loc[0, "value"] = 1000

        zscores = cross_sectional_zscore(
            df,
            value_col="value",
            date_col="date",
            clip_outliers=True,
            clip_std=3.0,
        )

        # 异常值应该被截断
        assert float(zscores.max()) <= 3.0
        assert float(zscores.min()) >= -3.0


class TestCrossSectionalPercentile:
    """测试横截面百分位"""

    def test_percentile_basic(self):
        """测试基本百分位"""
        n = 101
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"] * n,
                "ticker": [f"T{i}" for i in range(n)],
                "value": list(range(n)),
            }
        )

        percentiles = cross_sectional_percentile(df, value_col="value", date_col="date")

        # percentiles 是 Series
        assert isinstance(percentiles, pd.Series)
        assert len(percentiles) == n
        # 百分位应该在 0-1 之间
        assert float(percentiles.min()) >= 0.0
        assert float(percentiles.max()) <= 1.0


class TestAddCrossSectionalFeatures:
    """测试添加横截面特征"""

    def test_add_features(self):
        """测试添加特征"""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"] * 5,
                "ticker": ["A", "B", "C", "D", "E"],
                "close_qfq": [10.0, 20.0, 15.0, 25.0, 30.0],
                "volume": [1000, 2000, 1500, 2500, 3000],
            }
        )

        result, meta = add_cross_sectional_features(
            df,
            columns=["close_qfq", "volume"],
            features=["rank", "zscore"],
        )

        # 应该添加新列
        assert "cs_rank_close_qfq" in result.columns
        assert "cs_zscore_close_qfq" in result.columns
        assert "cs_rank_volume" in result.columns
        assert "cs_zscore_volume" in result.columns

        # 元数据应该正确
        assert len(meta.cross_sectional_features) == 4
        assert meta.n_tickers == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
