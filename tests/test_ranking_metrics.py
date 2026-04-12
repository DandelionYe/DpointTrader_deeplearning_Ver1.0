# test_ranking_metrics.py
"""
排序评估指标测试模块
==================

测试 ranking_metrics.py 的功能。

运行测试:
    pytest test_ranking_metrics.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# 添加父目录到路径以便导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ranking_metrics import (
    compute_all_ranking_metrics,
    compute_ic,
    compute_layered_returns,
    compute_rank_ic,
    compute_topk_return,
)


class TestComputeIC:
    """测试 IC 计算"""

    def test_compute_ic(self):
        """测试 IC 计算"""
        # 需要更多数据点来计算 IC
        np.random.seed(42)
        n = 50
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"] * n,
                "ticker": [f"T{i}" for i in range(n)],
                "score": np.random.randn(n),
                "label": np.random.randint(0, 2, n),
            }
        )
        # 让 score 和 label 有一定相关性
        df["label"] = (df["score"] + np.random.randn(n) * 0.5) > 0

        ic_series = compute_ic(df, score_col="score", label_col="label", date_col="date")

        assert len(ic_series) == 1
        # IC 值应该在 -1 到 1 之间
        assert -1 <= ic_series.iloc[0] <= 1


class TestComputeRankIC:
    """测试 RankIC 计算"""

    def test_compute_rank_ic(self):
        """测试 RankIC 计算"""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01", "2024-01-01"],
                "ticker": ["A", "B", "C"],
                "score": [0.9, 0.6, 0.3],
                "label": [1, 0.5, 0],
            }
        )

        rank_ic_series = compute_rank_ic(df, score_col="score", label_col="label", date_col="date")

        assert len(rank_ic_series) == 1
        # 完全正相关，RankIC 应该接近 1
        assert rank_ic_series.iloc[0] > 0.9


class TestComputeTopkReturn:
    """测试 TopK 收益计算"""

    def test_compute_topk_return(self):
        """测试 TopK 收益计算"""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"] * 5,
                "ticker": ["A", "B", "C", "D", "E"],
                "score": [0.9, 0.8, 0.7, 0.6, 0.5],
                "label": [0.05, 0.03, 0.02, 0.01, -0.01],  # 收益率
            }
        )

        topk_return = compute_topk_return(
            df,
            score_col="score",
            label_col="label",
            date_col="date",
            top_k=3,
        )

        assert len(topk_return) == 1
        # Top3 平均收益应该介于最高和最低之间
        assert 0.01 < topk_return.iloc[0] < 0.05


class TestComputeLayeredReturns:
    """测试分层收益计算"""

    def test_compute_layered_returns(self):
        """测试分层收益计算"""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"] * n,
                "ticker": [f"T{i}" for i in range(n)],
                "score": np.random.randn(n),
                "label": np.random.randn(n) * 0.1,
            }
        )

        # 让 score 和 label 正相关
        df["label"] = df["score"] * 0.1 + np.random.randn(n) * 0.01

        layer_returns = compute_layered_returns(
            df,
            score_col="score",
            label_col="label",
            date_col="date",
            n_layers=5,
        )

        assert len(layer_returns) == 5
        # 理论上 L5 > L1（分数越高收益越高）
        assert layer_returns["L5"].iloc[0] > layer_returns["L1"].iloc[0]


class TestComputeAllMetrics:
    """测试所有指标计算"""

    def test_compute_all_ranking_metrics(self):
        """测试计算所有排序指标"""
        np.random.seed(42)
        n_dates = 20
        n_tickers = 10

        dates = pd.date_range("2024-01-01", periods=n_dates)
        rows = []

        for date in dates:
            for i in range(n_tickers):
                score = np.random.randn()
                label = score * 0.3 + np.random.randn() * 0.1  # 正相关
                rows.append(
                    {
                        "date": date,
                        "ticker": f"T{i}",
                        "score": score,
                        "label": 1 if label > 0 else 0,
                    }
                )

        df = pd.DataFrame(rows)

        metrics = compute_all_ranking_metrics(
            df,
            score_col="score",
            label_col="label",
            date_col="date",
            ticker_col="ticker",
            top_k=3,
            n_layers=5,
        )

        assert metrics.ic_mean is not None
        assert metrics.rank_ic_mean is not None
        assert metrics.topk_return_mean is not None
        assert metrics.layered_returns is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
