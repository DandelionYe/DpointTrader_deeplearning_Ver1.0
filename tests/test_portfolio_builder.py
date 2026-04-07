# test_portfolio_builder.py
"""
组合构建器测试模块
================

测试 portfolio_builder.py 的功能。

运行测试:
    pytest test_portfolio_builder.py -v
"""
import os
import sys
import pytest
import pandas as pd

# 添加父目录到路径以便导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portfolio_builder import (
    PortfolioConfig,
    Portfolio,
    select_topk,
    compute_weights_equal,
    compute_weights_score,
    build_portfolio,
)


class TestPortfolioConfig:
    """测试组合配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = PortfolioConfig()

        assert config.top_k == 5
        assert config.weighting == "equal"
        assert config.max_weight == 0.20


class TestSelectTopk:
    """测试 TopK 选择"""

    def test_select_topk(self):
        """测试选择 TopK 股票"""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"] * 5),
            "ticker": ["A", "B", "C", "D", "E"],
            "score": [0.8, 0.6, 0.9, 0.7, 0.5],
        })

        topk = select_topk(df, date=pd.to_datetime("2024-01-01"), top_k=3)

        assert len(topk) == 3
        assert list(topk["ticker"]) == ["C", "A", "D"]  # 按 score 降序

    def test_select_topk_with_min_score(self):
        """测试带最低分数阈值"""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"] * 3),
            "ticker": ["A", "B", "C"],
            "score": [0.8, 0.3, 0.9],
        })

        topk = select_topk(df, date=pd.to_datetime("2024-01-01"), top_k=3, min_score=0.5)

        assert len(topk) == 2
        assert all(topk["score"] >= 0.5)


class TestComputeWeights:
    """测试权重计算"""

    def test_equal_weights(self):
        """测试等权"""
        tickers = ["A", "B", "C", "D", "E"]
        weights = compute_weights_equal(tickers)

        assert len(weights) == 5
        assert all(abs(w - 0.2) < 0.001 for w in weights)

    def test_equal_weights_with_max(self):
        """测试带权重上限的等权"""
        tickers = ["A", "B", "C"]
        weights = compute_weights_equal(tickers, max_weight=0.25)

        assert len(weights) == 3
        # 3 只股票等权是 0.33，上限 0.25 时应严格不超限，剩余留作现金
        assert all(w <= 0.25 + 1e-9 for w in weights)
        assert abs(sum(weights) - 0.75) < 0.001

    def test_score_weights(self):
        """测试按分数加权"""
        tickers = ["A", "B", "C"]
        scores = [0.9, 0.6, 0.3]
        weights = compute_weights_score(tickers, scores, max_weight=1.0)

        assert len(weights) == 3
        # 分数越高权重越大（允许小的浮点误差）
        assert weights[0] >= weights[1] >= weights[2]
        assert abs(sum(weights) - 1.0) < 0.001

    def test_score_weights_respect_max_weight(self):
        """测试分数加权严格遵守单票上限"""
        tickers = ["A", "B"]
        scores = [0.9, 0.8]
        weights = compute_weights_score(tickers, scores, max_weight=0.2)

        assert all(w <= 0.2 + 1e-9 for w in weights)
        assert abs(sum(weights) - 0.4) < 0.001


class TestBuildPortfolio:
    """测试构建组合"""

    def test_build_portfolio(self):
        """测试构建组合"""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"] * 5),
            "ticker": ["A", "B", "C", "D", "E"],
            "score": [0.8, 0.6, 0.9, 0.7, 0.5],
        })

        config = PortfolioConfig(top_k=3, weighting="equal")

        portfolio = build_portfolio(
            df,
            date=pd.to_datetime("2024-01-01"),
            config=config,
        )

        assert portfolio.n_holdings == 3
        assert len(portfolio.tickers) == 3
        assert len(portfolio.weights) == 3
        assert abs(sum(portfolio.weights) - 0.6) < 0.001
        assert abs(portfolio.cash - 0.4) < 0.001

    def test_build_portfolio_reserves_cash_buffer(self):
        """测试组合对象保留现金缓冲"""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"] * 3),
            "ticker": ["A", "B", "C"],
            "score": [0.9, 0.8, 0.7],
        })

        config = PortfolioConfig(top_k=3, weighting="equal", cash_buffer=0.10)
        portfolio = build_portfolio(
            df,
            date=pd.to_datetime("2024-01-01"),
            config=config,
        )

        assert sum(portfolio.weights) <= 0.9 + 0.001
        assert portfolio.cash >= 0.1 - 0.001
    
    def test_portfolio_respects_max_weight_under_score_weighting(self):
        """测试score-weighting下遵守max_weight约束"""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"] * 5),
            "ticker": ["A", "B", "C", "D", "E"],
            "score": [0.9, 0.8, 0.7, 0.6, 0.5],
        })
        
        config = PortfolioConfig(
            top_k=5,
            weighting="score",
            max_weight=0.3,  # 限制单个股票最大权重
            cash_buffer=0.1,
        )
        
        portfolio = build_portfolio(
            df,
            date=pd.to_datetime("2024-01-01"),
            config=config,
        )
        
        # 检查所有权重都不超过max_weight
        for weight in portfolio.weights:
            assert weight <= config.max_weight + 0.001, (
                f"Weight {weight} exceeds max_weight {config.max_weight}"
            )
        
        # 检查总权重+现金=1
        total = sum(portfolio.weights) + portfolio.cash
        assert abs(total - 1.0) < 0.001
    
    def test_portfolio_empty_when_scores_empty(self):
        """测试scores为空时组合不会崩溃"""
        # 空DataFrame
        df_empty = pd.DataFrame(columns=["date", "ticker", "score"])
        
        config = PortfolioConfig(top_k=5, weighting="equal")
        
        # 不应该抛出异常
        portfolio = build_portfolio(
            df_empty,
            date=pd.to_datetime("2024-01-01"),
            config=config,
        )
        
        assert portfolio.n_holdings == 0
        assert len(portfolio.tickers) == 0
        assert len(portfolio.weights) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
