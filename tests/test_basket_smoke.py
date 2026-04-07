# test_basket_smoke.py
"""
Basket 模式冒烟测试
==================

测试 basket 模式的端到端流程。

运行测试:
    pytest test_basket_smoke.py -v
"""
import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np

# 添加父目录到路径以便导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from constants import (
    DEFAULT_DATA_ROOT,
    DEFAULT_BASKET_NAME,
)


def create_test_basket_data(basket_path: str, n_tickers: int = 3, n_days: int = 100):
    """创建测试 basket 数据"""
    os.makedirs(basket_path, exist_ok=True)

    tickers = [f"TEST{i:06d}" for i in range(n_tickers)]
    dates = pd.date_range("2024-01-01", periods=n_days)

    for ticker in tickers:
        np.random.seed(hash(ticker) % 2**32)

        # 生成随机价格序列
        close = 10.0 * np.cumprod(1 + np.random.randn(n_days) * 0.02)
        open_ = close * (1 + np.random.randn(n_days) * 0.01)
        high = np.maximum(open_, close) * (1 + np.abs(np.random.randn(n_days) * 0.01))
        low = np.minimum(open_, close) * (1 - np.abs(np.random.randn(n_days) * 0.01))
        volume = np.random.randint(1000, 10000, n_days)

        df = pd.DataFrame({
            "Date": dates,
            "Open (CNY, qfq)": open_,
            "High (CNY, qfq)": high,
            "Low (CNY, qfq)": low,
            "Close (CNY, qfq)": close,
            "Volume (shares)": volume,
        })

        df.to_csv(os.path.join(basket_path, f"{ticker}.csv"), index=False)

    return tickers


class TestBasketSmoke:
    """Basket 冒烟测试"""

    def test_basket_data_creation(self):
        """测试 basket 数据创建"""
        with tempfile.TemporaryDirectory() as tmpdir:
            basket_path = os.path.join(tmpdir, "basket_test")
            tickers = create_test_basket_data(basket_path, n_tickers=3, n_days=50)

            # 验证文件创建
            assert os.path.isdir(basket_path)
            assert len(os.listdir(basket_path)) == 3

            # 验证文件内容
            for ticker in tickers:
                file_path = os.path.join(basket_path, f"{ticker}.csv")
                assert os.path.exists(file_path)

                df = pd.read_csv(file_path)
                assert len(df) == 50
                assert "Date" in df.columns
                assert "Close (CNY, qfq)" in df.columns

    def test_basket_loader_integration(self):
        """测试 basket 加载器集成"""
        from basket_loader import load_basket_folder

        with tempfile.TemporaryDirectory() as tmpdir:
            basket_path = os.path.join(tmpdir, "basket_test")
            tickers = create_test_basket_data(basket_path, n_tickers=3, n_days=50)

            panel_df, report, meta = load_basket_folder(basket_path)

            # 验证加载结果
            assert meta.n_tickers == 3
            assert report.total_rows == 150  # 3 * 50
            assert "ticker" in panel_df.columns
            assert "date" in panel_df.columns
            assert panel_df["ticker"].nunique() == 3

    def test_panel_builder_integration(self):
        """测试 panel 构建器集成"""
        from basket_loader import load_basket_folder
        from panel_builder import validate_panel

        with tempfile.TemporaryDirectory() as tmpdir:
            basket_path = os.path.join(tmpdir, "basket_test")
            create_test_basket_data(basket_path, n_tickers=3, n_days=50)

            panel_df, _, _ = load_basket_folder(basket_path)
            valid, issues = validate_panel(panel_df)

            # 验证 panel 结构
            assert valid is True or len(issues) == 0

    def test_feature_engineering_integration(self):
        """测试特征工程集成"""
        from basket_loader import load_basket_folder
        from feature_dpoint import build_features_and_labels_panel

        with tempfile.TemporaryDirectory() as tmpdir:
            basket_path = os.path.join(tmpdir, "basket_test")
            create_test_basket_data(basket_path, n_tickers=3, n_days=100)

            panel_df, _, _ = load_basket_folder(basket_path)

            config = {
                "basket_name": "test",
                "windows": [5, 10],
                "use_momentum": True,
                "use_volatility": True,
                "use_volume": True,
                "use_candle": True,
                "use_ta_indicators": False,
            }

            X, y, meta = build_features_and_labels_panel(
                panel_df,
                config,
                date_col="date",
                ticker_col="ticker",
                include_cross_section=True,
            )

            # 验证特征
            assert "date" in X.columns
            assert "ticker" in X.columns
            assert len(meta.feature_names) > 0
            assert meta.n_tickers == 3

    def test_portfolio_builder_integration(self):
        """测试组合构建器集成"""
        from portfolio_builder import PortfolioConfig, build_portfolio

        # 创建模拟分数
        dates = pd.date_range("2024-01-01", periods=10)
        tickers = ["A", "B", "C", "D", "E"]

        rows = []
        np.random.seed(42)
        for date in dates:
            for ticker in tickers:
                rows.append({
                    "date": date,
                    "ticker": ticker,
                    "score": np.random.randn(),
                })

        scores_df = pd.DataFrame(rows)

        # 构建组合
        config = PortfolioConfig(top_k=3, weighting="equal")
        portfolio = build_portfolio(
            scores_df,
            date=dates[0],
            config=config,
            score_col="score",
            ticker_col="ticker",
            date_col="date",
        )

        # 验证组合
        assert portfolio.n_holdings == 3
        assert len(portfolio.tickers) == 3
        assert abs(sum(portfolio.weights) - 0.6) < 0.001
        assert abs(portfolio.cash - 0.4) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
