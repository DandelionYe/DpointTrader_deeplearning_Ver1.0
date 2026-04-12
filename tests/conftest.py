# conftest.py
"""
Basket 模式测试夹具
==================

提供测试所需的样本数据和配置。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def minimal_basket_data():
    """
    Minimal basket data for testing (3 stocks, 100 days each).
    """
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")

    frames = []
    for ticker in ["TEST1", "TEST2", "TEST3"]:
        base_price = 10.0 + np.random.uniform(0, 5)

        returns = np.random.normal(0.0005, 0.02, n)
        close_prices = base_price * np.exp(np.cumsum(returns))
        open_prices = close_prices * (1 + np.random.uniform(-0.01, 0.01, n))
        high_prices = np.maximum(open_prices, close_prices) * (
            1 + np.abs(np.random.uniform(0, 0.02, n))
        )
        low_prices = np.minimum(open_prices, close_prices) * (
            1 - np.abs(np.random.uniform(0, 0.02, n))
        )
        volumes = np.random.uniform(1_000_000, 10_000_000, n)

        df = pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "open_qfq": open_prices,
                "high_qfq": high_prices,
                "low_qfq": low_prices,
                "close_qfq": close_prices,
                "volume": volumes,
            }
        )
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def minimal_price_data():
    """
    Minimal price data for testing (100 trading days).
    """
    np.random.seed(42)
    n = 100
    base_price = 10.0

    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")

    returns = np.random.normal(0.0005, 0.02, n)
    close_prices = base_price * np.exp(np.cumsum(returns))
    open_prices = close_prices * (1 + np.random.uniform(-0.01, 0.01, n))
    high_prices = np.maximum(open_prices, close_prices) * (
        1 + np.abs(np.random.uniform(0, 0.02, n))
    )
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.uniform(0, 0.02, n)))
    volumes = np.random.uniform(1_000_000, 10_000_000, n)
    amounts = volumes * close_prices

    df = pd.DataFrame(
        {
            "date": dates,
            "open_qfq": open_prices,
            "high_qfq": high_prices,
            "low_qfq": low_prices,
            "close_qfq": close_prices,
            "volume": volumes,
            "amount": amounts,
        }
    )

    return df


@pytest.fixture
def sample_portfolio_config():
    """
    Sample portfolio configuration for testing.
    """
    from portfolio_builder import PortfolioConfig

    return PortfolioConfig(
        top_k=3,
        weighting="equal",
        max_weight=0.33,
        cash_buffer=0.05,
    )


@pytest.fixture
def temp_basket_dir(tmp_path):
    """
    Temporary basket directory for testing.
    """
    basket_dir = tmp_path / "basket_test"
    basket_dir.mkdir()

    # Create sample CSV files
    np.random.seed(42)
    n = 50
    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")

    for ticker in ["600001", "600002", "600003"]:
        base_price = 10.0 + np.random.uniform(0, 5)
        returns = np.random.normal(0.0005, 0.02, n)
        close_prices = base_price * np.exp(np.cumsum(returns))
        open_prices = close_prices * (1 + np.random.uniform(-0.01, 0.01, n))
        high_prices = np.maximum(open_prices, close_prices) * (
            1 + np.abs(np.random.uniform(0, 0.02, n))
        )
        low_prices = np.minimum(open_prices, close_prices) * (
            1 - np.abs(np.random.uniform(0, 0.02, n))
        )
        volumes = np.random.uniform(1_000_000, 10_000_000, n)

        df = pd.DataFrame(
            {
                "Date": dates,
                "Open (CNY qfq)": open_prices,
                "High (CNY qfq)": high_prices,
                "Low (CNY qfq)": low_prices,
                "Close (CNY qfq)": close_prices,
                "Volume (shares)": volumes,
            }
        )
        df.to_csv(basket_dir / f"{ticker}.csv", index=False)

    return str(basket_dir)
