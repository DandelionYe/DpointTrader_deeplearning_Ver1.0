import numpy as np
import pandas as pd

from backtester_engine import backtest_from_scores, prepare_scores_for_backtest
from portfolio_builder import PortfolioConfig


def _sample_panel(n_days: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    tickers = ["A", "B"]
    rows = []
    for ticker_idx, ticker in enumerate(tickers):
        base = 10.0 + ticker_idx
        for day_idx, date in enumerate(dates):
            open_price = base + day_idx * 0.05
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open_qfq": open_price,
                    "high_qfq": open_price * 1.02,
                    "low_qfq": open_price * 0.98,
                    "close_qfq": open_price * 1.01,
                    "volume": 1_000_000 + day_idx * 100,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_scores_for_backtest_shifts_to_next_trade_day():
    panel_df = _sample_panel()
    signal_dates = sorted(panel_df["date"].unique())[:5]
    scores_df = pd.DataFrame(
        {
            "date": signal_dates * 2,
            "ticker": ["A"] * 5 + ["B"] * 5,
            "score": np.linspace(0.1, 1.0, 10),
        }
    )

    prepared = prepare_scores_for_backtest(panel_df, scores_df)
    assert "signal_date" in prepared.columns
    assert "trade_date" in prepared.columns
    assert (pd.to_datetime(prepared["trade_date"]) > pd.to_datetime(prepared["signal_date"])).all()


def test_backtest_marks_to_market_every_panel_date():
    panel_df = _sample_panel()
    signal_dates = sorted(panel_df["date"].unique())[5:10]
    scores_df = pd.DataFrame(
        {
            "date": signal_dates * 2,
            "ticker": ["A"] * 5 + ["B"] * 5,
            "score": np.linspace(0.1, 1.0, 10),
        }
    )
    config = PortfolioConfig(top_k=1, weighting="score", max_weight=1.0, cash_buffer=0.0, rebalance_freq="daily")
    result = backtest_from_scores(
        panel_df,
        scores_df,
        portfolio_config=config,
        start_date=panel_df["date"].min(),
        end_date=panel_df["date"].max(),
    )
    assert len(result.equity_curve) == panel_df["date"].nunique()


def test_monthly_rebalance_gates_order_generation():
    panel_df = _sample_panel(50)
    signal_dates = sorted(panel_df["date"].unique())[:-1]
    scores_df = pd.DataFrame(
        {
            "date": signal_dates * 2,
            "ticker": ["A"] * len(signal_dates) + ["B"] * len(signal_dates),
            "score": np.tile([0.9, 0.1], len(signal_dates)),
        }
    )
    daily_result = backtest_from_scores(
        panel_df,
        scores_df,
        portfolio_config=PortfolioConfig(top_k=1, weighting="score", max_weight=1.0, cash_buffer=0.0, rebalance_freq="daily"),
    )
    monthly_result = backtest_from_scores(
        panel_df,
        scores_df,
        portfolio_config=PortfolioConfig(top_k=1, weighting="score", max_weight=1.0, cash_buffer=0.0, rebalance_freq="monthly"),
    )
    assert len(monthly_result.orders) < len(daily_result.orders)
    assert monthly_result.equity_curve["is_rebalance_day"].sum() < daily_result.equity_curve["is_rebalance_day"].sum()
