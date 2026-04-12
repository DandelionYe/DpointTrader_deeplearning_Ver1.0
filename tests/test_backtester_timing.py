import numpy as np
import pandas as pd

from backtester_engine import (
    _build_rebalance_calendar,
    _current_execution_prices_no_carry,
    backtest_from_scores,
    prepare_scores_for_backtest,
    validate_prepared_scores,
)
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


def test_prepare_scores_for_backtest_drops_untradable_terminal_signal():
    panel_df = _sample_panel()
    last_date = pd.to_datetime(panel_df["date"].max())
    scores_df = pd.DataFrame(
        {
            "date": [last_date],
            "ticker": ["A"],
            "score": [0.5],
        }
    )

    prepared = prepare_scores_for_backtest(
        panel_df, scores_df, execution_lag_days=1, drop_untradable_signals=True
    )
    assert prepared.empty


def test_prepare_scores_for_backtest_supports_execution_lag_two():
    panel_df = _sample_panel()
    signal_date = sorted(panel_df["date"].unique())[0]
    scores_df = pd.DataFrame({"date": [signal_date], "ticker": ["A"], "score": [1.0]})

    prepared = prepare_scores_for_backtest(panel_df, scores_df, execution_lag_days=2)

    expected_trade_date = sorted(pd.to_datetime(panel_df["date"].unique()))[2]
    assert pd.Timestamp(prepared["trade_date"].iloc[0]) == expected_trade_date


def test_prepare_scores_for_backtest_is_idempotent_for_prepared_scores():
    panel_df = _sample_panel()
    scores_df = pd.DataFrame({"date": [panel_df["date"].iloc[0]], "ticker": ["A"], "score": [0.5]})
    prepared = prepare_scores_for_backtest(panel_df, scores_df, execution_lag_days=1)

    prepared_again = prepare_scores_for_backtest(panel_df, prepared, execution_lag_days=2)

    pd.testing.assert_series_equal(
        pd.to_datetime(prepared["trade_date"]).reset_index(drop=True),
        pd.to_datetime(prepared_again["trade_date"]).reset_index(drop=True),
    )


def test_execution_prices_do_not_carry_forward_missing_open():
    panel_df = _sample_panel(5)
    day = pd.Timestamp(panel_df["date"].iloc[1])
    day_prices = panel_df[panel_df["date"] == day].copy()
    day_prices.loc[day_prices["ticker"] == "B", "open_qfq"] = np.nan

    execution_prices = _current_execution_prices_no_carry(
        day_prices,
        ticker_col="ticker",
        price_col="open_qfq",
    )

    assert "A" in execution_prices
    assert "B" not in execution_prices


def test_build_rebalance_calendar_monthly_uses_first_trade_day():
    panel_df = _sample_panel(50)
    dates = sorted(pd.to_datetime(panel_df["date"].unique()))

    calendar = _build_rebalance_calendar(dates, rebalance_freq="monthly", anchor="first")

    assert len(calendar) >= 2
    first_jan = min([d for d in dates if d.month == 1])
    first_feb = min([d for d in dates if d.month == 2])
    assert first_jan in calendar
    assert first_feb in calendar


def test_build_rebalance_calendar_monthly_uses_last_trade_day():
    panel_df = _sample_panel(50)
    dates = sorted(pd.to_datetime(panel_df["date"].unique()))

    calendar = _build_rebalance_calendar(dates, rebalance_freq="monthly", anchor="last")

    last_jan = max([d for d in dates if d.month == 1])
    last_feb = max([d for d in dates if d.month == 2])
    assert last_jan in calendar
    assert last_feb in calendar


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
    prepared_scores = prepare_scores_for_backtest(panel_df, scores_df)
    config = PortfolioConfig(
        top_k=1, weighting="score", max_weight=1.0, cash_buffer=0.0, rebalance_freq="daily"
    )
    result = backtest_from_scores(
        panel_df,
        prepared_scores,
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
    prepared_scores = prepare_scores_for_backtest(panel_df, scores_df)
    daily_result = backtest_from_scores(
        panel_df,
        prepared_scores,
        portfolio_config=PortfolioConfig(
            top_k=1, weighting="score", max_weight=1.0, cash_buffer=0.0, rebalance_freq="daily"
        ),
    )
    monthly_result = backtest_from_scores(
        panel_df,
        prepared_scores,
        portfolio_config=PortfolioConfig(
            top_k=1, weighting="score", max_weight=1.0, cash_buffer=0.0, rebalance_freq="monthly"
        ),
    )
    assert len(monthly_result.orders) < len(daily_result.orders)
    assert (
        monthly_result.equity_curve["is_rebalance_day"].sum()
        < daily_result.equity_curve["is_rebalance_day"].sum()
    )


def test_backtest_requires_prepared_scores():
    panel_df = _sample_panel()
    raw_scores = pd.DataFrame(
        {
            "date": [panel_df["date"].iloc[0]],
            "ticker": ["A"],
            "score": [0.2],
        }
    )

    try:
        backtest_from_scores(
            panel_df,
            raw_scores,
            portfolio_config=PortfolioConfig(
                top_k=1, weighting="score", max_weight=1.0, cash_buffer=0.0, rebalance_freq="daily"
            ),
        )
    except ValueError as exc:
        assert "prepared columns" in str(exc)
    else:
        raise AssertionError(
            "backtest_from_scores should reject raw scores without trade_date/signal_date"
        )


def test_validate_prepared_scores_rejects_missing_trade_date():
    scores_df = pd.DataFrame({"signal_date": [pd.Timestamp("2024-01-01")]})
    try:
        validate_prepared_scores(scores_df)
    except ValueError as exc:
        assert "prepared columns" in str(exc)
    else:
        raise AssertionError("validate_prepared_scores should reject missing trade_date")


def test_backtest_without_execution_price_only_marks_to_market_no_order():
    panel_df = _sample_panel(6)
    target_day = sorted(pd.to_datetime(panel_df["date"].unique()))[2]
    panel_df.loc[
        (panel_df["date"] == target_day) & (panel_df["ticker"] == "A"),
        "open_qfq",
    ] = np.nan
    scores_df = pd.DataFrame(
        {
            "date": [sorted(pd.to_datetime(panel_df["date"].unique()))[1]],
            "ticker": ["A"],
            "score": [1.0],
        }
    )
    prepared_scores = prepare_scores_for_backtest(panel_df, scores_df)

    result = backtest_from_scores(
        panel_df,
        prepared_scores,
        portfolio_config=PortfolioConfig(
            top_k=1, weighting="score", max_weight=1.0, cash_buffer=0.0, rebalance_freq="daily"
        ),
    )

    assert result.orders.empty
    assert len(result.equity_curve) == panel_df["date"].nunique()
    assert result.execution_stats["rebalance_days_without_tradable_scores"] >= 1
