import numpy as np
import pandas as pd

from allocator import Order
from backtester_engine import prepare_scores_for_backtest
from execution_engine import ExecutionEngine, TradingConstraints


def _price_panel(
    *,
    prev_close: float = 10.0,
    open_price: float = 10.0,
    volume: float = 1_000_000,
    board: str = "",
    is_st: int = 0,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    prev_date = pd.Timestamp("2024-01-02")
    trade_date = pd.Timestamp("2024-01-03")
    panel_df = pd.DataFrame(
        [
            {
                "date": prev_date,
                "ticker": "A",
                "open_qfq": prev_close,
                "high_qfq": prev_close,
                "low_qfq": prev_close,
                "close_qfq": prev_close,
                "volume": volume,
                "board": board,
                "is_st": is_st,
            },
            {
                "date": trade_date,
                "ticker": "A",
                "open_qfq": open_price,
                "high_qfq": open_price,
                "low_qfq": open_price,
                "close_qfq": open_price,
                "volume": volume,
                "board": board,
                "is_st": is_st,
            },
        ]
    )
    return panel_df, prev_date, trade_date


def test_st_limit_up_rejects_buy():
    price_df, prev_date, trade_date = _price_panel(prev_close=10.0, open_price=10.5, is_st=1)
    engine = ExecutionEngine(constraints=TradingConstraints(min_trade_value=0.0))
    prev_close = float(price_df.loc[price_df["date"] == prev_date, "close_qfq"].iloc[0])

    fill = engine.execute_order(Order(ticker="A", action="buy", shares=100, date=trade_date), price_df, prev_close)

    assert fill.status == "rejected"
    assert fill.reject_reason == "limit_up"


def test_chinext_allows_move_below_twenty_percent_limit():
    price_df, prev_date, trade_date = _price_panel(prev_close=10.0, open_price=11.5, board="创业板")
    engine = ExecutionEngine(constraints=TradingConstraints(min_trade_value=0.0))
    prev_close = float(price_df.loc[price_df["date"] == prev_date, "close_qfq"].iloc[0])

    fill = engine.execute_order(Order(ticker="A", action="buy", shares=100, date=trade_date), price_df, prev_close)

    assert fill.status == "filled"
    assert fill.filled_shares == 100


def test_suspended_security_rejects_order():
    price_df, prev_date, trade_date = _price_panel(prev_close=10.0, open_price=np.nan)
    engine = ExecutionEngine(constraints=TradingConstraints(min_trade_value=0.0))
    prev_close = float(price_df.loc[price_df["date"] == prev_date, "close_qfq"].iloc[0])

    fill = engine.execute_order(Order(ticker="A", action="buy", shares=100, date=trade_date), price_df, prev_close)

    assert fill.status == "rejected"
    assert fill.reject_reason == "suspended"


def test_t_plus_one_rejects_same_day_sell_after_buy():
    price_df, prev_date, trade_date = _price_panel(prev_close=10.0, open_price=10.0)
    engine = ExecutionEngine(constraints=TradingConstraints(min_trade_value=0.0))
    prev_close = float(price_df.loc[price_df["date"] == prev_date, "close_qfq"].iloc[0])

    buy_fill = engine.execute_order(Order(ticker="A", action="buy", shares=100, date=trade_date), price_df, prev_close)
    sell_fill = engine.execute_order(Order(ticker="A", action="sell", shares=100, date=trade_date), price_df, prev_close)

    assert buy_fill.status == "filled"
    assert sell_fill.status == "rejected"
    assert sell_fill.reject_reason == "t_plus_1"


def test_volume_cap_allows_partial_fill_when_enabled():
    price_df, prev_date, trade_date = _price_panel(prev_close=10.0, open_price=10.0, volume=3_500)
    constraints = TradingConstraints(max_participation_rate=0.10, allow_partial_fill=True, min_trade_value=0.0)
    engine = ExecutionEngine(constraints=constraints)
    prev_close = float(price_df.loc[price_df["date"] == prev_date, "close_qfq"].iloc[0])

    fill = engine.execute_order(Order(ticker="A", action="buy", shares=1_000, date=trade_date), price_df, prev_close)

    assert fill.status == "filled"
    assert fill.filled_shares == 300


def test_board_lot_round_to_zero_rejects_small_order():
    price_df, prev_date, trade_date = _price_panel(prev_close=10.0, open_price=10.0)
    engine = ExecutionEngine(constraints=TradingConstraints(min_trade_value=0.0))
    prev_close = float(price_df.loc[price_df["date"] == prev_date, "close_qfq"].iloc[0])

    fill = engine.execute_order(Order(ticker="A", action="buy", shares=50, date=trade_date), price_df, prev_close)

    assert fill.status == "rejected"
    assert fill.reject_reason == "board_lot_round_to_zero"


def test_sell_stamp_duty_is_accounted_for_separately():
    price_df, prev_date, trade_date = _price_panel(prev_close=10.0, open_price=10.0)
    next_date = pd.Timestamp("2024-01-04")
    next_row = price_df[price_df["date"] == trade_date].copy()
    next_row["date"] = next_date
    next_row["open_qfq"] = 11.0
    next_row["high_qfq"] = 11.0
    next_row["low_qfq"] = 11.0
    next_row["close_qfq"] = 11.0
    full_price_df = pd.concat([price_df, next_row], ignore_index=True)

    constraints = TradingConstraints(min_trade_value=0.0)
    engine = ExecutionEngine(constraints=constraints)
    first_prev_close = float(full_price_df.loc[full_price_df["date"] == prev_date, "close_qfq"].iloc[0])
    buy_fill = engine.execute_order(Order(ticker="A", action="buy", shares=100, date=trade_date), full_price_df, first_prev_close)
    engine.reset_daily()
    second_prev_close = float(full_price_df.loc[full_price_df["date"] == trade_date, "close_qfq"].iloc[0])
    sell_fill = engine.execute_order(Order(ticker="A", action="sell", shares=100, date=next_date), full_price_df, second_prev_close)

    assert buy_fill.status == "filled"
    assert sell_fill.status == "filled"
    assert sell_fill.stamp_duty > 0
    assert abs(sell_fill.stamp_duty - sell_fill.filled_shares * sell_fill.fill_price * engine.stamp_duty_sell) < 1e-9
    assert engine.get_stats()["total_stamp_duty"] == sell_fill.stamp_duty


def test_missing_volume_rejects_order():
    price_df, prev_date, trade_date = _price_panel(prev_close=10.0, open_price=10.0, volume=np.nan)
    engine = ExecutionEngine(constraints=TradingConstraints(min_trade_value=0.0))
    prev_close = float(price_df.loc[price_df["date"] == prev_date, "close_qfq"].iloc[0])

    fill = engine.execute_order(Order(ticker="A", action="buy", shares=100, date=trade_date), price_df, prev_close)

    assert fill.status == "rejected"
    assert fill.reject_reason == "volume_cap"


def test_prepare_scores_for_backtest_adds_tradeability_diagnostics():
    panel_df, _, signal_trade_date = _price_panel(prev_close=10.0, open_price=11.0)
    scores_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")],
            "ticker": ["A"],
            "score": [1.0],
        }
    )

    prepared = prepare_scores_for_backtest(panel_df, scores_df, execution_lag_days=1, drop_untradable_signals=False)

    assert "is_tradeable" in prepared.columns
    assert "expected_drop_reason" in prepared.columns
    assert "resolved_limit_up_price" in prepared.columns
    assert "resolved_limit_down_price" in prepared.columns
    assert pd.Timestamp(prepared["trade_date"].iloc[0]) == signal_trade_date
