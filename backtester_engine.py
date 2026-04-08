from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from allocator import allocate_orders, rebalance_orders
from execution_engine import ExecutionEngine
from portfolio_builder import PortfolioConfig, build_portfolio

logger = logging.getLogger(__name__)


def _current_prices_with_carry(
    day_prices: pd.DataFrame,
    prev_prices: Dict[str, float],
    *,
    ticker_col: str,
    price_col: str,
) -> Dict[str, float]:
    prices = prev_prices.copy()
    if not day_prices.empty:
        prices.update(dict(zip(day_prices[ticker_col], day_prices[price_col])))
    return prices


def _seed_prev_closes(
    panel_df: pd.DataFrame,
    start_date: pd.Timestamp,
    *,
    date_col: str,
    ticker_col: str,
    close_col: str,
) -> Dict[str, float]:
    history = panel_df[panel_df[date_col] < start_date].copy()
    if history.empty:
        return {}
    history = history.sort_values([ticker_col, date_col])
    last_close = history.groupby(ticker_col, as_index=False).tail(1)
    return dict(zip(last_close[ticker_col], last_close[close_col]))


def prepare_scores_for_backtest(
    panel_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    *,
    date_col: str = "date",
    trade_date_col: str = "trade_date",
    signal_date_col: str = "signal_date",
    execution_lag_days: int = 1,
    drop_untradable_signals: bool = True,
) -> pd.DataFrame:
    if scores_df.empty:
        prepared = scores_df.copy()
        if signal_date_col not in prepared.columns and date_col in prepared.columns:
            prepared[signal_date_col] = prepared[date_col]
        if trade_date_col not in prepared.columns:
            prepared[trade_date_col] = pd.NaT
        return prepared

    prepared = scores_df.copy()
    prepared[date_col] = pd.to_datetime(prepared[date_col])
    if signal_date_col not in prepared.columns:
        prepared[signal_date_col] = prepared[date_col]
    else:
        prepared[signal_date_col] = pd.to_datetime(prepared[signal_date_col])

    if trade_date_col in prepared.columns:
        prepared[trade_date_col] = pd.to_datetime(prepared[trade_date_col])
        if drop_untradable_signals:
            return prepared[prepared[trade_date_col].notna()].copy()
        return prepared.copy()

    trading_dates = pd.Index(sorted(pd.to_datetime(panel_df[date_col].unique())))
    if trading_dates.empty:
        prepared[trade_date_col] = pd.NaT
        if drop_untradable_signals:
            return prepared.iloc[0:0].copy()
        return prepared

    signal_dates = pd.DatetimeIndex(prepared[date_col])
    lag = max(1, int(execution_lag_days))
    next_positions = trading_dates.searchsorted(signal_dates, side="right") + (lag - 1)
    prepared[trade_date_col] = [
        trading_dates[pos] if pos < len(trading_dates) else pd.NaT
        for pos in next_positions
    ]
    if drop_untradable_signals:
        return prepared[prepared[trade_date_col].notna()].copy()
    return prepared


def _current_execution_prices_no_carry(
    day_prices: pd.DataFrame,
    *,
    ticker_col: str,
    price_col: str,
) -> Dict[str, float]:
    if day_prices.empty:
        return {}
    px = day_prices[[ticker_col, price_col]].dropna()
    px = px[px[price_col] > 0]
    return dict(zip(px[ticker_col], px[price_col]))


def _is_rebalance_day(
    current_date: pd.Timestamp,
    previous_date: Optional[pd.Timestamp],
    *,
    rebalance_freq: str,
) -> bool:
    if previous_date is None:
        return True
    if rebalance_freq == "daily":
        return True
    if rebalance_freq == "weekly":
        current_iso = current_date.isocalendar()
        previous_iso = previous_date.isocalendar()
        return (current_iso.year, current_iso.week) != (previous_iso.year, previous_iso.week)
    if rebalance_freq == "monthly":
        return (current_date.year, current_date.month) != (previous_date.year, previous_date.month)
    raise ValueError(f"Unsupported rebalance_freq: {rebalance_freq}")


def _rebalance_bucket_key(date: pd.Timestamp, *, rebalance_freq: str) -> str:
    if rebalance_freq == "daily":
        return str(pd.Timestamp(date).date())
    if rebalance_freq == "weekly":
        iso = date.isocalendar()
        return f"{iso.year}-W{int(iso.week):02d}"
    if rebalance_freq == "monthly":
        return f"{date.year}-{date.month:02d}"
    raise ValueError(f"Unsupported rebalance_freq: {rebalance_freq}")


def _build_rebalance_calendar(
    dates: List[pd.Timestamp],
    *,
    rebalance_freq: str,
    anchor: str = "first",
) -> Dict[pd.Timestamp, str]:
    dates = sorted(pd.to_datetime(dates))
    if rebalance_freq == "daily":
        return {pd.Timestamp(date): _rebalance_bucket_key(pd.Timestamp(date), rebalance_freq=rebalance_freq) for date in dates}

    buckets: Dict[str, List[pd.Timestamp]] = {}
    for date in dates:
        bucket = _rebalance_bucket_key(pd.Timestamp(date), rebalance_freq=rebalance_freq)
        buckets.setdefault(bucket, []).append(pd.Timestamp(date))

    rebalance_calendar: Dict[pd.Timestamp, str] = {}
    for bucket, bucket_dates in buckets.items():
        chosen = bucket_dates[0] if anchor == "first" else bucket_dates[-1]
        rebalance_calendar[pd.Timestamp(chosen)] = bucket
    return rebalance_calendar


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    orders: pd.DataFrame
    fills: pd.DataFrame
    portfolio_config: PortfolioConfig
    execution_stats: Dict
    notes: List[str] = field(default_factory=list)


def backtest_from_scores(
    panel_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    *,
    portfolio_config: PortfolioConfig,
    score_col: str = "score",
    ticker_col: str = "ticker",
    date_col: str = "date",
    trade_date_col: str = "trade_date",
    price_cols: Optional[Dict[str, str]] = None,
    initial_cash: float = 100000.0,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> BacktestResult:
    notes: List[str] = []
    raw_panel_df = panel_df.copy()

    if price_cols is None:
        price_cols = {
            "open": "open_qfq",
            "close": "close_qfq",
            "high": "high_qfq",
            "low": "low_qfq",
        }

    scores_df = prepare_scores_for_backtest(
        raw_panel_df,
        scores_df,
        date_col=date_col,
        trade_date_col=trade_date_col,
    )

    if start_date is not None:
        panel_df = panel_df[panel_df[date_col] >= start_date].copy()
        scores_df = scores_df[scores_df[trade_date_col] >= start_date].copy()
    if end_date is not None:
        panel_df = panel_df[panel_df[date_col] <= end_date].copy()
        scores_df = scores_df[scores_df[trade_date_col] <= end_date].copy()

    engine = ExecutionEngine(initial_cash=initial_cash)
    dates = sorted(pd.to_datetime(panel_df[date_col].unique()))

    if not dates:
        empty_equity = pd.DataFrame(columns=["date", "equity", "cash", "market_value", "n_holdings", "is_rebalance_day"])
        return BacktestResult(
            equity_curve=empty_equity,
            trades=pd.DataFrame(),
            positions=pd.DataFrame(),
            orders=pd.DataFrame(),
            fills=pd.DataFrame(),
            portfolio_config=portfolio_config,
            execution_stats=engine.get_stats(),
            notes=notes,
        )

    score_groups = {
        pd.Timestamp(trade_date): group.copy()
        for trade_date, group in scores_df.groupby(trade_date_col)
    }

    all_orders: List[Dict[str, object]] = []
    all_fills: List[Dict[str, object]] = []
    all_positions: List[Dict[str, object]] = []
    equity_rows: List[Dict[str, object]] = []

    current_holdings: Dict[str, int] = {}
    prev_closes = _seed_prev_closes(
        raw_panel_df,
        dates[0],
        date_col=date_col,
        ticker_col=ticker_col,
        close_col=price_cols["close"],
    )
    rebalance_calendar = _build_rebalance_calendar(dates, rebalance_freq=portfolio_config.rebalance_freq, anchor="first")

    for date in dates:
        day_prices = panel_df[panel_df[date_col] == date]
        close_prices = _current_prices_with_carry(
            day_prices,
            prev_closes,
            ticker_col=ticker_col,
            price_col=price_cols["close"],
        )
        execution_prices = _current_execution_prices_no_carry(
            day_prices,
            ticker_col=ticker_col,
            price_col=price_cols["open"],
        )

        day_scores = score_groups.get(pd.Timestamp(date), pd.DataFrame())
        should_rebalance = bool(
            pd.Timestamp(date) in rebalance_calendar and not day_scores.empty
        )
        rebalance_bucket = rebalance_calendar.get(pd.Timestamp(date))

        if should_rebalance:
            tradable_scores = day_scores[day_scores[ticker_col].isin(execution_prices.keys())].copy()
            if tradable_scores.empty:
                notes.append(f"{pd.Timestamp(date).date()}: rebalance day had no tradable scores at execution open.")
            else:
                target_portfolio = build_portfolio(
                    tradable_scores,
                    date=pd.Timestamp(date),
                    config=portfolio_config,
                    score_col=score_col,
                    ticker_col=ticker_col,
                    date_col=trade_date_col,
                )
                total_equity = engine.cash + engine.position_book.total_market_value(close_prices)
                if not current_holdings:
                    alloc_result = allocate_orders(
                        target_portfolio,
                        execution_prices,
                        total_equity,
                    )
                else:
                    alloc_result = rebalance_orders(
                        current_holdings,
                        target_portfolio,
                        execution_prices,
                        total_equity,
                    )

                if alloc_result.orders:
                    fills = engine.execute_orders(
                        alloc_result.orders,
                        panel_df,
                        prev_closes=prev_closes.copy() if prev_closes else None,
                    )
                    for order in alloc_result.orders:
                        all_orders.append(
                            {
                                "date": order.date,
                                "ticker": order.ticker,
                                "action": order.action,
                                "shares": order.shares,
                                "target_weight": order.target_weight,
                                "estimated_value": order.estimated_value,
                            }
                        )
                    for fill in fills:
                        if fill.status == "filled":
                            all_fills.append(
                                {
                                    "date": date,
                                    "ticker": fill.order.ticker,
                                    "action": fill.order.action,
                                    "filled_shares": fill.filled_shares,
                                    "fill_price": fill.fill_price,
                                    "commission": fill.commission,
                                    "slippage_cost": fill.slippage_cost,
                                }
                            )

        current_holdings = {
            pos.ticker: pos.shares
            for pos in engine.position_book.get_positions()
        }

        equity = engine.position_book.total_equity(close_prices)
        equity_rows.append(
            {
                "date": date,
                "equity": equity,
                "cash": engine.cash,
                "market_value": engine.position_book.total_market_value(close_prices),
                "n_holdings": len(current_holdings),
                "is_rebalance_day": should_rebalance,
                "rebalance_bucket": rebalance_bucket if should_rebalance else None,
            }
        )

        for pos in engine.position_book.get_positions():
            all_positions.append(
                {
                    "date": date,
                    "ticker": pos.ticker,
                    "shares": pos.shares,
                    "avg_cost": pos.avg_cost,
                    "market_value": pos.shares * close_prices.get(pos.ticker, 0.0),
                }
            )

        prev_closes = close_prices.copy()
        engine.reset_daily()

    equity_curve = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(all_fills) if all_fills else pd.DataFrame()
    positions = pd.DataFrame(all_positions) if all_positions else pd.DataFrame()
    orders = pd.DataFrame(all_orders) if all_orders else pd.DataFrame()
    fills = pd.DataFrame(all_fills) if all_fills else pd.DataFrame()

    if not equity_curve.empty:
        equity_curve["cum_return"] = equity_curve["equity"] / equity_curve["equity"].iloc[0] - 1
        equity_curve["daily_return"] = equity_curve["equity"].pct_change()

    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        positions=positions,
        orders=orders,
        fills=fills,
        portfolio_config=portfolio_config,
        execution_stats=engine.get_stats(),
        notes=notes,
    )


def compute_buy_and_hold_benchmark(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close_qfq",
    initial_cash: float = 100000.0,
    commission_rate: float = 0.0003,
) -> pd.DataFrame:
    dates = sorted(panel_df[date_col].unique())
    tickers = panel_df[ticker_col].unique().tolist()
    n_tickers = len(tickers)

    if n_tickers == 0 or len(dates) == 0:
        return pd.DataFrame()

    first_date = dates[0]
    first_day_prices = panel_df[panel_df[date_col] == first_date]
    per_ticker_cash = initial_cash / n_tickers

    holdings: Dict[str, int] = {}
    cash = float(initial_cash)
    for _, row in first_day_prices.iterrows():
        ticker = row[ticker_col]
        price = row["open_qfq"]
        shares = int((per_ticker_cash / price / (1 + commission_rate)) / 100) * 100
        if shares > 0:
            holdings[ticker] = shares
            cash -= shares * price * (1 + commission_rate)

    equity_curve: List[Dict[str, object]] = []
    prev_prices: Dict[str, float] = {}
    for date in dates:
        day_prices = panel_df[panel_df[date_col] == date]
        price_dict = _current_prices_with_carry(
            day_prices,
            prev_prices,
            ticker_col=ticker_col,
            price_col=close_col,
        )

        market_value = sum(
            shares * price_dict.get(ticker, 0.0)
            for ticker, shares in holdings.items()
        )
        equity = cash + market_value
        equity_curve.append(
            {
                "date": date,
                "bnh_equity": equity,
                "bnh_cum_return": equity / initial_cash - 1,
            }
        )
        prev_prices = price_dict.copy()

    return pd.DataFrame(equity_curve)


__all__ = [
    "BacktestResult",
    "_build_rebalance_calendar",
    "backtest_from_scores",
    "compute_buy_and_hold_benchmark",
    "prepare_scores_for_backtest",
]
