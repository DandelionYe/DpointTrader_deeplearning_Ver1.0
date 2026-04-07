# backtester_engine.py
"""
组合回测引擎
============

本模块提供多股票组合回测功能。

主要功能:
    - backtest_from_scores: 从预测分数进行回测
    - BacktestResult: 回测结果数据类

使用示例:
    >>> from backtester_engine import backtest_from_scores
    >>> result = backtest_from_scores(panel_df, scores_df, portfolio_config)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from portfolio_builder import PortfolioConfig, build_portfolio, build_portfolio_series, portfolio_to_df
from allocator import rebalance_orders, compute_turnover
from execution_engine import ExecutionEngine, Fill
from position_book import PositionBook

logger = logging.getLogger(__name__)


def _current_prices_with_carry(
    day_prices: pd.DataFrame,
    prev_closes: Dict[str, float],
    *,
    ticker_col: str,
    close_col: str,
) -> Dict[str, float]:
    """使用当日收盘价更新可用价格，缺失 ticker 沿用上一可用收盘价。"""
    prices = prev_closes.copy()
    if not day_prices.empty:
        prices.update(dict(zip(day_prices[ticker_col], day_prices[close_col])))
    return prices


@dataclass
class BacktestResult:
    """回测结果。

    Attributes:
        equity_curve: 权益曲线 DataFrame
        trades: 交易记录 DataFrame
        positions: 持仓记录 DataFrame
        orders: 订单记录 DataFrame
        fills: 成交记录 DataFrame
        portfolio_config: 组合配置
        execution_stats: 执行统计
        notes: 注释
    """
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
    price_cols: Optional[Dict[str, str]] = None,
    initial_cash: float = 100000.0,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> BacktestResult:
    """
    从预测分数进行组合回测。

    Args:
        panel_df: 行情 panel DataFrame
        scores_df: 预测分数 DataFrame
        portfolio_config: 组合配置
        score_col: 分数列名
        ticker_col: ticker 列名
        date_col: 日期列名
        price_cols: 价格列名映射
        initial_cash: 初始现金
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        BacktestResult
    """
    notes: List[str] = []

    if price_cols is None:
        price_cols = {
            "open": "open_qfq",
            "close": "close_qfq",
            "high": "high_qfq",
            "low": "low_qfq",
        }

    # 过滤日期范围
    if start_date:
        panel_df = panel_df[panel_df[date_col] >= start_date].copy()
        scores_df = scores_df[scores_df[date_col] >= start_date].copy()
    if end_date:
        panel_df = panel_df[panel_df[date_col] <= end_date].copy()
        scores_df = scores_df[scores_df[date_col] <= end_date].copy()

    # 初始化执行引擎
    engine = ExecutionEngine(initial_cash=initial_cash)

    # 获取交易日期
    dates = sorted(scores_df[date_col].unique())

    # 存储结果
    all_orders = []
    all_fills = []
    all_positions = []
    equity_rows = []

    # 当前持仓
    current_holdings: Dict[str, int] = {}
    prev_portfolio: Optional = None
    prev_closes: Dict[str, float] = {}

    for i, date in enumerate(dates):
        # 获取当日价格
        day_prices = panel_df[panel_df[date_col] == date]
        price_dict = _current_prices_with_carry(
            day_prices,
            prev_closes,
            ticker_col=ticker_col,
            close_col=price_cols["close"],
        )
        open_dict = dict(zip(day_prices[ticker_col], day_prices[price_cols["open"]]))

        # 获取当日分数
        day_scores = scores_df[scores_df[date_col] == date]
        if day_scores.empty:
            # 无预测，保持持仓
            engine.reset_daily()
            equity = engine.position_book.total_equity(price_dict)
            equity_rows.append({
                "date": date,
                "equity": equity,
                "cash": engine.cash,
                "market_value": engine.position_book.total_market_value(price_dict),
                "n_holdings": len(engine.position_book.get_positions()),
            })
            prev_closes = price_dict.copy()
            continue

        # 构建目标组合
        target_portfolio = build_portfolio(
            day_scores,
            date=date,
            config=portfolio_config,
            score_col=score_col,
            ticker_col=ticker_col,
            date_col=date_col,
        )

        # 计算调仓订单
        if i == 0:
            # 初始建仓
            from allocator import allocate_orders
            alloc_result = allocate_orders(
                target_portfolio,
                price_dict,
                engine.cash + engine.position_book.total_market_value(price_dict),
            )
        else:
            # 调仓
            alloc_result = rebalance_orders(
                current_holdings,
                target_portfolio,
                price_dict,
                engine.cash + engine.position_book.total_market_value(price_dict),
            )

        # 执行订单
        if alloc_result.orders:
            # 准备 prev_closes
            prev_closes_for_exec = prev_closes.copy() if prev_closes else None

            fills = engine.execute_orders(
                alloc_result.orders,
                panel_df,
                prev_closes=prev_closes_for_exec,
            )

            # 记录订单和成交
            for order in alloc_result.orders:
                all_orders.append({
                    "date": order.date,
                    "ticker": order.ticker,
                    "action": order.action,
                    "shares": order.shares,
                    "target_weight": order.target_weight,
                    "estimated_value": order.estimated_value,
                })

            for fill in fills:
                if fill.status == "filled":
                    all_fills.append({
                        "date": date,
                        "ticker": fill.order.ticker,
                        "action": fill.order.action,
                        "filled_shares": fill.filled_shares,
                        "fill_price": fill.fill_price,
                        "commission": fill.commission,
                        "slippage_cost": fill.slippage_cost,
                    })

        # 更新当前持仓
        current_holdings = {
            pos.ticker: pos.shares
            for pos in engine.position_book.get_positions()
        }

        # 记录权益
        equity = engine.position_book.total_equity(price_dict)
        equity_rows.append({
            "date": date,
            "equity": equity,
            "cash": engine.cash,
            "market_value": engine.position_book.total_market_value(price_dict),
            "n_holdings": len(current_holdings),
        })

        # 记录持仓快照
        for pos in engine.position_book.get_positions():
            all_positions.append({
                "date": date,
                "ticker": pos.ticker,
                "shares": pos.shares,
                "avg_cost": pos.avg_cost,
                "market_value": pos.shares * price_dict.get(pos.ticker, 0),
            })

        prev_portfolio = target_portfolio
        prev_closes = price_dict.copy()
        engine.reset_daily()

    # 生成结果 DataFrame
    equity_curve = pd.DataFrame(equity_rows)

    trades = pd.DataFrame(all_fills) if all_fills else pd.DataFrame()
    positions = pd.DataFrame(all_positions) if all_positions else pd.DataFrame()
    orders = pd.DataFrame(all_orders) if all_orders else pd.DataFrame()
    fills = pd.DataFrame(all_fills) if all_fills else pd.DataFrame()

    # 计算累计收益
    if not equity_curve.empty:
        equity_curve["cum_return"] = equity_curve["equity"] / equity_curve["equity"].iloc[0] - 1
        equity_curve["daily_return"] = equity_curve["equity"].pct_change()

    result = BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        positions=positions,
        orders=orders,
        fills=fills,
        portfolio_config=portfolio_config,
        execution_stats=engine.get_stats(),
        notes=notes,
    )

    return result


def compute_buy_and_hold_benchmark(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close_qfq",
    initial_cash: float = 100000.0,
    commission_rate: float = 0.0003,
) -> pd.DataFrame:
    """
    计算等权持有基准。

    Args:
        panel_df: 行情 panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        close_col: 收盘价列名
        initial_cash: 初始现金
        commission_rate: 佣金率

    Returns:
        基准权益曲线 DataFrame
    """
    dates = sorted(panel_df[date_col].unique())
    tickers = panel_df[ticker_col].unique().tolist()
    n_tickers = len(tickers)

    if n_tickers == 0 or len(dates) == 0:
        return pd.DataFrame()

    # 第一日等权买入所有股票
    first_date = dates[0]
    first_day_prices = panel_df[panel_df[date_col] == first_date]
    per_ticker_cash = initial_cash / n_tickers

    holdings = {}
    cash = float(initial_cash)
    for _, row in first_day_prices.iterrows():
        ticker = row[ticker_col]
        price = row["open_qfq"]
        shares = int((per_ticker_cash / price / (1 + commission_rate)) / 100) * 100
        if shares > 0:
            holdings[ticker] = shares
            cash -= shares * price * (1 + commission_rate)

    # 计算每日权益
    equity_curve = []
    prev_prices: Dict[str, float] = {}
    for date in dates:
        day_prices = panel_df[panel_df[date_col] == date]
        price_dict = _current_prices_with_carry(
            day_prices,
            prev_prices,
            ticker_col=ticker_col,
            close_col=close_col,
        )

        market_value = sum(
            shares * price_dict.get(ticker, 0)
            for ticker, shares in holdings.items()
        )
        equity = cash + market_value
        equity_curve.append({
            "date": date,
            "bnh_equity": equity,
            "bnh_cum_return": equity / initial_cash - 1,
        })
        prev_prices = price_dict.copy()

    return pd.DataFrame(equity_curve)


# =========================================================
# 公开 API 导出
# =========================================================
__all__ = [
    "BacktestResult",
    "backtest_from_scores",
    "compute_buy_and_hold_benchmark",
]
