# allocator.py
"""
分配器模块
==========

本模块提供订单分配功能。

主要功能:
    - allocate_orders: 根据组合配置生成订单
    - rebalance_orders: 计算调仓订单
    - adjust_for_lot_size: 调整手数（A 股 100 股/手）

使用示例:
    >>> from allocator import allocate_orders, rebalance_orders
    >>> orders = allocate_orders(portfolio, prices, total_equity)
    >>> rebalance_orders = compute_rebalance_orders(current_holdings, target_portfolio)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from portfolio_builder import Portfolio

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """订单数据类。

    Attributes:
        ticker: 股票代码
        action: 买卖方向 ("buy" / "sell")
        shares: 股数
        target_weight: 目标权重
        current_weight: 当前权重
        weight_diff: 权重变化
        estimated_value: 预估金额
        priority: 优先级
    """
    ticker: str
    action: str
    shares: int
    target_weight: float = 0.0
    current_weight: float = 0.0
    weight_diff: float = 0.0
    estimated_value: float = 0.0
    priority: int = 0
    date: Optional[pd.Timestamp] = None


@dataclass
class AllocationResult:
    """分配结果。

    Attributes:
        orders: 订单列表
        total_buy_value: 总买入金额
        total_sell_value: 总卖出金额
        cash_used: 净使用现金
        cash_remaining: 剩余现金
        notes: 注释
    """
    orders: List[Order] = field(default_factory=list)
    total_buy_value: float = 0.0
    total_sell_value: float = 0.0
    cash_used: float = 0.0
    cash_remaining: float = 0.0
    notes: List[str] = field(default_factory=list)


def adjust_for_lot_size(
    shares: int,
    *,
    lot_size: int = 100,
    round_down: bool = True,
) -> int:
    """
    调整股数为手数整数倍（A 股 100 股/手）。

    Args:
        shares: 原始股数
        lot_size: 每手股数
        round_down: 是否向下取整（True=向下，False=向上）

    Returns:
        调整后的股数
    """
    if shares <= 0:
        return 0

    lots = shares // lot_size
    remainder = shares % lot_size

    if remainder > 0 and not round_down:
        lots += 1

    return lots * lot_size


def allocate_orders(
    portfolio: Portfolio,
    prices: Dict[str, float],
    total_equity: float,
    *,
    lot_size: int = 100,
    commission_rate: float = 0.0003,
) -> AllocationResult:
    """
    根据组合配置生成订单。

    Args:
        portfolio: 投资组合
        prices: 价格字典 {ticker: price}
        total_equity: 总权益
        lot_size: 每手股数
        commission_rate: 佣金率

    Returns:
        AllocationResult
    """
    orders: List[Order] = []
    total_buy_value = 0.0
    notes: List[str] = []

    for i, ticker in enumerate(portfolio.tickers):
        if ticker not in prices:
            notes.append(f"Price not found for {ticker}, skipping")
            continue

        price = prices[ticker]
        weight = portfolio.weights[i]
        target_value = total_equity * weight * (1 - commission_rate)
        target_shares = int(target_value / price)

        # 调整手数
        adjusted_shares = adjust_for_lot_size(target_shares, lot_size=lot_size, round_down=True)

        if adjusted_shares <= 0:
            notes.append(f"Shares too small for {ticker}, skipping")
            continue

        order = Order(
            ticker=ticker,
            action="buy",
            shares=adjusted_shares,
            target_weight=weight,
            estimated_value=adjusted_shares * price,
            priority=i,
            date=portfolio.date,
        )
        orders.append(order)
        total_buy_value += order.estimated_value

    result = AllocationResult(
        orders=orders,
        total_buy_value=total_buy_value,
        total_sell_value=0.0,
        cash_used=total_buy_value,
        cash_remaining=total_equity * portfolio.cash,
        notes=notes,
    )

    return result


def rebalance_orders(
    current_holdings: Dict[str, int],
    target_portfolio: Portfolio,
    prices: Dict[str, float],
    total_equity: float,
    *,
    lot_size: int = 100,
    commission_rate: float = 0.0003,
) -> AllocationResult:
    """
    计算调仓订单。

    Args:
        current_holdings: 当前持仓 {ticker: shares}
        target_portfolio: 目标组合
        prices: 价格字典
        total_equity: 总权益
        lot_size: 每手股数
        commission_rate: 佣金率

    Returns:
        AllocationResult
    """
    orders: List[Order] = []
    total_buy_value = 0.0
    total_sell_value = 0.0
    notes: List[str] = []

    # 计算目标持仓
    target_holdings: Dict[str, int] = {}
    for i, ticker in enumerate(target_portfolio.tickers):
        if ticker not in prices:
            continue
        weight = target_portfolio.weights[i]
        price = prices[ticker]
        target_value = total_equity * weight * (1 - commission_rate)
        target_shares = int(target_value / price)
        target_shares = adjust_for_lot_size(target_shares, lot_size=lot_size, round_down=True)
        target_holdings[ticker] = target_shares

    # 所有涉及的 ticker
    all_tickers = set(current_holdings.keys()) | set(target_holdings.keys())

    for ticker in all_tickers:
        current_shares = current_holdings.get(ticker, 0)
        target_shares = target_holdings.get(ticker, 0)
        diff = target_shares - current_shares

        if diff == 0:
            continue

        price = prices.get(ticker, 0)
        if price == 0:
            notes.append(f"Price not found for {ticker}, skipping")
            continue

        action = "buy" if diff > 0 else "sell"
        shares = abs(diff)
        estimated_value = shares * price

        # 计算权重
        current_weight = (current_shares * price) / total_equity if total_equity > 0 else 0
        target_weight = (target_shares * price) / total_equity if total_equity > 0 else 0

        order = Order(
            ticker=ticker,
            action=action,
            shares=shares,
            target_weight=target_weight,
            current_weight=current_weight,
            weight_diff=target_weight - current_weight,
            estimated_value=estimated_value,
            date=target_portfolio.date,
        )
        orders.append(order)

        if action == "buy":
            total_buy_value += estimated_value
        else:
            total_sell_value += estimated_value

    # 排序：先卖后买，优化资金使用
    orders.sort(key=lambda x: (0 if x.action == "sell" else 1, -x.priority))

    result = AllocationResult(
        orders=orders,
        total_buy_value=total_buy_value,
        total_sell_value=total_sell_value,
        cash_used=total_buy_value - total_sell_value,
        cash_remaining=total_equity * target_portfolio.cash + total_sell_value,
        notes=notes,
    )

    return result


def compute_turnover(
    old_portfolio: Optional[Portfolio],
    new_portfolio: Portfolio,
    prices: Dict[str, float],
    total_equity: float,
) -> float:
    """
    计算调仓换手率。

    Args:
        old_portfolio: 旧组合
        new_portfolio: 新组合
        prices: 价格字典
        total_equity: 总权益

    Returns:
        换手率（0-1）
    """
    if old_portfolio is None:
        # 新建仓，不算换手
        return 0.0

    # 计算权重变化
    old_weights = {t: w for t, w in zip(old_portfolio.tickers, old_portfolio.weights)}
    new_weights = {t: w for t, w in zip(new_portfolio.tickers, new_portfolio.weights)}

    all_tickers = set(old_weights.keys()) | set(new_weights.keys())
    turnover = 0.0

    for ticker in all_tickers:
        old_w = old_weights.get(ticker, 0)
        new_w = new_weights.get(ticker, 0)
        turnover += abs(new_w - old_w)

    # 换手率 = 权重变化总和 / 2
    turnover = turnover / 2

    return turnover


# =========================================================
# 公开 API 导出
# =========================================================
__all__ = [
    "Order",
    "AllocationResult",
    "adjust_for_lot_size",
    "allocate_orders",
    "rebalance_orders",
    "compute_turnover",
]
