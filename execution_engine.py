# execution_engine.py
"""
执行引擎模块
============

本模块提供订单执行模拟功能。

主要功能:
    - ExecutionEngine: 执行引擎类
    - 模拟 A 股 T+1、涨跌停、停牌等约束
    - 滑点和成本计算

使用示例:
    >>> from execution_engine import ExecutionEngine
    >>> engine = ExecutionEngine(initial_cash=100000)
    >>> fills = engine.execute_orders(orders, price_df)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from allocator import Order, AllocationResult
from position_book import PositionBook, Position

logger = logging.getLogger(__name__)


# A 股执行常量
DEFAULT_SLIPPAGE_BPS: int = 20  # 20 bps = 0.2%
DEFAULT_LIMIT_UP_PCT: float = 0.10  # 涨停幅度 10%
DEFAULT_LIMIT_DOWN_PCT: float = 0.10  # 跌停幅度 10%
ST_LIMIT_PCT: float = 0.05  # ST 股 5%
COMMISSION_RATE_BUY: float = 0.0003  # 买入佣金 0.03%
COMMISSION_RATE_SELL: float = 0.0013  # 卖出佣金 + 印花税 0.13%


@dataclass
class Fill:
    """成交数据类。

    Attributes:
        order: 原始订单
        filled_shares: 成交股数
        fill_price: 成交价格
        commission: 佣金
        slippage_cost: 滑点成本
        status: 状态 ("filled" / "partial" / "rejected")
        reject_reason: 拒绝原因
    """
    order: Order
    filled_shares: int = 0
    fill_price: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0
    status: str = "pending"
    reject_reason: Optional[str] = None


@dataclass
class ExecutionStats:
    """执行统计。

    Attributes:
        orders_submitted: 提交订单数
        orders_filled: 成交订单数
        orders_rejected: 拒绝订单数
        total_commission: 总佣金
        total_slippage: 总滑点成本
        reject_reasons: 拒绝原因统计
    """
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    reject_reasons: Dict[str, int] = field(default_factory=dict)


class ExecutionEngine:
    """
    执行引擎。

    模拟 A 股交易约束：
        - T+1：当日买入次日才能卖出
        - 涨跌停限制：涨停买不进，跌停卖不出
        - 停牌：停牌无法交易
        - 手数：100 股整数倍
        - 资金约束：买入金额不能超过可用现金
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        *,
        slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
        limit_up_pct: float = DEFAULT_LIMIT_UP_PCT,
        limit_down_pct: float = DEFAULT_LIMIT_DOWN_PCT,
        commission_buy: float = COMMISSION_RATE_BUY,
        commission_sell: float = COMMISSION_RATE_SELL,
    ):
        """
        初始化执行引擎。

        Args:
            initial_cash: 初始现金
            slippage_bps: 滑点 (bps)
            limit_up_pct: 涨停幅度
            limit_down_pct: 跌停幅度
            commission_buy: 买入佣金率
            commission_sell: 卖出佣金率
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.slippage_bps = slippage_bps
        self.limit_up_pct = limit_up_pct
        self.limit_down_pct = limit_down_pct
        self.commission_buy = commission_buy
        self.commission_sell = commission_sell

        self.position_book = PositionBook(initial_cash=initial_cash)
        self.fills: List[Fill] = []
        self.stats = ExecutionStats()

        # T+1 约束：记录当日买入的 ticker
        self._today_buys: set = set()

    def reset_daily(self) -> None:
        """每日重置（清空当日买入记录）"""
        self._today_buys = set()

    def check_limit_up(
        self,
        ticker: str,
        current_price: float,
        prev_close: float,
    ) -> bool:
        """检查是否涨停"""
        if prev_close <= 0:
            return False
        limit_up = prev_close * (1 + self.limit_up_pct)
        return current_price >= limit_up

    def check_limit_down(
        self,
        ticker: str,
        current_price: float,
        prev_close: float,
    ) -> bool:
        """检查是否跌停"""
        if prev_close <= 0:
            return False
        limit_down = prev_close * (1 - self.limit_down_pct)
        return current_price <= limit_down

    def check_suspended(
        self,
        ticker: str,
        price: Optional[float],
    ) -> bool:
        """检查是否停牌（价格为 None 或 0 视为停牌）"""
        return price is None or price <= 0

    def apply_slippage(
        self,
        price: float,
        action: str,
    ) -> float:
        """
        应用滑点。

        Args:
            price: 基准价格
            action: 买卖方向

        Returns:
            滑点后价格
        """
        slippage_pct = self.slippage_bps / 10000
        if action == "buy":
            return price * (1 + slippage_pct)
        else:
            return price * (1 - slippage_pct)

    def execute_order(
        self,
        order: Order,
        price_df: pd.DataFrame,
        prev_close: Optional[float] = None,
    ) -> Fill:
        """
        执行单个订单。

        Args:
            order: 订单
            price_df: 价格 DataFrame（含 date/ticker/open/high/low/close）
            prev_close: 前收盘价

        Returns:
            Fill
        """
        self.stats.orders_submitted += 1

        ticker = order.ticker
        date = order.date
        action = order.action
        shares = order.shares

        # 获取当日价格
        day_prices = price_df[
            (price_df["date"] == date) & (price_df["ticker"] == ticker)
        ]

        if day_prices.empty:
            fill = Fill(
                order=order,
                status="rejected",
                reject_reason="no_price_data",
            )
            self.stats.orders_rejected += 1
            self.stats.reject_reasons["no_price_data"] = (
                self.stats.reject_reasons.get("no_price_data", 0) + 1
            )
            return fill

        row = day_prices.iloc[0]
        open_price = row.get("open_qfq", row.get("close_qfq", 0))
        high = row.get("high_qfq", open_price)
        low = row.get("low_qfq", open_price)
        close = row.get("close_qfq", open_price)

        # 检查停牌
        if self.check_suspended(ticker, open_price):
            fill = Fill(
                order=order,
                status="rejected",
                reject_reason="suspended",
            )
            self.stats.orders_rejected += 1
            self.stats.reject_reasons["suspended"] = (
                self.stats.reject_reasons.get("suspended", 0) + 1
            )
            return fill

        # 检查涨跌停
        if action == "buy" and self.check_limit_up(ticker, open_price, prev_close or close):
            fill = Fill(
                order=order,
                status="rejected",
                reject_reason="limit_up",
            )
            self.stats.orders_rejected += 1
            self.stats.reject_reasons["limit_up"] = (
                self.stats.reject_reasons.get("limit_up", 0) + 1
            )
            return fill

        if action == "sell" and self.check_limit_down(ticker, open_price, prev_close or close):
            fill = Fill(
                order=order,
                status="rejected",
                reject_reason="limit_down",
            )
            self.stats.orders_rejected += 1
            self.stats.reject_reasons["limit_down"] = (
                self.stats.reject_reasons.get("limit_down", 0) + 1
            )
            return fill

        # T+1 约束：当日买入不能当日卖出
        if action == "sell" and ticker in self._today_buys:
            fill = Fill(
                order=order,
                status="rejected",
                reject_reason="t+1_constraint",
            )
            self.stats.orders_rejected += 1
            self.stats.reject_reasons["t+1_constraint"] = (
                self.stats.reject_reasons.get("t+1_constraint", 0) + 1
            )
            return fill

        # 计算执行价格（使用开盘价，应用滑点）
        exec_price = self.apply_slippage(open_price, action)

        # 计算佣金
        if action == "buy":
            commission = shares * exec_price * self.commission_buy
            total_cost = shares * exec_price + commission

            # 检查资金约束
            if total_cost > self.cash:
                # 部分成交或拒绝
                max_shares = int((self.cash / (1 + self.commission_buy)) / exec_price)
                max_shares = max_shares - (max_shares % 100)  # 调整为手数整数倍

                if max_shares <= 0:
                    fill = Fill(
                        order=order,
                        status="rejected",
                        reject_reason="insufficient_cash",
                    )
                    self.stats.orders_rejected += 1
                    self.stats.reject_reasons["insufficient_cash"] = (
                        self.stats.reject_reasons.get("insufficient_cash", 0) + 1
                    )
                    return fill

                # 部分成交
                shares = max_shares
                commission = shares * exec_price * self.commission_buy
                total_cost = shares * exec_price + commission

            # 更新现金
            self.cash -= total_cost

            # 更新持仓
            self.position_book.open_position(
                ticker=ticker,
                shares=shares,
                price=exec_price,
                date=date,
                commission=self.commission_buy,
            )

            # 记录当日买入
            self._today_buys.add(ticker)

        else:  # sell
            # 检查持仓约束
            current_pos = self.position_book.get_position(ticker)
            if current_pos is None or current_pos.shares < shares:
                fill = Fill(
                    order=order,
                    status="rejected",
                    reject_reason="insufficient_shares",
                )
                self.stats.orders_rejected += 1
                self.stats.reject_reasons["insufficient_shares"] = (
                    self.stats.reject_reasons.get("insufficient_shares", 0) + 1
                )
                return fill

            revenue = shares * exec_price * (1 - self.commission_sell)
            commission = shares * exec_price * self.commission_sell  # 卖出佣金

            # 更新现金
            self.cash += revenue

            # 更新持仓
            self.position_book.close_position(
                ticker=ticker,
                shares=shares,
                price=exec_price,
                date=date,
                commission=self.commission_sell,
            )

        # 计算滑点成本
        slippage_cost = abs(exec_price - open_price) * shares

        # 更新统计
        self.stats.orders_filled += 1
        self.stats.total_commission += commission
        self.stats.total_slippage += slippage_cost

        fill = Fill(
            order=order,
            filled_shares=shares,
            fill_price=exec_price,
            commission=commission,
            slippage_cost=slippage_cost,
            status="filled",
        )

        self.fills.append(fill)
        return fill

    def execute_orders(
        self,
        orders: List[Order],
        price_df: pd.DataFrame,
        prev_closes: Optional[Dict[str, float]] = None,
    ) -> List[Fill]:
        """
        执行订单列表。

        Args:
            orders: 订单列表
            price_df: 价格 DataFrame
            prev_closes: 前收盘价字典

        Returns:
            Fill 列表
        """
        fills = []
        for order in orders:
            prev_close = prev_closes.get(order.ticker) if prev_closes else None
            fill = self.execute_order(order, price_df, prev_close)
            fills.append(fill)

        return fills

    def get_equity_curve(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        获取权益曲线。

        Args:
            price_df: 价格 DataFrame

        Returns:
            权益曲线 DataFrame
        """
        dates = sorted(price_df["date"].unique())
        rows = []

        for date in dates:
            day_prices = price_df[price_df["date"] == date]
            prices = dict(zip(day_prices["ticker"], day_prices["close_qfq"]))

            equity = self.position_book.total_equity(prices)
            cash = self.cash
            market_value = self.position_book.total_market_value(prices)

            rows.append({
                "date": date,
                "equity": equity,
                "cash": cash,
                "market_value": market_value,
            })

        return pd.DataFrame(rows)

    def get_stats(self) -> Dict:
        """获取执行统计"""
        return {
            "orders_submitted": self.stats.orders_submitted,
            "orders_filled": self.stats.orders_filled,
            "orders_rejected": self.stats.orders_rejected,
            "total_commission": self.stats.total_commission,
            "total_slippage": self.stats.total_slippage,
            "reject_reasons": self.stats.reject_reasons,
        }


# =========================================================
# 公开 API 导出
# =========================================================
__all__ = [
    "Fill",
    "ExecutionStats",
    "ExecutionEngine",
]
