# position_book.py
"""
持仓簿模块
==========

本模块提供持仓管理功能。

主要功能:
    - Position: 单个持仓数据类
    - PositionBook: 持仓簿管理类
    - 持仓跟踪、盈亏计算

使用示例:
    >>> from position_book import PositionBook
    >>> book = PositionBook(initial_cash=100000)
    >>> book.open_position("600036", shares=1000, price=35.0, date="2024-01-01")
    >>> book.get_positions()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """单个持仓。

    Attributes:
        ticker: 股票代码
        shares: 持仓股数
        avg_cost: 平均成本
        open_date: 开仓日期
        last_update: 最后更新日期
        realized_pnl: 已实现盈亏
        unrealized_pnl: 未实现盈亏
    """

    ticker: str
    shares: int = 0
    avg_cost: float = 0.0
    open_date: Optional[pd.Timestamp] = None
    last_update: Optional[pd.Timestamp] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    def market_value(self, current_price: float) -> float:
        """计算市值"""
        return self.shares * current_price

    def total_pnl(self, current_price: float) -> float:
        """计算总盈亏"""
        self.unrealized_pnl = (current_price - self.avg_cost) * self.shares
        return self.realized_pnl + self.unrealized_pnl

    def pnl_pct(self, current_price: float) -> float:
        """计算盈亏比例"""
        if self.avg_cost == 0:
            return 0.0
        return (current_price - self.avg_cost) / self.avg_cost


@dataclass
class PositionBook:
    """持仓簿。

    Attributes:
        initial_cash: 初始现金
        cash: 当前现金
        positions: 持仓字典 {ticker: Position}
        history: 交易历史
    """

    initial_cash: float = 100000.0
    cash: float = 100000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)

    def open_position(
        self,
        ticker: str,
        shares: int,
        price: float,
        date: pd.Timestamp,
        commission: float = 0.0003,
    ) -> bool:
        """
        开仓。

        Args:
            ticker: 股票代码
            shares: 股数
            price: 成交价格
            date: 日期
            commission: 佣金率

        Returns:
            是否成功
        """
        cost = shares * price * (1 + commission)
        if cost > self.cash:
            logger.warning(f"Insufficient cash for {ticker}: need {cost}, have {self.cash}")
            return False

        # 更新现金
        self.cash -= cost

        # 更新或创建持仓
        if ticker in self.positions:
            pos = self.positions[ticker]
            old_shares = pos.shares
            old_cost = pos.avg_cost * old_shares
            new_cost = shares * price
            pos.shares = old_shares + shares
            pos.avg_cost = (old_cost + new_cost) / pos.shares
            pos.last_update = date
        else:
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                avg_cost=price,
                open_date=date,
                last_update=date,
            )

        # 记录历史
        self.history.append(
            {
                "date": date,
                "ticker": ticker,
                "action": "buy",
                "shares": shares,
                "price": price,
                "cost": cost,
            }
        )

        return True

    def close_position(
        self,
        ticker: str,
        shares: Optional[int] = None,
        price: float = 0.0,
        date: pd.Timestamp = None,
        commission: float = 0.0013,
    ) -> bool:
        """
        平仓。

        Args:
            ticker: 股票代码
            shares: 股数（None 则全部平仓）
            price: 成交价格
            date: 日期
            commission: 佣金率

        Returns:
            是否成功
        """
        if ticker not in self.positions:
            logger.warning(f"Position not found: {ticker}")
            return False

        pos = self.positions[ticker]
        if shares is None:
            shares = pos.shares
        elif shares > pos.shares:
            logger.warning(f"Insufficient shares for {ticker}: need {shares}, have {pos.shares}")
            return False

        # 计算收入
        revenue = shares * price * (1 - commission)
        cost = shares * pos.avg_cost
        pnl = revenue - cost

        # 更新现金
        self.cash += revenue

        # 更新持仓
        pos.shares -= shares
        pos.realized_pnl += pnl
        pos.last_update = date

        if pos.shares == 0:
            del self.positions[ticker]

        # 记录历史
        self.history.append(
            {
                "date": date,
                "ticker": ticker,
                "action": "sell",
                "shares": shares,
                "price": price,
                "revenue": revenue,
                "pnl": pnl,
            }
        )

        return True

    def update_unrealized_pnl(self, prices: Dict[str, float]) -> None:
        """
        更新未实现盈亏。

        Args:
            prices: 价格字典 {ticker: price}
        """
        for ticker, pos in self.positions.items():
            if ticker in prices:
                pos.unrealized_pnl = (prices[ticker] - pos.avg_cost) * pos.shares

    def get_positions(self) -> List[Position]:
        """获取所有持仓"""
        return list(self.positions.values())

    def get_position(self, ticker: str) -> Optional[Position]:
        """获取单个持仓"""
        return self.positions.get(ticker)

    def total_market_value(self, prices: Dict[str, float]) -> float:
        """
        计算总市值。

        Args:
            prices: 价格字典

        Returns:
            总市值
        """
        total = 0.0
        for ticker, pos in self.positions.items():
            if ticker in prices:
                total += pos.market_value(prices[ticker])
        return total

    def total_equity(self, prices: Dict[str, float]) -> float:
        """
        计算总权益。

        Args:
            prices: 价格字典

        Returns:
            总权益（现金 + 市值）
        """
        return self.cash + self.total_market_value(prices)

    def get_turnover(self, date: pd.Timestamp) -> Tuple[float, float]:
        """
        计算某日的买卖成交额。

        Args:
            date: 日期

        Returns:
            Tuple[buy_turnover, sell_turnover]
        """
        buy_turnover = sum(
            h["shares"] * h["price"]
            for h in self.history
            if h["date"] == date and h["action"] == "buy"
        )
        sell_turnover = sum(
            h["shares"] * h["price"]
            for h in self.history
            if h["date"] == date and h["action"] == "sell"
        )
        return buy_turnover, sell_turnover

    def to_df(self) -> pd.DataFrame:
        """
        转换成 DataFrame。

        Returns:
            持仓 DataFrame
        """
        if not self.positions:
            return pd.DataFrame()

        rows = []
        for pos in self.positions.values():
            rows.append(
                {
                    "ticker": pos.ticker,
                    "shares": pos.shares,
                    "avg_cost": pos.avg_cost,
                    "open_date": pos.open_date,
                    "last_update": pos.last_update,
                    "realized_pnl": pos.realized_pnl,
                    "unrealized_pnl": pos.unrealized_pnl,
                }
            )

        return pd.DataFrame(rows)

    def history_to_df(self) -> pd.DataFrame:
        """
        将交易历史转换成 DataFrame。

        Returns:
            交易历史 DataFrame
        """
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)


# =========================================================
# 公开 API 导出
# =========================================================
__all__ = [
    "Position",
    "PositionBook",
]
