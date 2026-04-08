from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from allocator import Order
from position_book import PositionBook

logger = logging.getLogger(__name__)

DEFAULT_SLIPPAGE_BPS: int = 20
DEFAULT_LIMIT_UP_PCT: float = 0.10
DEFAULT_LIMIT_DOWN_PCT: float = 0.10
ST_LIMIT_PCT: float = 0.05
COMMISSION_RATE_BUY: float = 0.0003
COMMISSION_RATE_SELL: float = 0.0013


@dataclass
class Fill:
    order: Order
    filled_shares: int = 0
    fill_price: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0
    status: str = "pending"
    reject_reason: Optional[str] = None


@dataclass
class ExecutionStats:
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    reject_reasons: Dict[str, int] = field(default_factory=dict)


class ExecutionEngine:
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
        self._today_buys: set = set()

    def reset_daily(self) -> None:
        self._today_buys = set()

    def check_limit_up(self, ticker: str, current_price: float, prev_close: float) -> bool:
        del ticker
        if pd.isna(prev_close) or prev_close <= 0:
            return False
        limit_up = prev_close * (1 + self.limit_up_pct)
        return current_price >= limit_up

    def check_limit_down(self, ticker: str, current_price: float, prev_close: float) -> bool:
        del ticker
        if pd.isna(prev_close) or prev_close <= 0:
            return False
        limit_down = prev_close * (1 - self.limit_down_pct)
        return current_price <= limit_down

    def check_suspended(self, ticker: str, price: Optional[float]) -> bool:
        del ticker
        return price is None or pd.isna(price) or price <= 0

    def apply_slippage(self, price: float, action: str) -> float:
        slippage_pct = self.slippage_bps / 10000
        if action == "buy":
            return price * (1 + slippage_pct)
        return price * (1 - slippage_pct)

    def _reject(self, order: Order, reason: str) -> Fill:
        self.stats.orders_rejected += 1
        self.stats.reject_reasons[reason] = self.stats.reject_reasons.get(reason, 0) + 1
        return Fill(order=order, status="rejected", reject_reason=reason)

    def execute_order(
        self,
        order: Order,
        price_df: pd.DataFrame,
        prev_close: Optional[float] = None,
    ) -> Fill:
        self.stats.orders_submitted += 1

        day_prices = price_df[
            (price_df["date"] == order.date) & (price_df["ticker"] == order.ticker)
        ]
        if day_prices.empty:
            return self._reject(order, "no_price_data")

        row = day_prices.iloc[0]
        open_price = row.get("open_qfq", row.get("close_qfq", 0))

        if self.check_suspended(order.ticker, open_price):
            return self._reject(order, "suspended")

        if prev_close is None or pd.isna(prev_close) or prev_close <= 0:
            return self._reject(order, "missing_prev_close")

        if order.action == "buy" and self.check_limit_up(order.ticker, open_price, float(prev_close)):
            return self._reject(order, "limit_up")
        if order.action == "sell" and self.check_limit_down(order.ticker, open_price, float(prev_close)):
            return self._reject(order, "limit_down")
        if order.action == "sell" and order.ticker in self._today_buys:
            return self._reject(order, "t+1_constraint")

        exec_price = self.apply_slippage(float(open_price), order.action)
        shares = int(order.shares)

        if order.action == "buy":
            commission = shares * exec_price * self.commission_buy
            total_cost = shares * exec_price + commission
            if total_cost > self.cash:
                max_shares = int((self.cash / (1 + self.commission_buy)) / exec_price)
                max_shares = max_shares - (max_shares % 100)
                if max_shares <= 0:
                    return self._reject(order, "insufficient_cash")
                shares = max_shares
                commission = shares * exec_price * self.commission_buy
                total_cost = shares * exec_price + commission

            self.cash -= total_cost
            self.position_book.open_position(
                ticker=order.ticker,
                shares=shares,
                price=exec_price,
                date=order.date,
                commission=self.commission_buy,
            )
            self._today_buys.add(order.ticker)
        else:
            current_pos = self.position_book.get_position(order.ticker)
            if current_pos is None or current_pos.shares < shares:
                return self._reject(order, "insufficient_shares")

            revenue = shares * exec_price * (1 - self.commission_sell)
            commission = shares * exec_price * self.commission_sell
            self.cash += revenue
            self.position_book.close_position(
                ticker=order.ticker,
                shares=shares,
                price=exec_price,
                date=order.date,
                commission=self.commission_sell,
            )

        slippage_cost = abs(exec_price - float(open_price)) * shares
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
        fills: List[Fill] = []
        for order in orders:
            prev_close = prev_closes.get(order.ticker) if prev_closes else None
            fills.append(self.execute_order(order, price_df, prev_close))
        return fills

    def get_equity_curve(self, price_df: pd.DataFrame) -> pd.DataFrame:
        dates = sorted(price_df["date"].unique())
        rows = []
        for date in dates:
            day_prices = price_df[price_df["date"] == date]
            prices = dict(zip(day_prices["ticker"], day_prices["close_qfq"]))
            rows.append(
                {
                    "date": date,
                    "equity": self.position_book.total_equity(prices),
                    "cash": self.cash,
                    "market_value": self.position_book.total_market_value(prices),
                }
            )
        return pd.DataFrame(rows)

    def get_stats(self) -> Dict:
        return {
            "orders_submitted": self.stats.orders_submitted,
            "orders_filled": self.stats.orders_filled,
            "orders_rejected": self.stats.orders_rejected,
            "total_commission": self.stats.total_commission,
            "total_slippage": self.stats.total_slippage,
            "reject_reasons": self.stats.reject_reasons,
        }


__all__ = [
    "Fill",
    "ExecutionStats",
    "ExecutionEngine",
]
