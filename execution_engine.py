from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from allocator import Order
from constants import (
    DEFAULT_BOARD_LOT,
    DEFAULT_BUY_COMMISSION_RATE,
    DEFAULT_LIMIT_DOWN_PCT,
    DEFAULT_LIMIT_DOWN_PCT_CHINEXT_STAR,
    DEFAULT_LIMIT_DOWN_PCT_MAIN,
    DEFAULT_LIMIT_DOWN_PCT_ST,
    DEFAULT_LIMIT_UP_PCT,
    DEFAULT_LIMIT_UP_PCT_CHINEXT_STAR,
    DEFAULT_LIMIT_UP_PCT_MAIN,
    DEFAULT_LIMIT_UP_PCT_ST,
    DEFAULT_MAX_PARTICIPATION_RATE,
    DEFAULT_MIN_TRADE_VALUE,
    DEFAULT_SELL_COMMISSION_RATE,
    DEFAULT_SELL_STAMP_DUTY_RATE,
)
from position_book import PositionBook

logger = logging.getLogger(__name__)

DEFAULT_SLIPPAGE_BPS: int = 20


@dataclass
class TradingConstraints:
    board_lot: int = DEFAULT_BOARD_LOT
    buy_commission_rate: float = DEFAULT_BUY_COMMISSION_RATE
    sell_commission_rate: float = DEFAULT_SELL_COMMISSION_RATE
    sell_stamp_duty_rate: float = DEFAULT_SELL_STAMP_DUTY_RATE
    max_participation_rate: float = DEFAULT_MAX_PARTICIPATION_RATE
    min_trade_value: float = DEFAULT_MIN_TRADE_VALUE
    allow_partial_fill: bool = True
    default_limit_up_pct_main: float = DEFAULT_LIMIT_UP_PCT_MAIN
    default_limit_down_pct_main: float = DEFAULT_LIMIT_DOWN_PCT_MAIN
    default_limit_up_pct_st: float = DEFAULT_LIMIT_UP_PCT_ST
    default_limit_down_pct_st: float = DEFAULT_LIMIT_DOWN_PCT_ST
    default_limit_up_pct_chinext_star: float = DEFAULT_LIMIT_UP_PCT_CHINEXT_STAR
    default_limit_down_pct_chinext_star: float = DEFAULT_LIMIT_DOWN_PCT_CHINEXT_STAR


def resolve_price_limit(
    row: pd.Series,
    prev_close: float,
    *,
    default_limit_up_pct: float = DEFAULT_LIMIT_UP_PCT,
    default_limit_down_pct: float = DEFAULT_LIMIT_DOWN_PCT,
    constraints: Optional[TradingConstraints] = None,
) -> Tuple[Optional[float], Optional[float]]:
    if pd.isna(prev_close) or prev_close <= 0:
        return None, None
    constraints = constraints or TradingConstraints()

    explicit_up = row.get("up_limit_price")
    explicit_down = row.get("down_limit_price")
    if pd.notna(explicit_up) and pd.notna(explicit_down):
        return float(explicit_up), float(explicit_down)

    board = str(row.get("board", "")).strip()
    is_st = bool(row.get("is_st", 0))
    if is_st:
        up_pct = constraints.default_limit_up_pct_st
        down_pct = constraints.default_limit_down_pct_st
    elif board in {"创业板", "科创板"}:
        up_pct = constraints.default_limit_up_pct_chinext_star
        down_pct = constraints.default_limit_down_pct_chinext_star
    else:
        up_pct = constraints.default_limit_up_pct_main if constraints else default_limit_up_pct
        down_pct = constraints.default_limit_down_pct_main if constraints else default_limit_down_pct
    return float(prev_close) * (1 + up_pct), float(prev_close) * (1 - down_pct)


def compute_fillable_shares(
    row: pd.Series,
    desired_shares: int,
    constraints: TradingConstraints,
) -> int:
    if desired_shares <= 0:
        return 0
    daily_volume = row.get("volume", 0)
    if pd.isna(daily_volume) or float(daily_volume) <= 0:
        return 0
    cap_by_volume = int(float(daily_volume) * constraints.max_participation_rate)
    cap_by_volume -= cap_by_volume % constraints.board_lot
    return max(0, min(int(desired_shares), cap_by_volume))


@dataclass
class Fill:
    order: Order
    filled_shares: int = 0
    fill_price: float = 0.0
    commission: float = 0.0
    stamp_duty: float = 0.0
    slippage_cost: float = 0.0
    status: str = "pending"
    reject_reason: Optional[str] = None


@dataclass
class ExecutionStats:
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    total_commission: float = 0.0
    total_stamp_duty: float = 0.0
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
        commission_buy: float = DEFAULT_BUY_COMMISSION_RATE,
        commission_sell: float = DEFAULT_SELL_COMMISSION_RATE,
        stamp_duty_sell: float = DEFAULT_SELL_STAMP_DUTY_RATE,
        constraints: Optional[TradingConstraints] = None,
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.slippage_bps = slippage_bps
        self.limit_up_pct = limit_up_pct
        self.limit_down_pct = limit_down_pct
        self.commission_buy = commission_buy
        self.commission_sell = commission_sell
        self.stamp_duty_sell = stamp_duty_sell
        self.constraints = constraints or TradingConstraints(
            buy_commission_rate=commission_buy,
            sell_commission_rate=commission_sell,
            sell_stamp_duty_rate=stamp_duty_sell,
            default_limit_up_pct_main=limit_up_pct,
            default_limit_down_pct_main=limit_down_pct,
        )

        self.position_book = PositionBook(initial_cash=initial_cash)
        self.fills: List[Fill] = []
        self.stats = ExecutionStats()
        self._today_buys: set = set()

    def reset_daily(self) -> None:
        self._today_buys = set()

    def check_limit_up(self, row: pd.Series, current_price: float, prev_close: float) -> bool:
        limit_up, _ = resolve_price_limit(
            row,
            prev_close,
            default_limit_up_pct=self.limit_up_pct,
            default_limit_down_pct=self.limit_down_pct,
            constraints=self.constraints,
        )
        if limit_up is None:
            return False
        return current_price >= limit_up

    def check_limit_down(self, row: pd.Series, current_price: float, prev_close: float) -> bool:
        _, limit_down = resolve_price_limit(
            row,
            prev_close,
            default_limit_up_pct=self.limit_up_pct,
            default_limit_down_pct=self.limit_down_pct,
            constraints=self.constraints,
        )
        if limit_down is None:
            return False
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

        if order.action == "buy" and self.check_limit_up(row, open_price, float(prev_close)):
            return self._reject(order, "limit_up")
        if order.action == "sell" and self.check_limit_down(row, open_price, float(prev_close)):
            return self._reject(order, "limit_down")
        if order.action == "sell" and order.ticker in self._today_buys:
            return self._reject(order, "t_plus_1")

        exec_price = self.apply_slippage(float(open_price), order.action)
        shares = int(order.shares)
        shares = shares - (shares % self.constraints.board_lot)
        if shares <= 0:
            return self._reject(order, "board_lot_round_to_zero")

        fillable_shares = compute_fillable_shares(row, shares, self.constraints)
        if fillable_shares <= 0:
            return self._reject(order, "volume_cap")
        if fillable_shares < shares:
            if not self.constraints.allow_partial_fill:
                return self._reject(order, "volume_cap")
            shares = fillable_shares

        estimated_value = shares * exec_price

        if order.action == "buy":
            if estimated_value < self.constraints.min_trade_value:
                return self._reject(order, "min_trade_value_not_met")
            commission = shares * exec_price * self.commission_buy
            stamp_duty = 0.0
            total_cost = shares * exec_price + commission
            if total_cost > self.cash:
                max_shares = int((self.cash / (1 + self.commission_buy)) / exec_price)
                max_shares = max_shares - (max_shares % self.constraints.board_lot)
                if max_shares <= 0:
                    return self._reject(order, "insufficient_cash")
                shares = max_shares
                if shares * exec_price < self.constraints.min_trade_value:
                    return self._reject(order, "min_trade_value_not_met")
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
                return self._reject(order, "insufficient_position")
            if estimated_value < self.constraints.min_trade_value:
                return self._reject(order, "min_trade_value_not_met")

            commission = shares * exec_price * self.commission_sell
            stamp_duty = shares * exec_price * self.stamp_duty_sell
            revenue = shares * exec_price - commission - stamp_duty
            self.cash += revenue
            self.position_book.close_position(
                ticker=order.ticker,
                shares=shares,
                price=exec_price,
                date=order.date,
                commission=self.commission_sell + self.stamp_duty_sell,
            )

        slippage_cost = abs(exec_price - float(open_price)) * shares
        self.stats.orders_filled += 1
        self.stats.total_commission += commission
        self.stats.total_stamp_duty += stamp_duty
        self.stats.total_slippage += slippage_cost

        fill = Fill(
            order=order,
            filled_shares=shares,
            fill_price=exec_price,
            commission=commission,
            stamp_duty=stamp_duty,
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
            "total_stamp_duty": self.stats.total_stamp_duty,
            "total_slippage": self.stats.total_slippage,
            "reject_reasons": self.stats.reject_reasons,
        }


__all__ = [
    "Fill",
    "ExecutionStats",
    "TradingConstraints",
    "ExecutionEngine",
    "compute_fillable_shares",
    "resolve_price_limit",
]
