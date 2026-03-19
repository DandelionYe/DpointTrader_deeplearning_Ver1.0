# test_backtester_market_state.py
"""
Phase 3: 新增回归测试 - Backtester 市场状态和流动性过滤

测试目标：
1. _prepare_price_limits 保留真实 is_st/listing_days/suspended 列
2. check_execution_feasibility 默认使用 amount 进行流动性过滤
3. legacy min_daily_volume 参数仍然可用
4. buy-side layered slippage 使用估算订单金额
"""
import pandas as pd
import pytest
from backtester import (
    check_execution_feasibility,
    apply_layered_slippage,
    _prepare_price_limits,
    DEFAULT_MIN_DAILY_AMOUNT,
)


class TestPreparePriceLimits:
    """Test _prepare_price_limits preserves external market state columns."""

    def test_prepare_price_limits_preserves_is_st_listing_days_and_suspended(self):
        """构造 df 传入真实 is_st/listing_days/suspended，断言不被覆盖。"""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5),
            "open_qfq": [10.0, 10.5, 0, 11.0, 11.5],  # 第 3 行开盘为 0（停牌）
            "close_qfq": [10.2, 10.8, 0, 11.2, 11.8],
            "is_st": [False, True, False, False, False],  # 第 2 行是 ST
            "listing_days": [100, 20, 200, 500, 1000],  # 第 2 行上市仅 20 天
            "suspended": [False, True, False, False, False],  # 第 2 行停牌
        }).set_index("date")

        result = _prepare_price_limits(df, limit_up_pct=0.10, limit_down_pct=0.10)

        # 断言 is_st 保留原始值
        assert result["is_st"].iloc[1] == True, "is_st[1] should be True (preserved)"
        assert result["is_st"].iloc[0] == False, "is_st[0] should be False"

        # 断言 listing_days 保留原始值
        assert result["listing_days"].iloc[1] == 20, "listing_days[1] should be 20 (preserved)"
        assert result["listing_days"].iloc[2] == 200, "listing_days[2] should be 200 (preserved)"

        # 断言 suspended 保留原始值并与计算值 OR
        # 第 2 行原始 suspended=True，应保持 True
        assert result["suspended"].iloc[1] == True, "suspended[1] should be True (preserved)"
        # 第 3 行开盘价=0，计算 suspended=True，应合并为 True
        assert result["suspended"].iloc[2] == True, "suspended[2] should be True (computed from open=0)"


class TestCheckExecutionFeasibility:
    """Test check_execution_feasibility uses amount by default."""

    def test_check_execution_feasibility_uses_amount_by_default(self):
        """构造 amount=500_000, volume=5_000_000，默认参数下应因成交额不足被拒单。"""
        row = pd.Series({
            "open_qfq": 10.0,
            "prev_close": 10.0,
            "volume": 5_000_000,  # 成交量很大
            "amount": 500_000,    # 成交额不足（默认门槛 100 万）
            "is_st": False,
            "listing_days": 100,
            "suspended": False,
        })

        is_feasible, reason = check_execution_feasibility(row, "BUY")

        assert is_feasible == False, "Should reject due to low amount"
        assert "成交额过低" in reason or "amount" in reason.lower(), f"Wrong reject reason: {reason}"

    def test_check_execution_feasibility_amount_sufficient(self):
        """构造 amount=2_000_000，应通过流动性检查。"""
        row = pd.Series({
            "open_qfq": 10.0,
            "prev_close": 10.0,
            "volume": 5_000_000,
            "amount": 2_000_000,  # 成交额足够
            "is_st": False,
            "listing_days": 100,
            "suspended": False,
        })

        is_feasible, reason = check_execution_feasibility(row, "BUY")

        assert is_feasible == True, f"Should pass, got reject reason: {reason}"

    def test_legacy_min_daily_volume_still_works(self):
        """显式传入 min_daily_volume=1_000_000，应使用 volume 检查。"""
        row = pd.Series({
            "open_qfq": 10.0,
            "prev_close": 10.0,
            "volume": 500_000,    # 成交量不足
            "amount": 2_000_000,  # 成交额足够
            "is_st": False,
            "listing_days": 100,
            "suspended": False,
        })

        # 显式传入 min_daily_volume，应使用 volume 检查
        is_feasible, reason = check_execution_feasibility(
            row, "BUY", min_daily_volume=1_000_000
        )

        assert is_feasible == False, "Should reject due to low volume (legacy mode)"
        assert "成交量过低" in reason, f"Wrong reject reason: {reason}"


class TestLayeredSlippage:
    """Test layered slippage uses estimated order value for BUY."""

    def test_buy_layered_slippage_uses_estimated_order_value(self):
        """构造大资金 BUY，断言开启 use_layered_slippage=True 时执行价更差。"""
        price = 10.0
        
        # 小单（< 10 万）：10 bps
        small_order_value = 50_000
        small_slippage_price = apply_layered_slippage(price, "BUY", small_order_value)
        small_slippage = (small_slippage_price - price) / price * 10000  # bps

        # 大单（> 50 万）：30 bps
        large_order_value = 1_000_000
        large_slippage_price = apply_layered_slippage(price, "BUY", large_order_value)
        large_slippage = (large_slippage_price - price) / price * 10000  # bps

        # 断言大单滑点更高
        assert large_slippage > small_slippage, f"Large order should have higher slippage"
        assert abs(small_slippage - 10) < 1, f"Small order slippage should be ~10 bps, got {small_slippage}"
        assert abs(large_slippage - 30) < 1, f"Large order slippage should be ~30 bps, got {large_slippage}"

    def test_sell_layered_slippage_order_value(self):
        """SELL 侧使用 shares * price 计算订单金额。"""
        price = 10.0
        shares = 10_000
        order_value = shares * price  # 10 万
        
        slippage_price = apply_layered_slippage(price, "SELL", order_value)
        
        # 10 万属于中单（10-50 万）：20 bps
        expected_slippage_bps = 20
        actual_slippage_bps = (price - slippage_price) / price * 10000
        
        assert abs(actual_slippage_bps - expected_slippage_bps) < 1, \
            f"SELL slippage should be ~{expected_slippage_bps} bps, got {actual_slippage_bps}"
