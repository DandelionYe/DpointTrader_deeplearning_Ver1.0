# test_panel_builder.py
"""
Panel 构建器测试模块
==================

测试 panel_builder.py 的功能。

运行测试:
    pytest test_panel_builder.py -v
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

# 添加父目录到路径以便导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from panel_builder import (
    add_ticker_column,
    align_calendar,
    build_panel,
    validate_panel,
    panel_to_wide,
)
from labeler import build_binary_label
from backtester_engine import compute_buy_and_hold_benchmark


class TestAddTickerColumn:
    """测试添加 ticker 列"""

    def test_add_ticker(self):
        """测试添加 ticker 列"""
        df = pd.DataFrame({"date": ["2024-01-01"], "close": [10.0]})
        result = add_ticker_column(df, "600036")

        assert "ticker" in result.columns
        assert result["ticker"].iloc[0] == "600036"
        assert len(result) == 1


class TestAlignCalendar:
    """测试日历对齐"""

    def test_inner_align(self):
        """测试交集对齐"""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03"],
            "ticker": ["A", "A", "B", "B"],
            "close": [10.0, 10.1, 20.0, 20.1],
        })

        result = align_calendar(df, method="inner")

        # 只有 2024-01-01 是共同日期
        assert len(result) == 2
        assert all(result["date"] == "2024-01-01")

    def test_outer_align(self):
        """测试并集对齐"""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "ticker": ["A", "A"],
            "close": [10.0, 10.1],
        })

        result = align_calendar(df, method="outer")

        assert len(result) == 2


class TestBuildPanel:
    """测试构建 panel"""

    def test_build_panel_from_frames(self):
        """测试从多个 DataFrame 构建 panel"""
        df1 = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3),
            "close": [10.0, 10.1, 10.2],
            "ticker": "A",
        })
        df2 = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3),
            "close": [20.0, 20.1, 20.2],
            "ticker": "B",
        })

        panel = build_panel([df1, df2], basket_name="test")

        assert len(panel) == 6
        assert panel["ticker"].nunique() == 2
        assert "date" in panel.columns
        assert "ticker" in panel.columns


class TestValidatePanel:
    """测试 panel 验证"""

    def test_valid_panel(self):
        """测试有效 panel"""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "ticker": ["A", "A"],
            "close": [10.0, 10.1],
        })

        valid, issues = validate_panel(df)

        assert valid is True
        assert len(issues) == 0

    def test_duplicate_detection(self):
        """测试重复检测"""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01"],
            "ticker": ["A", "A"],
            "close": [10.0, 10.1],
        })

        valid, issues = validate_panel(df)

        # 应该有重复警告
        assert any("duplicate" in issue.lower() for issue in issues)


class TestPanelToWide:
    """测试 panel 转宽格式"""

    def test_panel_to_wide(self):
        """测试转宽格式"""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "ticker": ["A", "B", "A", "B"],
            "close": [10.0, 20.0, 10.1, 20.1],
        })

        wide = panel_to_wide(df, value_col="close")

        assert wide.shape == (2, 2)  # 2 日期 × 2 ticker
        assert "A" in wide.columns
        assert "B" in wide.columns


class TestPanelSemantics:
    """测试 panel 语义性边界"""

    def test_binary_label_drops_last_row_per_ticker(self):
        """每个 ticker 的末行应没有未来标签"""
        df = pd.DataFrame({
            "date": pd.to_datetime([
                "2024-01-01", "2024-01-02", "2024-01-03",
                "2024-01-01", "2024-01-02", "2024-01-03",
            ]),
            "ticker": ["A", "A", "A", "B", "B", "B"],
            "close_qfq": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        })

        labels, _ = build_binary_label(df)

        assert len(labels) == 4
        assert labels.tolist() == [1, 1, 0, 0]

    def test_buy_and_hold_benchmark_keeps_cash(self):
        """benchmark 应保留整手约束下剩余现金"""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "ticker": ["A", "B", "A", "B"],
            "open_qfq": [333.0, 333.0, 333.0, 333.0],
            "close_qfq": [333.0, 333.0, 333.0, 333.0],
        })

        bench = compute_buy_and_hold_benchmark(df, initial_cash=100000.0)

        assert not bench.empty
        assert bench["bnh_equity"].iloc[0] > 99000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
