# excel_reporter.py
"""
Excel 报告生成器
================

本模块提供将回测结果输出为 Excel 文件的功能。

主要功能:
    - save_to_excel: 保存结果到 Excel
    - 支持多个 sheet：PortfolioEquity, Orders, Trades, Positions 等

使用示例:
    >>> from excel_reporter import save_to_excel
    >>> save_to_excel("output.xlsx", backtest_result, scores_df)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def save_to_excel(
    output_path: str,
    *,
    equity_curve: Optional[pd.DataFrame] = None,
    benchmark_curve: Optional[pd.DataFrame] = None,
    execution_stats: Optional[Dict[str, Any]] = None,
    orders: Optional[pd.DataFrame] = None,
    trades: Optional[pd.DataFrame] = None,
    positions: Optional[pd.DataFrame] = None,
    scores_df: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    log_notes: Optional[List[str]] = None,
) -> None:
    """
    保存结果到 Excel 文件。

    Args:
        output_path: 输出文件路径
        equity_curve: 权益曲线 DataFrame
        orders: 订单记录 DataFrame
        trades: 交易记录 DataFrame
        positions: 持仓记录 DataFrame
        scores_df: 分数 DataFrame
        config: 配置字典
        metrics: 指标字典
        log_notes: 日志注释列表
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # 权益曲线
        if equity_curve is not None and not equity_curve.empty:
            df = equity_curve.copy()
            if "date" in df.columns:
                df["date"] = df["date"].astype(str)
            df.to_excel(writer, sheet_name="PortfolioEquity", index=False)

        if benchmark_curve is not None and not benchmark_curve.empty:
            df = benchmark_curve.copy()
            if "date" in df.columns:
                df["date"] = df["date"].astype(str)
            df.to_excel(writer, sheet_name="BenchmarkEquity", index=False)

        # 订单
        if orders is not None and not orders.empty:
            df = orders.copy()
            if "date" in df.columns:
                df["date"] = df["date"].astype(str)
            df.to_excel(writer, sheet_name="Orders", index=False)

        # 交易
        if trades is not None and not trades.empty:
            df = trades.copy()
            if "date" in df.columns:
                df["date"] = df["date"].astype(str)
            df.to_excel(writer, sheet_name="Trades", index=False)

        # 持仓
        if positions is not None and not positions.empty:
            df = positions.copy()
            if "date" in df.columns:
                df["date"] = df["date"].astype(str)
            df.to_excel(writer, sheet_name="Positions", index=False)

        # 分数
        if scores_df is not None and not scores_df.empty:
            df = scores_df.copy()
            if "date" in df.columns:
                df["date"] = df["date"].astype(str)
            df.to_excel(writer, sheet_name="DailyScores", index=False)

        # 配置
        if config:
            config_df = pd.DataFrame(
                list(config.items()),
                columns=["key", "value"]
            )
            config_df.to_excel(writer, sheet_name="Config", index=False)

        # 指标
        if metrics:
            metrics_df = pd.DataFrame(
                list(metrics.items()),
                columns=["metric", "value"]
            )
            metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

        if execution_stats:
            summary_items = [(k, v) for k, v in execution_stats.items() if k != "reject_reasons"]
            if summary_items:
                execution_df = pd.DataFrame(summary_items, columns=["metric", "value"])
                execution_df.to_excel(writer, sheet_name="ExecutionStats", index=False)
            reject_reasons = execution_stats.get("reject_reasons", {}) or {}
            if reject_reasons:
                reject_df = pd.DataFrame(list(reject_reasons.items()), columns=["reason", "count"])
                reject_df.to_excel(writer, sheet_name="RejectReasons", index=False)

        # 日志
        if log_notes:
            log_df = pd.DataFrame({"note": log_notes})
            log_df.to_excel(writer, sheet_name="Log", index=False)

    logger.info(f"Saved results to {output_path}")


def save_basket_manifest(
    output_path: str,
    basket_name: str,
    tickers: List[str],
    basket_path: str,
    n_tickers: int,
    date_range: tuple,
    notes: Optional[List[str]] = None,
) -> None:
    """
    保存 basket manifest 到 Excel。

    Args:
        output_path: 输出文件路径
        basket_name: Basket 名称
        tickers: ticker 列表
        basket_path: Basket 路径
        n_tickers: 股票数量
        date_range: 日期范围
        notes: 注释
    """
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # 基本信息
        info_df = pd.DataFrame({
            "basket_name": [basket_name],
            "basket_path": [basket_path],
            "n_tickers": [n_tickers],
            "date_start": [date_range[0] if date_range else None],
            "date_end": [date_range[1] if date_range else None],
        })
        info_df.to_excel(writer, sheet_name="BasketInfo", index=False)

        # ticker 列表
        ticker_df = pd.DataFrame({"ticker": tickers})
        ticker_df.to_excel(writer, sheet_name="Tickers", index=False)

        # 注释
        if notes:
            notes_df = pd.DataFrame({"note": notes})
            notes_df.to_excel(writer, sheet_name="Notes", index=False)

    logger.info(f"Saved basket manifest to {output_path}")


# =========================================================
# 公开 API 导出
# =========================================================
__all__ = [
    "save_to_excel",
    "save_basket_manifest",
]
