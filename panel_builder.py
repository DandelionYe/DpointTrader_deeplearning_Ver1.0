# panel_builder.py
"""
Panel 数据构建器模块
===================

本模块提供将多股票数据合并成 date × ticker panel 的功能。

主要功能:
    - build_panel: 将股票列表合并成 panel DataFrame
    - align_calendar: 日历对齐（确保所有股票有相同的交易日）
    - add_ticker_column: 为 DataFrame 添加 ticker 列

使用示例:
    >>> from panel_builder import build_panel
    >>> panel_df = build_panel(stock_frames, basket_name="basket_1")
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import pandas as pd

from constants import REQUIRED_COLS_PANEL

logger = logging.getLogger(__name__)


def add_ticker_column(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    为 DataFrame 添加 ticker 列。

    Args:
        df: 输入 DataFrame
        ticker: 股票代码

    Returns:
        添加 ticker 列后的 DataFrame
    """
    df = df.copy()
    df["ticker"] = ticker
    return df


def align_calendar(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    method: str = "inner",
) -> pd.DataFrame:
    """
    日历对齐：确保所有股票有相同的交易日。

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        method: 对齐方法
            - "inner": 只保留所有股票都有的交易日（交集）
            - "outer": 保留所有交易日（并集），缺失值填 NaN
            - "majority": 保留超过 50% 股票有的交易日

    Returns:
        对齐后的 panel DataFrame
    """
    if method == "inner":
        # 获取所有股票的日期交集
        all_dates = set(panel_df[date_col].unique())
        for ticker in panel_df[ticker_col].unique():
            ticker_dates = set(
                panel_df[panel_df[ticker_col] == ticker][date_col].unique()
            )
            all_dates &= ticker_dates
            if not all_dates:
                logger.warning(
                    "Calendar alignment resulted in no common dates. "
                    "Consider using method='outer' or 'majority'."
                )
                break

        aligned_df = panel_df[panel_df[date_col].isin(all_dates)].copy()

    elif method == "outer":
        # 保留所有日期，不过滤
        aligned_df = panel_df.copy()

    elif method == "majority":
        # 统计每个日期出现的股票数
        date_counts = panel_df.groupby(date_col)[ticker_col].nunique()
        n_tickers = panel_df[ticker_col].nunique()
        majority_dates = date_counts[date_counts >= n_tickers * 0.5].index
        aligned_df = panel_df[panel_df[date_col].isin(majority_dates)].copy()

    else:
        raise ValueError(f"Unknown alignment method: {method}")

    return aligned_df.sort_values([date_col, ticker_col]).reset_index(drop=True)


def build_panel(
    stock_frames: List[pd.DataFrame],
    basket_name: str = "basket",
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    align_calendar_method: str = "inner",
    add_missing_tickers: bool = True,
) -> pd.DataFrame:
    """
    将多个股票 DataFrame 合并成 panel 数据。

    Args:
        stock_frames: 股票 DataFrame 列表
        basket_name: Basket 名称
        date_col: 日期列名
        ticker_col: ticker 列名
        align_calendar_method: 日历对齐方法
        add_missing_tickers: 是否为缺失 ticker 列的 DataFrame 添加

    Returns:
        panel DataFrame（包含 date 和 ticker 列）
    """
    if not stock_frames:
        raise ValueError("stock_frames is empty")

    processed_frames = []
    for i, df in enumerate(stock_frames):
        df_copy = df.copy()

        # 确保有 ticker 列
        if ticker_col not in df_copy.columns:
            if add_missing_tickers:
                # 尝试从已有列推断
                if "code" in df_copy.columns:
                    df_copy[ticker_col] = df_copy["code"]
                else:
                    df_copy[ticker_col] = f"stock_{i}"
            else:
                raise ValueError(
                    f"DataFrame {i} missing '{ticker_col}' column "
                    "and add_missing_tickers=False"
                )

        # 确保有 date 列
        if date_col not in df_copy.columns:
            if "Date" in df_copy.columns:
                df_copy = df_copy.rename(columns={"Date": date_col})
            else:
                raise ValueError(f"DataFrame {i} missing '{date_col}' column")

        # 标准化列顺序
        core_cols = [c for c in REQUIRED_COLS_PANEL if c in df_copy.columns]
        other_cols = [c for c in df_copy.columns if c not in core_cols]
        df_copy = df_copy[core_cols + other_cols]

        processed_frames.append(df_copy)

    # 合并
    panel_df = pd.concat(processed_frames, ignore_index=True)

    # 日历对齐
    if align_calendar_method:
        panel_df = align_calendar(
            panel_df,
            date_col=date_col,
            ticker_col=ticker_col,
            method=align_calendar_method,
        )

    # 排序
    panel_df = panel_df.sort_values([date_col, ticker_col]).reset_index(drop=True)

    logger.info(
        "Built panel: %d rows, %d tickers, date range: [%s, %s]",
        len(panel_df),
        panel_df[ticker_col].nunique(),
        panel_df[date_col].min(),
        panel_df[date_col].max(),
    )

    return panel_df


def validate_panel(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    required_cols: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    验证 panel 数据结构。

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        required_cols: 必需列列表

    Returns:
        Tuple[valid, issues]:
            - valid: 是否有效
            - issues: 问题列表
    """
    issues: List[str] = []

    # 检查必需列
    check_cols = [date_col, ticker_col]
    if required_cols:
        check_cols.extend(required_cols)

    missing_cols = [c for c in check_cols if c not in panel_df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    # 检查重复 (date, ticker) 组合
    if date_col in panel_df.columns and ticker_col in panel_df.columns:
        duplicates = panel_df.duplicated(subset=[date_col, ticker_col]).sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate (date, ticker) pairs")

    # 检查日期连续性（可选警告）
    if date_col in panel_df.columns:
        unique_dates = panel_df[date_col].nunique()
        n_tickers = panel_df[ticker_col].nunique()
        expected_rows = unique_dates * n_tickers
        actual_rows = len(panel_df)
        if actual_rows < expected_rows:
            missing_pct = (expected_rows - actual_rows) / expected_rows * 100
            issues.append(
                f"Panel is {missing_pct:.1f}% sparse "
                f"(expected {expected_rows} rows, got {actual_rows})"
            )

    return len(issues) == 0, issues


def panel_to_wide(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    value_col: str = "close_qfq",
) -> pd.DataFrame:
    """
    将 panel 数据转换成宽格式（pivot）。

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        value_col: 值列名

    Returns:
        宽格式 DataFrame（index=date, columns=ticker, values=value_col）
    """
    wide_df = panel_df.pivot(
        index=date_col,
        columns=ticker_col,
        values=value_col,
    )
    return wide_df


def panel_to_long(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    确保 panel 数据是长格式。

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名

    Returns:
        长格式 DataFrame（已排序）
    """
    long_df = panel_df.sort_values([date_col, ticker_col]).reset_index(drop=True)
    return long_df


# =========================================================
# 公开 API 导出
# =========================================================
__all__ = [
    "add_ticker_column",
    "align_calendar",
    "build_panel",
    "validate_panel",
    "panel_to_wide",
    "panel_to_long",
]
