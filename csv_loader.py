# csv_loader.py
"""
CSV 数据加载器模块
==================

本模块提供单文件 CSV 股票数据的读取与列映射功能。

主要功能:
    - load_single_csv: 从单个 CSV 文件加载股票数据
    - standardize_columns: 列名标准化与映射
    - validate_csv_structure: CSV 结构验证

使用示例:
    >>> from csv_loader import load_single_csv
    >>> df, report = load_single_csv("600036.csv", ticker="600036")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from constants import (
    REQUIRED_COLS_SINGLE,
    OPTIONAL_COLS,
    DEFAULT_COLUMN_MAP,
)

logger = logging.getLogger(__name__)


@dataclass
class SingleStockReport:
    """单股票数据质量报告。

    Attributes:
        ticker: 股票代码
        source_file: 源文件路径
        rows_raw: 原始数据行数
        rows_after_clean: 清洗后的行数
        missing_optional_cols: 缺失的可选列列表
        derived_cols: 衍生列列表
        notes: 处理过程中的注释和警告
    """
    ticker: str
    source_file: str
    rows_raw: int
    rows_after_clean: int
    missing_optional_cols: List[str] = field(default_factory=list)
    derived_cols: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def standardize_columns(
    df: pd.DataFrame,
    column_map: Optional[Dict[str, str]] = None,
    *,
    encoding: str = "utf-8-sig",
) -> pd.DataFrame:
    """
    标准化 DataFrame 列名。

    Args:
        df: 输入 DataFrame
        column_map: 列名映射字典，若为 None 则使用 DEFAULT_COLUMN_MAP
        encoding: 文件编码（仅用于日志记录）

    Returns:
        列名标准化后的 DataFrame
    """
    if column_map is None:
        column_map = DEFAULT_COLUMN_MAP

    # 创建列名映射表
    col_mapping = {}
    for orig_col in df.columns:
        # 去除空格
        cleaned = orig_col.strip()
        # 查找映射
        if cleaned in column_map:
            col_mapping[orig_col] = column_map[cleaned]
        elif orig_col in column_map:
            col_mapping[orig_col] = column_map[orig_col]
        else:
            # 保持原列名
            col_mapping[orig_col] = cleaned

    # 重命名列
    df = df.rename(columns=col_mapping)
    return df


def validate_csv_structure(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    optional_cols: Optional[List[str]] = None,
    *,
    strict: bool = True,
) -> Tuple[bool, List[str], List[str]]:
    """
    验证 CSV 结构是否满足要求。

    Args:
        df: 输入 DataFrame
        required_cols: 必需列列表，若为 None 则使用 REQUIRED_COLS_SINGLE
        optional_cols: 可选列列表，若为 None 则使用 OPTIONAL_COLS
        strict: 是否严格检查，True 时缺少必需列会抛出异常

    Returns:
        Tuple[valid, missing_required, missing_optional]:
            - valid: 是否有效
            - missing_required: 缺失的必需列
            - missing_optional: 缺失的可选列

    Raises:
        ValueError: 当 strict=True 且缺少必需列时
    """
    if required_cols is None:
        required_cols = REQUIRED_COLS_SINGLE
    if optional_cols is None:
        optional_cols = OPTIONAL_COLS

    present_cols = set(df.columns)
    missing_required = [c for c in required_cols if c not in present_cols]
    missing_optional = [c for c in optional_cols if c not in present_cols]

    valid = len(missing_required) == 0

    if not valid and strict:
        raise ValueError(
            f"Missing required columns: {missing_required}. "
            f"Found columns: {list(df.columns)}"
        )

    return valid, missing_required, missing_optional


def load_single_csv(
    file_path: str,
    *,
    ticker: Optional[str] = None,
    column_map: Optional[Dict[str, str]] = None,
    strict_columns: bool = True,
    allow_missing_optional: bool = True,
    encoding: str = "utf-8-sig",
) -> Tuple[pd.DataFrame, SingleStockReport]:
    """
    从单个 CSV 文件加载股票数据。

    支持的 CSV 列名映射（默认）:
        - "Date" -> "date"
        - "Open (CNY, qfq)" -> "open_qfq"
        - "High (CNY, qfq)" -> "high_qfq"
        - "Low (CNY, qfq)" -> "low_qfq"
        - "Close (CNY, qfq)" -> "close_qfq"
        - "Volume (shares)" -> "volume"
        - "Amount (CNY)" -> "amount" (可选)
        - "Turnover Rate" -> "turnover_rate" (可选)

    Args:
        file_path: CSV 文件路径
        ticker: 股票代码，若为 None 则从文件名提取（去除扩展名）
        column_map: 自定义列名映射字典
        strict_columns: 是否严格检查必需列
        allow_missing_optional: 是否允许缺失可选列
        encoding: 文件编码

    Returns:
        Tuple[pd.DataFrame, SingleStockReport]: 清洗后的 DataFrame 和数据报告

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 缺少必需列且 strict_columns=True
    """
    notes: List[str] = []
    derived_cols: List[str] = []

    # 提取 ticker
    if ticker is None:
        import os
        ticker = os.path.splitext(os.path.basename(file_path))[0]
        notes.append(f"Ticker extracted from filename: {ticker}")

    # 读取 CSV
    df = pd.read_csv(file_path, encoding=encoding)
    rows_raw = len(df)

    # 列标准化
    df = standardize_columns(df, column_map, encoding=encoding)

    # 验证结构
    valid, missing_required, missing_optional = validate_csv_structure(
        df,
        strict=strict_columns,
    )

    if not allow_missing_optional and missing_optional:
        raise ValueError(
            f"Missing optional columns (not allowed): {missing_optional}"
        )

    # 日期解析
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = int(df["date"].isna().sum())
    if bad_dates > 0:
        notes.append(f"Dropped rows with unparseable dates: {bad_dates}")
        df = df.dropna(subset=["date"]).copy()

    # 排序
    df = df.sort_values("date").reset_index(drop=True)

    # 重复日期检查
    duplicate_dates = int(df["date"].duplicated().sum())
    if duplicate_dates > 0:
        notes.append(
            f"Found duplicate dates: {duplicate_dates}. Keeping last occurrence."
        )
        df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    # 数值列转换
    num_cols = [c for c in REQUIRED_COLS_SINGLE if c != "date" and c in df.columns]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 去除核心 OHLC 缺失值
    core_cols = ["open_qfq", "high_qfq", "low_qfq", "close_qfq"]
    rows_before = len(df)
    df = df.dropna(subset=core_cols).copy()
    if len(df) < rows_before:
        notes.append(f"Dropped rows with NaN core OHLC: {rows_before - len(df)}")

    # 填充缺失的非核心字段
    if "volume" in df.columns:
        n_missing = int(df["volume"].isna().sum())
        if n_missing > 0:
            notes.append(f"Filled missing volume with 0: {n_missing}")
            df["volume"] = df["volume"].fillna(0.0)

    # 衍生列：amount_proxy（如果 amount 缺失）
    if "amount" not in df.columns and "volume" in df.columns and "close_qfq" in df.columns:
        df["amount"] = df["close_qfq"] * df["volume"]
        derived_cols.append("amount_proxy")
        notes.append("Derived 'amount' column: close_qfq * volume")

    # 有效性检查
    bad_price = int(
        ((df["open_qfq"] <= 0) | (df["high_qfq"] <= 0) | 
         (df["low_qfq"] <= 0) | (df["close_qfq"] <= 0)).sum()
    )
    if bad_price > 0:
        notes.append(f"Dropped non-positive price rows: {bad_price}")
        df = df[
            (df["open_qfq"] > 0) & (df["high_qfq"] > 0) &
            (df["low_qfq"] > 0) & (df["close_qfq"] > 0)
        ].copy()

    # OHLC 一致性检查
    bad_ohlc_mask = ~(
        (df["high_qfq"] >= df[["open_qfq", "close_qfq", "low_qfq"]].max(axis=1)) &
        (df["low_qfq"] <= df[["open_qfq", "close_qfq", "high_qfq"]].min(axis=1))
    )
    bad_ohlc_rows = int(bad_ohlc_mask.sum())
    if bad_ohlc_rows > 0:
        notes.append(f"Dropped OHLC inconsistent rows: {bad_ohlc_rows}")
        df = df[~bad_ohlc_mask].copy()

    # 最终排序
    df = df.sort_values("date").reset_index(drop=True)

    # 数据警告
    if len(df) < 300:
        notes.append(
            f"Warning: data length {len(df)} < 300 trading days. "
            "ML may be unstable."
        )

    report = SingleStockReport(
        ticker=ticker,
        source_file=file_path,
        rows_raw=rows_raw,
        rows_after_clean=len(df),
        missing_optional_cols=missing_optional,
        derived_cols=derived_cols,
        notes=notes,
    )

    return df, report


# =========================================================
# 公开 API 导出
# =========================================================
__all__ = [
    "SingleStockReport",
    "standardize_columns",
    "validate_csv_structure",
    "load_single_csv",
]
