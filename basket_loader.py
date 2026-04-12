# basket_loader.py
"""
Basket 数据加载器模块
=====================

本模块提供读取 basket 文件夹（多股票 CSV 集合）的功能。

主要功能:
    - load_basket_folder: 读取整个 basket 文件夹
    - discover_basket_files: 发现 basket 中的 CSV 文件
    - load_basket_manifest: 从 manifest 文件加载 basket 配置

使用示例:
    >>> from basket_loader import load_basket_folder
    >>> panel_df, reports, basket_meta = load_basket_folder("./data/basket_1")
"""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from constants import (
    DATA_CONTRACT_VERSION,
    DEFAULT_FILE_PATTERN,
)
from csv_loader import SingleStockReport, load_single_csv
from panel_builder import build_panel

logger = logging.getLogger(__name__)


@dataclass
class BasketMeta:
    """Basket 元数据。

    Attributes:
        basket_name: Basket 名称
        basket_path: Basket 文件夹路径
        n_tickers: 股票数量
        file_format: 文件格式（csv/parquet 等）
        data_contract_version: 数据契约版本
        tickers: 股票代码列表
        files: 文件路径列表
        date_range: 日期范围
        notes: 注释
    """
    basket_name: str
    basket_path: str
    n_tickers: int
    file_format: str = "csv"
    data_contract_version: str = DATA_CONTRACT_VERSION
    tickers: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    date_range: Tuple[Optional[str], Optional[str]] = (None, None)
    notes: List[str] = field(default_factory=list)


@dataclass
class BasketReport:
    """Basket 数据质量报告。

    Attributes:
        basket_name: Basket 名称
        total_rows: 总行数
        n_tickers: 股票数量
        stock_reports: 单股票报告列表
        missing_files: 缺失的文件列表
        load_errors: 加载错误列表
        notes: 注释
    """
    basket_name: str
    total_rows: int
    n_tickers: int
    stock_reports: List[SingleStockReport] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)
    load_errors: List[Dict[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def discover_basket_files(
    basket_path: str,
    file_pattern: str = DEFAULT_FILE_PATTERN,
) -> List[str]:
    """
    发现 basket 文件夹中的文件。

    Args:
        basket_path: Basket 文件夹路径
        file_pattern: 文件匹配模式，如 "*.csv"

    Returns:
        文件路径列表
    """
    pattern = os.path.join(basket_path, file_pattern)
    files = glob.glob(pattern)
    return sorted(files)


def extract_ticker_from_filename(file_path: str) -> str:
    """
    从文件名提取 ticker。

    Args:
        file_path: 文件路径

    Returns:
        ticker 字符串
    """
    basename = os.path.basename(file_path)
    ticker = os.path.splitext(basename)[0]
    return ticker


def load_basket_manifest(manifest_path: str) -> Dict:
    """
    从 manifest 文件加载 basket 配置。

    Manifest 文件格式 (JSON):
    {
        "basket_name": "basket_1",
        "tickers": ["600036", "601318", ...],
        "file_pattern": "*.csv",
        "ticker_from": "manifest",
        "notes": "..."
    }

    Args:
        manifest_path: manifest 文件路径

    Returns:
        manifest 字典
    """
    import json
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_basket_folder(
    basket_path: str,
    *,
    file_pattern: str = DEFAULT_FILE_PATTERN,
    ticker_from: str = "filename",
    manifest_path: Optional[str] = None,
    column_map: Optional[Dict[str, str]] = None,
    strict_columns: bool = False,
    allow_missing_optional: bool = True,
    encoding: str = "utf-8-sig",
) -> Tuple[pd.DataFrame, BasketReport, BasketMeta]:
    """
    读取整个 basket 文件夹，合并成 panel 数据。

    Args:
        basket_path: Basket 文件夹路径
        file_pattern: 文件匹配模式
        ticker_from: ticker 来源，"filename" 或 "manifest"
        manifest_path: manifest 文件路径（可选）
        column_map: 列名映射字典
        strict_columns: 是否严格检查必需列
        allow_missing_optional: 是否允许缺失可选列
        encoding: 文件编码

    Returns:
        Tuple[pd.DataFrame, BasketReport, BasketMeta]:
            - panel_df: 合并后的 panel DataFrame（含 date/ticker 列）
            - report: 数据质量报告
            - meta: basket 元数据

    Raises:
        FileNotFoundError: basket 路径不存在
        ValueError: 没有找到任何文件
    """
    notes: List[str] = []
    load_errors: List[Dict[str, str]] = []

    # 检查路径
    if not os.path.isdir(basket_path):
        raise FileNotFoundError(f"Basket path not found: {basket_path}")

    # 加载 manifest（如果有）
    manifest_data = None
    if manifest_path and os.path.exists(manifest_path):
        manifest_data = load_basket_manifest(manifest_path)
        notes.append(f"Loaded manifest from: {manifest_path}")

    # 发现文件
    files = discover_basket_files(basket_path, file_pattern)
    if not files:
        raise ValueError(
            f"No files found matching pattern '{file_pattern}' in {basket_path}"
        )

    # 提取 basket 名称
    basket_name = os.path.basename(basket_path.rstrip("/\\"))

    # 加载每只股票
    stock_frames: List[pd.DataFrame] = []
    stock_reports: List[SingleStockReport] = []
    tickers: List[str] = []

    for file_path in files:
        try:
            # 确定 ticker
            if ticker_from == "manifest" and manifest_data:
                # 从 manifest 获取 ticker 列表
                ticker = None
                for t in manifest_data.get("tickers", []):
                    if t in file_path:
                        ticker = t
                        break
                if ticker is None:
                    ticker = extract_ticker_from_filename(file_path)
            else:
                # 从文件名提取
                ticker = extract_ticker_from_filename(file_path)

            # 加载单股票
            df, report = load_single_csv(
                file_path,
                ticker=ticker,
                column_map=column_map,
                strict_columns=strict_columns,
                allow_missing_optional=allow_missing_optional,
                encoding=encoding,
            )

            # 添加 ticker 列
            df["ticker"] = ticker

            stock_frames.append(df)
            stock_reports.append(report)
            tickers.append(ticker)

        except Exception as e:
            load_errors.append({
                "file": file_path,
                "error": str(e),
            })
            notes.append(f"Error loading {file_path}: {e}")

    if not stock_frames:
        raise ValueError("No valid stock data loaded from basket folder")

    panel_df = build_panel(
        stock_frames,
        basket_name=basket_name,
        date_col="date",
        ticker_col="ticker",
        align_calendar_method="outer",
        add_missing_tickers=True,
    )

    # 计算日期范围
    if "date" in panel_df.columns:
        date_min = panel_df["date"].min()
        date_max = panel_df["date"].max()
        date_range = (str(date_min), str(date_max))
    else:
        date_range = (None, None)

    # 构建 meta
    meta = BasketMeta(
        basket_name=basket_name,
        basket_path=basket_path,
        n_tickers=len(tickers),
        file_format="csv",
        data_contract_version=DATA_CONTRACT_VERSION,
        tickers=tickers,
        files=files,
        date_range=date_range,
        notes=notes,
    )

    # 构建 report
    total_rows = sum(r.rows_after_clean for r in stock_reports)
    report = BasketReport(
        basket_name=basket_name,
        total_rows=total_rows,
        n_tickers=len(tickers),
        stock_reports=stock_reports,
        missing_files=[],
        load_errors=load_errors,
        notes=notes,
    )

    return panel_df, report, meta


# =========================================================
# 公开 API 导出
# =========================================================
__all__ = [
    "BasketMeta",
    "BasketReport",
    "discover_basket_files",
    "extract_ticker_from_filename",
    "load_basket_manifest",
    "load_basket_folder",
]
