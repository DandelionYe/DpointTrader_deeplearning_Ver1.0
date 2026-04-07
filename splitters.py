# splitters.py
"""
Panel 数据切分器模块
==================

本模块提供基于日期的时序数据切分功能，支持多股票 panel 数据。

主要功能:
    - walkforward_splits_by_date: 标准 Walk-Forward 日期切分
    - walkforward_splits_with_embargo: 带 embargo gap 的切分
    - final_holdout_split: 最终 holdout 集切分
    - recommend_n_folds: 根据数据量推荐折数

与单股票切分的区别:
    - 切分对象是日期，不是样本行
    - 返回 split spec（日期范围），不是 X/y 切分
    - 训练阶段再根据 split spec 过滤 panel 数据

使用示例:
    >>> from splitters import walkforward_splits_by_date
    >>> splits = walkforward_splits_by_date(panel_df, n_folds=4)
    >>> for train_dates, val_dates in splits:
    ...     train_df = panel_df[panel_df["date"].isin(train_dates)]
    ...     val_df = panel_df[panel_df["date"].isin(val_dates)]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitSpec:
    """切分规格说明。

    Attributes:
        train_start: 训练集开始日期
        train_end: 训练集结束日期
        val_start: 验证集开始日期
        val_end: 验证集结束日期
        embargo_days: embargo 天数（如果有）
        fold_id: 折 ID
    """
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    embargo_days: int = 0
    fold_id: int = 0

    def to_date_sets(
        self,
        unique_dates: List[pd.Timestamp],
    ) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
        """
        将规格转换成实际日期列表。

        Args:
            unique_dates: 所有唯一日期的列表

        Returns:
            Tuple[train_dates, val_dates]
        """
        train_dates = [
            d for d in unique_dates
            if self.train_start <= d <= self.train_end
        ]
        val_dates = [
            d for d in unique_dates
            if self.val_start <= d <= self.val_end
        ]
        return train_dates, val_dates


@dataclass
class SplitResult:
    """切分结果。

    Attributes:
        train_df: 训练集 DataFrame
        val_df: 验证集 DataFrame
        spec: 切分规格
    """
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    spec: SplitSpec


def walkforward_splits_by_date(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    n_folds: int = 4,
    train_start_ratio: float = 0.5,
    min_rows: int = 50,
) -> List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]]:
    """
    生成 walk-forward 日期切分。

    这是多股票 panel 版本的切分器，返回日期列表而不是 X/y 切分。

    参数说明：
        n_folds          : 验证折数，默认 4
        train_start_ratio: 第一折训练集占全部数据的比例，默认 0.5
        min_rows         : 每折的最小行数约束，不足时跳过该折

    切分示意（n_folds=4, train_start_ratio=0.5）：
        折 1: train=[0%~50%]   val=[50%~62.5%]
        折 2: train=[0%~62%]   val=[62.5%~75%]
        折 3: train=[0%~75%]   val=[75%~87.5%]
        折 4: train=[0%~87%]   val=[87.5%~100%]

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        n_folds: 验证折数
        train_start_ratio: 初始训练集比例
        min_rows: 每折最小行数约束

    Returns:
        List[Tuple[train_dates, val_dates]]: 日期列表元组的列表
    """
    # 获取唯一日期
    unique_dates = sorted(panel_df[date_col].unique())
    n_dates = len(unique_dates)

    if n_dates < 2:
        logger.warning("Not enough unique dates for walk-forward splits")
        return []

    # 计算切分点
    cuts = [
        train_start_ratio + (1.0 - train_start_ratio) * i / n_folds
        for i in range(n_folds + 1)
    ]

    splits = []
    for k in range(len(cuts) - 1):
        train_end_idx = int(n_dates * cuts[k])
        val_end_idx = int(n_dates * cuts[k + 1])

        train_dates = unique_dates[:train_end_idx]
        val_dates = unique_dates[train_end_idx:val_end_idx]

        # 计算行数约束（考虑多股票）
        n_tickers = panel_df[ticker_col].nunique()
        train_rows = len(train_dates) * n_tickers
        val_rows = len(val_dates) * n_tickers

        if train_rows < min_rows or val_rows < min_rows:
            logger.warning(
                "walkforward_splits_by_date: fold %d skipped "
                "(train_rows=%d, val_rows=%d, min_rows=%d)",
                k + 1, train_rows, val_rows, min_rows
            )
            continue

        splits.append((train_dates, val_dates))

    if not splits:
        logger.warning(
            "walkforward_splits_by_date: ALL %d folds skipped. "
            "Total dates=%d, train_start_ratio=%.2f, min_rows=%d",
            n_folds, n_dates, train_start_ratio, min_rows
        )

    return splits


def walkforward_splits_with_embargo(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    n_folds: int = 4,
    train_start_ratio: float = 0.5,
    min_rows: int = 60,
    embargo_days: int = 5,
) -> List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]]:
    """
    带 embargo gap 的 Walk-Forward 切分。

    在训练集和验证集之间留出 gap，防止滚动窗口特征导致的信息泄露。

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        n_folds: 验证折数
        train_start_ratio: 初始训练集比例
        min_rows: 最小行数约束
        embargo_days: embargo 天数

    Returns:
        List[Tuple[train_dates, val_dates]]
    """
    # 获取唯一日期
    unique_dates = sorted(panel_df[date_col].unique())
    n_dates = len(unique_dates)

    if n_dates < embargo_days + 2:
        logger.warning("Not enough dates for embargo=%d", embargo_days)
        return []

    # 计算切分点
    cuts = [
        train_start_ratio + (1.0 - train_start_ratio) * i / n_folds
        for i in range(n_folds + 1)
    ]

    splits = []
    for k in range(len(cuts) - 1):
        train_end_idx = int(n_dates * cuts[k])
        val_end_idx = int(n_dates * cuts[k + 1])

        # 应用 embargo：验证集向后推移
        val_start_idx = train_end_idx + embargo_days

        if val_start_idx >= val_end_idx:
            logger.warning(
                "walkforward_splits_with_embargo: fold %d skipped (embargo=%d)",
                k + 1, embargo_days
            )
            continue

        train_dates = unique_dates[:train_end_idx]
        val_dates = unique_dates[val_start_idx:val_end_idx]

        # 行数约束
        n_tickers = panel_df[ticker_col].nunique()
        train_rows = len(train_dates) * n_tickers
        val_rows = len(val_dates) * n_tickers

        if train_rows < min_rows or val_rows < min_rows:
            logger.warning(
                "walkforward_splits_with_embargo: fold %d skipped "
                "(train_rows=%d, val_rows=%d)",
                k + 1, train_rows, val_rows
            )
            continue

        splits.append((train_dates, val_dates))

    if not splits:
        logger.warning(
            "walkforward_splits_with_embargo: ALL %d folds skipped",
            n_folds
        )

    return splits


def nested_walkforward_splits_by_date(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    n_outer_folds: int = 3,
    n_inner_folds: int = 2,
    train_start_ratio: float = 0.5,
    min_rows: int = 60,
    embargo_days: int = 5,
) -> List[Tuple[
    List[pd.Timestamp],  # outer_train_dates
    List[pd.Timestamp],  # outer_val_dates
    List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]],  # inner_splits
]]:
    """
    嵌套 Walk-Forward 切分（日期版本）。

    外层：标准 walk-forward split
    内层：在外层训练集上再做 expanding-window walk-forward

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        n_outer_folds: 外层折数
        n_inner_folds: 内层折数
        train_start_ratio: 外层初始训练集比例
        min_rows: 最小行数约束
        embargo_days: embargo 天数

    Returns:
        List[Tuple[outer_train_dates, outer_val_dates, inner_splits]]
    """
    unique_dates = sorted(panel_df[date_col].unique())
    n_dates = len(unique_dates)

    cuts = [
        train_start_ratio + (1.0 - train_start_ratio) * i / n_outer_folds
        for i in range(n_outer_folds + 1)
    ]

    splits = []
    for k in range(len(cuts) - 1):
        outer_train_end_idx = int(n_dates * cuts[k])
        outer_val_end_idx = int(n_dates * cuts[k + 1])

        # 应用 embargo
        outer_val_start_idx = outer_train_end_idx + embargo_days
        if outer_val_start_idx >= outer_val_end_idx:
            logger.warning(
                "nested_walkforward: fold %d skipped due to embargo_days=%d",
                k + 1, embargo_days
            )
            continue

        outer_train_dates = unique_dates[:outer_train_end_idx]
        outer_val_dates = unique_dates[outer_val_start_idx:outer_val_end_idx]

        # 行数约束
        n_tickers = panel_df[ticker_col].nunique()
        outer_train_rows = len(outer_train_dates) * n_tickers
        outer_val_rows = len(outer_val_dates) * n_tickers

        if outer_train_rows < min_rows or outer_val_rows < min_rows:
            logger.warning(
                "nested_walkforward: fold %d skipped (train_rows=%d, val_rows=%d)",
                k + 1, outer_train_rows, outer_val_rows
            )
            continue

        # 内层 walk-forward：在外层训练集日期上切分
        # 复用标准 walk-forward 逻辑
        inner_panel = panel_df[panel_df[date_col].isin(outer_train_dates)]
        inner_splits = walkforward_splits_by_date(
            inner_panel,
            date_col=date_col,
            ticker_col=ticker_col,
            n_folds=n_inner_folds,
            train_start_ratio=train_start_ratio,
            min_rows=min_rows,
        )

        if not inner_splits:
            logger.warning(
                "nested_walkforward: fold %d skipped (no valid inner splits)",
                k + 1
            )
            continue

        splits.append((outer_train_dates, outer_val_dates, inner_splits))

    if not splits:
        logger.warning("nested_walkforward: ALL %d folds skipped", n_outer_folds)

    return splits


def final_holdout_split_by_date(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    holdout_ratio: float = 0.15,
    min_holdout_rows: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从数据末尾切出 holdout 集（日期版本）。

    三阶段验证流程：
        1. Search OOS: walk-forward splits 在 search 数据上评估
        2. Selection OOS: top-K 候选在 search 数据上重新验证
        3. Final Holdout OOS: 最优配置在 holdout 集上做最终评估

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        holdout_ratio: holdout 集比例
        min_holdout_rows: holdout 集最小行数

    Returns:
        Tuple[search_df, holdout_df]

    Raises:
        ValueError: 当 holdout_size < min_holdout_rows 时
    """
    unique_dates = sorted(panel_df[date_col].unique())
    n_dates = len(unique_dates)
    n_tickers = panel_df[ticker_col].nunique() if "ticker" in panel_df.columns else 1

    holdout_size = int(n_dates * holdout_ratio)
    holdout_rows = holdout_size * n_tickers

    if holdout_rows < min_holdout_rows:
        raise ValueError(
            f"holdout_rows={holdout_rows} < min_holdout_rows={min_holdout_rows}. "
            f"Increase holdout_ratio or use more data."
        )

    holdout_start_idx = n_dates - holdout_size
    holdout_dates = unique_dates[holdout_start_idx:]
    search_dates = unique_dates[:holdout_start_idx]

    search_df = panel_df[panel_df[date_col].isin(search_dates)].copy()
    holdout_df = panel_df[panel_df[date_col].isin(holdout_dates)].copy()

    logger.info(
        "P0 Final Holdout Split: search=%d rows (%d dates), "
        "holdout=%d rows (%d dates, %.1f%%)",
        len(search_df), len(search_dates),
        len(holdout_df), len(holdout_dates), holdout_ratio * 100
    )

    return search_df, holdout_df


def recommend_n_folds(
    n_dates: int,
    n_tickers: int = 1,
    train_start_ratio: float = 0.5,
    target_trades_per_fold: int = 4,
    assumed_trade_freq: float = 1.0 / 15.0,
    min_rows: int = 50,
    min_folds: int = 2,
    max_folds: int = 8,
) -> int:
    """
    根据数据量自适应推算合理的 walk-forward 折数（日期版本）。

    推算原则：
        在满足以下三个约束的前提下，选取尽可能大的折数：
            ① 每折验证期行数 ≥ min_rows
            ② 每折期望交易次数 ≈ target_trades_per_fold
            ③ 折数在 [min_folds, max_folds] 范围内

    Args:
        n_dates: 唯一日期数量
        n_tickers: 股票数量
        train_start_ratio: 初始训练集比例
        target_trades_per_fold: 每折目标交易次数
        assumed_trade_freq: 假设的交易频率
        min_rows: 每折最小行数
        min_folds: 最少折数
        max_folds: 最多折数

    Returns:
        int: 推荐的折数
    """
    val_pool = n_dates * (1.0 - train_start_ratio)

    best_n = min_folds
    for n in range(max_folds, min_folds - 1, -1):
        val_dates_per_fold = val_pool / n
        val_rows_per_fold = val_dates_per_fold * n_tickers

        # 约束①：验证折行数 ≥ min_rows
        if val_rows_per_fold < min_rows:
            continue

        # 约束②：期望交易次数 ≥ target_trades_per_fold
        expected_trades = val_rows_per_fold * assumed_trade_freq
        if expected_trades < target_trades_per_fold:
            continue

        # 满足所有约束
        best_n = n
        break

    return max(min_folds, min(max_folds, best_n))


def filter_panel_by_dates(
    panel_df: pd.DataFrame,
    train_dates: List[pd.Timestamp],
    val_dates: List[pd.Timestamp],
    *,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    根据日期列表过滤 panel 数据。

    Args:
        panel_df: panel DataFrame
        train_dates: 训练集日期列表
        val_dates: 验证集日期列表
        date_col: 日期列名

    Returns:
        Tuple[train_df, val_df]
    """
    train_df = panel_df[panel_df[date_col].isin(train_dates)].copy()
    val_df = panel_df[panel_df[date_col].isin(val_dates)].copy()
    return train_df, val_df


# =========================================================
# 公开 API 导出
# =========================================================
__all__ = [
    "SplitSpec",
    "SplitResult",
    "walkforward_splits_by_date",
    "walkforward_splits_with_embargo",
    "nested_walkforward_splits_by_date",
    "final_holdout_split_by_date",
    "recommend_n_folds",
    "filter_panel_by_dates",
]
