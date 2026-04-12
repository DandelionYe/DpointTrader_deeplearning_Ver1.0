# labeler.py
"""
标签构造器模块
==============

本模块提供多股票 panel 数据的标签构造功能。

主要功能:
    - build_binary_label: 构建二分类标签（次日涨跌）
    - build_multiclass_label: 构建多分类标签
    - build_regression_label: 构建回归标签（次日收益率）
    - build_label: 统一标签构造入口

标签模式:
    - "binary_next_close_up": 二分类，close_{t+1} > close_t
    - "binary_return_positive": 二分类，return_{t+1} > 0
    - "multiclass_3": 三分类（涨/平/跌）
    - "regression_return": 回归（次日收益率）

使用示例:
    >>> from labeler import build_label
    >>> y = build_label(panel_df, mode="binary_next_close_up")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from constants import DEFAULT_LABEL_MODE
from tasks import LabelSpec

logger = logging.getLogger(__name__)


@dataclass
class LabelMeta:
    """标签元数据。

    Attributes:
        label_mode: 标签模式
        n_samples: 样本数
        n_positive: 正样本数（二分类）
        positive_ratio: 正样本比例
        class_distribution: 类别分布（多分类）
        label_mean: 标签均值（回归）
        label_std: 标签标准差（回归）
        notes: 注释
    """

    label_mode: str
    n_samples: int
    n_positive: Optional[int] = None
    positive_ratio: Optional[float] = None
    class_distribution: Optional[Dict[int, int]] = None
    label_mean: Optional[float] = None
    label_std: Optional[float] = None
    notes: List[str] = field(default_factory=list)


def build_binary_label(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close_qfq",
    threshold: float = 0.0,
    shift: int = 1,
) -> Tuple[pd.Series, LabelMeta]:
    """
    构建二分类标签。

    默认模式：close_{t+1} > close_t

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        close_col: 收盘价列名
        threshold: 阈值，默认 0.0
        shift: 前向 shift 期数，默认 1

    Returns:
        Tuple[label_series, meta]: 标签 Series 和元数据
    """
    df = panel_df.copy()

    close = df[close_col]
    future_close = df.groupby(ticker_col)[close_col].shift(-shift)
    df["label"] = pd.Series(np.nan, index=df.index, dtype=float)
    valid_mask = future_close.notna()
    df.loc[valid_mask, "label"] = (future_close.loc[valid_mask] > close.loc[valid_mask]).astype(
        float
    )

    # 去除末尾 NaN
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # 计算元数据
    n_samples = len(df)
    n_positive = int(df["label"].sum())
    positive_ratio = n_positive / n_samples if n_samples > 0 else None

    meta = LabelMeta(
        label_mode="binary",
        n_samples=n_samples,
        n_positive=n_positive,
        positive_ratio=positive_ratio,
        notes=[f"Threshold: {threshold}, Shift: {shift}"],
    )

    label_series = df["label"].copy()
    label_series.index = df.index

    return label_series, meta


def build_multiclass_label(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close_qfq",
    n_classes: int = 3,
    threshold_up: float = 0.01,
    threshold_down: float = -0.01,
    shift: int = 1,
) -> Tuple[pd.Series, LabelMeta]:
    """
    构建多分类标签。

    三分类：
        - 2: 涨（return > threshold_up）
        - 1: 平（threshold_down <= return <= threshold_up）
        - 0: 跌（return < threshold_down）

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        close_col: 收盘价列名
        n_classes: 类别数，默认 3
        threshold_up: 上涨阈值
        threshold_down: 下跌阈值
        shift: 前向 shift 期数

    Returns:
        Tuple[label_series, meta]
    """
    df = panel_df.copy()

    close = df[close_col]
    future_close = df.groupby(ticker_col)[close_col].shift(-shift)
    df["return"] = (future_close - close) / close

    # 多分类
    if n_classes == 3:
        conditions = [
            df["return"] > threshold_up,
            df["return"] < threshold_down,
        ]
        choices = [2, 0]
        df["label"] = np.select(conditions, choices, default=1)
    else:
        # 通用多分类：按收益率分位数
        df = df.dropna(subset=["return"])
        quantiles = np.linspace(0, 1, n_classes + 1)
        df["label"] = pd.qcut(df["return"], q=quantiles, labels=False, duplicates="drop")

    # 去除 NaN
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # 元数据
    n_samples = len(df)
    class_dist = df["label"].value_counts().to_dict()

    meta = LabelMeta(
        label_mode=f"multiclass_{n_classes}",
        n_samples=n_samples,
        class_distribution=class_dist,
        notes=[
            f"n_classes: {n_classes}",
            f"threshold_up: {threshold_up}",
            f"threshold_down: {threshold_down}",
        ],
    )

    label_series = df["label"].copy()
    label_series.index = df.index

    return label_series, meta


def build_regression_label(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close_qfq",
    shift: int = 1,
    log_return: bool = True,
) -> Tuple[pd.Series, LabelMeta]:
    """
    构建回归标签（次日收益率）。

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        close_col: 收盘价列名
        shift: 前向 shift 期数
        log_return: 是否使用对数收益率

    Returns:
        Tuple[label_series, meta]
    """
    df = panel_df.copy()

    close = df[close_col]
    future_close = df.groupby(ticker_col)[close_col].shift(-shift)
    if log_return:
        df["label"] = np.log(future_close / close)
    else:
        df["label"] = (future_close - close) / close

    # 去除 NaN
    df = df.dropna(subset=["label"])

    # 元数据
    n_samples = len(df)
    label_mean = float(df["label"].mean())
    label_std = float(df["label"].std())

    meta = LabelMeta(
        label_mode="regression",
        n_samples=n_samples,
        label_mean=label_mean,
        label_std=label_std,
        notes=[f"Log return: {log_return}, Shift: {shift}"],
    )

    label_series = df["label"].copy()
    label_series.index = df.index

    return label_series, meta


def build_label(
    panel_df: pd.DataFrame,
    *,
    mode: str = DEFAULT_LABEL_MODE,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close_qfq",
    **kwargs,
) -> Tuple[pd.Series, LabelMeta]:
    """
    统一标签构造入口。

    Args:
        panel_df: panel DataFrame
        mode: 标签模式
            - "binary_next_close_up": 二分类
            - "binary_return_positive": 二分类（收益率>0）
            - "multiclass_3": 三分类
            - "regression_return": 回归
        date_col: 日期列名
        ticker_col: ticker 列名
        close_col: 收盘价列名
        **kwargs: 传递给具体构造函数的额外参数

    Returns:
        Tuple[label_series, meta]

    Raises:
        ValueError: 未知标签模式
    """
    if mode in ("binary_next_close_up", "binary_return_positive"):
        return build_binary_label(
            panel_df,
            date_col=date_col,
            ticker_col=ticker_col,
            close_col=close_col,
            **kwargs,
        )

    elif mode.startswith("multiclass"):
        n_classes = int(mode.split("_")[-1]) if "_" in mode else 3
        return build_multiclass_label(
            panel_df,
            date_col=date_col,
            ticker_col=ticker_col,
            close_col=close_col,
            n_classes=n_classes,
            **kwargs,
        )

    elif mode.startswith("regression"):
        return build_regression_label(
            panel_df,
            date_col=date_col,
            ticker_col=ticker_col,
            close_col=close_col,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown label mode: {mode}")


def attach_label_to_panel(
    panel_df: pd.DataFrame,
    label: pd.Series,
    *,
    label_col: str = "label",
) -> pd.DataFrame:
    """
    将标签附加到 panel DataFrame。

    Args:
        panel_df: panel DataFrame
        label: 标签 Series
        label_col: 标签列名

    Returns:
        附加标签后的 DataFrame
    """
    df = panel_df.copy()
    df[label_col] = label
    return df


def build_labels(
    panel_df: pd.DataFrame,
    label_spec: LabelSpec,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close_qfq",
) -> Tuple[pd.Series, pd.DataFrame, LabelMeta]:
    extra_kwargs = {}
    if str(label_spec.label_mode).startswith("regression"):
        extra_kwargs["log_return"] = False
    target, meta = build_label(
        panel_df,
        mode=label_spec.label_mode,
        date_col=date_col,
        ticker_col=ticker_col,
        close_col=close_col,
        shift=max(1, int(label_spec.horizon_days)),
        **extra_kwargs,
    )
    label_end_date = panel_df.groupby(ticker_col)[date_col].shift(
        -max(1, int(label_spec.horizon_days))
    )
    label_meta = panel_df[[date_col, ticker_col]].copy()
    label_meta["label_end_date"] = pd.to_datetime(label_end_date)
    label_meta["target"] = target.reindex(label_meta.index)
    return target, label_meta, meta


# =========================================================
# 公开 API 导出
# =========================================================
__all__ = [
    "LabelMeta",
    "build_binary_label",
    "build_multiclass_label",
    "build_regression_label",
    "build_label",
    "build_labels",
    "attach_label_to_panel",
]
