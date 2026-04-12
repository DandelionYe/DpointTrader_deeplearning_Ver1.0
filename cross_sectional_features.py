# cross_sectional_features.py
"""
横截面特征模块
==============

本模块提供按日横截面特征计算功能。

主要功能:
    - cross_sectional_rank: 横截面排序
    - cross_sectional_zscore: 横截面标准化
    - cross_sectional_industry_rank: 行业相对强弱
    - add_cross_sectional_features: 添加横截面特征

横截面特征 vs 时序特征:
    - 时序特征：在 ticker 内按时间滚动计算（如 MA、RSI）
    - 横截面特征：在日期内跨 ticker 计算（如 rank、zscore）

使用示例:
    >>> from cross_sectional_features import add_cross_sectional_features
    >>> panel_df = add_cross_sectional_features(panel_df, columns=["close_qfq", "volume"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CrossSectionalMeta:
    """横截面特征元数据。

    Attributes:
        cross_sectional_features: 横截面特征列表
        n_dates: 日期数
        n_tickers: 股票数
        notes: 注释
    """

    cross_sectional_features: List[str] = field(default_factory=list)
    n_dates: int = 0
    n_tickers: int = 0
    notes: List[str] = field(default_factory=list)


def cross_sectional_rank(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    value_col: str,
    output_col: Optional[str] = None,
    ascending: bool = False,
    method: str = "average",
) -> pd.Series:
    """
    横截面排序（Rank）。

    对每个日期，计算股票在该日的横截面排名。

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        value_col: 值列名
        output_col: 输出列名（若为 None 则返回 Series）
        ascending: 是否升序（默认 False，即降序，值越大 rank 越小）
        method: 排名方法，同 pandas.rank()

    Returns:
        rank Series 或修改后的 DataFrame
    """

    def rank_func(group):
        return group[value_col].rank(method=method, ascending=ascending)

    # 使用 include_groups=False 避免 pandas 新版本警告
    ranks = panel_df.groupby(date_col, group_keys=False).apply(rank_func, include_groups=False)

    if output_col:
        result = panel_df.copy()
        result[output_col] = ranks
        return result
    # ranks 现在是 DataFrame，需要转换回 Series
    if isinstance(ranks, pd.DataFrame):
        # 展平为 Series，保持原始 index
        ranks = ranks.stack().reset_index(drop=True)
    return ranks


def cross_sectional_zscore(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    value_col: str,
    output_col: Optional[str] = None,
    clip_outliers: bool = True,
    clip_std: float = 3.0,
) -> pd.Series:
    """
    横截面 Z-score 标准化。

    对每个日期，计算股票在该日的横截面标准化值。

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        value_col: 值列名
        output_col: 输出列名
        clip_outliers: 是否截断异常值
        clip_std: 截断标准差倍数

    Returns:
        zscore Series 或修改后的 DataFrame
    """

    def zscore_func(group):
        values = group[value_col]
        mu = values.mean()
        sigma = values.std()

        if sigma == 0 or np.isnan(sigma):
            return pd.Series(0.0, index=group.index)

        zscore = (values - mu) / sigma

        if clip_outliers:
            zscore = zscore.clip(-clip_std, clip_std)

        return zscore

    # 使用 include_groups=False 避免 pandas 新版本警告
    zscores = panel_df.groupby(date_col, group_keys=False).apply(zscore_func, include_groups=False)

    if output_col:
        result = panel_df.copy()
        result[output_col] = zscores
        return result
    # zscores 现在是 DataFrame，需要转换回 Series
    if isinstance(zscores, pd.DataFrame):
        zscores = zscores.stack().reset_index(drop=True)
    return zscores


def cross_sectional_percentile(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    value_col: str,
    output_col: Optional[str] = None,
) -> pd.Series:
    """
    横截面百分位数。

    对每个日期，计算股票在该日的横截面百分位排名（0-1）。

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        value_col: 值列名
        output_col: 输出列名

    Returns:
        percentile Series 或修改后的 DataFrame
    """

    def percentile_func(group):
        values = group[value_col]
        n = len(values)
        if n == 0:
            return pd.Series(0.5, index=group.index)

        ranks = values.rank(method="average", ascending=False)
        percentiles = (ranks - 1) / (n - 1) if n > 1 else pd.Series(0.5, index=group.index)
        return percentiles

    # 使用 include_groups=False 避免 pandas 新版本警告
    percentiles = panel_df.groupby(date_col, group_keys=False).apply(
        percentile_func, include_groups=False
    )

    if output_col:
        result = panel_df.copy()
        result[output_col] = percentiles
        return result
    # percentiles 现在是 DataFrame，需要转换回 Series
    if isinstance(percentiles, pd.DataFrame):
        percentiles = percentiles.stack().reset_index(drop=True)
    return percentiles


def cross_sectional_industry_rank(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    value_col: str,
    industry_col: str = "industry",
    output_col: Optional[str] = None,
) -> pd.Series:
    """
    行业相对强弱排名。

    对每个日期，计算股票在所属行业内的排名。

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        value_col: 值列名
        industry_col: 行业列名
        output_col: 输出列名

    Returns:
        industry rank Series 或修改后的 DataFrame
    """

    def industry_rank_func(group):
        return group[value_col].rank(method="average", ascending=False)

    ranks = panel_df.groupby([date_col, industry_col], group_keys=False).apply(industry_rank_func)

    if output_col:
        result = panel_df.copy()
        result[output_col] = ranks
        return result
    return ranks


def cross_sectional_momentum(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close_qfq",
    lookback: int = 20,
    output_col: Optional[str] = None,
) -> pd.Series:
    """
    横截面动量因子。

    计算过去 N 日的累计收益率，然后做横截面标准化。

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        close_col: 收盘价列名
        lookback: 回看期数
        output_col: 输出列名

    Returns:
        momentum factor Series 或修改后的 DataFrame
    """
    df = panel_df.copy()

    # 计算时序动量（按 ticker）
    def calc_momentum(group):
        close = group[close_col]
        ret = close.pct_change(lookback)
        return ret

    df["momentum_raw"] = df.groupby(ticker_col, group_keys=False).apply(calc_momentum)

    # 横截面标准化
    momentum_cs = cross_sectional_zscore(
        df,
        date_col=date_col,
        ticker_col=ticker_col,
        value_col="momentum_raw",
    )

    if output_col:
        result = df.copy()
        result[output_col] = momentum_cs
        return result

    df["momentum_cs"] = momentum_cs
    return df["momentum_cs"]


def cross_sectional_volatility(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close_qfq",
    lookback: int = 20,
    output_col: Optional[str] = None,
) -> pd.Series:
    """
    横截面波动率因子。

    计算过去 N 日的收益率波动率，然后做横截面标准化。

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        close_col: 收盘价列名
        lookback: 回看期数
        output_col: 输出列名

    Returns:
        volatility factor Series 或修改后的 DataFrame
    """
    df = panel_df.copy()

    # 计算时序波动率（按 ticker）
    def calc_volatility(group):
        ret = group[close_col].pct_change()
        vol = ret.rolling(lookback, min_periods=lookback).std()
        return vol

    df["volatility_raw"] = df.groupby(ticker_col, group_keys=False).apply(calc_volatility)

    # 横截面标准化
    vol_cs = cross_sectional_zscore(
        df,
        date_col=date_col,
        ticker_col=ticker_col,
        value_col="volatility_raw",
    )

    if output_col:
        result = df.copy()
        result[output_col] = vol_cs
        return result

    df["volatility_cs"] = vol_cs
    return df["volatility_cs"]


def add_cross_sectional_features(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    columns: Optional[List[str]] = None,
    prefix: str = "cs",
    features: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, CrossSectionalMeta]:
    """
    添加横截面特征。

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        columns: 要计算横截面特征的列列表
        prefix: 输出列前缀
        features: 特征类型列表
            - "rank": 横截面排序
            - "zscore": 横截面标准化
            - "percentile": 横截面百分位
        **kwargs: 额外参数

    Returns:
        Tuple[panel_df, meta]: 添加特征后的 DataFrame 和元数据
    """
    if columns is None:
        columns = ["close_qfq", "volume", "amount"]

    if features is None:
        features = ["rank", "zscore", "percentile"]

    df = panel_df.copy()
    added_features: List[str] = []
    notes: List[str] = []

    for col in columns:
        if col not in df.columns:
            notes.append(f"Column '{col}' not found, skipping")
            continue

        if "rank" in features:
            rank_col = f"{prefix}_rank_{col}"
            df[rank_col] = cross_sectional_rank(
                df,
                date_col=date_col,
                ticker_col=ticker_col,
                value_col=col,
            )
            added_features.append(rank_col)

        if "zscore" in features:
            zscore_col = f"{prefix}_zscore_{col}"
            df[zscore_col] = cross_sectional_zscore(
                df,
                date_col=date_col,
                ticker_col=ticker_col,
                value_col=col,
            )
            added_features.append(zscore_col)

        if "percentile" in features:
            pct_col = f"{prefix}_pct_{col}"
            df[pct_col] = cross_sectional_percentile(
                df,
                date_col=date_col,
                ticker_col=ticker_col,
                value_col=col,
            )
            added_features.append(pct_col)

    meta = CrossSectionalMeta(
        cross_sectional_features=added_features,
        n_dates=df[date_col].nunique(),
        n_tickers=df[ticker_col].nunique(),
        notes=notes,
    )

    return df, meta


# =========================================================
# 公开 API 导出
# =========================================================
__all__ = [
    "CrossSectionalMeta",
    "cross_sectional_rank",
    "cross_sectional_zscore",
    "cross_sectional_percentile",
    "cross_sectional_industry_rank",
    "cross_sectional_momentum",
    "cross_sectional_volatility",
    "add_cross_sectional_features",
]
