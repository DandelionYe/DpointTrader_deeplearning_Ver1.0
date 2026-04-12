# portfolio_builder.py
"""
组合构建器模块
==============

本模块提供投资组合构建功能。

主要功能:
    - select_topk: 选择 TopK 股票
    - compute_weights: 计算权重
    - build_portfolio: 构建组合

权重方式:
    - "equal": 等权
    - "score": 按 score 归一化加权
    - "vol_inv": 按波动率倒数加权

使用示例:
    >>> from portfolio_builder import build_portfolio
    >>> portfolio = build_portfolio(scores_df, top_k=5, weighting="equal")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from constants import (
    DEFAULT_TOP_K,
    DEFAULT_WEIGHTING,
    DEFAULT_MAX_WEIGHT,
)

logger = logging.getLogger(__name__)


def _normalize_with_max_weight(
    raw_weights: np.ndarray,
    *,
    max_weight: float,
    atol: float = 1e-12,
) -> np.ndarray:
    """
    在满足单票上限的前提下归一化权重。

    当 n * max_weight < 1 时，满仓不可行，此时返回按上限截断后的权重，
    留下剩余现金。
    """
    raw = np.asarray(raw_weights, dtype=float)
    raw = np.clip(raw, 0.0, None)
    n = len(raw)

    if n == 0 or raw.sum() <= atol:
        return np.array([], dtype=float)

    if max_weight <= 0:
        return np.zeros(n, dtype=float)

    feasible_total = min(1.0, n * max_weight)
    scaled = raw / raw.sum() * feasible_total
    capped = np.minimum(scaled, max_weight)

    remainder = feasible_total - capped.sum()
    while remainder > atol:
        eligible = capped < (max_weight - atol)
        if not np.any(eligible):
            break

        eligible_raw = raw[eligible]
        if eligible_raw.sum() <= atol:
            add = np.full(eligible.sum(), remainder / eligible.sum())
        else:
            add = remainder * eligible_raw / eligible_raw.sum()

        eligible_indices = np.where(eligible)[0]
        for idx, inc in zip(eligible_indices, add):
            room = max_weight - capped[idx]
            delta = min(room, inc)
            capped[idx] += delta
            remainder -= delta

        if eligible_raw.sum() <= atol:
            break

    return capped


@dataclass
class PortfolioConfig:
    """组合配置。

    Attributes:
        top_k: TopK 数量
        weighting: 权重方式
        max_weight: 单票权重上限
        cash_buffer: 现金缓冲比例
        rebalance_freq: 调仓频率
        min_score: 最低分数阈值
        exclude_tickers: 排除的 ticker 列表
    """
    top_k: int = DEFAULT_TOP_K
    weighting: str = DEFAULT_WEIGHTING
    max_weight: float = DEFAULT_MAX_WEIGHT
    cash_buffer: float = 0.05
    rebalance_freq: str = "daily"
    min_score: Optional[float] = None
    skip_untradeable_on_rebalance: bool = True
    exclude_tickers: List[str] = field(default_factory=list)


@dataclass
class Portfolio:
    """投资组合。

    Attributes:
        date: 日期
        tickers: 持仓 ticker 列表
        weights: 权重列表
        scores: 分数列表
        total_weight: 总权重
        cash: 现金比例
        n_holdings: 持仓数
    """
    date: pd.Timestamp
    tickers: List[str]
    weights: List[float]
    scores: List[float]
    total_weight: float
    cash: float
    n_holdings: int


def select_topk(
    scores_df: pd.DataFrame,
    *,
    date: pd.Timestamp,
    score_col: str = "score",
    ticker_col: str = "ticker",
    date_col: str = "date",
    top_k: int = DEFAULT_TOP_K,
    min_score: Optional[float] = None,
    exclude_tickers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    选择 TopK 股票。

    Args:
        scores_df: 包含 score 的 DataFrame
        date: 日期
        score_col: 分数列名
        ticker_col: ticker 列名
        date_col: 日期列名
        top_k: TopK 数量
        min_score: 最低分数阈值
        exclude_tickers: 排除的 ticker 列表

    Returns:
        TopK 股票 DataFrame
    """
    # 过滤日期
    day_df = scores_df[scores_df[date_col] == date].copy()

    # 过滤排除的 ticker
    if exclude_tickers:
        day_df = day_df[~day_df[ticker_col].isin(exclude_tickers)]

    # 过滤最低分数
    if min_score is not None:
        day_df = day_df[day_df[score_col] >= min_score]

    # 按 score 排序取 TopK
    if len(day_df) > top_k:
        topk_df = day_df.nlargest(top_k, score_col)
    else:
        topk_df = day_df

    return topk_df


def compute_weights_equal(
    tickers: List[str],
    scores: Optional[List[float]] = None,
    *,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> List[float]:
    """
    计算等权权重。

    Args:
        tickers: ticker 列表
        scores: 分数列表（未使用）
        max_weight: 单票权重上限

    Returns:
        权重列表
    """
    n = len(tickers)
    if n == 0:
        return []

    base_weight = np.ones(n, dtype=float)
    return _normalize_with_max_weight(base_weight, max_weight=max_weight).tolist()


def compute_weights_score(
    tickers: List[str],
    scores: List[float],
    *,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    score_power: float = 1.0,
) -> List[float]:
    """
    按 score 归一化计算权重。

    Args:
        tickers: ticker 列表
        scores: 分数列表
        max_weight: 单票权重上限
        score_power: 分数幂次（>1 放大差异，<1 缩小差异）

    Returns:
        权重列表
    """
    n = len(tickers)
    if n == 0:
        return []

    # 分数转正值
    scores_arr = np.array(scores)
    scores_pos = scores_arr - scores_arr.min() + 1e-6

    # 幂次变换
    scores_transformed = scores_pos ** score_power

    weights = _normalize_with_max_weight(scores_transformed, max_weight=max_weight)
    return weights.tolist()


def compute_weights_vol_inv(
    tickers: List[str],
    scores: List[float],
    volatilities: List[float],
    *,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    vol_window: int = 20,
) -> List[float]:
    """
    按波动率倒数计算权重（低波动率加权）。

    Args:
        tickers: ticker 列表
        scores: 分数列表
        volatilities: 波动率列表
        max_weight: 单票权重上限
        vol_window: 波动率计算窗口

    Returns:
        权重列表
    """
    n = len(tickers)
    if n == 0:
        return []

    # 波动率倒数
    vol_arr = np.array(volatilities)
    vol_inv = 1.0 / (vol_arr + 1e-6)

    weights = _normalize_with_max_weight(vol_inv, max_weight=max_weight)
    return weights.tolist()


def compute_weights(
    tickers: List[str],
    scores: List[float],
    *,
    weighting: str = DEFAULT_WEIGHTING,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    volatilities: Optional[List[float]] = None,
    **kwargs,
) -> List[float]:
    """
    统一权重计算入口。

    Args:
        tickers: ticker 列表
        scores: 分数列表
        weighting: 权重方式
        max_weight: 单票权重上限
        volatilities: 波动率列表（vol_inv 方式需要）
        **kwargs: 额外参数

    Returns:
        权重列表
    """
    if weighting == "equal":
        return compute_weights_equal(tickers, scores, max_weight=max_weight)

    elif weighting == "score":
        return compute_weights_score(tickers, scores, max_weight=max_weight, **kwargs)

    elif weighting == "vol_inv":
        if volatilities is None:
            logger.warning("vol_inv weighting requires volatilities, falling back to equal")
            return compute_weights_equal(tickers, scores, max_weight=max_weight)
        return compute_weights_vol_inv(tickers, scores, volatilities, max_weight=max_weight, **kwargs)

    else:
        logger.warning(f"Unknown weighting '{weighting}', falling back to equal")
        return compute_weights_equal(tickers, scores, max_weight=max_weight)


def build_portfolio(
    scores_df: pd.DataFrame,
    *,
    date: pd.Timestamp,
    config: PortfolioConfig,
    score_col: str = "score",
    ticker_col: str = "ticker",
    date_col: str = "date",
) -> Portfolio:
    """
    构建单日投资组合。

    Args:
        scores_df: 包含 score 的 DataFrame
        date: 日期
        config: 组合配置
        score_col: 分数列名
        ticker_col: ticker 列名
        date_col: 日期列名

    Returns:
        Portfolio 数据类
    """
    candidate_scores = scores_df
    if config.skip_untradeable_on_rebalance and "is_tradeable" in scores_df.columns:
        candidate_scores = scores_df[scores_df["is_tradeable"].fillna(False)].copy()

    # 选择 TopK
    topk_df = select_topk(
        candidate_scores,
        date=date,
        score_col=score_col,
        ticker_col=ticker_col,
        date_col=date_col,
        top_k=config.top_k,
        min_score=config.min_score,
        exclude_tickers=config.exclude_tickers,
    )

    tickers = topk_df[ticker_col].tolist()
    scores = topk_df[score_col].tolist()

    # 计算权重
    weights = compute_weights(
        tickers,
        scores,
        weighting=config.weighting,
        max_weight=config.max_weight,
    )

    investable_fraction = max(0.0, 1.0 - config.cash_buffer)
    current_total = float(sum(weights))
    if current_total > 0 and current_total > investable_fraction:
        scale = investable_fraction / current_total
        weights = [w * scale for w in weights]

    # 计算现金比例
    total_weight = float(sum(weights))
    cash = max(0.0, 1.0 - total_weight)

    portfolio = Portfolio(
        date=date,
        tickers=tickers,
        weights=weights,
        scores=scores,
        total_weight=total_weight,
        cash=cash,
        n_holdings=len(tickers),
    )

    return portfolio


def build_portfolio_series(
    scores_df: pd.DataFrame,
    *,
    config: PortfolioConfig,
    score_col: str = "score",
    ticker_col: str = "ticker",
    date_col: str = "date",
    dates: Optional[List[pd.Timestamp]] = None,
) -> List[Portfolio]:
    """
    构建多日投资组合序列。

    Args:
        scores_df: 包含 score 的 DataFrame
        config: 组合配置
        score_col: 分数列名
        ticker_col: ticker 列名
        date_col: 日期列名
        dates: 日期列表（若为 None 则使用所有日期）

    Returns:
        Portfolio 列表
    """
    if dates is None:
        dates = sorted(scores_df[date_col].unique())

    portfolios = []
    for date in dates:
        portfolio = build_portfolio(
            scores_df,
            date=date,
            config=config,
            score_col=score_col,
            ticker_col=ticker_col,
            date_col=date_col,
        )
        portfolios.append(portfolio)

    return portfolios


def portfolio_to_df(portfolios: List[Portfolio]) -> pd.DataFrame:
    """
    将 Portfolio 列表转换成 DataFrame。

    Args:
        portfolios: Portfolio 列表

    Returns:
        DataFrame（每行一个持仓）
    """
    rows = []
    for port in portfolios:
        for i, ticker in enumerate(port.tickers):
            rows.append({
                "date": port.date,
                "ticker": ticker,
                "weight": port.weights[i],
                "score": port.scores[i],
                "cash": port.cash,
                "n_holdings": port.n_holdings,
            })

    return pd.DataFrame(rows)


# =========================================================
# 公开 API 导出
# =========================================================
__all__ = [
    "PortfolioConfig",
    "Portfolio",
    "select_topk",
    "compute_weights_equal",
    "compute_weights_score",
    "compute_weights_vol_inv",
    "compute_weights",
    "build_portfolio",
    "build_portfolio_series",
    "portfolio_to_df",
]
