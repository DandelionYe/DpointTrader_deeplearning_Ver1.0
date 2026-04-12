# feature_groups.py
"""
特征组模块
==========

本模块提供多股票 panel 数据的时序特征计算功能。

主要功能:
    - 动量特征
    - 波动率特征
    - K 线特征
    - 量价特征
    - TA 技术指标

与单股票特征的区别:
    - 所有 rolling 操作都在 groupby("ticker") 内进行
    - 避免跨 ticker 数据泄露

使用示例:
    >>> from feature_groups import add_all_features
    >>> feature_df = add_all_features(panel_df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureGroupMeta:
    """特征组元数据。

    Attributes:
        feature_names: 特征名称列表
        n_features: 特征数量
        notes: 注释
    """

    feature_names: List[str] = field(default_factory=list)
    n_features: int = 0
    notes: List[str] = field(default_factory=list)


def _apply_per_ticker(df: pd.DataFrame, ticker_col: str, func) -> pd.DataFrame:
    pieces = []
    for _, group in df.groupby(ticker_col, sort=False):
        pieces.append(func(group))
    if not pieces:
        return pd.DataFrame(index=df.index)
    return pd.concat(pieces).sort_index()


# =========================================================
# 基础工具函数
# =========================================================


def _safe_log1p(x: pd.Series) -> pd.Series:
    """对序列做 log1p 变换"""
    return np.log1p(np.clip(x.astype(float), 0.0, None))


def _rolling_mad(x: pd.Series, window: int) -> pd.Series:
    """滚动中位数绝对偏差"""
    med = x.rolling(window, min_periods=window).median()
    mad = (x - med).abs().rolling(window, min_periods=window).median()
    return mad


def _rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    """滚动 Z-score 标准化"""
    mu = x.rolling(window, min_periods=window).mean()
    sd = x.rolling(window, min_periods=window).std()
    return (x - mu) / sd.replace(0, np.nan)


# =========================================================
# 动量特征
# =========================================================


def add_momentum_features(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close_qfq",
    windows: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, FeatureGroupMeta]:
    """
    添加动量特征。

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        close_col: 收盘价列名
        windows: 回看窗口列表

    Returns:
        Tuple[panel_df, meta]
    """
    if windows is None:
        windows = [5, 10, 20, 60]

    df = panel_df.copy()
    feature_names = []

    def calc_momentum(group):
        close = group[close_col]
        result = pd.DataFrame(index=group.index)

        for w in windows:
            # 收益率动量
            ret = close.pct_change(w)
            result[f"mom_{w}d"] = ret
            feature_names.append(f"mom_{w}d")

            # 标准化动量
            result[f"mom_{w}d_z"] = _rolling_zscore(ret, w * 2)
            feature_names.append(f"mom_{w}d_z")

        return result

    momentum_features = _apply_per_ticker(df, ticker_col, calc_momentum)

    for col in momentum_features.columns:
        df[col] = momentum_features[col]

    meta = FeatureGroupMeta(
        feature_names=feature_names,
        n_features=len(feature_names),
    )

    return df, meta


# =========================================================
# 波动率特征
# =========================================================


def add_volatility_features(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close_qfq",
    high_col: str = "high_qfq",
    low_col: str = "low_qfq",
    windows: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, FeatureGroupMeta]:
    """
    添加波动率特征。

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        close_col: 收盘价列名
        high_col: 最高价列名
        low_col: 最低价列名
        windows: 窗口列表

    Returns:
        Tuple[panel_df, meta]
    """
    if windows is None:
        windows = [5, 10, 20, 60]

    df = panel_df.copy()
    feature_names = []

    def calc_volatility(group):
        close = group[close_col]
        high = group[high_col]
        low = group[low_col]

        result = pd.DataFrame(index=group.index)
        ret = close.pct_change()

        for w in windows:
            # 收益率标准差
            vol = ret.rolling(w, min_periods=w).std()
            result[f"vol_{w}d"] = vol
            feature_names.append(f"vol_{w}d")

            # 对数收益率标准差
            log_ret = np.log(close / close.shift(1))
            log_vol = log_ret.rolling(w, min_periods=w).std()
            result[f"log_vol_{w}d"] = log_vol
            feature_names.append(f"log_vol_{w}d")

            # 振幅
            amplitude = (high - low) / close.shift(1)
            result[f"amp_{w}d"] = amplitude.rolling(w, min_periods=w).mean()
            feature_names.append(f"amp_{w}d")

        return result

    vol_features = _apply_per_ticker(df, ticker_col, calc_volatility)

    for col in vol_features.columns:
        df[col] = vol_features[col]

    meta = FeatureGroupMeta(
        feature_names=feature_names,
        n_features=len(feature_names),
    )

    return df, meta


# =========================================================
# K 线特征
# =========================================================


def add_candlestick_features(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    open_col: str = "open_qfq",
    high_col: str = "high_qfq",
    low_col: str = "low_qfq",
    close_col: str = "close_qfq",
) -> Tuple[pd.DataFrame, FeatureGroupMeta]:
    """
    添加 K 线特征。

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        open_col: 开盘价列名
        high_col: 最高价列名
        low_col: 最低价列名
        close_col: 收盘价列名

    Returns:
        Tuple[panel_df, meta]
    """
    df = panel_df.copy()
    feature_names = []

    def calc_candlestick(group):
        o = group[open_col]
        h = group[high_col]
        low = group[low_col]
        c = group[close_col]

        result = pd.DataFrame(index=group.index)

        # 实体大小
        body = abs(c - o) / o
        result["body"] = body
        feature_names.append("body")

        # 上影线
        upper_shadow = (h - pd.concat([o, c], axis=1).max(axis=1)) / o
        result["upper_shadow"] = upper_shadow
        feature_names.append("upper_shadow")

        # 下影线
        lower_shadow = (pd.concat([o, c], axis=1).min(axis=1) - low) / o
        result["lower_shadow"] = lower_shadow
        feature_names.append("lower_shadow")

        # 涨跌方向
        result["direction"] = (c > o).astype(int)
        feature_names.append("direction")

        # 涨跌幅
        result["pct_change"] = c.pct_change()
        feature_names.append("pct_change")

        # 跳空
        result["gap"] = o / c.shift(1) - 1
        feature_names.append("gap")

        return result

    cs_features = _apply_per_ticker(df, ticker_col, calc_candlestick)

    for col in cs_features.columns:
        df[col] = cs_features[col]

    meta = FeatureGroupMeta(
        feature_names=feature_names,
        n_features=len(feature_names),
    )

    return df, meta


# =========================================================
# 量价特征
# =========================================================


def add_volume_price_features(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close_qfq",
    volume_col: str = "volume",
    amount_col: Optional[str] = None,
    windows: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, FeatureGroupMeta]:
    """
    添加量价特征。

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        close_col: 收盘价列名
        volume_col: 成交量列名
        amount_col: 成交额列名（可选）
        windows: 窗口列表

    Returns:
        Tuple[panel_df, meta]
    """
    if windows is None:
        windows = [5, 10, 20]

    df = panel_df.copy()
    feature_names = []
    notes = []

    def calc_volume_price(group):
        close = group[close_col]
        volume = group[volume_col]

        result = pd.DataFrame(index=group.index)

        for w in windows:
            # 量比
            vol_ma = volume.rolling(w, min_periods=w).mean()
            result[f"vol_ratio_{w}d"] = volume / vol_ma
            feature_names.append(f"vol_ratio_{w}d")

            # 成交量标准化
            result[f"vol_z_{w}d"] = _rolling_zscore(volume, w)
            feature_names.append(f"vol_z_{w}d")

            # 价量相关性
            ret = close.pct_change()
            vol_ret = volume.pct_change().fillna(0)
            corr = ret.rolling(w, min_periods=w).corr(vol_ret)
            result[f"vol_price_corr_{w}d"] = corr
            feature_names.append(f"vol_price_corr_{w}d")

        return result

    vp_features = _apply_per_ticker(df, ticker_col, calc_volume_price)

    for col in vp_features.columns:
        df[col] = vp_features[col]

    # 如果有成交额，添加额外特征
    if amount_col and amount_col in df.columns:

        def calc_amount(group):
            amt = group[amount_col]
            result = pd.DataFrame(index=group.index)

            for w in windows:
                amt_ma = amt.rolling(w, min_periods=w).mean()
                result[f"amt_ratio_{w}d"] = amt / amt_ma
                feature_names.append(f"amt_ratio_{w}d")

            return result

        amt_features = _apply_per_ticker(df, ticker_col, calc_amount)
        for col in amt_features.columns:
            df[col] = amt_features[col]
    else:
        notes.append("amount column not found, skipping amount features")

    meta = FeatureGroupMeta(
        feature_names=feature_names,
        n_features=len(feature_names),
        notes=notes,
    )

    return df, meta


# =========================================================
# TA 技术指标
# =========================================================


def add_ta_indicators(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close_qfq",
    high_col: str = "high_qfq",
    low_col: str = "low_qfq",
    volume_col: str = "volume",
    rsi_windows: Optional[List[int]] = None,
    bb_windows: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, FeatureGroupMeta]:
    """
    添加 TA 技术指标。

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        close_col: 收盘价列名
        high_col: 最高价列名
        low_col: 最低价列名
        volume_col: 成交量列名
        rsi_windows: RSI 窗口列表
        bb_windows: 布林带窗口列表

    Returns:
        Tuple[panel_df, meta]
    """
    if rsi_windows is None:
        rsi_windows = [6, 14, 20]
    if bb_windows is None:
        bb_windows = [20]

    df = panel_df.copy()
    feature_names = []

    def calc_rsi(group):
        close = group[close_col]
        result = pd.DataFrame(index=group.index)

        for w in rsi_windows:
            delta = close.diff(1)
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)

            avg_gain = gain.ewm(com=w - 1, min_periods=w, adjust=False).mean()
            avg_loss = loss.ewm(com=w - 1, min_periods=w, adjust=False).mean()

            with np.errstate(invalid="ignore", divide="ignore"):
                rs = avg_gain / avg_loss.where(avg_loss != 0, other=np.nan)

            rsi_raw = 100.0 - 100.0 / (1.0 + rs)
            rsi_filled = np.where(
                avg_loss == 0,
                np.where(avg_gain == 0, 50.0, 100.0),
                np.where(avg_gain == 0, 0.0, rsi_raw),
            )

            result[f"rsi_{w}d"] = pd.Series(rsi_filled / 100.0, index=close.index)
            feature_names.append(f"rsi_{w}d")

        return result

    def calc_macd(group):
        close = group[close_col]
        result = pd.DataFrame(index=group.index)

        ema_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
        macd_hist = macd_line - signal_line

        norm_window = 35
        result["macd_line_z"] = _rolling_zscore(macd_line, norm_window)
        feature_names.append("macd_line_z")
        result["macd_hist_z"] = _rolling_zscore(macd_hist, norm_window)
        feature_names.append("macd_hist_z")

        return result

    def calc_bband(group):
        close = group[close_col]
        result = pd.DataFrame(index=group.index)

        for w in bb_windows:
            sma = close.rolling(w, min_periods=w).mean()
            std = close.rolling(w, min_periods=w).std()
            bband_width = 2 * 2 * std / sma  # n_std=2

            result[f"bband_width_{w}d"] = bband_width
            feature_names.append(f"bband_width_{w}d")

            # 价格位置
            price_pos = (close - (sma - 2 * std)) / (4 * std)
            result[f"bband_pos_{w}d"] = price_pos.clip(0, 1)
            feature_names.append(f"bband_pos_{w}d")

        return result

    def calc_obv(group):
        close = group[close_col]
        volume = group[volume_col]
        result = pd.DataFrame(index=group.index)

        direction = np.sign(close.diff(1))
        obv = (volume * direction).cumsum()
        result["obv_z"] = _rolling_zscore(obv, 20)
        feature_names.append("obv_z")

        return result

    # 计算所有 TA 指标
    rsi_features = _apply_per_ticker(df, ticker_col, calc_rsi)
    macd_features = _apply_per_ticker(df, ticker_col, calc_macd)
    bband_features = _apply_per_ticker(df, ticker_col, calc_bband)
    obv_features = _apply_per_ticker(df, ticker_col, calc_obv)

    for features in [rsi_features, macd_features, bband_features, obv_features]:
        for col in features.columns:
            df[col] = features[col]

    meta = FeatureGroupMeta(
        feature_names=feature_names,
        n_features=len(feature_names),
    )

    return df, meta


# =========================================================
# 统一入口
# =========================================================


def add_all_features(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    include_groups: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, Dict[str, FeatureGroupMeta]]:
    """
    添加所有特征。

    Args:
        panel_df: panel DataFrame
        date_col: 日期列名
        ticker_col: ticker 列名
        include_groups: 要包含的特征组
        **kwargs: 传递给各特征组的参数

    Returns:
        Tuple[panel_df, metas]
    """
    if include_groups is None:
        include_groups = ["momentum", "volatility", "candlestick", "volume_price", "ta"]

    df = panel_df.copy()
    metas = {}

    if "momentum" in include_groups:
        df, meta = add_momentum_features(df, date_col=date_col, ticker_col=ticker_col, **kwargs)
        metas["momentum"] = meta

    if "volatility" in include_groups:
        df, meta = add_volatility_features(df, date_col=date_col, ticker_col=ticker_col, **kwargs)
        metas["volatility"] = meta

    if "candlestick" in include_groups:
        df, meta = add_candlestick_features(df, date_col=date_col, ticker_col=ticker_col, **kwargs)
        metas["candlestick"] = meta

    if "volume_price" in include_groups:
        df, meta = add_volume_price_features(df, date_col=date_col, ticker_col=ticker_col, **kwargs)
        metas["volume_price"] = meta

    if "ta" in include_groups:
        df, meta = add_ta_indicators(df, date_col=date_col, ticker_col=ticker_col, **kwargs)
        metas["ta"] = meta

    return df, metas


# =========================================================
# 公开 API 导出
# =========================================================
__all__ = [
    "FeatureGroupMeta",
    "add_momentum_features",
    "add_volatility_features",
    "add_candlestick_features",
    "add_volume_price_features",
    "add_ta_indicators",
    "add_all_features",
]
