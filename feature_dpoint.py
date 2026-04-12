from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

from labeler import build_labels
from tasks import LabelSpec, infer_task_type


@dataclass
class PanelFeatureMeta:
    feature_names: List[str] = field(default_factory=list)
    time_series_feature_names: List[str] = field(default_factory=list)
    cross_section_feature_names: List[str] = field(default_factory=list)
    optional_inputs_used: List[str] = field(default_factory=list)
    label_mode: str = "binary_next_close_up"
    basket_name: str = ""
    n_tickers: int = 0
    n_samples: int = 0
    params: Dict[str, object] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _coerce_int_list(value: object, default: List[int]) -> List[int]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return [int(item) for item in value]
    return list(default)


def build_features_and_labels_panel(
    panel_df: pd.DataFrame,
    config: Dict[str, object],
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    label_mode: str = "binary_next_close_up",
    include_cross_section: bool = True,
    feature_config: Optional[Dict] = None,
    label_horizon_days: int = 1,
    max_label_date: Optional[pd.Timestamp] = None,
    return_label_end_date: bool = False,
) -> Union[
    Tuple[pd.DataFrame, pd.Series, PanelFeatureMeta],
    Tuple[pd.DataFrame, pd.Series, PanelFeatureMeta, pd.DataFrame],
]:
    """
    Build panel features and labels for basket mode.

    `feature_config` is kept only for backward compatibility with existing call sites.
    """
    del feature_config

    from cross_sectional_features import add_cross_sectional_features
    from feature_groups import (
        add_candlestick_features,
        add_momentum_features,
        add_ta_indicators,
        add_volatility_features,
        add_volume_price_features,
    )

    df = panel_df.copy().sort_values([date_col, ticker_col]).reset_index(drop=True)

    windows = _coerce_int_list(config.get("windows", [5, 10, 20, 60]), [5, 10, 20, 60])
    label_horizon_days = max(1, int(label_horizon_days))
    use_momentum = bool(config.get("use_momentum", True))
    use_volatility = bool(config.get("use_volatility", True))
    use_volume = bool(config.get("use_volume", True))
    use_candle = bool(config.get("use_candle", True))
    use_ta = bool(config.get("use_ta_indicators", False))
    ta_windows = _coerce_int_list(config.get("ta_windows", [6, 14, 20]), [6, 14, 20])

    optional_inputs_used: List[str] = []
    notes: List[str] = []

    if "amount" in df.columns:
        optional_inputs_used.append("amount")
    else:
        notes.append("amount column not found, will use amount_proxy if needed")

    if "turnover_rate" in df.columns:
        optional_inputs_used.append("turnover_rate")
    else:
        notes.append("turnover_rate column not found, skipping turnover features")

    all_feature_names: List[str] = []
    ts_feature_names: List[str] = []
    cs_feature_names: List[str] = []

    df["_ret_1"] = df.groupby(ticker_col)["close_qfq"].pct_change(1)
    all_feature_names.append("_ret_1")
    ts_feature_names.append("_ret_1")

    if use_momentum:
        df, momentum_meta = add_momentum_features(
            df,
            date_col=date_col,
            ticker_col=ticker_col,
            windows=windows,
        )
        all_feature_names.extend(momentum_meta.feature_names)
        ts_feature_names.extend(momentum_meta.feature_names)

    if use_volatility:
        df, volatility_meta = add_volatility_features(
            df,
            date_col=date_col,
            ticker_col=ticker_col,
            windows=windows,
        )
        all_feature_names.extend(volatility_meta.feature_names)
        ts_feature_names.extend(volatility_meta.feature_names)

    if use_candle:
        df, candle_meta = add_candlestick_features(
            df,
            date_col=date_col,
            ticker_col=ticker_col,
        )
        all_feature_names.extend(candle_meta.feature_names)
        ts_feature_names.extend(candle_meta.feature_names)

    if use_volume:
        df, volume_meta = add_volume_price_features(
            df,
            date_col=date_col,
            ticker_col=ticker_col,
            windows=windows,
        )
        all_feature_names.extend(volume_meta.feature_names)
        ts_feature_names.extend(volume_meta.feature_names)

    if use_ta:
        df, ta_meta = add_ta_indicators(
            df,
            date_col=date_col,
            ticker_col=ticker_col,
            rsi_windows=ta_windows,
            bb_windows=ta_windows,
        )
        all_feature_names.extend(ta_meta.feature_names)
        ts_feature_names.extend(ta_meta.feature_names)

    if include_cross_section:
        cs_columns = ["close_qfq", "volume"]
        if "amount" in df.columns:
            cs_columns.append("amount")

        df, cs_meta = add_cross_sectional_features(
            df,
            date_col=date_col,
            ticker_col=ticker_col,
            columns=cs_columns,
            features=["rank", "zscore", "percentile"],
        )
        all_feature_names.extend(cs_meta.cross_sectional_features)
        cs_feature_names.extend(cs_meta.cross_sectional_features)

    feature_cols = [c for c in all_feature_names if c in df.columns]
    X = df[[date_col, ticker_col] + feature_cols].copy()

    label_spec = LabelSpec(
        task_type=infer_task_type(label_mode),
        label_mode=label_mode,
        horizon_days=label_horizon_days,
    )
    y, full_label_meta, _ = build_labels(
        df,
        label_spec,
        date_col=date_col,
        ticker_col=ticker_col,
        close_col="close_qfq",
    )
    y = y.rename("label")
    label_end_date = full_label_meta["label_end_date"]

    valid = X[feature_cols].notna().all(axis=1) & y.notna() & label_end_date.notna()
    if max_label_date is not None:
        valid &= label_end_date <= pd.Timestamp(max_label_date)
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()
    label_meta = full_label_meta.loc[valid, [date_col, ticker_col, "label_end_date"]].copy()

    meta = PanelFeatureMeta(
        feature_names=feature_cols,
        time_series_feature_names=ts_feature_names,
        cross_section_feature_names=cs_feature_names,
        optional_inputs_used=optional_inputs_used,
        label_mode=label_mode,
        basket_name=str(config.get("basket_name", "")),
        n_tickers=int(df[ticker_col].nunique()),
        n_samples=len(X),
        params={
            "windows": windows,
            "use_momentum": use_momentum,
            "use_volatility": use_volatility,
            "use_volume": use_volume,
            "use_candle": use_candle,
            "use_ta_indicators": use_ta,
            "ta_windows": ta_windows,
            "include_cross_section": include_cross_section,
            "label_horizon_days": label_horizon_days,
        },
        notes=notes,
    )

    if return_label_end_date:
        return X, y, meta, label_meta
    return X, y, meta


__all__ = [
    "PanelFeatureMeta",
    "build_features_and_labels_panel",
]
