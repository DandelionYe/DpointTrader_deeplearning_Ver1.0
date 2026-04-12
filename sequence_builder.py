from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SequenceDatasetBundle:
    X_seq: np.ndarray
    y_seq: Optional[np.ndarray]
    meta_df: pd.DataFrame
    seq_len: int
    feature_names: List[str]


@dataclass
class PanelSequenceStore:
    feature_by_ticker: Dict[Hashable, np.ndarray]
    label_by_ticker: Optional[Dict[Hashable, np.ndarray]]
    window_keys: List[Tuple[Hashable, int]]
    meta_df: pd.DataFrame
    seq_len: int
    feature_names: List[str]


def build_panel_sequence_store(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    seq_len: int = 20,
) -> PanelSequenceStore:
    if date_col not in X.columns:
        raise ValueError(f"X must contain column: {date_col}")
    if ticker_col not in X.columns:
        raise ValueError(f"X must contain column: {ticker_col}")

    feature_cols = [c for c in X.columns if c not in [date_col, ticker_col]]
    if not feature_cols:
        raise ValueError("No feature columns found in X")

    X_ordered = X.sort_values([ticker_col, date_col]).copy()

    feature_by_ticker: Dict[Hashable, np.ndarray] = {}
    label_by_ticker: Optional[Dict[Hashable, np.ndarray]] = {} if y is not None else None
    window_keys: List[Tuple[Hashable, int]] = []
    meta_rows: List[dict] = []

    for ticker, group_df in X_ordered.groupby(ticker_col, sort=True):
        group_df = group_df.sort_values(date_col).copy()
        n_windows = len(group_df) - seq_len + 1
        if n_windows <= 0:
            logger.warning(
                "Ticker %s has only %d samples, less than seq_len=%d, skipping",
                ticker,
                len(group_df),
                seq_len,
            )
            continue

        feature_values = group_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        feature_by_ticker[ticker] = feature_values
        if y is not None and label_by_ticker is not None:
            label_by_ticker[ticker] = y.loc[group_df.index].to_numpy(dtype=np.float32, copy=False)

        source_indices = group_df.index.to_numpy()
        date_values = group_df[date_col].to_numpy()
        for start in range(n_windows):
            last_pos = start + seq_len - 1
            window_keys.append((ticker, start))
            meta_rows.append(
                {
                    "source_index": int(source_indices[last_pos]),
                    "date": date_values[last_pos],
                    "ticker": ticker,
                }
            )

    if not window_keys:
        raise ValueError(
            f"No sequences generated. Check if seq_len={seq_len} is too large for the data."
        )

    meta_df = pd.DataFrame(meta_rows, columns=["source_index", "date", "ticker"])
    logger.info(
        "Indexed panel sequences: %d samples, seq_len=%d, n_features=%d, n_tickers=%d",
        len(window_keys),
        seq_len,
        len(feature_cols),
        X[ticker_col].nunique(),
    )
    return PanelSequenceStore(
        feature_by_ticker=feature_by_ticker,
        label_by_ticker=label_by_ticker,
        window_keys=window_keys,
        meta_df=meta_df,
        seq_len=seq_len,
        feature_names=feature_cols,
    )


def build_panel_sequences(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    seq_len: int = 20,
) -> SequenceDatasetBundle:
    store = build_panel_sequence_store(
        X,
        y,
        date_col=date_col,
        ticker_col=ticker_col,
        seq_len=seq_len,
    )
    all_sequences: List[np.ndarray] = []
    all_labels: List[float] = []
    for _idx, (ticker, start) in enumerate(store.window_keys):
        feature_values = store.feature_by_ticker[ticker]
        all_sequences.append(feature_values[start : start + seq_len])
        if store.label_by_ticker is not None:
            all_labels.append(float(store.label_by_ticker[ticker][start + seq_len - 1]))

    X_seq = np.asarray(all_sequences, dtype=np.float32)
    y_seq = np.asarray(all_labels, dtype=np.float32) if y is not None else None
    meta_df = store.meta_df.copy()

    logger.info(
        "Built panel sequences: %d samples, seq_len=%d, n_features=%d, n_tickers=%d",
        len(X_seq),
        seq_len,
        len(store.feature_names),
        X[ticker_col].nunique(),
    )

    return SequenceDatasetBundle(
        X_seq=X_seq,
        y_seq=y_seq,
        meta_df=meta_df,
        seq_len=seq_len,
        feature_names=store.feature_names,
    )
