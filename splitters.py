from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitSpec:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    embargo_days: int = 0
    fold_id: int = 0

    def to_date_sets(self, unique_dates: List[pd.Timestamp]) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
        train_dates = [d for d in unique_dates if self.train_start <= d <= self.train_end]
        val_dates = [d for d in unique_dates if self.val_start <= d <= self.val_end]
        return train_dates, val_dates


@dataclass
class SplitResult:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    spec: SplitSpec


def _unique_dates(panel_df: pd.DataFrame, date_col: str) -> List[pd.Timestamp]:
    return sorted(pd.to_datetime(panel_df[date_col].unique()))


def _rows_for_dates(panel_df: pd.DataFrame, dates: List[pd.Timestamp], *, date_col: str) -> int:
    if not dates:
        return 0
    return int(panel_df[panel_df[date_col].isin(dates)].shape[0])


def _tickers_for_dates(panel_df: pd.DataFrame, dates: List[pd.Timestamp], *, date_col: str, ticker_col: str) -> int:
    if not dates:
        return 0
    return int(panel_df.loc[panel_df[date_col].isin(dates), ticker_col].nunique())


def _log_split_summary(
    name: str,
    panel_df: pd.DataFrame,
    splits: List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]],
    *,
    date_col: str,
    ticker_col: str,
    embargo_days: int = 0,
) -> None:
    if not splits:
        logger.warning("%s: no valid folds generated from %d dates", name, len(_unique_dates(panel_df, date_col)))
        return
    logger.info(
        "%s: %d folds generated from %d dates (embargo=%d)",
        name,
        len(splits),
        len(_unique_dates(panel_df, date_col)),
        embargo_days,
    )
    for idx, (train_dates, val_dates) in enumerate(splits, start=1):
        logger.info(
            "  Fold %d: train=%d dates/%d rows/%d tickers, val=%d dates/%d rows/%d tickers",
            idx,
            len(train_dates),
            _rows_for_dates(panel_df, train_dates, date_col=date_col),
            _tickers_for_dates(panel_df, train_dates, date_col=date_col, ticker_col=ticker_col),
            len(val_dates),
            _rows_for_dates(panel_df, val_dates, date_col=date_col),
            _tickers_for_dates(panel_df, val_dates, date_col=date_col, ticker_col=ticker_col),
        )


def walkforward_splits_by_date(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    n_folds: int = 4,
    train_start_ratio: float = 0.5,
    min_rows: int = 50,
) -> List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]]:
    unique_dates = _unique_dates(panel_df, date_col)
    n_dates = len(unique_dates)
    if n_dates < 2:
        logger.warning("walkforward_splits_by_date: not enough unique dates")
        return []

    cuts = [
        train_start_ratio + (1.0 - train_start_ratio) * i / n_folds
        for i in range(n_folds + 1)
    ]

    splits: List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]] = []
    for fold_idx in range(len(cuts) - 1):
        train_end_idx = int(n_dates * cuts[fold_idx])
        val_end_idx = int(n_dates * cuts[fold_idx + 1])

        train_dates = unique_dates[:train_end_idx]
        val_dates = unique_dates[train_end_idx:val_end_idx]
        train_rows = _rows_for_dates(panel_df, train_dates, date_col=date_col)
        val_rows = _rows_for_dates(panel_df, val_dates, date_col=date_col)

        if train_rows < min_rows or val_rows < min_rows:
            logger.warning(
                "walkforward_splits_by_date: fold %d skipped (train_rows=%d, val_rows=%d, min_rows=%d)",
                fold_idx + 1,
                train_rows,
                val_rows,
                min_rows,
            )
            continue
        splits.append((train_dates, val_dates))

    _log_split_summary(
        "walkforward_splits_by_date",
        panel_df,
        splits,
        date_col=date_col,
        ticker_col=ticker_col,
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
    unique_dates = _unique_dates(panel_df, date_col)
    n_dates = len(unique_dates)
    if n_dates < embargo_days + 2:
        logger.warning("walkforward_splits_with_embargo: not enough dates for embargo=%d", embargo_days)
        return []

    cuts = [
        train_start_ratio + (1.0 - train_start_ratio) * i / n_folds
        for i in range(n_folds + 1)
    ]

    splits: List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]] = []
    for fold_idx in range(len(cuts) - 1):
        train_end_idx = int(n_dates * cuts[fold_idx])
        val_end_idx = int(n_dates * cuts[fold_idx + 1])
        val_start_idx = train_end_idx + embargo_days
        if val_start_idx >= val_end_idx:
            logger.warning(
                "walkforward_splits_with_embargo: fold %d skipped due to embargo_days=%d",
                fold_idx + 1,
                embargo_days,
            )
            continue

        train_dates = unique_dates[:train_end_idx]
        val_dates = unique_dates[val_start_idx:val_end_idx]
        train_rows = _rows_for_dates(panel_df, train_dates, date_col=date_col)
        val_rows = _rows_for_dates(panel_df, val_dates, date_col=date_col)

        if train_rows < min_rows or val_rows < min_rows:
            logger.warning(
                "walkforward_splits_with_embargo: fold %d skipped (train_rows=%d, val_rows=%d, min_rows=%d)",
                fold_idx + 1,
                train_rows,
                val_rows,
                min_rows,
            )
            continue
        splits.append((train_dates, val_dates))

    _log_split_summary(
        "walkforward_splits_with_embargo",
        panel_df,
        splits,
        date_col=date_col,
        ticker_col=ticker_col,
        embargo_days=embargo_days,
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
    inner_use_embargo: bool = True,
) -> List[Tuple[List[pd.Timestamp], List[pd.Timestamp], List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]]]]:
    unique_dates = _unique_dates(panel_df, date_col)
    n_dates = len(unique_dates)
    cuts = [
        train_start_ratio + (1.0 - train_start_ratio) * i / n_outer_folds
        for i in range(n_outer_folds + 1)
    ]

    splits: List[Tuple[List[pd.Timestamp], List[pd.Timestamp], List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]]]] = []
    for fold_idx in range(len(cuts) - 1):
        outer_train_end_idx = int(n_dates * cuts[fold_idx])
        outer_val_end_idx = int(n_dates * cuts[fold_idx + 1])
        outer_val_start_idx = outer_train_end_idx + embargo_days
        if outer_val_start_idx >= outer_val_end_idx:
            logger.warning(
                "nested_walkforward: fold %d skipped due to embargo_days=%d",
                fold_idx + 1,
                embargo_days,
            )
            continue

        outer_train_dates = unique_dates[:outer_train_end_idx]
        outer_val_dates = unique_dates[outer_val_start_idx:outer_val_end_idx]
        outer_train_rows = _rows_for_dates(panel_df, outer_train_dates, date_col=date_col)
        outer_val_rows = _rows_for_dates(panel_df, outer_val_dates, date_col=date_col)
        if outer_train_rows < min_rows or outer_val_rows < min_rows:
            logger.warning(
                "nested_walkforward: fold %d skipped (train_rows=%d, val_rows=%d, min_rows=%d)",
                fold_idx + 1,
                outer_train_rows,
                outer_val_rows,
                min_rows,
            )
            continue

        inner_panel = panel_df[panel_df[date_col].isin(outer_train_dates)].copy()
        if inner_use_embargo:
            inner_splits = walkforward_splits_with_embargo(
                inner_panel,
                date_col=date_col,
                ticker_col=ticker_col,
                n_folds=n_inner_folds,
                train_start_ratio=train_start_ratio,
                min_rows=min_rows,
                embargo_days=embargo_days,
            )
        else:
            inner_splits = walkforward_splits_by_date(
                inner_panel,
                date_col=date_col,
                ticker_col=ticker_col,
                n_folds=n_inner_folds,
                train_start_ratio=train_start_ratio,
                min_rows=min_rows,
            )
        if not inner_splits:
            logger.warning("nested_walkforward: fold %d skipped (no valid inner splits)", fold_idx + 1)
            continue
        splits.append((outer_train_dates, outer_val_dates, inner_splits))

    if not splits:
        logger.warning("nested_walkforward: no valid outer folds generated from %d dates", n_dates)
        return []

    logger.info(
        "nested_walkforward: %d outer folds generated from %d dates (outer_embargo=%d, inner_embargo=%s)",
        len(splits),
        n_dates,
        embargo_days,
        "on" if inner_use_embargo else "off",
    )
    for idx, (outer_train_dates, outer_val_dates, inner_splits) in enumerate(splits, start=1):
        logger.info(
            "  Outer Fold %d: train=%d dates/%d rows/%d tickers, val=%d dates/%d rows/%d tickers, inner_splits=%d",
            idx,
            len(outer_train_dates),
            _rows_for_dates(panel_df, outer_train_dates, date_col=date_col),
            _tickers_for_dates(panel_df, outer_train_dates, date_col=date_col, ticker_col=ticker_col),
            len(outer_val_dates),
            _rows_for_dates(panel_df, outer_val_dates, date_col=date_col),
            _tickers_for_dates(panel_df, outer_val_dates, date_col=date_col, ticker_col=ticker_col),
            len(inner_splits),
        )
    return splits


def final_holdout_split_by_date(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    holdout_ratio: float = 0.15,
    min_holdout_rows: int = 60,
    enforce_non_empty_search: bool = True,
    gap_days: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = _unique_dates(panel_df, date_col)
    n_dates = len(unique_dates)
    holdout_size = int(n_dates * holdout_ratio)
    holdout_start_idx = n_dates - holdout_size
    search_end_idx = max(0, holdout_start_idx - max(0, gap_days))

    holdout_dates = unique_dates[holdout_start_idx:]
    search_dates = unique_dates[:search_end_idx]
    holdout_df = panel_df[panel_df[date_col].isin(holdout_dates)].copy()
    search_df = panel_df[panel_df[date_col].isin(search_dates)].copy()

    if len(holdout_df) < min_holdout_rows:
        raise ValueError(
            f"holdout_rows={len(holdout_df)} < min_holdout_rows={min_holdout_rows}. "
            "Increase holdout_ratio or use more data."
        )
    if enforce_non_empty_search and search_df.empty:
        raise ValueError(
            "Search set is empty after holdout split. Reduce holdout_ratio/gap_days or use more data."
        )

    logger.info(
        "Final Holdout Split: search=%d rows (%d dates), gap=%d dates, holdout=%d rows (%d dates, %.1f%%)",
        len(search_df),
        len(search_dates),
        max(0, holdout_start_idx - search_end_idx),
        len(holdout_df),
        len(holdout_dates),
        holdout_ratio * 100.0,
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
    val_pool = n_dates * (1.0 - train_start_ratio)
    best_n = min_folds
    for n in range(max_folds, min_folds - 1, -1):
        val_dates_per_fold = val_pool / n
        val_rows_per_fold = val_dates_per_fold * n_tickers
        if val_rows_per_fold < min_rows:
            continue
        expected_trades = val_rows_per_fold * assumed_trade_freq
        if expected_trades < target_trades_per_fold:
            continue
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
    train_df = panel_df[panel_df[date_col].isin(train_dates)].copy()
    val_df = panel_df[panel_df[date_col].isin(val_dates)].copy()
    return train_df, val_df


def build_date_splits(
    panel_df: pd.DataFrame,
    *,
    split_mode: str,
    date_col: str = "date",
    ticker_col: str = "ticker",
    n_folds: int = 4,
    n_outer_folds: int = 3,
    n_inner_folds: int = 2,
    train_start_ratio: float = 0.5,
    min_rows: int = 60,
    embargo_days: int = 5,
) -> List:
    if split_mode == "wf":
        return walkforward_splits_by_date(
            panel_df,
            date_col=date_col,
            ticker_col=ticker_col,
            n_folds=n_folds,
            train_start_ratio=train_start_ratio,
            min_rows=min_rows,
        )
    if split_mode == "wf_embargo":
        return walkforward_splits_with_embargo(
            panel_df,
            date_col=date_col,
            ticker_col=ticker_col,
            n_folds=n_folds,
            train_start_ratio=train_start_ratio,
            min_rows=min_rows,
            embargo_days=embargo_days,
        )
    if split_mode == "nested_wf":
        return nested_walkforward_splits_by_date(
            panel_df,
            date_col=date_col,
            ticker_col=ticker_col,
            n_outer_folds=n_outer_folds,
            n_inner_folds=n_inner_folds,
            train_start_ratio=train_start_ratio,
            min_rows=min_rows,
            embargo_days=embargo_days,
            inner_use_embargo=True,
        )
    raise ValueError(f"Invalid split_mode: {split_mode}. Must be one of ['wf', 'wf_embargo', 'nested_wf']")


__all__ = [
    "SplitSpec",
    "SplitResult",
    "walkforward_splits_by_date",
    "walkforward_splits_with_embargo",
    "nested_walkforward_splits_by_date",
    "final_holdout_split_by_date",
    "recommend_n_folds",
    "filter_panel_by_dates",
    "build_date_splits",
]
