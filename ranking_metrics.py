from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _has_variation(values: pd.Series) -> bool:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return False
    return not np.allclose(arr, arr[0], equal_nan=True)


def _pearson_corr(x: pd.Series, y: pd.Series) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 2:
        return float("nan")
    if not np.isfinite(x_arr).all() or not np.isfinite(y_arr).all():
        return float("nan")
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return float("nan")
    x_centered = x_arr - x_arr.mean()
    y_centered = y_arr - y_arr.mean()
    denom = np.sqrt((x_centered * x_centered).sum() * (y_centered * y_centered).sum())
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float((x_centered * y_centered).sum() / denom)


def _spearman_corr(x: pd.Series, y: pd.Series) -> float:
    x_rank = pd.Series(x).rank(method="average")
    y_rank = pd.Series(y).rank(method="average")
    return _pearson_corr(x_rank, y_rank)


@dataclass
class RankingMetrics:
    ic_mean: Optional[float] = None
    ic_std: Optional[float] = None
    ic_ir: Optional[float] = None
    rank_ic_mean: Optional[float] = None
    rank_ic_std: Optional[float] = None
    rank_ic_ir: Optional[float] = None
    topk_return_mean: Optional[float] = None
    topk_return_annual: Optional[float] = None
    layered_returns: Optional[Dict[str, float]] = None
    notes: List[str] = field(default_factory=list)


def compute_ic(
    scores_df: pd.DataFrame,
    *,
    score_col: str = "score",
    label_col: str = "label",
    date_col: str = "date",
    ticker_col: str = "ticker",
    method: str = "pearson",
) -> pd.Series:
    del ticker_col

    def calc_ic(group: pd.DataFrame) -> float:
        scores = group[score_col]
        labels = group[label_col]
        mask = scores.notna() & labels.notna()
        if mask.sum() < 3:
            return float("nan")

        filtered_scores = scores[mask]
        filtered_labels = labels[mask]
        if not _has_variation(filtered_scores) or not _has_variation(filtered_labels):
            return float("nan")

        if method == "pearson":
            return _pearson_corr(filtered_scores, filtered_labels)
        if method == "spearman":
            return _spearman_corr(filtered_scores, filtered_labels)
        raise ValueError(f"Unknown method: {method}")

    rows = []
    for current_date, group in scores_df.groupby(date_col, sort=True):
        rows.append((current_date, calc_ic(group)))
    return pd.Series(
        [value for _, value in rows],
        index=[current_date for current_date, _ in rows],
        dtype=float,
    )


def compute_rank_ic(
    scores_df: pd.DataFrame,
    *,
    score_col: str = "score",
    label_col: str = "label",
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.Series:
    del ticker_col

    def calc_rank_ic(group: pd.DataFrame) -> float:
        scores = group[score_col]
        labels = group[label_col]
        mask = scores.notna() & labels.notna()
        if mask.sum() < 3:
            return float("nan")

        score_rank = scores[mask].rank(method="average")
        label_rank = labels[mask].rank(method="average")
        if not _has_variation(score_rank) or not _has_variation(label_rank):
            return float("nan")
        return _pearson_corr(score_rank, label_rank)

    rows = []
    for current_date, group in scores_df.groupby(date_col, sort=True):
        rows.append((current_date, calc_rank_ic(group)))
    return pd.Series(
        [value for _, value in rows],
        index=[current_date for current_date, _ in rows],
        dtype=float,
    )


def compute_ic_summary(
    ic_series: pd.Series,
    *,
    window: Optional[int] = None,
) -> Dict[str, float]:
    ic_valid = ic_series.dropna()
    if len(ic_valid) == 0:
        return {
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "ic_tstat": np.nan,
            "ic_pvalue": np.nan,
        }

    ic_mean = float(ic_valid.mean())
    ic_std = float(ic_valid.std())
    ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan
    if len(ic_valid) > 1 and ic_std > 0:
        t_stat = ic_mean / (ic_std / np.sqrt(len(ic_valid)))
    else:
        t_stat = np.nan

    summary = {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "ic_tstat": float(t_stat),
        "ic_pvalue": np.nan,
        "n_periods": len(ic_valid),
    }

    if window and len(ic_valid) >= window:
        rolling_ic = ic_valid.rolling(window).mean()
        summary["ic_last"] = float(ic_valid.iloc[-1])
        summary["ic_rolling_mean"] = float(rolling_ic.iloc[-1])

    return summary


def compute_topk_return(
    scores_df: pd.DataFrame,
    *,
    score_col: str = "score",
    label_col: str = "label",
    date_col: str = "date",
    ticker_col: str = "ticker",
    top_k: int = 5,
    weighting: str = "equal",
) -> pd.Series:
    del ticker_col

    def calc_topk_return(group: pd.DataFrame) -> float:
        if len(group) < top_k:
            return float("nan")

        sorted_group = group.nlargest(top_k, score_col)
        if weighting == "equal":
            weights = np.ones(top_k, dtype=float) / top_k
        elif weighting == "score":
            scores = sorted_group[score_col].to_numpy(dtype=float)
            scores_pos = scores - scores.min() + 1e-6
            weights = scores_pos / scores_pos.sum()
        else:
            raise ValueError(f"Unknown weighting: {weighting}")

        labels = sorted_group[label_col].to_numpy(dtype=float)
        return float(np.dot(weights, labels))

    rows = []
    for current_date, group in scores_df.groupby(date_col, sort=True):
        rows.append((current_date, calc_topk_return(group)))
    return pd.Series(
        [value for _, value in rows],
        index=[current_date for current_date, _ in rows],
        dtype=float,
    )


def compute_layered_returns(
    scores_df: pd.DataFrame,
    *,
    score_col: str = "score",
    label_col: str = "label",
    date_col: str = "date",
    ticker_col: str = "ticker",
    n_layers: int = 5,
) -> Dict[str, pd.Series]:
    del ticker_col

    def calc_layer_return(group: pd.DataFrame) -> pd.Series:
        group = group.copy()
        if len(group) < n_layers:
            return pd.Series({f"L{i}": np.nan for i in range(1, n_layers + 1)})

        quantiles = np.linspace(0, 1, n_layers + 1)
        try:
            group["layer"] = pd.qcut(
                group[score_col],
                q=quantiles,
                labels=list(range(1, n_layers + 1)),
                duplicates="drop",
            )
        except ValueError:
            group["layer"] = pd.cut(
                group[score_col].rank(method="average"),
                bins=n_layers,
                labels=list(range(1, n_layers + 1)),
            )

        layer_returns = group.groupby("layer", observed=False)[label_col].mean()
        result = pd.Series({f"L{i}": np.nan for i in range(1, n_layers + 1)})
        for layer_idx, ret in layer_returns.items():
            result[f"L{layer_idx}"] = ret
        return result

    rows = []
    for current_date, group in scores_df.groupby(date_col, sort=True):
        layer_values = calc_layer_return(group)
        layer_values.name = current_date
        rows.append(layer_values)

    if rows:
        daily_layer_returns = pd.DataFrame(rows)
    else:
        daily_layer_returns = pd.DataFrame(columns=[f"L{i}" for i in range(1, n_layers + 1)])

    return {f"L{i}": daily_layer_returns[f"L{i}"] for i in range(1, n_layers + 1)}


def compute_long_short_return(
    scores_df: pd.DataFrame,
    *,
    score_col: str = "score",
    label_col: str = "label",
    date_col: str = "date",
    ticker_col: str = "ticker",
    n_layers: int = 5,
) -> pd.Series:
    layer_returns = compute_layered_returns(
        scores_df,
        score_col=score_col,
        label_col=label_col,
        date_col=date_col,
        ticker_col=ticker_col,
        n_layers=n_layers,
    )
    return layer_returns[f"L{n_layers}"] - layer_returns["L1"]


def compute_all_ranking_metrics(
    scores_df: pd.DataFrame,
    *,
    score_col: str = "score",
    label_col: str = "label",
    date_col: str = "date",
    ticker_col: str = "ticker",
    top_k: int = 5,
    n_layers: int = 5,
) -> RankingMetrics:
    notes: List[str] = []

    ic_series = compute_ic(
        scores_df,
        score_col=score_col,
        label_col=label_col,
        date_col=date_col,
        ticker_col=ticker_col,
    )
    ic_summary = compute_ic_summary(ic_series)

    rank_ic_series = compute_rank_ic(
        scores_df,
        score_col=score_col,
        label_col=label_col,
        date_col=date_col,
        ticker_col=ticker_col,
    )
    rank_ic_summary = compute_ic_summary(rank_ic_series)

    topk_returns = compute_topk_return(
        scores_df,
        score_col=score_col,
        label_col=label_col,
        date_col=date_col,
        ticker_col=ticker_col,
        top_k=top_k,
    )
    topk_mean = float(topk_returns.dropna().mean())
    topk_annual = topk_mean * 252

    layer_returns = compute_layered_returns(
        scores_df,
        score_col=score_col,
        label_col=label_col,
        date_col=date_col,
        ticker_col=ticker_col,
        n_layers=n_layers,
    )
    layered_summary = {k: float(v.dropna().mean()) for k, v in layer_returns.items()}

    ls_returns = compute_long_short_return(
        scores_df,
        score_col=score_col,
        label_col=label_col,
        date_col=date_col,
        ticker_col=ticker_col,
        n_layers=n_layers,
    )
    ls_mean = float(ls_returns.dropna().mean())
    ls_annual = ls_mean * 252

    if np.isnan(ic_summary.get("ic_mean", np.nan)):
        notes.append("IC is NaN for the available evaluation periods.")
    if np.isnan(rank_ic_summary.get("ic_mean", np.nan)):
        notes.append("RankIC is NaN for the available evaluation periods.")
    if np.isnan(ls_mean):
        notes.append("Long-short return is NaN for the available evaluation periods.")
    else:
        notes.append(f"Long-short annualized return: {ls_annual:.2%}")

    return RankingMetrics(
        ic_mean=ic_summary.get("ic_mean"),
        ic_std=ic_summary.get("ic_std"),
        ic_ir=ic_summary.get("ic_ir"),
        rank_ic_mean=rank_ic_summary.get("ic_mean"),
        rank_ic_std=rank_ic_summary.get("ic_std"),
        rank_ic_ir=rank_ic_summary.get("ic_ir"),
        topk_return_mean=topk_mean,
        topk_return_annual=topk_annual,
        layered_returns=layered_summary,
        notes=notes,
    )


__all__ = [
    "RankingMetrics",
    "compute_ic",
    "compute_rank_ic",
    "compute_ic_summary",
    "compute_topk_return",
    "compute_layered_returns",
    "compute_long_short_return",
    "compute_all_ranking_metrics",
]
