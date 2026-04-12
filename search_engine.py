from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from panel_trainer import train_with_nested_walkforward, train_with_walkforward

logger = logging.getLogger(__name__)
LOWER_IS_BETTER_METRICS = {"logloss", "mse", "rmse", "mae"}


@dataclass
class SearchCandidateResult:
    config: Dict[str, Any]
    seed: int
    metrics: Dict[str, float]
    notes: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    best_config: Dict[str, Any]
    best_seed: int
    best_metrics: Dict[str, float]
    candidates: List[SearchCandidateResult] = field(default_factory=list)
    best_oof_scores: Optional[pd.DataFrame] = None
    best_model: Optional[Any] = None


def score_candidate(metrics: Dict[str, float], selection_metric: str) -> float:
    if selection_metric not in metrics:
        raise ValueError(f"Unknown selection_metric: {selection_metric}")
    value = float(metrics[selection_metric])
    return -value if selection_metric in LOWER_IS_BETTER_METRICS else value


def run_search(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    args: argparse.Namespace,
    split_mode: str,
    indexed_splits: Any,
    base_config: Dict[str, Any],
    date_col: str = "date",
    ticker_col: str = "ticker",
    search_runs: Optional[int] = None,
) -> SearchResult:
    from search_space import sample_model_config

    total_runs = int(search_runs if search_runs is not None else args.runs)
    selection_metric = str(args.selection_metric)
    base_seed = int(args.seed)

    candidates: List[SearchCandidateResult] = []
    best_score = float("-inf")
    best_config: Optional[Dict[str, Any]] = None
    best_seed: Optional[int] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_oof_scores: Optional[pd.DataFrame] = None
    best_model: Optional[Any] = None

    rng = np.random.RandomState(base_seed)

    for run_idx in range(total_runs):
        candidate_config = sample_model_config(
            model_type=base_config.get("model_type", "mlp"),
            rng=rng,
            base_config=base_config,
        )
        candidate_seed = base_seed + run_idx
        logger.info(
            "Search run %d/%d: model_type=%s seed=%d",
            run_idx + 1,
            total_runs,
            candidate_config.get("model_type"),
            candidate_seed,
        )

        try:
            if split_mode == "nested_wf":
                train_result = train_with_nested_walkforward(
                    X,
                    y,
                    candidate_config,
                    indexed_splits,
                    date_col=date_col,
                    ticker_col=ticker_col,
                    seed=candidate_seed,
                )
            else:
                train_result = train_with_walkforward(
                    X,
                    y,
                    candidate_config,
                    indexed_splits,
                    date_col=date_col,
                    ticker_col=ticker_col,
                    seed=candidate_seed,
                )
            candidate_metrics = dict(train_result.val_metrics)
            candidate_notes = list(train_result.notes)
            candidate_oof = (
                train_result.oof_scores.copy()
                if train_result.oof_scores is not None
                else pd.DataFrame()
            )
            candidate_model = train_result.model
        except Exception as exc:
            logger.warning("Search run %d failed: %s", run_idx + 1, exc)
            candidate_metrics = {
                "rank_ic_mean": float("-inf"),
                "rank_ic_std": float("inf"),
                "ic_mean": float("-inf"),
                "topk_return_mean": float("-inf"),
            }
            candidate_notes = [f"Failed: {exc}"]
            candidate_oof = pd.DataFrame()
            candidate_model = None

        candidates.append(
            SearchCandidateResult(
                config=candidate_config,
                seed=candidate_seed,
                metrics=candidate_metrics,
                notes=candidate_notes,
            )
        )

        candidate_score = score_candidate(candidate_metrics, selection_metric)
        if candidate_score > best_score:
            best_score = candidate_score
            best_config = candidate_config
            best_seed = candidate_seed
            best_metrics = candidate_metrics
            best_oof_scores = candidate_oof
            best_model = candidate_model

    if best_config is None or best_seed is None or best_metrics is None:
        raise RuntimeError("Search produced no valid candidates")

    return SearchResult(
        best_config=best_config,
        best_seed=best_seed,
        best_metrics=best_metrics,
        candidates=candidates,
        best_oof_scores=best_oof_scores,
        best_model=best_model,
    )
