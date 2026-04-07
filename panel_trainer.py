from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from models import (
    get_torch_runtime_info,
    is_torch_model_instance,
    is_torch_model_type,
    make_model,
    predict_dpoint,
    resolve_torch_device,
    train_pytorch_model,
)
from ranking_metrics import RankingMetrics, compute_all_ranking_metrics

logger = logging.getLogger(__name__)


@dataclass
class PanelTrainResult:
    model: Any
    feature_names: List[str] = field(default_factory=list)
    oof_scores: Optional[pd.DataFrame] = None
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class ScoreResult:
    scores_df: pd.DataFrame
    label: Optional[pd.Series] = None
    split: str = "test"


def train_panel_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    seed: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    feature_cols = [c for c in X.columns if c not in [date_col, ticker_col]]
    X_train_df = X[feature_cols].copy()

    model_type = str(config.get("model_type", "xgb")).lower()
    model_params = dict(config.get("model_params", {}))

    if is_torch_model_type(model_type):
        runtime = get_torch_runtime_info()
        device = resolve_torch_device(str(config.get("device", "auto")))
        logger.info(
            "Training torch model type=%s on device=%s (torch=%s, cuda_available=%s)",
            model_type,
            device,
            runtime.get("torch_version"),
            runtime.get("cuda_available"),
        )
        torch_config = {"model_type": model_type, **model_params}
        model = train_pytorch_model(
            X_train_df,
            y,
            torch_config,
            device=device,
        )
        setattr(model, "_device_preference", str(config.get("device", "auto")))
    else:
        candidate = {"model_config": {"model_type": model_type, "params": model_params}}
        model = make_model(candidate, seed=seed)
        if not hasattr(model, "fit"):
            raise ValueError(f"Model {model_type} does not have fit method")
        model.fit(X_train_df.values, y.values)

    model_info = {
        "model_type": model_type,
        "feature_names": feature_cols,
        "n_features": len(feature_cols),
        "n_samples": len(X),
        "n_tickers": X[ticker_col].nunique(),
        "device": str(config.get("device", "auto")),
    }
    return model, model_info


def predict_panel(
    model: Any,
    X: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    return_proba: bool = True,
) -> pd.DataFrame:
    del return_proba

    feature_cols = [c for c in X.columns if c not in [date_col, ticker_col]]
    X_eval = X[feature_cols].values

    if is_torch_model_instance(model):
        score = predict_dpoint(model, X[feature_cols])
        proba = score
    else:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_eval)[:, 1]
            score = proba
        else:
            score = model.predict(X_eval)
            proba = score

    return pd.DataFrame(
        {
            date_col: X[date_col].values,
            ticker_col: X[ticker_col].values,
            "score": score,
            "proba": proba,
        }
    )


def compute_oof_scores(
    X: pd.DataFrame,
    y: pd.Series,
    splits: List[Tuple[List, List]],
    config: Dict[str, Any],
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    seed: int = 42,
) -> pd.DataFrame:
    oof_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"Training fold {fold_idx + 1}/{len(splits)}")

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        model, _ = train_panel_model(
            X_train,
            y_train,
            config,
            date_col=date_col,
            ticker_col=ticker_col,
            seed=seed + fold_idx,
        )

        val_pred = predict_panel(
            model,
            X_val,
            date_col=date_col,
            ticker_col=ticker_col,
        )
        val_pred["fold"] = fold_idx
        val_pred["is_oof"] = True
        val_pred["label"] = y_val.values

        for _, row in val_pred.iterrows():
            oof_rows.append(row.to_dict())

    return pd.DataFrame(oof_rows)


def evaluate_panel_model(
    scores_df: pd.DataFrame,
    *,
    score_col: str = "score",
    label_col: str = "label",
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> RankingMetrics:
    return compute_all_ranking_metrics(
        scores_df,
        score_col=score_col,
        label_col=label_col,
        date_col=date_col,
        ticker_col=ticker_col,
    )


def train_with_walkforward(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    splits: List[Tuple[List, List]],
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    seed: int = 42,
    compute_oof: bool = True,
) -> PanelTrainResult:
    notes: List[str] = []
    all_val_scores: List[pd.DataFrame] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"Training fold {fold_idx + 1}/{len(splits)}")

        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_val = X.loc[val_idx]
        y_val = y.loc[val_idx]

        model, model_info = train_panel_model(
            X_train,
            y_train,
            config,
            date_col=date_col,
            ticker_col=ticker_col,
            seed=seed + fold_idx,
        )

        train_pred = predict_panel(
            model,
            X_train,
            date_col=date_col,
            ticker_col=ticker_col,
        )
        train_pred["label"] = y_train.values

        val_pred = predict_panel(
            model,
            X_val,
            date_col=date_col,
            ticker_col=ticker_col,
        )
        val_pred["label"] = y_val.values

        train_metrics = evaluate_panel_model(
            train_pred,
            score_col="score",
            label_col="label",
            date_col=date_col,
            ticker_col=ticker_col,
        )
        val_metrics = evaluate_panel_model(
            val_pred,
            score_col="score",
            label_col="label",
            date_col=date_col,
            ticker_col=ticker_col,
        )

        all_val_scores.append(val_pred)
        notes.append(
            f"Fold {fold_idx + 1}: "
            f"train_rank_ic={train_metrics.rank_ic_mean:.4f}, "
            f"val_rank_ic={val_metrics.rank_ic_mean:.4f}"
        )

    oof_df = pd.concat(all_val_scores, ignore_index=True) if compute_oof else None

    final_model, final_info = train_panel_model(
        X,
        y,
        config,
        date_col=date_col,
        ticker_col=ticker_col,
        seed=seed,
    )

    all_val_combined = pd.concat(all_val_scores, ignore_index=True)
    final_val_metrics = evaluate_panel_model(
        all_val_combined,
        score_col="score",
        label_col="label",
        date_col=date_col,
        ticker_col=ticker_col,
    )

    return PanelTrainResult(
        model=final_model,
        feature_names=final_info["feature_names"],
        oof_scores=oof_df,
        train_metrics={
            "rank_ic_mean": final_val_metrics.rank_ic_mean,
            "rank_ic_std": final_val_metrics.rank_ic_std,
            "ic_mean": final_val_metrics.ic_mean,
            "topk_return_mean": final_val_metrics.topk_return_mean,
        },
        val_metrics={
            "rank_ic_mean": final_val_metrics.rank_ic_mean,
            "rank_ic_std": final_val_metrics.rank_ic_std,
            "ic_mean": final_val_metrics.ic_mean,
            "topk_return_mean": final_val_metrics.topk_return_mean,
        },
        config=config,
        notes=notes,
    )


__all__ = [
    "PanelTrainResult",
    "ScoreResult",
    "train_panel_model",
    "predict_panel",
    "compute_oof_scores",
    "evaluate_panel_model",
    "train_with_walkforward",
]
