from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models import (
    clear_torch_cuda_cache,
    get_torch_runtime_info,
    is_torch_model_instance,
    is_torch_model_type,
    make_model,
    predict_pytorch_model_sequence,
    predict_pytorch_model_tabular,
    resolve_torch_device,
    train_pytorch_model,
)
from ranking_metrics import RankingMetrics, compute_all_ranking_metrics
from sequence_builder import PanelSequenceStore, SequenceDatasetBundle, build_panel_sequence_store, build_panel_sequences

logger = logging.getLogger(__name__)

SEQUENCE_MODEL_TYPES = {"lstm", "gru", "cnn", "transformer"}


@dataclass
class PanelTrainResult:
    model: Any
    feature_names: List[str] = field(default_factory=list)
    oof_scores: Optional[pd.DataFrame] = None
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    holdout_metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    split_mode: str = "wf"
    evaluation_split: str = "oof"


@dataclass
class ScoreResult:
    scores_df: pd.DataFrame
    label: Optional[pd.Series] = None
    split: str = "test"


def _feature_cols(X: pd.DataFrame, date_col: str, ticker_col: str) -> List[str]:
    return [c for c in X.columns if c not in [date_col, ticker_col]]


def _is_sequence_model(config: Dict[str, Any]) -> bool:
    return str(config.get("model_type", "xgb")).lower() in SEQUENCE_MODEL_TYPES


def _fit_tabular_preprocessor(
    X_df: pd.DataFrame,
    *,
    model_type: str,
) -> Optional[StandardScaler]:
    if model_type not in {"mlp"}:
        return None
    scaler = StandardScaler()
    scaler.fit(X_df.to_numpy(dtype=np.float32, copy=False))
    return scaler


def _transform_tabular_features(
    X_df: pd.DataFrame,
    *,
    feature_cols: List[str],
    preprocessor: Optional[StandardScaler],
) -> pd.DataFrame:
    if preprocessor is None:
        return X_df
    transformed = preprocessor.transform(X_df.to_numpy(dtype=np.float32, copy=False))
    del feature_cols
    return pd.DataFrame(transformed, columns=list(X_df.columns), index=X_df.index)


def _predict_and_align_fold(
    model: Any,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    *,
    config: Dict[str, Any],
    date_col: str,
    ticker_col: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    eval_store = _build_sequence_store(
        X_eval,
        y_eval if _is_sequence_model(config) else None,
        config=config,
        date_col=date_col,
        ticker_col=ticker_col,
    ) if _is_sequence_model(config) else None
    pred_df = predict_panel(
        model,
        X_eval,
        date_col=date_col,
        ticker_col=ticker_col,
        sequence_store=eval_store,
    )
    scores_df = align_scores_with_labels(
        pred_df,
        X_eval,
        y_eval,
        config=config,
        date_col=date_col,
        ticker_col=ticker_col,
        sequence_store=eval_store,
    )
    metrics = evaluate_scores_df(scores_df, date_col=date_col, ticker_col=ticker_col)
    return scores_df, metrics


def _build_early_stop_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_type: str,
    date_col: str,
    ratio: float,
    min_dates: int,
    min_rows: int,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
    if model_type not in SEQUENCE_MODEL_TYPES and model_type != "mlp":
        return X, y, None, None
    unique_dates = sorted(pd.to_datetime(X[date_col].unique()))
    if len(unique_dates) < (min_dates * 2):
        return X, y, None, None

    val_dates_count = max(min_dates, int(round(len(unique_dates) * ratio)))
    if val_dates_count >= len(unique_dates):
        return X, y, None, None

    train_dates = unique_dates[:-val_dates_count]
    val_dates = unique_dates[-val_dates_count:]
    X_fit = X[X[date_col].isin(train_dates)].copy()
    y_fit = y.loc[X_fit.index].copy()
    X_val = X[X[date_col].isin(val_dates)].copy()
    y_val = y.loc[X_val.index].copy()
    if len(X_fit) < min_rows or len(X_val) < min_rows:
        return X, y, None, None
    return X_fit, y_fit, X_val, y_val


def _build_sequence_bundle(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    *,
    config: Dict[str, Any],
    date_col: str,
    ticker_col: str,
) -> SequenceDatasetBundle:
    model_params = dict(config.get("model_params", {}))
    seq_len = int(model_params.get("seq_len", 20))
    return build_panel_sequences(
        X,
        y,
        date_col=date_col,
        ticker_col=ticker_col,
        seq_len=seq_len,
    )


def _build_sequence_store(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    *,
    config: Dict[str, Any],
    date_col: str,
    ticker_col: str,
) -> PanelSequenceStore:
    model_params = dict(config.get("model_params", {}))
    seq_len = int(model_params.get("seq_len", 20))
    return build_panel_sequence_store(
        X,
        y,
        date_col=date_col,
        ticker_col=ticker_col,
        seq_len=seq_len,
    )


def align_scores_with_labels(
    scores_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    config: Dict[str, Any],
    date_col: str = "date",
    ticker_col: str = "ticker",
    sequence_store: Optional[PanelSequenceStore] = None,
) -> pd.DataFrame:
    result = scores_df.copy().reset_index(drop=True)
    if _is_sequence_model(config):
        store = sequence_store or _build_sequence_store(
            X,
            y,
            config=config,
            date_col=date_col,
            ticker_col=ticker_col,
        )
        if len(result) != len(store.meta_df):
            raise ValueError(
                f"Sequence prediction rows ({len(result)}) do not match sequence labels ({len(store.meta_df)})"
            )
        if store.label_by_ticker is None:
            raise ValueError("Sequence store is missing labels")
        labels = [
            float(store.label_by_ticker[ticker][start + store.seq_len - 1])
            for ticker, start in store.window_keys
        ]
        result["label"] = labels
        result["source_index"] = store.meta_df["source_index"].to_numpy()
        return result

    labels = y.loc[X.index].to_numpy()
    if len(result) != len(labels):
        raise ValueError(f"Prediction rows ({len(result)}) do not match labels ({len(labels)})")
    result["label"] = labels
    result["source_index"] = X.index.to_numpy()
    return result


def train_panel_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    seed: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    feature_cols = _feature_cols(X, date_col, ticker_col)
    model_type = str(config.get("model_type", "xgb")).lower()
    model_params = dict(config.get("model_params", {}))
    is_sequence = model_type in SEQUENCE_MODEL_TYPES
    early_stop_ratio = float(model_params.get("early_stop_ratio", 0.1))
    early_stop_min_dates = int(model_params.get("early_stop_min_dates", 20))
    early_stop_min_rows = int(model_params.get("early_stop_min_rows", 256))
    X_fit, y_fit, X_early_val, y_early_val = _build_early_stop_split(
        X,
        y,
        model_type=model_type,
        date_col=date_col,
        ratio=early_stop_ratio,
        min_dates=early_stop_min_dates,
        min_rows=early_stop_min_rows,
    )

    if is_sequence:
        store = _build_sequence_store(
            X_fit,
            y_fit,
            config=config,
            date_col=date_col,
            ticker_col=ticker_col,
        )
        early_val_store = None
        if X_early_val is not None and y_early_val is not None:
            early_val_store = _build_sequence_store(
                X_early_val,
                y_early_val,
                config=config,
                date_col=date_col,
                ticker_col=ticker_col,
            )
        runtime = get_torch_runtime_info()
        device = resolve_torch_device(str(config.get("device", "auto")))
        logger.info(
            "Sequence training setup: model_type=%s device=%s rows=%d sequences=%d seq_len=%d n_features=%d batch_size=%d hidden_dim=%s auto_batch_tune=%s train_target_vram=%.0f%% predict_target_vram=%.0f%% amp=%s tf32=%s early_stop_val_rows=%d",
            model_type,
            device,
            len(X_fit),
            len(store.window_keys),
            store.seq_len,
            len(store.feature_names),
            int(model_params.get("batch_size", 64)),
            model_params.get("hidden_dim", model_params.get("d_model", "n/a")),
            bool(model_params.get("auto_batch_tune", True)),
            float(model_params.get("train_target_vram_util", model_params.get("target_vram_util", 0.88))) * 100.0,
            float(model_params.get("predict_target_vram_util", model_params.get("target_vram_util", 0.88))) * 100.0,
            bool(model_params.get("use_amp", False)),
            bool(model_params.get("use_tf32", False)),
            0 if X_early_val is None else len(X_early_val),
        )
        logger.info(
            "Training sequence model type=%s on device=%s (torch=%s cuda_available=%s)",
            model_type,
            device,
            runtime.get("torch_version"),
            runtime.get("cuda_available"),
        )
        torch_config = {"model_type": model_type, **model_params}
        model = train_pytorch_model(store, None, torch_config, device=device, X_val=early_val_store, y_val=None)
        model._seq_len = store.seq_len
        model._feature_names = list(store.feature_names)
        model._is_panel_sequence_model = True
        setattr(model, "_device_preference", str(config.get("device", "auto")))
        model_info = {
            "model_type": model_type,
            "feature_names": list(store.feature_names),
            "n_features": len(store.feature_names),
            "n_samples": len(store.window_keys),
            "n_tickers": X[ticker_col].nunique(),
            "device": str(config.get("device", "auto")),
            "is_sequence": True,
            "seq_len": store.seq_len,
            "batch_size": int(getattr(model, "_trained_batch_size", model_params.get("batch_size", 64))),
        }
        return model, model_info

    X_train_df = X_fit[feature_cols].copy()
    preprocessor = _fit_tabular_preprocessor(X_train_df, model_type=model_type)
    X_train_df = _transform_tabular_features(
        X_train_df,
        feature_cols=feature_cols,
        preprocessor=preprocessor,
    )
    X_val_df = None
    if X_early_val is not None:
        X_val_df = _transform_tabular_features(
            X_early_val[feature_cols].copy(),
            feature_cols=feature_cols,
            preprocessor=preprocessor,
        )
    if is_torch_model_type(model_type):
        runtime = get_torch_runtime_info()
        device = resolve_torch_device(str(config.get("device", "auto")))
        logger.info(
            "Training torch model type=%s on device=%s (torch=%s cuda_available=%s early_stop_val_rows=%d)",
            model_type,
            device,
            runtime.get("torch_version"),
            runtime.get("cuda_available"),
            0 if X_early_val is None else len(X_early_val),
        )
        torch_config = {"model_type": model_type, **model_params}
        model = train_pytorch_model(X_train_df, y_fit, torch_config, device=device, X_val=X_val_df, y_val=y_early_val)
        setattr(model, "_device_preference", str(config.get("device", "auto")))
        setattr(model, "_feature_names", list(feature_cols))
        setattr(model, "_preprocessor", preprocessor)
    else:
        candidate = {"model_config": {"model_type": model_type, "params": model_params}}
        model = make_model(candidate, seed=seed)
        if not hasattr(model, "fit"):
            raise ValueError(f"Model {model_type} does not have fit method")
        model.fit(X_train_df.to_numpy(), y_fit.to_numpy())

    model_info = {
        "model_type": model_type,
        "feature_names": feature_cols,
        "n_features": len(feature_cols),
        "n_samples": len(X),
        "n_tickers": X[ticker_col].nunique(),
        "device": str(config.get("device", "auto")),
        "is_sequence": False,
    }
    return model, model_info


def predict_panel(
    model: Any,
    X: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    return_proba: bool = True,
    sequence_store: Optional[PanelSequenceStore] = None,
) -> pd.DataFrame:
    del return_proba
    feature_cols = _feature_cols(X, date_col, ticker_col)
    is_sequence = getattr(model, "_is_panel_sequence_model", False)

    if is_sequence:
        seq_len = int(getattr(model, "_seq_len", 20))
        device = resolve_torch_device(str(getattr(model, "_device_preference", "auto")))
        store = sequence_store or build_panel_sequence_store(
            X,
            None,
            date_col=date_col,
            ticker_col=ticker_col,
            seq_len=seq_len,
        )
        logger.info(
            "Predicting sequence model on device=%s with sequences=%d seq_len=%d train_batch_size=%d predict_batch_size=%d",
            device,
            len(store.window_keys),
            store.seq_len,
            int(getattr(model, "_trained_batch_size", 64)),
            int(getattr(model, "_predict_batch_size", getattr(model, "_trained_batch_size", 64))),
        )
        scores = predict_pytorch_model_sequence(model, store, store.meta_df, device)
        return pd.DataFrame(
            {
                date_col: store.meta_df["date"].to_numpy(),
                ticker_col: store.meta_df["ticker"].to_numpy(),
                "score": scores.to_numpy(),
                "proba": scores.to_numpy(),
            }
        )

    X_eval = X[feature_cols]
    preprocessor = getattr(model, "_preprocessor", None)
    X_eval = _transform_tabular_features(
        X_eval,
        feature_cols=feature_cols,
        preprocessor=preprocessor,
    )
    if is_torch_model_instance(model):
        device = resolve_torch_device(str(getattr(model, "_device_preference", "auto")))
        score = predict_pytorch_model_tabular(model, X_eval, device)
        proba = score
        probability_available = True
    else:
        X_values = X_eval.to_numpy()
        if hasattr(model, "predict_proba"):
            proba = pd.Series(model.predict_proba(X_values)[:, 1], index=X.index)
            score = proba
            probability_available = True
        elif hasattr(model, "decision_function"):
            score = pd.Series(model.decision_function(X_values), index=X.index)
            proba = pd.Series(np.nan, index=X.index, dtype=np.float32)
            probability_available = False
        else:
            score = pd.Series(model.predict(X_values), index=X.index)
            proba = pd.Series(np.nan, index=X.index, dtype=np.float32)
            probability_available = False

    return pd.DataFrame(
        {
            date_col: X[date_col].to_numpy(),
            ticker_col: X[ticker_col].to_numpy(),
            "score": score.to_numpy(),
            "proba": proba.to_numpy(),
            "probability_available": probability_available,
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
    oof_parts: List[pd.DataFrame] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_val = X.loc[val_idx]
        y_val = y.loc[val_idx]
        model, _ = train_panel_model(
            X_train,
            y_train,
            config,
            date_col=date_col,
            ticker_col=ticker_col,
            seed=seed + fold_idx,
        )
        val_scores, _ = _predict_and_align_fold(
            model,
            X_val,
            y_val,
            config=config,
            date_col=date_col,
            ticker_col=ticker_col,
        )
        val_scores["fold"] = fold_idx
        val_scores["is_oof"] = True
        oof_parts.append(val_scores)
        if _is_sequence_model(config):
            clear_torch_cuda_cache()
    return pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()


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


def evaluate_scores_df(
    scores_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    score_col: str = "score",
    label_col: str = "label",
) -> Dict[str, float]:
    if scores_df.empty:
        return {
            "rank_ic_mean": 0.0,
            "rank_ic_std": 0.0,
            "ic_mean": 0.0,
            "topk_return_mean": 0.0,
        }
    metrics = compute_all_ranking_metrics(
        scores_df,
        score_col=score_col,
        label_col=label_col,
        date_col=date_col,
        ticker_col=ticker_col,
    )
    return {
        "rank_ic_mean": float(metrics.rank_ic_mean),
        "rank_ic_std": float(metrics.rank_ic_std),
        "ic_mean": float(metrics.ic_mean),
        "topk_return_mean": float(metrics.topk_return_mean),
    }


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
        logger.info("Training fold %d/%d", fold_idx + 1, len(splits))
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_val = X.loc[val_idx]
        y_val = y.loc[val_idx]

        model, _ = train_panel_model(
            X_train,
            y_train,
            config,
            date_col=date_col,
            ticker_col=ticker_col,
            seed=seed + fold_idx,
        )
        val_scores, val_metrics = _predict_and_align_fold(
            model,
            X_val,
            y_val,
            config=config,
            date_col=date_col,
            ticker_col=ticker_col,
        )
        val_scores["fold"] = fold_idx
        val_scores["is_oof"] = True
        all_val_scores.append(val_scores)
        notes.append(
            f"Fold {fold_idx + 1}: val_rank_ic={val_metrics['rank_ic_mean']:.4f}, "
            f"val_topk_return={val_metrics['topk_return_mean']:.4f}"
        )
        if _is_sequence_model(config):
            clear_torch_cuda_cache()

    oof_df = pd.concat(all_val_scores, ignore_index=True) if compute_oof and all_val_scores else pd.DataFrame()
    final_model, final_info = train_panel_model(
        X,
        y,
        config,
        date_col=date_col,
        ticker_col=ticker_col,
        seed=seed,
    )
    oof_metrics = evaluate_scores_df(oof_df, date_col=date_col, ticker_col=ticker_col)

    return PanelTrainResult(
        model=final_model,
        feature_names=final_info["feature_names"],
        oof_scores=oof_df,
        train_metrics={},
        val_metrics=oof_metrics,
        config=config,
        notes=notes,
        split_mode="wf",
        evaluation_split="oof",
    )


def train_with_nested_walkforward(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    nested_splits: List[Dict[str, Any]],
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    seed: int = 42,
) -> PanelTrainResult:
    notes: List[str] = []
    outer_scores: List[pd.DataFrame] = []

    for outer_fold_idx, outer_split in enumerate(nested_splits):
        logger.info("Outer fold %d/%d", outer_fold_idx + 1, len(nested_splits))
        X_outer_train = X.loc[outer_split["outer_train_idx"]]
        y_outer_train = y.loc[outer_split["outer_train_idx"]]
        X_outer_val = X.loc[outer_split["outer_val_idx"]]
        y_outer_val = y.loc[outer_split["outer_val_idx"]]

        best_inner_metric = float("-inf")
        for inner_idx, (inner_train_idx, inner_val_idx) in enumerate(outer_split["inner_splits"]):
            inner_model, _ = train_panel_model(
                X.loc[inner_train_idx],
                y.loc[inner_train_idx],
                config,
                date_col=date_col,
                ticker_col=ticker_col,
                seed=seed + outer_fold_idx * 100 + inner_idx,
            )
            X_inner_val = X.loc[inner_val_idx]
            y_inner_val = y.loc[inner_val_idx]
            inner_scores, inner_metrics = _predict_and_align_fold(
                inner_model,
                X_inner_val,
                y_inner_val,
                config=config,
                date_col=date_col,
                ticker_col=ticker_col,
            )
            best_inner_metric = max(best_inner_metric, inner_metrics["rank_ic_mean"])

        outer_model, _ = train_panel_model(
            X_outer_train,
            y_outer_train,
            config,
            date_col=date_col,
            ticker_col=ticker_col,
            seed=seed + outer_fold_idx,
        )
        outer_scores_df, outer_metrics = _predict_and_align_fold(
            outer_model,
            X_outer_val,
            y_outer_val,
            config=config,
            date_col=date_col,
            ticker_col=ticker_col,
        )
        outer_scores_df["outer_fold"] = outer_fold_idx
        outer_scores_df["is_oof"] = True
        outer_scores.append(outer_scores_df)
        notes.append(
            f"Outer fold {outer_fold_idx + 1}: inner_best_rank_ic={best_inner_metric:.4f}, "
            f"outer_rank_ic={outer_metrics['rank_ic_mean']:.4f}"
        )
        if _is_sequence_model(config):
            clear_torch_cuda_cache()

    oof_df = pd.concat(outer_scores, ignore_index=True) if outer_scores else pd.DataFrame()
    final_model, final_info = train_panel_model(
        X,
        y,
        config,
        date_col=date_col,
        ticker_col=ticker_col,
        seed=seed,
    )
    oof_metrics = evaluate_scores_df(oof_df, date_col=date_col, ticker_col=ticker_col)

    return PanelTrainResult(
        model=final_model,
        feature_names=final_info["feature_names"],
        oof_scores=oof_df,
        train_metrics={},
        val_metrics=oof_metrics,
        config=config,
        notes=notes,
        split_mode="nested_wf",
        evaluation_split="oof",
    )


__all__ = [
    "PanelTrainResult",
    "ScoreResult",
    "align_scores_with_labels",
    "train_panel_model",
    "predict_panel",
    "compute_oof_scores",
    "evaluate_panel_model",
    "train_with_walkforward",
    "evaluate_scores_df",
    "train_with_nested_walkforward",
]
