from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backtester_engine import backtest_from_scores, compute_buy_and_hold_benchmark, prepare_scores_for_backtest
from basket_loader import BasketMeta, BasketReport, load_basket_folder
from constants import (
    DATA_CONTRACT_VERSION,
    DEFAULT_BASKET_NAME,
    DEFAULT_DATA_ROOT,
    DEFAULT_FILE_PATTERN,
    DEFAULT_LABEL_MODE,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_REBALANCE_FREQ,
    DEFAULT_TOP_K,
    DEFAULT_WEIGHTING,
)
from excel_reporter import save_to_excel
from feature_dpoint import build_features_and_labels_panel
from html_reporter import generate_html_report
from models import get_torch_runtime_info, load_saved_model, save_trained_model
from panel_builder import validate_panel
from panel_trainer import align_scores_with_labels, evaluate_scores_df, predict_panel, train_panel_model
from portfolio_builder import PortfolioConfig
from ranking_metrics import compute_all_ranking_metrics
from rolling_retrainer import RollingConfig, RollingRetrainer
from search_engine import run_search
from search_space import build_base_model_config
from splitters import build_date_splits, final_holdout_split_by_date
from utils import (
    create_experiment_dir,
    create_manifest,
    get_git_commit_hash,
    get_package_versions,
    resolve_basket_path,
    set_global_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

GENERIC_HIDDEN_DIM_DEFAULT = 1024
GENERIC_BATCH_SIZE_DEFAULT = 8192
SEQUENCE_HIDDEN_DIM_DEFAULT = 128
SEQUENCE_BATCH_SIZE_DEFAULT = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A-share basket ML/DL Dpoint trader.")
    parser.add_argument("--basket", type=str, default=DEFAULT_BASKET_NAME)
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--basket_path", type=str, default=None)
    parser.add_argument("--file_pattern", type=str, default=DEFAULT_FILE_PATTERN)
    parser.add_argument("--ticker_from", type=str, choices=["filename", "manifest"], default="filename")
    parser.add_argument("--output_dir", type=str, default="./output_basket")

    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=4)

    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rebalance_freq", type=str, choices=["daily", "weekly", "monthly"], default=DEFAULT_REBALANCE_FREQ)
    parser.add_argument("--weighting", type=str, choices=["equal", "score", "vol_inv"], default=DEFAULT_WEIGHTING)
    parser.add_argument("--max_weight", type=float, default=DEFAULT_MAX_WEIGHT)
    parser.add_argument("--cash_buffer", type=float, default=0.05)

    parser.add_argument("--label_mode", type=str, default=DEFAULT_LABEL_MODE)
    parser.add_argument("--include_cross_section", type=int, default=1)

    parser.add_argument("--initial_cash", type=float, default=100000.0)
    parser.add_argument("--benchmark_mode", choices=["none", "equal_weight"], default="equal_weight")
    parser.add_argument("--research_start_date", type=str, default=None)
    parser.add_argument("--research_end_date", type=str, default=None)
    parser.add_argument("--report_start_date", type=str, default=None)
    parser.add_argument("--report_end_date", type=str, default=None)
    parser.add_argument("--backtest_start_date", type=str, default=None)
    parser.add_argument("--backtest_end_date", type=str, default=None)

    parser.add_argument("--mode", type=str, choices=["first", "continue"], default="first")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--additional_runs", type=int, default=50)

    parser.add_argument(
        "--model_type",
        choices=["mlp", "xgb", "lstm", "gru", "cnn", "transformer"],
        default="mlp",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cpu_threads", type=int, default=4)

    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--hidden_dims", type=str, default="1024,512,256")
    parser.add_argument("--dropout_rate", type=float, default=0.10)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--predict_batch_size", type=int, default=0)
    parser.add_argument("--auto_batch_tune", type=int, default=1)
    parser.add_argument("--target_vram_util", type=float, default=0.88)
    parser.add_argument("--train_target_vram_util", type=float, default=None)
    parser.add_argument("--predict_target_vram_util", type=float, default=None)
    parser.add_argument("--use_amp", type=int, default=0)
    parser.add_argument("--use_tf32", type=int, default=0)

    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--bidirectional", type=int, default=0)
    parser.add_argument("--num_filters", type=int, default=64)
    parser.add_argument("--kernel_sizes", type=str, default="2,3,5")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=128)

    parser.add_argument("--xgb_n_estimators", type=int, default=200)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_subsample", type=float, default=0.9)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.9)

    parser.add_argument("--split_mode", choices=["wf", "wf_embargo", "nested_wf"], default="wf")
    parser.add_argument("--train_start_ratio", type=float, default=0.5)
    parser.add_argument("--split_min_rows", type=int, default=60)

    parser.add_argument("--use_holdout", type=int, default=1)
    parser.add_argument("--holdout_ratio", type=float, default=0.15)
    parser.add_argument("--min_holdout_rows", type=int, default=60)
    parser.add_argument("--holdout_gap_days", type=int, default=0)

    parser.add_argument("--embargo_days", type=int, default=5)
    parser.add_argument("--inner_embargo_days", type=int, default=None)
    parser.add_argument("--n_outer_folds", type=int, default=3)
    parser.add_argument("--n_inner_folds", type=int, default=2)
    parser.add_argument("--selection_metric", choices=["rank_ic_mean", "topk_return_mean"], default="rank_ic_mean")

    parser.add_argument("--run_mode", choices=["single", "rolling"], default="single")
    parser.add_argument("--rolling_mode", choices=["expanding", "rolling"], default="expanding")
    parser.add_argument("--rolling_window_length", type=int, default=252)
    parser.add_argument("--retrain_frequency", choices=["monthly"], default="monthly")
    parser.add_argument("--min_history_days", type=int, default=120)
    return parser.parse_args()


def _next_experiment_id(output_dir: str) -> int:
    exp_ids: List[int] = []
    for path in glob.glob(os.path.join(output_dir, "exp_*")):
        name = os.path.basename(path)
        try:
            exp_ids.append(int(name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return max(exp_ids, default=0) + 1


def _next_continue_run_id(experiment_dir: str) -> int:
    run_ids: List[int] = []
    for path in glob.glob(os.path.join(experiment_dir, "continue_run_*")):
        name = os.path.basename(path)
        try:
            run_ids.append(int(name.split("_")[-1]))
        except ValueError:
            continue
    return max(run_ids, default=0) + 1


def create_continue_run_dir(base_experiment_dir: str) -> str:
    continue_dir = os.path.join(base_experiment_dir, f"continue_run_{_next_continue_run_id(base_experiment_dir):03d}")
    os.makedirs(continue_dir, exist_ok=True)
    os.makedirs(os.path.join(continue_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(continue_dir, "artifacts"), exist_ok=True)
    return continue_dir


def log_runtime_status(args: argparse.Namespace) -> None:
    torch_info = get_torch_runtime_info()
    logger.info(
        "Torch runtime: version=%s cuda_available=%s cuda_build=%s device_count=%s device_name=%s",
        torch_info.get("torch_version"),
        torch_info.get("cuda_available"),
        torch_info.get("cuda_version"),
        torch_info.get("device_count"),
        torch_info.get("device_name"),
    )
    if args.model_type == "mlp" and args.device in {"auto", "cuda"} and not torch_info.get("cuda_available"):
        logger.warning(
            "GPU will not be used because the current PyTorch runtime has no CUDA support. Current torch build: %s",
            torch_info.get("torch_version"),
        )


def load_and_validate_data(
    basket_path: str,
    file_pattern: str,
    ticker_from: str,
) -> Tuple[pd.DataFrame, BasketReport, BasketMeta]:
    logger.info("Loading basket from: %s", basket_path)
    panel_df, report, meta = load_basket_folder(
        basket_path,
        file_pattern=file_pattern,
        ticker_from=ticker_from,
    )
    logger.info("Basket: %s", meta.basket_name)
    logger.info("Tickers: %s", meta.n_tickers)
    logger.info("Total rows: %s", report.total_rows)
    logger.info("Raw data date range: %s", meta.date_range)

    valid, issues = validate_panel(panel_df)
    critical_issues = [issue for issue in issues if "sparse" not in issue.lower()]
    if not valid and critical_issues:
        raise ValueError(f"Panel validation failed: {critical_issues}")
    for issue in issues:
        logger.info("Panel note: %s", issue)
    return panel_df, report, meta


def build_feature_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "basket_name": getattr(args, "basket", DEFAULT_BASKET_NAME),
        "windows": [5, 10, 20, 60],
        "use_momentum": True,
        "use_volatility": True,
        "use_volume": True,
        "use_candle": True,
        "use_ta_indicators": True,
        "ta_windows": [6, 14, 20],
    }


def build_model_config(args: argparse.Namespace) -> Dict[str, Any]:
    cpu_threads = max(1, args.cpu_threads)
    sequence_hidden_dim = args.hidden_dim
    sequence_batch_size = args.batch_size
    predict_batch_size = int(getattr(args, "predict_batch_size", 0))
    auto_batch_tune = bool(getattr(args, "auto_batch_tune", 1))
    target_vram_util = float(getattr(args, "target_vram_util", 0.88))
    train_target_vram_util = float(
        getattr(args, "train_target_vram_util", None)
        if getattr(args, "train_target_vram_util", None) is not None
        else target_vram_util
    )
    predict_target_vram_util = float(
        getattr(args, "predict_target_vram_util", None)
        if getattr(args, "predict_target_vram_util", None) is not None
        else target_vram_util
    )
    use_amp = bool(getattr(args, "use_amp", 0))
    use_tf32 = bool(getattr(args, "use_tf32", 0))
    if args.model_type in {"lstm", "gru", "cnn", "transformer"}:
        if sequence_hidden_dim == GENERIC_HIDDEN_DIM_DEFAULT:
            sequence_hidden_dim = SEQUENCE_HIDDEN_DIM_DEFAULT
        if sequence_batch_size == GENERIC_BATCH_SIZE_DEFAULT:
            sequence_batch_size = SEQUENCE_BATCH_SIZE_DEFAULT
    if args.model_type == "mlp":
        hidden_dims = [int(part.strip()) for part in str(args.hidden_dims).split(",") if part.strip()]
        if not hidden_dims:
            hidden_dims = [args.hidden_dim]
        return {
            "model_type": "mlp",
            "device": args.device,
            "model_params": {
                "hidden_dim": args.hidden_dim,
                "hidden_dims": hidden_dims,
                "dropout_rate": args.dropout_rate,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "predict_batch_size": predict_batch_size,
                "auto_batch_tune": auto_batch_tune,
                "target_vram_util": target_vram_util,
                "train_target_vram_util": train_target_vram_util,
                "predict_target_vram_util": predict_target_vram_util,
                "use_amp": use_amp,
                "use_tf32": use_tf32,
            },
        }
    if args.model_type in {"lstm", "gru", "cnn", "transformer"}:
        return {
            "model_type": args.model_type,
            "device": args.device,
            "model_params": {
                "dropout_rate": args.dropout_rate,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "batch_size": sequence_batch_size,
                "predict_batch_size": predict_batch_size,
                "auto_batch_tune": auto_batch_tune,
                "target_vram_util": target_vram_util,
                "train_target_vram_util": train_target_vram_util,
                "predict_target_vram_util": predict_target_vram_util,
                "use_amp": use_amp,
                "use_tf32": use_tf32,
                "seq_len": args.seq_len,
                "num_layers": args.num_layers,
                "bidirectional": bool(args.bidirectional),
                "hidden_dim": sequence_hidden_dim,
                "num_filters": args.num_filters,
                "kernel_sizes": [int(part.strip()) for part in str(args.kernel_sizes).split(",") if part.strip()],
                "d_model": args.d_model,
                "nhead": args.nhead,
                "dim_feedforward": args.dim_feedforward,
            },
        }
    return {
        "model_type": "xgb",
        "device": "cpu",
        "model_params": {
            "n_estimators": args.xgb_n_estimators,
            "max_depth": args.xgb_max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.xgb_subsample,
            "colsample_bytree": args.xgb_colsample_bytree,
            "random_state": args.seed,
            "n_jobs": cpu_threads,
            "tree_method": "hist",
            "eval_metric": "logloss",
            "verbosity": 0,
        },
    }


def dates_to_indices(
    X: pd.DataFrame,
    train_dates: List[pd.Timestamp],
    val_dates: List[pd.Timestamp],
    date_col: str = "date",
) -> Tuple[List[int], List[int]]:
    train_mask = X[date_col].isin(train_dates)
    val_mask = X[date_col].isin(val_dates)
    return X.index[train_mask].tolist(), X.index[val_mask].tolist()


def nested_dates_to_indices(
    X: pd.DataFrame,
    nested_splits: List[Tuple[List[pd.Timestamp], List[pd.Timestamp], List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]]]],
    date_col: str = "date",
) -> List[Dict[str, Any]]:
    indexed: List[Dict[str, Any]] = []
    for outer_train_dates, outer_val_dates, inner_splits in nested_splits:
        indexed.append(
            {
                "outer_train_idx": X.index[X[date_col].isin(outer_train_dates)].tolist(),
                "outer_val_idx": X.index[X[date_col].isin(outer_val_dates)].tolist(),
                "inner_splits": [
                    (
                        X.index[X[date_col].isin(inner_train_dates)].tolist(),
                        X.index[X[date_col].isin(inner_val_dates)].tolist(),
                    )
                    for inner_train_dates, inner_val_dates in inner_splits
                ],
            }
        )
    return indexed


def build_split_plan(
    X: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> Dict[str, Any]:
    use_holdout = args.use_holdout == 1
    if use_holdout:
        effective_holdout_gap = max(int(getattr(args, "holdout_gap_days", 0)), int(getattr(args, "embargo_days", 0)))
        search_df, holdout_df = final_holdout_split_by_date(
            X,
            date_col=date_col,
            ticker_col=ticker_col,
            holdout_ratio=args.holdout_ratio,
            min_holdout_rows=args.min_holdout_rows,
            enforce_non_empty_search=True,
            gap_days=effective_holdout_gap,
        )
        search_X = search_df
        search_y = y.loc[search_df.index]
        holdout_X = holdout_df
        holdout_y = y.loc[holdout_df.index]
    else:
        search_X = X
        search_y = y
        holdout_X = None
        holdout_y = None

    date_splits = build_date_splits(
        search_X,
        split_mode=args.split_mode,
        date_col=date_col,
        ticker_col=ticker_col,
        n_folds=args.n_folds,
        n_outer_folds=args.n_outer_folds,
        n_inner_folds=args.n_inner_folds,
        train_start_ratio=args.train_start_ratio,
        min_rows=args.split_min_rows,
        embargo_days=args.embargo_days,
        inner_embargo_days=getattr(args, "inner_embargo_days", None),
    )

    if args.split_mode == "nested_wf":
        indexed_splits = nested_dates_to_indices(search_X, date_splits, date_col=date_col)
    else:
        indexed_splits = [
            dates_to_indices(search_X, train_dates, val_dates, date_col=date_col)
            for train_dates, val_dates in date_splits
        ]

    split_summary = {
        "split_mode": args.split_mode,
        "use_holdout": use_holdout,
        "holdout_ratio": args.holdout_ratio if use_holdout else None,
        "holdout_gap_days": effective_holdout_gap if use_holdout else None,
        "embargo_days": args.embargo_days if args.split_mode == "wf_embargo" else None,
        "n_folds": len(date_splits),
        "n_outer_folds": args.n_outer_folds if args.split_mode == "nested_wf" else None,
        "n_inner_folds": args.n_inner_folds if args.split_mode == "nested_wf" else None,
        "inner_embargo_days": (
            args.embargo_days if getattr(args, "inner_embargo_days", None) is None else args.inner_embargo_days
        ) if args.split_mode == "nested_wf" else None,
        "train_start_ratio": args.train_start_ratio,
        "split_min_rows": args.split_min_rows,
    }
    return {
        "search_X": search_X,
        "search_y": search_y,
        "holdout_X": holdout_X,
        "holdout_y": holdout_y,
        "indexed_splits": indexed_splits,
        "split_summary": split_summary,
    }


def _parse_optional_date(date_str: Optional[str]) -> Optional[pd.Timestamp]:
    if date_str is None:
        return None
    ts = pd.Timestamp(date_str)
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {date_str}")
    return ts


def _coalesce_window_arg(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value is not None:
            return value
    return None


def resolve_window_config(
    panel_df: pd.DataFrame,
    args: argparse.Namespace,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> Dict[str, pd.Timestamp]:
    first_trade_by_ticker = panel_df.groupby(ticker_col)[date_col].min()
    default_start = pd.Timestamp(first_trade_by_ticker.max())
    default_end = pd.Timestamp(panel_df[date_col].max())

    research_start = _parse_optional_date(_coalesce_window_arg(args.research_start_date))
    research_end = _parse_optional_date(_coalesce_window_arg(args.research_end_date))
    report_start = _parse_optional_date(_coalesce_window_arg(args.report_start_date, args.backtest_start_date))
    report_end = _parse_optional_date(_coalesce_window_arg(args.report_end_date, args.backtest_end_date))

    effective_research_start = research_start or default_start
    effective_research_end = research_end or default_end
    effective_report_start = report_start or effective_research_start
    effective_report_end = report_end or effective_research_end

    if effective_research_start > effective_research_end:
        raise ValueError(
            f"research_start_date {effective_research_start} cannot be later than research_end_date {effective_research_end}"
        )
    if effective_report_start > effective_report_end:
        raise ValueError(
            f"report_start_date {effective_report_start} cannot be later than report_end_date {effective_report_end}"
        )
    if effective_report_start < effective_research_start or effective_report_end > effective_research_end:
        raise ValueError("Report window must stay within the research window.")

    logger.info("Universe-aligned default research window: (%s, %s)", default_start, default_end)
    logger.info("Effective research window: (%s, %s)", effective_research_start, effective_research_end)
    logger.info("Effective report window: (%s, %s)", effective_report_start, effective_report_end)
    return {
        "default_start": default_start,
        "default_end": default_end,
        "research_start": effective_research_start,
        "research_end": effective_research_end,
        "report_start": effective_report_start,
        "report_end": effective_report_end,
    }


def filter_feature_window(
    X: pd.DataFrame,
    y: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.Series]:
    mask = (X[date_col] >= start_date) & (X[date_col] <= end_date)
    X_window = X.loc[mask].copy()
    y_window = y.loc[X_window.index].copy()
    return X_window, y_window


def resolve_label_mode_alias(label_mode: str) -> str:
    alias_map = {
        "classification": "binary_next_close_up",
        "binary": "binary_next_close_up",
        "regression": "regression_return",
    }
    return alias_map.get(label_mode, label_mode)


def normalize_mode_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.mode == "continue":
        if not args.continue_from:
            args.continue_from = "latest"
    elif args.continue_from:
        raise ValueError("--continue_from can only be used when --mode continue")
    return args


def load_previous_experiment(
    args: argparse.Namespace,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str], int]:
    if not args.continue_from:
        return None, None, None, 0
    if args.continue_from == "latest":
        exp_dirs = sorted(glob.glob(os.path.join(args.output_dir, "exp_*")))
        if not exp_dirs:
            raise FileNotFoundError("No previous experiments found")
        prev_exp_dir = exp_dirs[-1]
    else:
        prev_exp_dir = args.continue_from
    if not os.path.isdir(prev_exp_dir):
        raise FileNotFoundError(f"Previous experiment dir not found: {prev_exp_dir}")

    manifest_path = os.path.join(prev_exp_dir, "manifest.json")
    prev_best_config: Optional[Dict[str, Any]] = None
    completed_search_runs = 0
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        prev_best_config = manifest.get("best_config")
        completed_search_runs = int(manifest.get("search_runs_completed", 0) or 0)

    prev_model_path = None
    models_dir = os.path.join(prev_exp_dir, "models")
    if os.path.isdir(models_dir):
        model_entries = sorted(
            os.path.join(models_dir, name)
            for name in os.listdir(models_dir)
            if name.endswith(".joblib") or os.path.isdir(os.path.join(models_dir, name))
        )
        for candidate in reversed(model_entries):
            if candidate.endswith(".joblib"):
                prev_model_path = candidate
                break
            if os.path.exists(os.path.join(candidate, "model_state.pt")):
                prev_model_path = candidate
                break
    return prev_exp_dir, prev_best_config, prev_model_path, completed_search_runs


def _build_search_base_config(args: argparse.Namespace, prev_best_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if prev_best_config is not None:
        return prev_best_config
    try:
        return build_base_model_config(args)
    except ValueError:
        return build_model_config(args)


def main() -> None:
    args = normalize_mode_args(parse_args())
    set_global_seed(args.seed)
    logger.info("Global seed set to %s", args.seed)
    log_runtime_status(args)

    basket_path = args.basket_path or resolve_basket_path(args.data_root, args.basket)
    logger.info("Basket path: %s", basket_path)
    logger.info("Output dir: %s", args.output_dir)

    panel_df, basket_report, basket_meta = load_and_validate_data(basket_path, args.file_pattern, args.ticker_from)
    window_config = resolve_window_config(
        panel_df,
        args,
        date_col="date",
        ticker_col="ticker",
    )

    if args.dry_run:
        logger.info("Dry run mode - data validation only")
        logger.info("Panel shape: %s", panel_df.shape)
        logger.info("Unique dates: %s", panel_df["date"].nunique())
        logger.info("Unique tickers: %s", panel_df["ticker"].nunique())
        return

    if args.run_mode == "rolling":
        logger.info("Running in rolling retrain mode")
        _run_rolling_retrain(panel_df, args, basket_meta)
        return

    _run_single_experiment(panel_df, basket_report, basket_meta, args, window_config)


def _run_rolling_retrain(
    panel_df: pd.DataFrame,
    args: argparse.Namespace,
    basket_meta: BasketMeta,
) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    experiment_dir = create_experiment_dir(args.output_dir, _next_experiment_id(args.output_dir))
    rolling_config = RollingConfig(
        window_type=args.rolling_mode,
        rolling_window_length=args.rolling_window_length if args.rolling_mode == "rolling" else None,
        retrain_frequency=args.retrain_frequency,
        min_history_days=args.min_history_days,
    )
    retrainer = RollingRetrainer(config=rolling_config, experiment_dir=experiment_dir)
    snapshots = retrainer.run(panel_df, args)
    create_manifest(
        experiment_dir,
        run_id=1,
        timestamp=datetime.now().isoformat(),
        git_commit_hash=get_git_commit_hash(),
        package_versions=get_package_versions(),
        seed=args.seed,
        data_info={
            "basket_path": args.basket_path or resolve_basket_path(args.data_root, args.basket),
            "n_tickers": basket_meta.n_tickers,
            "n_rows": len(panel_df),
            "raw_date_range": f"{panel_df['date'].min()} ~ {panel_df['date'].max()}",
        },
        cli_args=vars(args),
        best_config=None,
        metrics={"n_snapshots": len(snapshots), "evaluation_split": "oof"},
        search_runs_completed=len(snapshots),
        split_info={
            "run_mode": "rolling",
            "window_type": args.rolling_mode,
            "retrain_frequency": args.retrain_frequency,
            "min_history_days": args.min_history_days,
        },
    )
    logger.info("Results saved to: %s", experiment_dir)
    logger.info("Done!")


def _run_single_experiment(
    panel_df: pd.DataFrame,
    basket_report: BasketReport,
    basket_meta: BasketMeta,
    args: argparse.Namespace,
    window_config: Dict[str, pd.Timestamp],
) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    prev_exp_dir, prev_best_config, prev_model_path, completed_search_runs = load_previous_experiment(args)
    continue_mode = prev_exp_dir is not None
    if continue_mode:
        logger.info("Continue mode - source experiment: %s", prev_exp_dir)
        logger.info("Completed search runs from previous manifest: %s", completed_search_runs)

    basket_path = args.basket_path or resolve_basket_path(args.data_root, args.basket)
    experiment_dir = create_continue_run_dir(prev_exp_dir) if continue_mode else create_experiment_dir(args.output_dir, _next_experiment_id(args.output_dir))
    logger.info("Experiment dir: %s", experiment_dir)

    research_start = window_config["research_start"]
    research_end = window_config["research_end"]
    report_start = window_config["report_start"]
    report_end = window_config["report_end"]
    default_start = window_config["default_start"]
    default_end = window_config["default_end"]

    feature_config = build_feature_config(args)
    X, y, feature_meta = build_features_and_labels_panel(
        panel_df,
        feature_config,
        date_col="date",
        ticker_col="ticker",
        label_mode=resolve_label_mode_alias(args.label_mode),
        include_cross_section=bool(args.include_cross_section),
    )
    X, y = filter_feature_window(X, y, research_start, research_end, date_col="date")
    logger.info("Features: %s", len(feature_meta.feature_names))
    logger.info("Samples: %s", len(X))
    logger.info("Tickers: %s", feature_meta.n_tickers)
    if X.empty:
        raise ValueError(f"No samples remain inside research window [{research_start}, {research_end}]")

    split_plan = build_split_plan(X, y, args, date_col="date", ticker_col="ticker")
    search_X = split_plan["search_X"]
    search_y = split_plan["search_y"]
    holdout_X = split_plan["holdout_X"]
    holdout_y = split_plan["holdout_y"]
    indexed_splits = split_plan["indexed_splits"]
    split_summary = split_plan["split_summary"]
    if not indexed_splits:
        raise ValueError(
            "No walk-forward splits were generated for the selected research window. "
            "Use an earlier research_start_date or reduce n_folds."
        )

    search_runs = args.additional_runs if continue_mode else args.runs
    best_config = _build_search_base_config(args, prev_best_config)
    best_seed = args.seed
    search_summary = {
        "n_candidates": 0,
        "selection_metric": args.selection_metric,
        "best_seed": best_seed,
        "best_model_type": best_config.get("model_type"),
    }
    train_notes: List[str] = []
    train_metrics: Dict[str, Any] = {}

    if continue_mode and prev_model_path and search_runs <= 0:
        final_model = load_saved_model(prev_model_path)
        if holdout_X is None or holdout_y is None:
            raise ValueError("Holdout data not available for evaluation-only mode")
        holdout_pred = predict_panel(final_model, holdout_X, date_col="date", ticker_col="ticker")
        search_scores_df = pd.DataFrame()
        holdout_scores_df = align_scores_with_labels(
            holdout_pred,
            holdout_X,
            holdout_y,
            config=best_config,
            date_col="date",
            ticker_col="ticker",
        )
        holdout_scores_df["split"] = "holdout"
        final_eval_scores_df = holdout_scores_df
        train_metrics = evaluate_scores_df(final_eval_scores_df, date_col="date", ticker_col="ticker")
        train_metrics["evaluation_split"] = "holdout"
        train_metrics["search_rank_ic_mean"] = train_metrics.get("rank_ic_mean")
        train_metrics["final_rank_ic_mean"] = train_metrics.get("rank_ic_mean")
        train_notes.append("Continue mode: reused previously saved model without retraining.")
    else:
        if search_runs <= 0:
            raise ValueError("search_runs must be positive for training mode")

        search_result = run_search(
            search_X,
            search_y,
            args=args,
            split_mode=split_summary["split_mode"],
            indexed_splits=indexed_splits,
            base_config=best_config,
            date_col="date",
            ticker_col="ticker",
            search_runs=search_runs,
        )
        best_config = search_result.best_config
        best_seed = search_result.best_seed
        search_summary = {
            "n_candidates": len(search_result.candidates),
            "selection_metric": args.selection_metric,
            "best_seed": best_seed,
            "best_model_type": best_config.get("model_type"),
        }
        search_scores_df = (
            search_result.best_oof_scores.copy()
            if search_result.best_oof_scores is not None
            else pd.DataFrame()
        )
        if not search_scores_df.empty:
            search_scores_df["split"] = "oof"

        final_model = search_result.best_model
        if final_model is None:
            final_model, _ = train_panel_model(
                search_X,
                search_y,
                best_config,
                date_col="date",
                ticker_col="ticker",
                seed=best_seed,
            )
            train_notes.append("Search did not return a reusable final model; retrained best configuration on search data.")
        else:
            train_notes.append("Reused best candidate final model trained during search.")

        search_metrics = dict(search_result.best_metrics)
        search_metrics["best_seed"] = best_seed
        search_metrics["search_runs_completed"] = len(search_result.candidates)

        holdout_scores_df = pd.DataFrame()
        if split_summary["use_holdout"] and holdout_X is not None and holdout_y is not None:
            holdout_pred = predict_panel(final_model, holdout_X, date_col="date", ticker_col="ticker")
            holdout_scores_df = align_scores_with_labels(
                holdout_pred,
                holdout_X,
                holdout_y,
                config=best_config,
                date_col="date",
                ticker_col="ticker",
            )
            holdout_scores_df["split"] = "holdout"
            final_eval_scores_df = holdout_scores_df
            final_metrics = evaluate_scores_df(final_eval_scores_df, date_col="date", ticker_col="ticker")
            train_metrics = dict(final_metrics)
            train_metrics["evaluation_split"] = "holdout"
            train_metrics["search_rank_ic_mean"] = search_metrics.get("rank_ic_mean")
            train_metrics["final_rank_ic_mean"] = final_metrics.get("rank_ic_mean")
            train_notes.append(f"Evaluated on holdout ({len(final_eval_scores_df)} samples)")
        else:
            final_eval_scores_df = search_scores_df
            train_metrics = dict(search_metrics)
            train_metrics["evaluation_split"] = "oof"
            train_metrics["search_rank_ic_mean"] = search_metrics.get("rank_ic_mean")
            train_metrics["final_rank_ic_mean"] = search_metrics.get("rank_ic_mean")
            train_notes.append(f"Evaluated on OOF ({len(final_eval_scores_df)} samples)")

        models_dir = os.path.join(experiment_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = save_trained_model(
            final_model,
            best_config,
            os.path.join(models_dir, f"model_{model_timestamp}"),
        )
        logger.info("Model saved to: %s", model_path)

    final_eval_scores_df = prepare_scores_for_backtest(
        panel_df,
        final_eval_scores_df,
        date_col="date",
        signal_date_col="signal_date",
        trade_date_col="trade_date",
        execution_lag_days=1,
        drop_untradable_signals=True,
    )
    report_scores_df = final_eval_scores_df[
        (final_eval_scores_df["trade_date"] >= report_start) & (final_eval_scores_df["trade_date"] <= report_end)
    ].copy()
    portfolio_config = PortfolioConfig(
        top_k=args.top_k,
        weighting=args.weighting,
        max_weight=args.max_weight,
        cash_buffer=args.cash_buffer,
        rebalance_freq=args.rebalance_freq,
    )
    backtest_result = backtest_from_scores(
        panel_df,
        report_scores_df,
        portfolio_config=portfolio_config,
        score_col="score",
        ticker_col="ticker",
        date_col="date",
        trade_date_col="trade_date",
        initial_cash=args.initial_cash,
        start_date=report_start,
        end_date=report_end,
    )
    logger.info("Backtest final equity: %s", f"{backtest_result.equity_curve['equity'].iloc[-1]:,.2f}")
    logger.info("Total trades: %s", len(backtest_result.trades))
    execution_stats = dict(backtest_result.execution_stats)
    train_metrics.update(
        {
            "orders_submitted": int(execution_stats.get("orders_submitted", 0)),
            "orders_filled": int(execution_stats.get("orders_filled", 0)),
            "orders_rejected": int(execution_stats.get("orders_rejected", 0)),
            "total_commission": float(execution_stats.get("total_commission", 0.0)),
            "total_slippage": float(execution_stats.get("total_slippage", 0.0)),
        }
    )
    reject_reasons = execution_stats.get("reject_reasons", {}) or {}
    for reason, count in sorted(reject_reasons.items()):
        train_metrics[f"reject_{reason}"] = int(count)
    if reject_reasons:
        train_notes.append(
            "Execution rejects: " + ", ".join(f"{reason}={count}" for reason, count in sorted(reject_reasons.items()))
        )

    benchmark_curve = None
    if args.benchmark_mode == "equal_weight":
        benchmark_curve = compute_buy_and_hold_benchmark(
            panel_df[(panel_df["date"] >= report_start) & (panel_df["date"] <= report_end)].copy(),
            initial_cash=args.initial_cash,
        )
        if not benchmark_curve.empty and not backtest_result.equity_curve.empty:
            train_metrics["benchmark_total_return"] = float(benchmark_curve["bnh_cum_return"].iloc[-1])
            train_metrics["excess_return_vs_benchmark"] = float(backtest_result.equity_curve["cum_return"].iloc[-1]) - float(
                benchmark_curve["bnh_cum_return"].iloc[-1]
            )

    if not final_eval_scores_df.empty and "label" in final_eval_scores_df.columns:
        ranking_metrics = compute_all_ranking_metrics(
            final_eval_scores_df,
            score_col="score",
            label_col="label",
            date_col="date",
            ticker_col="ticker",
        )
        train_metrics.update(asdict(ranking_metrics))

    artifacts_dir = os.path.join(experiment_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    report_scores_df.to_csv(os.path.join(artifacts_dir, "scores.csv"), index=False)
    final_eval_scores_df.to_csv(os.path.join(artifacts_dir, "scores_research.csv"), index=False)
    backtest_result.equity_curve.to_csv(os.path.join(artifacts_dir, "equity_curve.csv"), index=False)
    backtest_result.trades.to_csv(os.path.join(artifacts_dir, "trades.csv"), index=False)
    if benchmark_curve is not None and not benchmark_curve.empty:
        benchmark_curve.to_csv(os.path.join(artifacts_dir, "benchmark_equity_curve.csv"), index=False)

    report_config = {
        "basket": args.basket,
        "top_k": args.top_k,
        "weighting": args.weighting,
        "max_weight": args.max_weight,
        "rebalance_freq": args.rebalance_freq,
        "seed": args.seed,
        "n_folds": args.n_folds,
        "research_start_date": str(research_start),
        "research_end_date": str(research_end),
        "report_start_date": str(report_start),
        "report_end_date": str(report_end),
        "model_type": best_config.get("model_type"),
        "device": best_config.get("device", "n/a"),
        "benchmark_mode": args.benchmark_mode,
        "split_mode": split_summary["split_mode"],
        "use_holdout": split_summary["use_holdout"],
        "embargo_days": split_summary["embargo_days"],
        "n_outer_folds": split_summary["n_outer_folds"],
        "n_inner_folds": split_summary["n_inner_folds"],
        "evaluation_split": train_metrics.get("evaluation_split"),
    }

    excel_path = os.path.join(experiment_dir, "results.xlsx")
    save_to_excel(
        excel_path,
        equity_curve=backtest_result.equity_curve,
        orders=backtest_result.orders,
        trades=backtest_result.trades,
        positions=backtest_result.positions,
        scores_df=report_scores_df,
        benchmark_curve=benchmark_curve,
        execution_stats=execution_stats,
        config=report_config,
        metrics=train_metrics,
        log_notes=train_notes,
    )
    html_path = os.path.join(experiment_dir, "report.html")
    generate_html_report(
        html_path,
        equity_curve=backtest_result.equity_curve,
        benchmark_curve=benchmark_curve,
        execution_stats=execution_stats,
        metrics=train_metrics,
        config=report_config,
        basket_info={
            "basket_name": basket_meta.basket_name,
            "n_tickers": basket_meta.n_tickers,
            "date_range": f"{report_start} ~ {report_end}",
            "data_contract_version": DATA_CONTRACT_VERSION,
        },
        notes=train_notes,
    )

    create_manifest(
        experiment_dir,
        run_id=1,
        timestamp=datetime.now().isoformat(),
        git_commit_hash=get_git_commit_hash(),
        package_versions=get_package_versions(),
        seed=args.seed,
        data_info={
            "basket_path": basket_path,
            "n_tickers": basket_meta.n_tickers,
            "n_rows": basket_report.total_rows,
            "raw_date_range": basket_meta.date_range,
            "default_backtest_start_date": str(default_start),
            "default_backtest_end_date": str(default_end),
            "effective_backtest_start_date": str(report_start),
            "effective_backtest_end_date": str(report_end),
            "research_start_date": str(research_start),
            "research_end_date": str(research_end),
            "report_start_date": str(report_start),
            "report_end_date": str(report_end),
        },
        cli_args=vars(args),
        best_config=best_config,
        metrics=train_metrics,
        search_runs_completed=completed_search_runs + search_summary["n_candidates"],
        split_info=split_summary,
        search_summary=search_summary,
    )
    logger.info("Results saved to: %s", experiment_dir)
    logger.info("Done!")


if __name__ == "__main__":
    main()
