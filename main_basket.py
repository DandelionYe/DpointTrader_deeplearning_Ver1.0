from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

from backtester_engine import backtest_from_scores
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
from models import get_torch_runtime_info
from panel_builder import validate_panel
from panel_trainer import predict_panel, train_with_walkforward
from portfolio_builder import PortfolioConfig
from ranking_metrics import compute_all_ranking_metrics
from splitters import final_holdout_split_by_date, walkforward_splits_by_date
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A-share basket ML/DL Dpoint trader."
    )

    parser.add_argument("--basket", type=str, default=DEFAULT_BASKET_NAME)
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--basket_path", type=str, default=None)
    parser.add_argument("--file_pattern", type=str, default=DEFAULT_FILE_PATTERN)
    parser.add_argument(
        "--ticker_from",
        type=str,
        choices=["filename", "manifest"],
        default="filename",
    )
    parser.add_argument("--output_dir", type=str, default="./output_basket")

    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=4)

    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--rebalance_freq",
        type=str,
        choices=["daily", "weekly", "monthly"],
        default=DEFAULT_REBALANCE_FREQ,
    )
    parser.add_argument(
        "--weighting",
        type=str,
        choices=["equal", "score", "vol_inv"],
        default=DEFAULT_WEIGHTING,
    )
    parser.add_argument("--max_weight", type=float, default=DEFAULT_MAX_WEIGHT)
    parser.add_argument("--cash_buffer", type=float, default=0.05)

    parser.add_argument("--label_mode", type=str, default=DEFAULT_LABEL_MODE)
    parser.add_argument("--include_cross_section", type=int, default=1)

    parser.add_argument("--initial_cash", type=float, default=100000.0)
    parser.add_argument("--benchmark_mode", type=str, default="equal_weight")
    parser.add_argument("--backtest_start_date", type=str, default=None)
    parser.add_argument("--backtest_end_date", type=str, default=None)

    parser.add_argument("--mode", type=str, choices=["first", "continue"], default="first")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--additional_runs", type=int, default=50)

    parser.add_argument("--model_type", choices=["mlp", "xgb"], default="mlp")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cpu_threads", type=int, default=4)

    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--hidden_dims", type=str, default="1024,512,256")
    parser.add_argument("--dropout_rate", type=float, default=0.10)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8192)

    parser.add_argument("--xgb_n_estimators", type=int, default=200)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_subsample", type=float, default=0.9)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.9)

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
            "GPU will not be used because the current PyTorch runtime has no CUDA support. "
            "Current torch build: %s",
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
        "basket_name": args.basket,
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
    if args.model_type == "mlp":
        hidden_dims = [
            int(part.strip())
            for part in str(args.hidden_dims).split(",")
            if part.strip()
        ]
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
            },
        }

    xgb_params: Dict[str, Any] = {
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
    }
    return {
        "model_type": "xgb",
        "device": "cpu",
        "model_params": xgb_params,
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


def _parse_optional_date(date_str: Optional[str]) -> Optional[pd.Timestamp]:
    if date_str is None:
        return None
    ts = pd.Timestamp(date_str)
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {date_str}")
    return ts


def resolve_backtest_window(
    panel_df: pd.DataFrame,
    args: argparse.Namespace,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    first_trade_by_ticker = panel_df.groupby(ticker_col)[date_col].min()
    default_start = pd.Timestamp(first_trade_by_ticker.max())
    default_end = pd.Timestamp(panel_df[date_col].max())

    effective_start = _parse_optional_date(args.backtest_start_date) or default_start
    effective_end = _parse_optional_date(args.backtest_end_date) or default_end

    if effective_start > effective_end:
        raise ValueError(
            f"backtest_start_date {effective_start} cannot be later than backtest_end_date {effective_end}"
        )

    logger.info(
        "Universe-aligned default backtest window: (%s, %s)",
        default_start,
        default_end,
    )
    if effective_start < default_start:
        logger.warning(
            "Using backtest_start_date=%s earlier than the universe-aligned default %s. "
            "Some basket constituents will not be tradable during the early part of the backtest.",
            effective_start,
            default_start,
        )
    logger.info("Effective backtest window: (%s, %s)", effective_start, effective_end)
    return effective_start, effective_end, default_start, default_end


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

    prev_model_path: Optional[str] = None
    models_dir = os.path.join(prev_exp_dir, "models")
    if os.path.isdir(models_dir):
        model_files = sorted(
            os.path.join(models_dir, name)
            for name in os.listdir(models_dir)
            if name.endswith(".joblib")
        )
        if model_files:
            prev_model_path = model_files[-1]

    return prev_exp_dir, prev_best_config, prev_model_path, completed_search_runs


def main() -> None:
    args = parse_args()

    set_global_seed(args.seed)
    logger.info("Global seed set to %s", args.seed)
    log_runtime_status(args)

    basket_path = args.basket_path or resolve_basket_path(args.data_root, args.basket)
    logger.info("Basket path: %s", basket_path)
    logger.info("Output dir: %s", args.output_dir)

    panel_df, basket_report, basket_meta = load_and_validate_data(
        basket_path,
        args.file_pattern,
        args.ticker_from,
    )
    effective_start, effective_end, default_start, default_end = resolve_backtest_window(
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

    os.makedirs(args.output_dir, exist_ok=True)

    prev_exp_dir, prev_best_config, prev_model_path, completed_search_runs = load_previous_experiment(args)
    continue_mode = prev_exp_dir is not None
    if continue_mode:
        logger.info("Continue mode - source experiment: %s", prev_exp_dir)
        if prev_model_path:
            logger.info("Previous model found: %s", prev_model_path)
        logger.info("Completed search runs from previous manifest: %s", completed_search_runs)

    experiment_dir = prev_exp_dir if continue_mode else create_experiment_dir(
        args.output_dir, _next_experiment_id(args.output_dir)
    )
    logger.info("Experiment dir: %s", experiment_dir)

    logger.info("Building features and labels...")
    feature_config = build_feature_config(args)
    X, y, feature_meta = build_features_and_labels_panel(
        panel_df,
        feature_config,
        date_col="date",
        ticker_col="ticker",
        label_mode=args.label_mode,
        include_cross_section=bool(args.include_cross_section),
    )
    X, y = filter_feature_window(
        X,
        y,
        effective_start,
        effective_end,
        date_col="date",
    )
    logger.info("Features: %s", len(feature_meta.feature_names))
    logger.info("Samples: %s", len(X))
    logger.info("Tickers: %s", feature_meta.n_tickers)
    logger.info(
        "Windowed samples: %s rows across %s dates",
        len(X),
        X["date"].nunique(),
    )
    if X.empty:
        raise ValueError(
            f"No samples remain inside backtest window [{effective_start}, {effective_end}]"
        )

    logger.info("Generating walk-forward splits...")
    splits = walkforward_splits_by_date(
        X,
        date_col="date",
        ticker_col="ticker",
        n_folds=args.n_folds,
        train_start_ratio=0.5,
    )
    indexed_splits = [dates_to_indices(X, train_dates, val_dates) for train_dates, val_dates in splits]
    logger.info("Number of folds: %s", len(indexed_splits))
    if not indexed_splits:
        raise ValueError(
            "No walk-forward splits were generated for the selected backtest window. "
            "Use an earlier backtest_start_date or reduce n_folds."
        )

    model_config = prev_best_config or build_model_config(args)
    train_notes: List[str] = []
    train_metrics: Dict[str, Any] = {}

    search_runs = args.additional_runs if continue_mode else args.runs

    if continue_mode and prev_model_path and search_runs <= 0:
        logger.info("Loading existing model for evaluation only: %s", prev_model_path)
        model = joblib.load(prev_model_path)
        _, holdout_X = final_holdout_split_by_date(
            X,
            date_col="date",
            holdout_ratio=0.15,
            min_holdout_rows=60,
        )
        holdout_y = y.loc[holdout_X.index]
        scores_df = predict_panel(model, holdout_X, date_col="date", ticker_col="ticker")
        scores_df["label"] = holdout_y.values
        scores_df["split"] = "holdout"
        train_notes.append("Continue mode: reused previously saved model without retraining.")
    else:
        logger.info(
            "Training model type=%s device=%s search_runs=%s",
            model_config.get("model_type"),
            model_config.get("device", "n/a"),
            search_runs,
        )
        if search_runs <= 0:
            raise ValueError("search_runs must be positive for training mode")

        best_rank_ic = float("-inf")
        best_seed = None
        best_train_result = None

        for run_idx in range(search_runs):
            run_seed = args.seed + completed_search_runs + run_idx
            logger.info("Search run %s/%s with seed=%s", run_idx + 1, search_runs, run_seed)
            candidate_result = train_with_walkforward(
                X,
                y,
                model_config,
                indexed_splits,
                date_col="date",
                ticker_col="ticker",
                seed=run_seed,
            )
            candidate_rank_ic = candidate_result.val_metrics.get("rank_ic_mean")
            candidate_rank_ic = float(candidate_rank_ic) if candidate_rank_ic is not None else float("-inf")
            logger.info("Search run %s validation RankIC: %s", run_idx + 1, candidate_rank_ic)
            if candidate_rank_ic > best_rank_ic:
                best_rank_ic = candidate_rank_ic
                best_seed = run_seed
                best_train_result = candidate_result

        if best_train_result is None:
            raise RuntimeError("Training produced no valid candidate result")

        train_result = best_train_result
        model = train_result.model
        scores_df = train_result.oof_scores.copy() if train_result.oof_scores is not None else pd.DataFrame()
        train_notes = list(train_result.notes)
        train_notes.append(f"Best search seed: {best_seed}")
        train_metrics = dict(train_result.val_metrics)
        train_metrics["best_seed"] = best_seed
        train_metrics["search_runs_completed"] = completed_search_runs + search_runs
        logger.info("Best validation RankIC: %s", train_metrics.get("rank_ic_mean", "N/A"))
        logger.info("OOF scores shape: %s", scores_df.shape)

        models_dir = os.path.join(experiment_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
        joblib.dump(model, model_path)
        logger.info("Model saved to: %s", model_path)

    logger.info("Building portfolio and backtesting...")
    portfolio_config = PortfolioConfig(
        top_k=args.top_k,
        weighting=args.weighting,
        max_weight=args.max_weight,
        cash_buffer=args.cash_buffer,
        rebalance_freq=args.rebalance_freq,
    )
    backtest_result = backtest_from_scores(
        panel_df,
        scores_df,
        portfolio_config=portfolio_config,
        score_col="score",
        ticker_col="ticker",
        date_col="date",
        initial_cash=args.initial_cash,
        start_date=effective_start,
        end_date=effective_end,
    )
    logger.info("Backtest final equity: %s", f"{backtest_result.equity_curve['equity'].iloc[-1]:,.2f}")
    logger.info("Total trades: %s", len(backtest_result.trades))

    if not scores_df.empty and "label" in scores_df.columns:
        ranking_metrics = compute_all_ranking_metrics(
            scores_df,
            score_col="score",
            label_col="label",
            date_col="date",
            ticker_col="ticker",
        )
        train_metrics.update(asdict(ranking_metrics))
        logger.info("RankIC: %.4f", ranking_metrics.rank_ic_mean)
        if args.label_mode == "regression_return":
            logger.info("TopK Return (annual): %s", f"{ranking_metrics.topk_return_annual:.2%}")
        else:
            logger.info("TopK hit rate: %.4f", ranking_metrics.topk_return_mean)

    logger.info("Saving results...")
    excel_path = os.path.join(experiment_dir, "results.xlsx")
    save_to_excel(
        excel_path,
        equity_curve=backtest_result.equity_curve,
        orders=backtest_result.orders,
        trades=backtest_result.trades,
        positions=backtest_result.positions,
        scores_df=scores_df,
        config={
            "basket": args.basket,
            "top_k": args.top_k,
            "weighting": args.weighting,
            "max_weight": args.max_weight,
            "rebalance_freq": args.rebalance_freq,
            "seed": args.seed,
            "n_folds": args.n_folds,
            "backtest_start_date": str(effective_start),
            "backtest_end_date": str(effective_end),
            "model_type": model_config.get("model_type"),
            "device": model_config.get("device", "n/a"),
        },
        metrics=train_metrics,
        log_notes=train_notes,
    )

    html_path = os.path.join(experiment_dir, "report.html")
    generate_html_report(
        html_path,
        equity_curve=backtest_result.equity_curve,
        metrics=train_metrics,
        config={
            "basket": args.basket,
            "top_k": args.top_k,
            "weighting": args.weighting,
            "backtest_start_date": str(effective_start),
            "backtest_end_date": str(effective_end),
            "model_type": model_config.get("model_type"),
            "device": model_config.get("device", "n/a"),
        },
        basket_info={
            "basket_name": basket_meta.basket_name,
            "n_tickers": basket_meta.n_tickers,
            "date_range": f"{effective_start} ~ {effective_end}",
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
            "effective_backtest_start_date": str(effective_start),
            "effective_backtest_end_date": str(effective_end),
        },
        cli_args=vars(args),
        best_config=model_config,
        metrics=train_metrics,
        search_runs_completed=completed_search_runs + max(search_runs, 0),
    )

    logger.info("Results saved to: %s", experiment_dir)
    logger.info("Done!")


if __name__ == "__main__":
    main()
