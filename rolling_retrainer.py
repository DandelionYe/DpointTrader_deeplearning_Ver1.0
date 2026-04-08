from __future__ import annotations

import argparse
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from backtester_engine import backtest_from_scores, prepare_scores_for_backtest
from models import save_trained_model
from panel_trainer import align_scores_with_labels, evaluate_scores_df, predict_panel, train_panel_model
from portfolio_builder import PortfolioConfig
from search_engine import run_search
from utils import create_manifest, create_snapshot_dir

logger = logging.getLogger(__name__)


@dataclass
class RollingConfig:
    window_type: str = "expanding"
    rolling_window_length: Optional[int] = None
    retrain_frequency: str = "monthly"
    min_history_days: int = 120


@dataclass
class SnapshotManifest:
    snapshot_id: str
    train_end_date: str
    model_path: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]


class RollingRetrainer:
    def __init__(self, config: RollingConfig, experiment_dir: str):
        self.config = config
        self.experiment_dir = experiment_dir
        self.snapshots_dir = os.path.join(experiment_dir, "snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)

    def iter_retrain_dates(self, panel_df: pd.DataFrame) -> List[pd.Timestamp]:
        if self.config.retrain_frequency != "monthly":
            raise ValueError(f"Unsupported retrain_frequency: {self.config.retrain_frequency}")
        unique_dates = sorted(pd.to_datetime(panel_df["date"].unique()))
        if not unique_dates:
            return []
        max_date = unique_dates[-1]
        retrain_dates: List[pd.Timestamp] = []
        min_history_dates = max(1, int(self.config.min_history_days))
        if self.config.window_type == "rolling" and self.config.rolling_window_length is not None:
            min_history_dates = max(min_history_dates, int(self.config.rolling_window_length))
        if len(unique_dates) < min_history_dates:
            return []
        min_date = unique_dates[min_history_dates - 1]
        current = pd.Timestamp(min_date.year, min_date.month, 1) + pd.DateOffset(months=1)
        while current <= max_date:
            valid_dates = [d for d in unique_dates if d <= current]
            if len(valid_dates) >= min_history_dates:
                retrain_dates.append(pd.Timestamp(valid_dates[-1]))
            current += pd.DateOffset(months=1)
        return retrain_dates

    def get_training_window(self, panel_df: pd.DataFrame, retrain_date: pd.Timestamp) -> pd.DataFrame:
        if self.config.window_type == "expanding":
            return panel_df[panel_df["date"] <= retrain_date].copy()
        if self.config.window_type == "rolling":
            if self.config.rolling_window_length is None:
                raise ValueError("rolling_window_length must be set for rolling window")
            unique_dates = sorted(pd.to_datetime(panel_df["date"].unique()))
            eligible_dates = [date for date in unique_dates if date <= retrain_date]
            if not eligible_dates:
                return panel_df.iloc[0:0].copy()
            window_dates = eligible_dates[-self.config.rolling_window_length :]
            return panel_df[panel_df["date"].isin(window_dates)].copy()
        raise ValueError(f"Invalid window_type: {self.config.window_type}")

    def run(self, panel_df: pd.DataFrame, args: argparse.Namespace) -> List[SnapshotManifest]:
        from main_basket import build_feature_config, build_model_config, build_split_plan, resolve_label_mode_alias

        retrain_dates = self.iter_retrain_dates(panel_df)
        if not retrain_dates:
            logger.warning("No retrain dates generated")
            return []

        snapshots: List[SnapshotManifest] = []
        for snapshot_idx, retrain_date in enumerate(retrain_dates):
            logger.info("=== Snapshot %d/%d: retrain_date=%s ===", snapshot_idx + 1, len(retrain_dates), retrain_date)
            next_retrain_date = retrain_dates[snapshot_idx + 1] if snapshot_idx + 1 < len(retrain_dates) else None
            eval_end_date = next_retrain_date if next_retrain_date is not None else pd.Timestamp(panel_df["date"].max())
            train_panel = self.get_training_window(panel_df, retrain_date)
            eval_panel = panel_df[(panel_df["date"] > retrain_date) & (panel_df["date"] <= eval_end_date)].copy()
            if train_panel.empty:
                continue
            if eval_panel.empty:
                logger.info("Skipping snapshot %s because no forward evaluation data exists", retrain_date)
                continue

            feature_config = build_feature_config(args)
            from feature_dpoint import build_features_and_labels_panel

            X, y, _ = build_features_and_labels_panel(
                panel_df[panel_df["date"] <= eval_end_date].copy(),
                feature_config,
                date_col="date",
                ticker_col="ticker",
                label_mode=resolve_label_mode_alias(args.label_mode),
                include_cross_section=bool(args.include_cross_section),
            )
            if X.empty:
                continue

            train_X = X[X["date"] <= retrain_date].copy()
            train_y = y.loc[train_X.index].copy()
            eval_X = X[(X["date"] > retrain_date) & (X["date"] <= eval_end_date)].copy()
            if train_X.empty or eval_X.empty:
                continue
            eval_y = y.loc[eval_X.index].copy()

            search_args = deepcopy(args)
            search_args.use_holdout = 0
            split_plan = build_split_plan(train_X, train_y, search_args, date_col="date", ticker_col="ticker")
            indexed_splits = split_plan["indexed_splits"]
            search_X = split_plan["search_X"]
            search_y = split_plan["search_y"]
            if not indexed_splits:
                continue

            model_config = build_model_config(args)
            search_result = run_search(
                search_X,
                search_y,
                args=args,
                split_mode=split_plan["split_summary"]["split_mode"],
                indexed_splits=indexed_splits,
                base_config=model_config,
                date_col="date",
                ticker_col="ticker",
                search_runs=max(1, min(int(getattr(args, "runs", 1)), 3)),
            )
            final_model, _ = train_panel_model(
                train_X,
                train_y,
                search_result.best_config,
                date_col="date",
                ticker_col="ticker",
                seed=search_result.best_seed,
            )

            eval_pred = predict_panel(final_model, eval_X, date_col="date", ticker_col="ticker")
            scores_df = align_scores_with_labels(
                eval_pred,
                eval_X,
                eval_y,
                config=search_result.best_config,
                date_col="date",
                ticker_col="ticker",
            )
            scores_df["split"] = "forward_eval"

            snapshot_id = f"snapshot_{snapshot_idx + 1:03d}"
            snapshot_dir = create_snapshot_dir(self.experiment_dir, snapshot_id)
            model_path = os.path.join(snapshot_dir, "model.joblib")
            scores_path = os.path.join(snapshot_dir, "scores.csv")
            equity_path = os.path.join(snapshot_dir, "equity_curve.csv")

            model_path = save_trained_model(final_model, search_result.best_config, model_path)
            scores_df, prep_stats = prepare_scores_for_backtest(
                panel_df,
                scores_df,
                date_col="date",
                signal_date_col="signal_date",
                trade_date_col="trade_date",
                execution_lag_days=int(getattr(args, "execution_lag_days", 1)),
                return_stats=True,
            )
            scores_df.to_csv(scores_path, index=False)

            portfolio_config = PortfolioConfig(
                top_k=getattr(args, "top_k", 5),
                weighting=getattr(args, "weighting", "equal"),
                max_weight=getattr(args, "max_weight", 0.2),
                cash_buffer=getattr(args, "cash_buffer", 0.05),
                rebalance_freq=getattr(args, "rebalance_freq", "monthly"),
            )
            backtest_result = backtest_from_scores(
                panel_df,
                scores_df,
                portfolio_config=portfolio_config,
                initial_cash=float(getattr(args, "initial_cash", 100000.0)),
                start_date=pd.Timestamp(scores_df["trade_date"].min()),
                end_date=pd.Timestamp(scores_df["trade_date"].max()),
                trade_date_col="trade_date",
                signal_date_col="signal_date",
                rebalance_anchor=str(getattr(args, "rebalance_anchor", "first")),
            )
            equity_curve = backtest_result.equity_curve.copy()
            equity_curve.to_csv(equity_path, index=False)

            metrics = evaluate_scores_df(scores_df, date_col="date", ticker_col="ticker")
            metrics["search_rank_ic_mean"] = float(search_result.best_metrics.get("rank_ic_mean", 0.0))
            metrics["final_rank_ic_mean"] = float(metrics.get("rank_ic_mean", 0.0))
            metrics["evaluation_split"] = "forward_eval"
            metrics["evaluation_start_date"] = str(scores_df["trade_date"].min())
            metrics["evaluation_end_date"] = str(scores_df["trade_date"].max())
            metrics["raw_signals"] = int(prep_stats.get("raw_signals", 0))
            metrics["prepared_signals"] = int(prep_stats.get("prepared_signals", 0))
            metrics["dropped_signals"] = int(prep_stats.get("dropped_signals", 0))
            metrics["execution_lag_days"] = int(prep_stats.get("execution_lag_days", getattr(args, "execution_lag_days", 1)))
            create_manifest(
                snapshot_dir,
                run_id=snapshot_idx + 1,
                timestamp=datetime.now().isoformat(),
                git_commit_hash="rolling_retrain",
                package_versions={},
                seed=args.seed,
                data_info={
                    "train_end_date": str(retrain_date),
                    "n_rows": len(train_X),
                    "n_tickers": train_X["ticker"].nunique(),
                },
                cli_args=vars(args),
                best_config=search_result.best_config,
                metrics=metrics,
                search_runs_completed=len(search_result.candidates),
                split_info=split_plan["split_summary"],
                search_summary={
                    "n_candidates": len(search_result.candidates),
                    "selection_metric": args.selection_metric,
                    "best_seed": search_result.best_seed,
                    "best_model_type": search_result.best_config.get("model_type"),
                },
            )

            snapshots.append(
                SnapshotManifest(
                    snapshot_id=snapshot_id,
                    train_end_date=str(retrain_date),
                    model_path=model_path,
                    config=search_result.best_config,
                    metrics=metrics,
                )
            )

        logger.info("Rolling retrain completed: %d snapshots", len(snapshots))
        return snapshots

    @staticmethod
    def _build_snapshot_equity_curve(scores_df: pd.DataFrame) -> pd.DataFrame:
        if scores_df.empty or "date" not in scores_df.columns or "score" not in scores_df.columns:
            return pd.DataFrame({"date": [], "equity": []})
        grouped = scores_df.groupby("date", as_index=False)["score"].mean().sort_values("date")
        grouped["equity"] = 100000.0 + grouped["score"].fillna(0.0).cumsum()
        return grouped[["date", "equity"]]
