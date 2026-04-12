from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd


def get_git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


def get_package_versions() -> Dict[str, str]:
    packages = [
        "torch",
        "numpy",
        "pandas",
        "sklearn",
        "joblib",
        "xgboost",
        "openpyxl",
        "xlsxwriter",
    ]
    versions: Dict[str, str] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
    versions["python"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    return versions


def set_global_seed(seed: int) -> Dict[str, Any]:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return {
        "seed": seed,
        "python_hashseed": str(seed),
        "torch_deterministic": True,
        "torch_benchmark": False,
    }


def compute_data_hash(df: pd.DataFrame) -> str:
    raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(raw).hexdigest()


def resolve_basket_path(data_root: str, basket_name: str) -> str:
    basket_path = os.path.join(data_root, basket_name)
    if not os.path.isdir(basket_path):
        raise FileNotFoundError(f"Basket not found: {basket_path}")
    return basket_path


def create_experiment_dir(output_dir: str, experiment_id: int) -> str:
    exp_dir = os.path.join(output_dir, f"exp_{experiment_id:03d}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "artifacts"), exist_ok=True)
    return exp_dir


def create_snapshot_dir(experiment_dir: str, snapshot_id: str) -> str:
    snapshot_dir = os.path.join(experiment_dir, "snapshots", snapshot_id)
    os.makedirs(snapshot_dir, exist_ok=True)
    return snapshot_dir


def create_manifest(
    experiment_dir: str,
    run_id: int,
    timestamp: str,
    git_commit_hash: str,
    package_versions: Dict[str, str],
    seed: int,
    data_info: Dict[str, Any],
    cli_args: Dict[str, Any],
    best_config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    search_runs_completed: Optional[int] = None,
    split_info: Optional[Dict[str, Any]] = None,
    search_summary: Optional[Dict[str, Any]] = None,
    contracts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "manifest_version": "1.0",
        "run_id": run_id,
        "experiment_id": run_id,
        "created_at": timestamp,
        "git_commit_hash": git_commit_hash,
        "package_versions": package_versions,
        "seed": seed,
        "data": {
            "data_path": data_info.get("data_path"),
            "data_hash": data_info.get("data_hash"),
            "n_rows": data_info.get("n_rows"),
            "n_columns": data_info.get("n_columns"),
            "date_range": data_info.get("date_range"),
            "tickers": data_info.get("tickers", []),
            "columns": data_info.get("columns", []),
            "basket_path": data_info.get("basket_path"),
            "raw_date_range": data_info.get("raw_date_range"),
            "default_backtest_start_date": data_info.get("default_backtest_start_date"),
            "default_backtest_end_date": data_info.get("default_backtest_end_date"),
            "effective_backtest_start_date": data_info.get("effective_backtest_start_date"),
            "effective_backtest_end_date": data_info.get("effective_backtest_end_date"),
            "research_start_date": data_info.get("research_start_date"),
            "research_end_date": data_info.get("research_end_date"),
            "report_start_date": data_info.get("report_start_date"),
            "report_end_date": data_info.get("report_end_date"),
            "n_tickers": data_info.get("n_tickers"),
            "train_end_date": data_info.get("train_end_date"),
        },
        "cli_args": cli_args,
    }
    if contracts is not None:
        manifest["contracts"] = contracts
        data_section = cast(Dict[str, Any], manifest["data"])
        data_section["data_hash"] = data_section.get("data_hash") or contracts.get("data", {}).get(
            "data_hash"
        )

    if best_config is not None:
        manifest["best_config"] = best_config
    if metrics is not None:
        manifest["metrics"] = metrics
        manifest["evaluation_split"] = metrics.get("evaluation_split", "oof")
    if search_runs_completed is not None:
        manifest["search_runs_completed"] = int(search_runs_completed)
    if split_info is not None:
        manifest["split_info"] = split_info
    if search_summary is not None:
        manifest["search_summary"] = search_summary

    manifest_path = os.path.join(experiment_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def load_manifest(experiment_dir: str) -> Optional[Dict[str, Any]]:
    manifest_path = os.path.join(experiment_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(experiment_dir: str) -> Optional[Dict[str, Any]]:
    config_path = os.path.join(experiment_dir, "config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_experiment(output_dir: str) -> Optional[Tuple[int, str]]:
    if not os.path.isdir(output_dir):
        return None
    candidates: List[Tuple[int, str]] = []
    for dn in os.listdir(output_dir):
        if dn.startswith("exp_") and os.path.isdir(os.path.join(output_dir, dn)):
            try:
                candidates.append((int(dn.split("_")[1]), os.path.join(output_dir, dn)))
            except (IndexError, ValueError):
                continue
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[0])[-1]


__all__ = [
    "compute_data_hash",
    "create_experiment_dir",
    "create_manifest",
    "create_snapshot_dir",
    "find_latest_experiment",
    "get_git_commit_hash",
    "get_package_versions",
    "load_config",
    "load_manifest",
    "resolve_basket_path",
    "set_global_seed",
]
