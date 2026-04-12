from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import pandas as pd

from utils import compute_data_hash


@dataclass
class DataContract:
    data_hash: str
    date_min: str
    date_max: str
    n_rows: int
    n_tickers: int


@dataclass
class FeatureContract:
    feature_names: list[str]
    feature_schema_hash: str
    include_cross_section: bool
    seq_len: Optional[int]
    feature_config_hash: str


@dataclass
class TrainingContract:
    task_type: str
    label_mode: str
    label_horizon_days: int
    model_type: str
    target_version: str


@dataclass
class RunContract:
    data: DataContract
    feature: FeatureContract
    training: TrainingContract


class ContinueCompatibilityError(ValueError):
    pass


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _task_type_from_label_mode(label_mode: str) -> str:
    normalized = str(label_mode).lower()
    if normalized in {"classification", "binary", "binary_next_close_up"}:
        return "binary_classification"
    if normalized in {"regression", "regression_return"}:
        return "regression"
    return "unknown"


def compute_feature_schema_hash(
    feature_names: list[str], extra: Optional[Dict[str, Any]] = None
) -> str:
    payload = {
        "feature_names": list(feature_names),
        "extra": extra or {},
    }
    return _stable_hash(payload)


def compute_feature_config_hash(feature_cfg: Dict[str, Any]) -> str:
    return _stable_hash(feature_cfg)


def contract_to_dict(contract: RunContract) -> Dict[str, Any]:
    return asdict(contract)


def build_run_contract(
    panel_df: pd.DataFrame,
    *,
    feature_meta: Any,
    args: Any,
    model_config: Dict[str, Any],
    feature_config: Optional[Dict[str, Any]] = None,
) -> RunContract:
    sorted_panel = panel_df.copy()
    sort_cols = [col for col in ["date", "ticker"] if col in sorted_panel.columns]
    if sort_cols:
        sorted_panel = sorted_panel.sort_values(sort_cols).reset_index(drop=True)

    date_min = ""
    date_max = ""
    if "date" in sorted_panel.columns and not sorted_panel.empty:
        date_min = str(pd.to_datetime(sorted_panel["date"]).min())
        date_max = str(pd.to_datetime(sorted_panel["date"]).max())

    feature_names = list(getattr(feature_meta, "feature_names", []))
    feature_params = dict(getattr(feature_meta, "params", {}))
    include_cross_section = bool(
        feature_params.get("include_cross_section", getattr(args, "include_cross_section", 1))
    )
    model_params = dict(model_config.get("model_params", {}))
    seq_len = model_params.get("seq_len")
    label_mode = str(
        getattr(feature_meta, "label_mode", getattr(args, "label_mode", "binary_next_close_up"))
    )
    task_type = str(getattr(args, "task_type", "") or _task_type_from_label_mode(label_mode))
    label_horizon_days = max(1, int(getattr(args, "label_horizon_days", 1)))

    return RunContract(
        data=DataContract(
            data_hash=compute_data_hash(sorted_panel),
            date_min=date_min,
            date_max=date_max,
            n_rows=int(len(sorted_panel)),
            n_tickers=int(sorted_panel["ticker"].nunique())
            if "ticker" in sorted_panel.columns
            else 0,
        ),
        feature=FeatureContract(
            feature_names=feature_names,
            feature_schema_hash=compute_feature_schema_hash(
                feature_names,
                extra={"seq_len": seq_len, "include_cross_section": include_cross_section},
            ),
            include_cross_section=include_cross_section,
            seq_len=int(seq_len) if seq_len is not None else None,
            feature_config_hash=compute_feature_config_hash(feature_config or feature_params),
        ),
        training=TrainingContract(
            task_type=task_type,
            label_mode=label_mode,
            label_horizon_days=label_horizon_days,
            model_type=str(model_config.get("model_type", "")),
            target_version="1",
        ),
    )


def _load_model_contract(previous_model_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not previous_model_path:
        return None
    candidate_paths: list[str] = []
    if os.path.isdir(previous_model_path):
        candidate_paths.append(os.path.join(previous_model_path, "model_contract.json"))
    else:
        root, _ = os.path.splitext(previous_model_path)
        candidate_paths.append(f"{root}.contract.json")
        candidate_paths.append(f"{previous_model_path}.contract.json")
    for path in candidate_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def _extract_saved_contract(
    previous_manifest: Optional[Dict[str, Any]],
    previous_model_path: Optional[str],
) -> Optional[Dict[str, Any]]:
    if previous_manifest and previous_manifest.get("contracts"):
        return previous_manifest["contracts"]
    return _load_model_contract(previous_model_path)


def validate_continue_compatibility(
    *,
    current_contract: RunContract,
    previous_manifest: Optional[Dict[str, Any]],
    previous_model_path: Optional[str] = None,
    strict: bool = True,
    allow_feature_contract_mismatch: bool = False,
    allow_data_contract_mismatch: bool = False,
) -> None:
    saved_contract = _extract_saved_contract(previous_manifest, previous_model_path)
    if saved_contract is None:
        if strict:
            raise ContinueCompatibilityError(
                "Previous experiment is missing contract metadata. "
                "Re-run the source experiment with contract export or disable strict mode."
            )
        return

    current = contract_to_dict(current_contract)
    mismatches: list[str] = []

    if (
        not allow_data_contract_mismatch
        and current["data"]["data_hash"] != saved_contract["data"]["data_hash"]
    ):
        mismatches.append(
            "Data contract mismatch: data_hash differs "
            f"(saved={saved_contract['data']['data_hash']}, current={current['data']['data_hash']})"
        )

    feature_checks = [
        ("feature_schema_hash", "Feature contract mismatch"),
        ("feature_config_hash", "Feature contract mismatch"),
        ("include_cross_section", "Feature contract mismatch"),
    ]
    if not allow_feature_contract_mismatch:
        for key, label in feature_checks:
            if current["feature"][key] != saved_contract["feature"][key]:
                mismatches.append(
                    f"{label}: {key} differs "
                    f"(saved={saved_contract['feature'][key]}, current={current['feature'][key]})"
                )
        if current["feature"]["feature_names"] != saved_contract["feature"]["feature_names"]:
            mismatches.append("Feature contract mismatch: feature_names order differs")

    training_checks = [
        ("task_type", "Training contract mismatch"),
        ("label_mode", "Training contract mismatch"),
        ("label_horizon_days", "Training contract mismatch"),
        ("model_type", "Training contract mismatch"),
        ("target_version", "Training contract mismatch"),
    ]
    for key, label in training_checks:
        if current["training"][key] != saved_contract["training"][key]:
            mismatches.append(
                f"{label}: {key} differs "
                f"(saved={saved_contract['training'][key]}, current={current['training'][key]})"
            )

    saved_seq_len = saved_contract["feature"].get("seq_len")
    current_seq_len = current["feature"].get("seq_len")
    if saved_contract["training"].get("model_type") in {"lstm", "gru", "cnn", "transformer"}:
        if current_seq_len != saved_seq_len:
            mismatches.append(
                "Feature contract mismatch: seq_len differs "
                f"(saved={saved_seq_len}, current={current_seq_len})"
            )

    if mismatches:
        raise ContinueCompatibilityError("; ".join(mismatches))
