import argparse

import pandas as pd
import pytest

from experiment_contract import (
    ContinueCompatibilityError,
    build_run_contract,
    contract_to_dict,
    validate_continue_compatibility,
)
from feature_dpoint import PanelFeatureMeta


def _sample_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    rows = []
    for ticker in ["A", "B"]:
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open_qfq": 10.0 + idx,
                    "high_qfq": 10.5 + idx,
                    "low_qfq": 9.5 + idx,
                    "close_qfq": 10.2 + idx,
                    "volume": 1_000_000 + idx,
                }
            )
    return pd.DataFrame(rows)


def _feature_meta(*, feature_names=None, label_mode="binary_next_close_up", include_cross_section=True) -> PanelFeatureMeta:
    return PanelFeatureMeta(
        feature_names=list(feature_names or ["f1", "f2"]),
        label_mode=label_mode,
        params={
            "include_cross_section": include_cross_section,
            "windows": [5, 10],
        },
    )


def _args(**overrides):
    base = {
        "label_mode": "classification",
        "label_horizon_days": 1,
        "include_cross_section": 1,
        "seq_len": 20,
        "report_start_date": "2024-01-01",
        "report_end_date": "2024-01-31",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _model_config(model_type="xgb", **model_params):
    return {
        "model_type": model_type,
        "model_params": model_params,
    }


def _manifest_for(contract):
    return {"contracts": contract_to_dict(contract)}


def test_continue_validation_accepts_matching_contract():
    panel_df = _sample_panel()
    contract = build_run_contract(
        panel_df,
        feature_meta=_feature_meta(),
        args=_args(),
        model_config=_model_config("xgb"),
        feature_config={"windows": [5, 10], "include_cross_section": True},
    )

    validate_continue_compatibility(
        current_contract=contract,
        previous_manifest=_manifest_for(contract),
    )


def test_continue_validation_rejects_data_hash_mismatch():
    panel_df = _sample_panel()
    saved_contract = build_run_contract(
        panel_df,
        feature_meta=_feature_meta(),
        args=_args(),
        model_config=_model_config("xgb"),
    )
    changed_panel = panel_df.copy()
    changed_panel.loc[0, "close_qfq"] += 1.0
    current_contract = build_run_contract(
        changed_panel,
        feature_meta=_feature_meta(),
        args=_args(),
        model_config=_model_config("xgb"),
    )

    with pytest.raises(ContinueCompatibilityError, match="data_hash differs"):
        validate_continue_compatibility(
            current_contract=current_contract,
            previous_manifest=_manifest_for(saved_contract),
        )


def test_continue_validation_rejects_feature_order_mismatch():
    panel_df = _sample_panel()
    saved_contract = build_run_contract(
        panel_df,
        feature_meta=_feature_meta(feature_names=["f1", "f2", "f3"]),
        args=_args(),
        model_config=_model_config("xgb"),
    )
    current_contract = build_run_contract(
        panel_df,
        feature_meta=_feature_meta(feature_names=["f1", "f3", "f2"]),
        args=_args(),
        model_config=_model_config("xgb"),
    )

    with pytest.raises(ContinueCompatibilityError, match="feature_names order differs"):
        validate_continue_compatibility(
            current_contract=current_contract,
            previous_manifest=_manifest_for(saved_contract),
        )


def test_continue_validation_rejects_task_or_label_mismatch():
    panel_df = _sample_panel()
    saved_contract = build_run_contract(
        panel_df,
        feature_meta=_feature_meta(label_mode="binary_next_close_up"),
        args=_args(label_mode="classification"),
        model_config=_model_config("xgb"),
    )
    current_contract = build_run_contract(
        panel_df,
        feature_meta=_feature_meta(label_mode="regression_return"),
        args=_args(label_mode="regression"),
        model_config=_model_config("xgb"),
    )

    with pytest.raises(ContinueCompatibilityError, match="label_mode differs"):
        validate_continue_compatibility(
            current_contract=current_contract,
            previous_manifest=_manifest_for(saved_contract),
        )


def test_continue_validation_rejects_sequence_length_mismatch():
    panel_df = _sample_panel()
    saved_contract = build_run_contract(
        panel_df,
        feature_meta=_feature_meta(),
        args=_args(seq_len=10),
        model_config=_model_config("lstm", seq_len=10),
    )
    current_contract = build_run_contract(
        panel_df,
        feature_meta=_feature_meta(),
        args=_args(seq_len=20),
        model_config=_model_config("lstm", seq_len=20),
    )

    with pytest.raises(ContinueCompatibilityError, match="seq_len differs"):
        validate_continue_compatibility(
            current_contract=current_contract,
            previous_manifest=_manifest_for(saved_contract),
        )


def test_continue_validation_allows_report_parameter_changes():
    panel_df = _sample_panel()
    saved_contract = build_run_contract(
        panel_df,
        feature_meta=_feature_meta(),
        args=_args(report_start_date="2024-01-01", report_end_date="2024-01-15"),
        model_config=_model_config("xgb"),
    )
    current_contract = build_run_contract(
        panel_df,
        feature_meta=_feature_meta(),
        args=_args(report_start_date="2024-01-10", report_end_date="2024-01-31"),
        model_config=_model_config("xgb"),
    )

    validate_continue_compatibility(
        current_contract=current_contract,
        previous_manifest=_manifest_for(saved_contract),
    )
