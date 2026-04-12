import argparse

import pytest

from tasks import validate_primary_metric


def test_regression_rejects_auc_primary_metric():
    with pytest.raises(
        ValueError, match="primary_metric 'auc' is invalid for task_type 'regression'"
    ):
        validate_primary_metric("regression", "auc")


def test_binary_allows_auc_primary_metric():
    validate_primary_metric("binary_classification", "auc")


def test_multiclass_allows_rank_ic_primary_metric():
    validate_primary_metric("multiclass_classification", "rank_ic_mean")


def test_normalize_mode_args_populates_task_type_and_validates_metric():
    from main_basket import normalize_mode_args

    args = argparse.Namespace(
        mode="first",
        continue_from=None,
        label_mode="regression_return",
        task_type=None,
        label_horizon_days=1,
        primary_metric="rank_ic_mean",
    )

    normalized = normalize_mode_args(args)

    assert normalized.task_type == "regression"
