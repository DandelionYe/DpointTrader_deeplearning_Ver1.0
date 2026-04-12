import argparse

from tasks import infer_task_type, resolve_label_spec


def test_resolve_label_spec_defaults_binary_from_label_mode():
    spec = resolve_label_spec(argparse.Namespace(label_mode="binary_next_close_up", task_type=None, label_horizon_days=1))
    assert spec.task_type == "binary_classification"
    assert spec.label_mode == "binary_next_close_up"
    assert spec.horizon_days == 1


def test_resolve_label_spec_defaults_regression_from_label_mode():
    spec = resolve_label_spec(argparse.Namespace(label_mode="regression_return", task_type=None, label_horizon_days=5))
    assert spec.task_type == "regression"
    assert spec.horizon_days == 5


def test_resolve_label_spec_preserves_explicit_task_type():
    spec = resolve_label_spec(argparse.Namespace(label_mode="binary_next_close_up", task_type="binary_classification", label_horizon_days=2))
    assert spec.task_type == "binary_classification"
    assert spec.horizon_days == 2


def test_infer_task_type_handles_multiclass():
    assert infer_task_type("multiclass_3") == "multiclass_classification"


def test_resolve_label_spec_extracts_multiclass_n_classes():
    spec = resolve_label_spec(argparse.Namespace(label_mode="multiclass_5", task_type=None, label_horizon_days=1))
    assert spec.task_type == "multiclass_classification"
    assert spec.n_classes == 5
