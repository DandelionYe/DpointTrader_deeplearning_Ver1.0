import pytest

from models import get_loss_fn


def test_binary_task_uses_bce():
    torch = pytest.importorskip("torch")
    loss = get_loss_fn("binary_classification", {})
    assert isinstance(loss, torch.nn.BCEWithLogitsLoss)


def test_multiclass_task_uses_cross_entropy():
    torch = pytest.importorskip("torch")
    loss = get_loss_fn("multiclass_classification", {"n_classes": 3})
    assert isinstance(loss, torch.nn.CrossEntropyLoss)


def test_regression_task_uses_huber():
    torch = pytest.importorskip("torch")
    loss = get_loss_fn("regression", {})
    assert isinstance(loss, torch.nn.HuberLoss)
