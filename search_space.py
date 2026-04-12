from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any, Dict, cast

import numpy as np

MLP_HIDDEN_DIMS_OPTIONS = [
    [512, 256],
    [1024, 512, 256],
    [1024, 512],
    [768, 384, 192],
]
MLP_DROPOUT_OPTIONS = [0.05, 0.10, 0.15, 0.20, 0.30]
MLP_LR_MIN = 5e-4
MLP_LR_MAX = 5e-3
MLP_WEIGHT_DECAY_MIN = 1e-6
MLP_WEIGHT_DECAY_MAX = 1e-3
MLP_BATCH_SIZE_OPTIONS = [2048, 4096, 8192, 16384]
MLP_EPOCHS_OPTIONS = [20, 30, 40, 60]

XGB_N_ESTIMATORS_OPTIONS = [100, 200, 300, 500]
XGB_MAX_DEPTH_OPTIONS = [4, 6, 8]
XGB_LR_OPTIONS = [0.03, 0.05, 0.1]
XGB_SUBSAMPLE_OPTIONS = [0.7, 0.8, 0.9, 1.0]
XGB_COLSAMPLE_BYTREE_OPTIONS = [0.7, 0.8, 0.9, 1.0]
XGB_MIN_CHILD_WEIGHT_OPTIONS = [1, 3, 5]

GENERIC_HIDDEN_DIM_DEFAULT = 1024
GENERIC_BATCH_SIZE_DEFAULT = 8192
SEQUENCE_HIDDEN_DIM_DEFAULT = 128
SEQUENCE_BATCH_SIZE_DEFAULT = 64
SEQUENCE_BATCH_SIZE_OPTIONS = [16, 32, 64, 128]


def build_base_model_config(args: argparse.Namespace) -> Dict[str, Any]:
    task_type = str(getattr(args, "task_type", "binary_classification"))
    label_mode = str(getattr(args, "label_mode", "binary_next_close_up"))
    label_horizon_days = max(1, int(getattr(args, "label_horizon_days", 1)))
    primary_metric = str(getattr(args, "primary_metric", "auto"))
    xgb_eval_metric = (
        "rmse"
        if task_type == "regression"
        else "mlogloss"
        if task_type == "multiclass_classification"
        else "logloss"
    )
    n_classes = 3 if task_type == "multiclass_classification" else None
    cpu_threads = max(1, args.cpu_threads)
    predict_batch_size = int(getattr(args, "predict_batch_size", 0))
    auto_batch_tune = bool(getattr(args, "auto_batch_tune", 1))
    target_vram_util = float(cast(Any, getattr(args, "target_vram_util", 0.88)))
    train_target_vram_util = float(
        cast(Any, getattr(args, "train_target_vram_util", None))
        if getattr(args, "train_target_vram_util", None) is not None
        else target_vram_util
    )
    predict_target_vram_util = float(
        cast(Any, getattr(args, "predict_target_vram_util", None))
        if getattr(args, "predict_target_vram_util", None) is not None
        else target_vram_util
    )
    use_amp = bool(getattr(args, "use_amp", 0))
    use_tf32 = bool(getattr(args, "use_tf32", 0))
    sequence_hidden_dim = args.hidden_dim
    sequence_batch_size = args.batch_size
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
            "task_type": task_type,
            "n_classes": n_classes,
            "model_type": "mlp",
            "label_mode": label_mode,
            "label_horizon_days": label_horizon_days,
            "primary_metric": primary_metric,
            "device": args.device,
            "model_params": {
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
    if args.model_type == "xgb":
        return {
            "task_type": task_type,
            "model_type": "xgb",
            "n_classes": n_classes,
            "label_mode": label_mode,
            "label_horizon_days": label_horizon_days,
            "primary_metric": primary_metric,
            "device": "cpu",
            "model_params": {
                "n_estimators": args.xgb_n_estimators,
                "max_depth": args.xgb_max_depth,
                "learning_rate": args.learning_rate,
                "subsample": args.xgb_subsample,
                "colsample_bytree": args.xgb_colsample_bytree,
                "n_jobs": cpu_threads,
                "tree_method": "hist",
                "eval_metric": xgb_eval_metric,
                "verbosity": 0,
            },
        }
    return {
        "task_type": task_type,
        "n_classes": n_classes,
        "model_type": args.model_type,
        "label_mode": label_mode,
        "label_horizon_days": label_horizon_days,
        "primary_metric": primary_metric,
        "device": args.device,
        "model_params": {
            "hidden_dim": sequence_hidden_dim,
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
            "num_filters": args.num_filters,
            "kernel_sizes": [int(part.strip()) for part in str(args.kernel_sizes).split(",") if part.strip()],
            "d_model": args.d_model,
            "nhead": args.nhead,
            "dim_feedforward": args.dim_feedforward,
        },
    }


def sample_model_config(
    *,
    model_type: str,
    rng: np.random.RandomState,
    base_config: Dict[str, Any],
) -> Dict[str, Any]:
    if model_type == "mlp":
        return _sample_mlp_config(rng, base_config)
    if model_type == "xgb":
        return _sample_xgb_config(rng, base_config)
    if model_type in {"lstm", "gru"}:
        return _sample_rnn_config(model_type, rng, base_config)
    if model_type == "cnn":
        return _sample_cnn_config(rng, base_config)
    if model_type == "transformer":
        return _sample_transformer_config(rng, base_config)
    return deepcopy(base_config)


def _sample_mlp_config(rng: np.random.RandomState, base_config: Dict[str, Any]) -> Dict[str, Any]:
    base_params = deepcopy(base_config.get("model_params", {}))
    learning_rate = float(np.exp(rng.uniform(np.log(MLP_LR_MIN), np.log(MLP_LR_MAX))))
    weight_decay = float(np.exp(rng.uniform(np.log(MLP_WEIGHT_DECAY_MIN), np.log(MLP_WEIGHT_DECAY_MAX))))
    config = deepcopy(base_config)
    config["model_type"] = "mlp"
    config["device"] = base_config.get("device", "auto")
    config["model_params"] = {
        "hidden_dims": MLP_HIDDEN_DIMS_OPTIONS[rng.randint(0, len(MLP_HIDDEN_DIMS_OPTIONS))],
        "dropout_rate": MLP_DROPOUT_OPTIONS[rng.randint(0, len(MLP_DROPOUT_OPTIONS))],
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": MLP_BATCH_SIZE_OPTIONS[rng.randint(0, len(MLP_BATCH_SIZE_OPTIONS))],
        "epochs": MLP_EPOCHS_OPTIONS[rng.randint(0, len(MLP_EPOCHS_OPTIONS))],
        "predict_batch_size": base_params.get("predict_batch_size", 0),
        "auto_batch_tune": base_params.get("auto_batch_tune", True),
        "target_vram_util": base_params.get("target_vram_util", 0.88),
        "train_target_vram_util": base_params.get("train_target_vram_util", base_params.get("target_vram_util", 0.88)),
        "predict_target_vram_util": base_params.get("predict_target_vram_util", base_params.get("target_vram_util", 0.88)),
        "use_amp": base_params.get("use_amp", False),
        "use_tf32": base_params.get("use_tf32", False),
    }
    return config


def _sample_xgb_config(rng: np.random.RandomState, base_config: Dict[str, Any]) -> Dict[str, Any]:
    cpu_threads = base_config.get("model_params", {}).get("n_jobs", 4)
    config = deepcopy(base_config)
    config["model_type"] = "xgb"
    config["device"] = "cpu"
    config["n_classes"] = base_config.get("n_classes")
    config["model_params"] = {
        "n_estimators": XGB_N_ESTIMATORS_OPTIONS[rng.randint(0, len(XGB_N_ESTIMATORS_OPTIONS))],
        "max_depth": XGB_MAX_DEPTH_OPTIONS[rng.randint(0, len(XGB_MAX_DEPTH_OPTIONS))],
        "learning_rate": XGB_LR_OPTIONS[rng.randint(0, len(XGB_LR_OPTIONS))],
        "subsample": XGB_SUBSAMPLE_OPTIONS[rng.randint(0, len(XGB_SUBSAMPLE_OPTIONS))],
        "colsample_bytree": XGB_COLSAMPLE_BYTREE_OPTIONS[rng.randint(0, len(XGB_COLSAMPLE_BYTREE_OPTIONS))],
        "min_child_weight": XGB_MIN_CHILD_WEIGHT_OPTIONS[rng.randint(0, len(XGB_MIN_CHILD_WEIGHT_OPTIONS))],
        "n_jobs": cpu_threads,
        "tree_method": "hist",
        "eval_metric": (
            "rmse" if config.get("task_type") == "regression" else "mlogloss" if config.get("task_type") == "multiclass_classification" else "logloss"
        ),
        "verbosity": 0,
    }
    return config


def _sample_rnn_config(model_type: str, rng: np.random.RandomState, base_config: Dict[str, Any]) -> Dict[str, Any]:
    params = deepcopy(base_config.get("model_params", {}))
    params["hidden_dim"] = int(rng.choice([16, 32, 64, int(params.get("hidden_dim", 32))]))
    params["num_layers"] = int(rng.choice([1, 2, int(params.get("num_layers", 2))]))
    params["dropout_rate"] = float(rng.choice([0.05, 0.10, 0.20, float(params.get("dropout_rate", 0.1))]))
    params["bidirectional"] = bool(rng.choice([False, True]))
    params["batch_size"] = int(rng.choice(SEQUENCE_BATCH_SIZE_OPTIONS + [int(params.get("batch_size", 64))]))
    config = deepcopy(base_config)
    config["model_type"] = model_type
    config["device"] = base_config.get("device", "cpu")
    config["model_params"] = params
    return config


def _sample_cnn_config(rng: np.random.RandomState, base_config: Dict[str, Any]) -> Dict[str, Any]:
    params = deepcopy(base_config.get("model_params", {}))
    params["num_filters"] = int(rng.choice([16, 32, 64, int(params.get("num_filters", 32))]))
    kernel_options = [[2, 3], [3, 5], [2, 3, 5]]
    params["kernel_sizes"] = list(kernel_options[int(rng.choice(len(kernel_options), p=[0.3, 0.3, 0.4]))])
    params["dropout_rate"] = float(rng.choice([0.05, 0.10, 0.20, float(params.get("dropout_rate", 0.1))]))
    params["batch_size"] = int(rng.choice(SEQUENCE_BATCH_SIZE_OPTIONS + [int(params.get("batch_size", 64))]))
    config = deepcopy(base_config)
    config["model_type"] = "cnn"
    config["device"] = base_config.get("device", "cpu")
    config["model_params"] = params
    return config


def _sample_transformer_config(rng: np.random.RandomState, base_config: Dict[str, Any]) -> Dict[str, Any]:
    params = deepcopy(base_config.get("model_params", {}))
    params["d_model"] = int(rng.choice([32, 64, int(params.get("d_model", 64))]))
    params["nhead"] = int(rng.choice([2, 4, int(params.get("nhead", 4))]))
    params["num_layers"] = int(rng.choice([1, 2, int(params.get("num_layers", 2))]))
    params["dim_feedforward"] = int(rng.choice([64, 128, int(params.get("dim_feedforward", 128))]))
    params["dropout_rate"] = float(rng.choice([0.05, 0.10, 0.20, float(params.get("dropout_rate", 0.1))]))
    params["batch_size"] = int(rng.choice(SEQUENCE_BATCH_SIZE_OPTIONS + [int(params.get("batch_size", 64))]))
    config = deepcopy(base_config)
    config["model_type"] = "transformer"
    config["device"] = base_config.get("device", "cpu")
    config["model_params"] = params
    return config


def mutate_model_config(
    incumbent_config: Dict[str, Any],
    *,
    rng: np.random.RandomState,
    strength: float = 0.2,
) -> Dict[str, Any]:
    mutated = deepcopy(incumbent_config)
    model_type = mutated.get("model_type", "mlp")
    params = mutated.setdefault("model_params", {})
    if model_type == "mlp":
        if "learning_rate" in params:
            lr = float(params["learning_rate"])
            params["learning_rate"] = max(MLP_LR_MIN, min(MLP_LR_MAX, lr + rng.uniform(-lr * strength, lr * strength)))
        if "dropout_rate" in params:
            dr = float(params["dropout_rate"])
            params["dropout_rate"] = max(0.01, min(0.5, dr + rng.uniform(-dr * strength, dr * strength)))
        if "weight_decay" in params:
            wd = float(params["weight_decay"])
            params["weight_decay"] = max(MLP_WEIGHT_DECAY_MIN, min(MLP_WEIGHT_DECAY_MAX, wd + rng.uniform(-wd * strength, wd * strength)))
    elif model_type == "xgb":
        if "learning_rate" in params:
            lr = float(params["learning_rate"])
            params["learning_rate"] = max(0.01, min(0.3, lr + rng.uniform(-lr * strength, lr * strength)))
    return mutated
