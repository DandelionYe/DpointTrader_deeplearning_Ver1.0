from __future__ import annotations

import gc
import json
import logging
import math
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sequence_builder import PanelSequenceStore
from tasks import get_output_dim as task_output_dim
from tasks import multiclass_class_values, multiclass_probabilities_to_score, resolve_loss_spec

logger = logging.getLogger(__name__)

_AUTO_BATCH_TUNE_CACHE: Dict[Tuple[Any, ...], int] = {}

TORCH_AVAILABLE = False
TORCH_IMPORT_ERROR: Optional[Exception] = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Dataset, TensorDataset

    TORCH_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    GradScaler = None  # type: ignore[assignment]
    autocast = None  # type: ignore[assignment]
    DataLoader = Any  # type: ignore[assignment]
    Dataset = Any  # type: ignore[assignment]
    TensorDataset = Any  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc

TORCH_MODEL_TYPES = {"mlp", "lstm", "gru", "cnn", "transformer"}

__all__ = [
    "MLP",
    "LSTM",
    "GRU",
    "CNN1D",
    "Transformer",
    "_get_device",
    "train_pytorch_model",
    "predict_pytorch_model",
    "create_sequence_dataset",
    "make_sequence_dataloader",
    "predict_pytorch_model_tabular",
    "predict_pytorch_model_tabular_outputs",
    "predict_pytorch_model_sequence",
    "predict_pytorch_model_sequence_outputs",
    "make_model",
    "predict_dpoint",
    "get_loss_fn",
    "get_output_dim",
    "resolve_torch_device",
    "get_torch_runtime_info",
    "is_torch_model_type",
    "is_torch_model_instance",
    "TORCH_AVAILABLE",
    "TORCH_IMPORT_ERROR",
    "TORCH_MODEL_TYPES",
    "clear_torch_cuda_cache",
    "save_trained_model",
    "load_saved_model",
]


class _CpuFallbackDevice:
    type = "cpu"

    def __str__(self) -> str:
        return self.type


def _require_torch(feature: str) -> None:
    if TORCH_AVAILABLE:
        return
    detail = ""
    if TORCH_IMPORT_ERROR is not None:
        detail = f" Original import error: {TORCH_IMPORT_ERROR!r}"
    raise RuntimeError(
        f"{feature} requires PyTorch, but PyTorch is not installed or failed to import.{detail}"
    )


def _get_device() -> torch.device:
    if not TORCH_AVAILABLE:
        return _CpuFallbackDevice()  # type: ignore[return-value]
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_torch_model_type(model_type: str) -> bool:
    return str(model_type).lower() in TORCH_MODEL_TYPES


def is_torch_model_instance(model: Any) -> bool:
    if not TORCH_AVAILABLE:
        return False
    return isinstance(model, (MLP, LSTM, GRU, CNN1D, Transformer))


def get_torch_runtime_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "torch_available": TORCH_AVAILABLE,
        "torch_version": getattr(torch, "__version__", "not_installed")
        if TORCH_AVAILABLE
        else "not_installed",
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "device_name": None,
    }
    if TORCH_AVAILABLE:
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
    elif TORCH_IMPORT_ERROR is not None:
        info["import_error"] = repr(TORCH_IMPORT_ERROR)
    return info


def resolve_torch_device(preferred_device: str = "auto") -> torch.device:
    preferred = str(preferred_device).lower()
    if preferred not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device preference: {preferred_device}")
    if preferred == "cpu":
        _require_torch("resolve_torch_device")
        return torch.device("cpu")
    if preferred == "cuda":
        _require_torch("resolve_torch_device(device='cuda')")
        if not torch.cuda.is_available():
            runtime = get_torch_runtime_info()
            raise RuntimeError(
                "CUDA was requested but is not available in the current PyTorch runtime. "
                f"torch={runtime['torch_version']} cuda_build={runtime['cuda_version']} "
                f"cuda_available={runtime['cuda_available']}."
            )
        return torch.device("cuda")
    return _get_device()


def _to_numpy_features(X: Any) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def _to_numpy_labels(y: Any) -> Optional[np.ndarray]:
    if y is None:
        return None
    if isinstance(y, pd.Series):
        return y.to_numpy(dtype=np.float32, copy=False)
    return np.asarray(y, dtype=np.float32)


def get_output_dim(task_type: str, n_classes: Optional[int] = None) -> int:
    return task_output_dim(task_type, n_classes)


def get_loss_fn(task_type: str, config: Dict[str, Any]) -> Any:
    _require_torch("get_loss_fn")
    if task_type == "binary_classification":
        return nn.BCEWithLogitsLoss()
    if task_type == "multiclass_classification":
        return nn.CrossEntropyLoss()
    if task_type == "regression":
        return nn.HuberLoss()
    raise ValueError(f"Unsupported task_type: {task_type}")


def create_sequence_dataset(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    seq_len: int = 20,
    batch_size: int = 64,
    task_type: str = "binary_classification",
    device: Optional[torch.device] = None,
    shuffle: bool = True,
    pin_memory: bool = False,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    _require_torch("create_sequence_dataset")
    _ = device

    date_ticker_cols = [c for c in X.columns if c in ["date", "ticker"]]
    if date_ticker_cols:
        raise ValueError(
            f"X contains date/ticker columns {date_ticker_cols}. "
            "For panel data, use sequence_builder.build_panel_sequences() + "
            "make_sequence_dataloader() instead."
        )

    X_tensor = torch.tensor(X.to_numpy(dtype=np.float32, copy=False), dtype=torch.float32)
    y_tensor = None
    if y is not None:
        y_values = y.to_numpy(copy=False)
        if task_type == "multiclass_classification":
            y_tensor = torch.tensor(np.asarray(y_values, dtype=np.int64), dtype=torch.long)
        else:
            y_tensor = torch.tensor(
                np.asarray(y_values, dtype=np.float32), dtype=torch.float32
            ).unsqueeze(1)

    if seq_len > 1:
        X_seq, y_seq = _make_sequences(X_tensor, y_tensor, seq_len)
        dataset = TensorDataset(X_seq, y_seq) if y_seq is not None else TensorDataset(X_seq)
    else:
        dataset = (
            TensorDataset(X_tensor, y_tensor) if y_tensor is not None else TensorDataset(X_tensor)
        )

    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor if prefetch_factor is not None else 2
    return DataLoader(dataset, **kwargs)


def make_sequence_dataloader(
    X_seq: np.ndarray,
    y_seq: Optional[np.ndarray],
    *,
    batch_size: int,
    shuffle: bool,
    task_type: str = "binary_classification",
    pin_memory: bool = False,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    _require_torch("make_sequence_dataloader")

    X_tensor = torch.tensor(np.asarray(X_seq, dtype=np.float32), dtype=torch.float32)
    if y_seq is None:
        dataset = TensorDataset(X_tensor)
    else:
        if task_type == "multiclass_classification":
            y_tensor = torch.tensor(np.asarray(y_seq, dtype=np.int64).reshape(-1), dtype=torch.long)
        else:
            y_tensor = torch.tensor(np.asarray(y_seq, dtype=np.float32), dtype=torch.float32)
            if y_tensor.dim() == 1:
                y_tensor = y_tensor.unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)
    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor if prefetch_factor is not None else 2
    return DataLoader(dataset, **kwargs)


if TORCH_AVAILABLE:

    class PanelSequenceDataset(Dataset):
        def __init__(
            self, store: PanelSequenceStore, task_type: str = "binary_classification"
        ) -> None:
            self.store = store
            self.task_type = str(task_type)

        def __len__(self) -> int:
            return len(self.store.window_keys)

        def __getitem__(self, idx: int):
            ticker, start = self.store.window_keys[idx]
            seq_len = self.store.seq_len
            features = self.store.feature_by_ticker[ticker][start : start + seq_len]
            X_tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
            if self.store.label_by_ticker is None:
                return X_tensor
            y_value = float(self.store.label_by_ticker[ticker][start + seq_len - 1])
            if self.task_type == "multiclass_classification":
                y_tensor = torch.tensor(int(y_value), dtype=torch.long)
            else:
                y_tensor = torch.tensor([y_value], dtype=torch.float32)
            return X_tensor, y_tensor
else:  # pragma: no cover

    class PanelSequenceDataset:  # type: ignore[no-redef]
        def __init__(
            self, store: PanelSequenceStore, task_type: str = "binary_classification"
        ) -> None:
            self.store = store
            self.task_type = str(task_type)


def make_panel_sequence_dataloader(
    store: PanelSequenceStore,
    *,
    batch_size: int,
    shuffle: bool,
    task_type: str = "binary_classification",
    pin_memory: bool = False,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    _require_torch("make_panel_sequence_dataloader")
    dataset = PanelSequenceDataset(store, task_type=task_type)
    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor if prefetch_factor is not None else 2
    return DataLoader(dataset, **kwargs)


def _make_sequences(
    X: torch.Tensor,
    y: Optional[torch.Tensor],
    seq_len: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    n_samples = X.shape[0]
    n_sequences = n_samples - seq_len + 1
    if n_sequences <= 0:
        raise ValueError(f"seq_len ({seq_len}) exceeds sample count ({n_samples})")
    X_seq = X.unfold(0, seq_len, 1).permute(0, 2, 1).contiguous()
    y_seq = y[seq_len - 1 : n_samples] if y is not None else None
    return X_seq, y_seq


def _logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def _model_outputs_to_scores(logits: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type == "binary_classification":
        return torch.sigmoid(logits)
    if task_type == "multiclass_classification":
        probs = torch.softmax(logits, dim=-1)
        class_values = torch.as_tensor(
            multiclass_class_values(int(probs.shape[-1])),
            dtype=probs.dtype,
            device=probs.device,
        )
        return (probs * class_values).sum(dim=-1, keepdim=True)
    return logits


def _model_outputs_to_prediction_components(
    logits: torch.Tensor, task_type: str
) -> Dict[str, torch.Tensor]:
    if task_type == "binary_classification":
        proba = torch.sigmoid(logits).reshape(-1).to(torch.float32)
        prediction = (proba >= 0.5).to(torch.float32)
        return {
            "score": proba,
            "prediction": prediction,
            "raw_output": proba,
            "proba_up": proba,
            "proba": proba,
        }
    if task_type == "multiclass_classification":
        probs = torch.softmax(logits, dim=-1).to(torch.float32)
        class_values = torch.as_tensor(
            multiclass_class_values(int(probs.shape[-1])),
            dtype=probs.dtype,
            device=probs.device,
        )
        score = (probs * class_values).sum(dim=-1)
        prediction = probs.argmax(dim=-1).to(torch.int64)
        bullish_proba = probs[:, -1]
        confidence = probs.max(dim=-1).values
        return {
            "score": score,
            "prediction": prediction,
            "raw_output": confidence,
            "proba_up": bullish_proba,
            "proba": bullish_proba,
        }
    prediction = logits.reshape(-1).to(torch.float32)
    nan_tensor = torch.full_like(prediction, float("nan"))
    return {
        "score": prediction,
        "prediction": prediction,
        "raw_output": prediction,
        "proba_up": nan_tensor,
        "proba": nan_tensor,
    }


def _is_cuda_oom_error(exc: Exception) -> bool:
    if not TORCH_AVAILABLE or not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "cuda out of memory",
            "cuda error: out of memory",
            "cudaerrormemoryallocation",
            "cublas_status_alloc_failed",
            "out of memory",
        )
    )


def _clear_cuda_cache() -> None:
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()


def clear_torch_cuda_cache() -> None:
    _clear_cuda_cache()


def _round_batch_size(value: int, *, minimum: int = 8) -> int:
    rounded = max(minimum, int(value))
    if rounded <= 16:
        return rounded
    return max(minimum, int(round(rounded / 16.0) * 16))


def _sequence_dataset_size(X: Any) -> int:
    if isinstance(X, PanelSequenceStore):
        return len(X.window_keys)
    if isinstance(X, np.ndarray):
        return int(X.shape[0])
    return int(len(X))


def _build_sequence_loader_runtime_settings(
    device: torch.device,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    default_workers = 0 if os.name == "nt" else (2 if device.type == "cuda" else 0)
    num_workers = max(0, int(config.get("dataloader_workers", default_workers)))
    pin_memory = bool(config.get("pin_memory", device.type == "cuda"))
    settings: Dict[str, Any] = {
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        settings["persistent_workers"] = bool(config.get("persistent_workers", True))
        settings["prefetch_factor"] = max(2, int(config.get("prefetch_factor", 2)))
    return settings


def _make_sequence_loader_for_data(
    X: Any,
    y: Any,
    *,
    seq_len: int,
    batch_size: int,
    shuffle: bool,
    task_type: str = "binary_classification",
    loader_settings: Dict[str, Any],
) -> DataLoader:
    if isinstance(X, PanelSequenceStore):
        return make_panel_sequence_dataloader(
            X,
            batch_size=batch_size,
            shuffle=shuffle,
            task_type=task_type,
            **loader_settings,
        )
    if isinstance(X, np.ndarray) and X.ndim == 3:
        return make_sequence_dataloader(
            np.asarray(X, dtype=np.float32),
            _to_numpy_labels(y),
            batch_size=batch_size,
            shuffle=shuffle,
            task_type=task_type,
            **loader_settings,
        )
    return create_sequence_dataset(
        X,
        y,
        seq_len=seq_len,
        batch_size=batch_size,
        task_type=task_type,
        device=None,
        shuffle=shuffle,
        **loader_settings,
    )


def _cuda_total_memory(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return int(torch.cuda.get_device_properties(device).total_memory)


@contextmanager
def _torch_precision_context(device: torch.device, use_tf32: bool):
    if not TORCH_AVAILABLE or device.type != "cuda":
        yield
        return
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(use_tf32)
    torch.backends.cudnn.allow_tf32 = bool(use_tf32)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def _next_probe_batch_size(current: int, util: float, target: float, dataset_size: int) -> int:
    if dataset_size <= current:
        return current
    if util <= 0.0:
        growth = current * 2
    else:
        scale = min(2.0, max(1.25, (target / max(util, 1e-3)) * 0.95))
        growth = int(current * scale)
    return min(dataset_size, _round_batch_size(max(current + 8, growth)))


def _midpoint_batch_size(lower: int, upper: int) -> int:
    if upper <= lower:
        return lower
    midpoint = _round_batch_size((lower + upper) // 2)
    if midpoint <= lower:
        midpoint = min(upper - 1, lower + 16)
    return max(lower, midpoint)


def _resolve_target_util(config: Dict[str, Any], mode: str) -> float:
    if mode == "predict":
        return float(config.get("predict_target_vram_util", config.get("target_vram_util", 0.88)))
    return float(config.get("train_target_vram_util", config.get("target_vram_util", 0.88)))


def _resolve_safety_buffer_bytes(config: Dict[str, Any], mode: str) -> int:
    default_gib = 0.50 if mode == "predict" else 0.75
    if bool(config.get("use_amp", False)):
        default_gib += 0.25
    return int(float(config.get(f"{mode}_vram_buffer_gib", default_gib)) * (1024**3))


def _batch_tune_cache_key(
    *,
    mode: str,
    model_type: str,
    input_dim: int,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[Any, ...]:
    kernel_sizes = tuple(int(v) for v in config.get("kernel_sizes", [2, 3, 5]))
    return (
        mode,
        str(model_type).lower(),
        str(device),
        int(input_dim),
        int(config.get("seq_len", 20)),
        int(config.get("hidden_dim", 0)),
        int(config.get("num_layers", 0)),
        bool(config.get("bidirectional", False)),
        int(config.get("num_filters", 0)),
        kernel_sizes,
        int(config.get("d_model", 0)),
        int(config.get("nhead", 0)),
        int(config.get("dim_feedforward", 0)),
        bool(config.get("use_amp", False)),
        bool(config.get("use_tf32", False)),
        round(_resolve_target_util(config, mode), 4),
        _resolve_safety_buffer_bytes(config, mode),
    )


def _probe_warmup_batches(config: Dict[str, Any], for_inference: bool) -> int:
    if for_inference:
        return max(2, int(config.get("predict_probe_batches", 2)))
    return max(2, int(config.get("train_probe_batches", 2)))


def _build_retry_batch_sizes(start_batch_size: int) -> list[int]:
    sizes: list[int] = []

    def _add(value: int) -> None:
        rounded = _round_batch_size(value)
        if rounded >= 8 and rounded not in sizes:
            sizes.append(rounded)

    _add(start_batch_size)
    for ratio in (0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50):
        _add(int(start_batch_size * ratio))

    current = max(8, start_batch_size // 2)
    while current >= 8:
        _add(current)
        if current == 8:
            break
        next_current = max(8, current // 2)
        if next_current == current:
            break
        current = next_current
    return sizes


def _probe_sequence_batch_memory(
    *,
    model_type: str,
    input_dim: int,
    config: Dict[str, Any],
    X: Any,
    y: Any,
    device: torch.device,
    batch_size: int,
    for_inference: bool,
    model: Optional[nn.Module] = None,
    loader_settings: Optional[Dict[str, Any]] = None,
) -> Tuple[float, int, int]:
    _require_torch("_probe_sequence_batch_memory")
    seq_len = int(config.get("seq_len", 20))
    use_amp = bool(config.get("use_amp", False)) and device.type == "cuda"
    use_tf32 = bool(config.get("use_tf32", False))
    loader_settings = dict(loader_settings or {})
    total_memory = _cuda_total_memory(device)
    warmup_batches = _probe_warmup_batches(config, for_inference)
    torch.cuda.reset_peak_memory_stats(device)
    _clear_cuda_cache()
    gc.collect()

    local_model = model
    optimizer = None
    criterion = None
    loader = None
    batch = None
    logits = None
    loss = None
    try:
        if local_model is None:
            local_model = _make_sequence_model(model_type, input_dim, config).to(device)
        else:
            local_model = local_model.to(device)
        local_model.train(not for_inference)
        loader = _make_sequence_loader_for_data(
            X,
            y,
            seq_len=seq_len,
            batch_size=batch_size,
            shuffle=False,
            task_type=str(getattr(model, "_task_type", "binary_classification")),
            loader_settings=loader_settings,
        )
        if for_inference:
            with _torch_precision_context(device, use_tf32):
                with torch.no_grad():
                    for batch_idx, batch in enumerate(loader):
                        if batch_idx >= warmup_batches:
                            break
                        X_batch = batch[0] if isinstance(batch, (tuple, list)) else batch
                        X_batch = X_batch.to(
                            device, non_blocking=bool(loader_settings.get("pin_memory", False))
                        )
                        with autocast(device_type=device.type, enabled=use_amp):
                            logits = local_model(X_batch)
                            _ = _logits_to_probs(logits)
        else:
            optimizer = optim.AdamW(
                local_model.parameters(),
                lr=float(config.get("learning_rate", 0.001)),
                weight_decay=float(config.get("weight_decay", 1e-5)),
            )
            criterion = nn.BCEWithLogitsLoss()
            with _torch_precision_context(device, use_tf32):
                for batch_idx, batch in enumerate(loader):
                    if batch_idx >= warmup_batches:
                        break
                    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                        raise ValueError("Training probe requires labels")
                    X_batch, y_batch = batch
                    X_batch = X_batch.to(
                        device, non_blocking=bool(loader_settings.get("pin_memory", False))
                    )
                    y_batch = y_batch.to(
                        device, non_blocking=bool(loader_settings.get("pin_memory", False))
                    )
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(device_type=device.type, enabled=use_amp):
                        logits = local_model(X_batch)
                        loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        peak_allocated = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        util = float(peak_allocated / total_memory) if total_memory > 0 else 0.0
        return util, peak_allocated, peak_reserved
    finally:
        del loader, batch, logits, loss, optimizer, criterion
        if model is None:
            del local_model
        _clear_cuda_cache()
        gc.collect()


def _auto_tune_sequence_batch_size(
    *,
    mode: str,
    model_type: str,
    input_dim: int,
    config: Dict[str, Any],
    X: Any,
    y: Any,
    device: torch.device,
    base_batch_size: int,
    model: Optional[nn.Module] = None,
    loader_settings: Optional[Dict[str, Any]] = None,
) -> int:
    if device.type != "cuda" or not bool(config.get("auto_batch_tune", True)):
        return max(1, base_batch_size)

    dataset_size = _sequence_dataset_size(X)
    if dataset_size <= 0:
        return max(1, base_batch_size)

    cache_key = _batch_tune_cache_key(
        mode=mode,
        model_type=model_type,
        input_dim=input_dim,
        config=config,
        device=device,
    )
    cached_batch = _AUTO_BATCH_TUNE_CACHE.get(cache_key)
    if cached_batch is not None:
        selected = min(dataset_size, cached_batch)
        logger.info("Reusing cached %s batch_size=%d", mode, selected)
        return selected

    requested = min(dataset_size, _round_batch_size(max(1, base_batch_size)))
    target_util = _resolve_target_util(config, mode)
    total_memory = _cuda_total_memory(device)
    safety_buffer_bytes = _resolve_safety_buffer_bytes(config, mode)
    safe_cap_util = (
        float(max(0, total_memory - safety_buffer_bytes) / total_memory)
        if total_memory > 0
        else 0.96
    )
    max_util = min(0.96, target_util + 0.04, safe_cap_util)
    best_batch = min(requested, dataset_size)
    best_util = 0.0
    best_allocated = 0
    best_reserved = 0
    best_distance = float("inf")
    current = requested
    tried = set()
    for_inference = mode == "predict"
    lower_batch = 0
    upper_batch: Optional[int] = None

    while current not in tried:
        tried.add(current)
        try:
            util, peak_allocated, peak_reserved = _probe_sequence_batch_memory(
                model_type=model_type,
                input_dim=input_dim,
                config=config,
                X=X,
                y=y,
                device=device,
                batch_size=current,
                for_inference=for_inference,
                model=model,
                loader_settings=loader_settings,
            )
            logger.info(
                "Auto batch probe (%s): batch_size=%d peak_allocated=%.2f GiB peak_reserved=%.2f GiB util=%.1f%% safe_cap=%.1f%%",
                mode,
                current,
                peak_allocated / (1024**3),
                peak_reserved / (1024**3),
                util * 100.0,
                max_util * 100.0,
            )
            if util <= max_util:
                lower_batch = current
                distance = abs(util - target_util)
                if distance < best_distance or (
                    abs(distance - best_distance) <= 1e-9 and current < best_batch
                ):
                    best_batch = current
                    best_util = util
                    best_allocated = peak_allocated
                    best_reserved = peak_reserved
                    best_distance = distance
                if target_util <= util <= max_util:
                    if upper_batch is None or upper_batch - lower_batch <= 32:
                        break
                    current = _midpoint_batch_size(lower_batch, upper_batch)
                    if current <= lower_batch or current >= upper_batch:
                        break
                    continue
                if upper_batch is not None:
                    if upper_batch - lower_batch <= 32:
                        break
                    current = _midpoint_batch_size(lower_batch, upper_batch)
                    if current <= lower_batch or current >= upper_batch:
                        break
                    continue
                next_batch = _next_probe_batch_size(current, util, target_util, dataset_size)
                if next_batch <= current or next_batch > dataset_size:
                    break
                current = next_batch
                continue

            upper_batch = current
            logger.info(
                "Auto batch probe (%s): batch_size=%d exceeded effective ceiling %.1f%%, rolling back to last stable batch_size=%d",
                mode,
                current,
                max_util * 100.0,
                best_batch,
            )
            if lower_batch <= 0:
                next_batch = _round_batch_size(max(8, current // 2))
                if next_batch >= current:
                    break
                current = next_batch
                continue
            if upper_batch - lower_batch <= 32:
                break
            current = _midpoint_batch_size(lower_batch, upper_batch)
            if current <= lower_batch or current >= upper_batch:
                break
        except Exception as exc:
            if not _is_cuda_oom_error(exc):
                raise
            logger.warning(
                "Auto batch probe (%s) hit CUDA OOM at batch_size=%d; keeping previous stable batch_size=%d",
                mode,
                current,
                best_batch,
            )
            _clear_cuda_cache()
            upper_batch = current
            if lower_batch <= 0:
                next_batch = _round_batch_size(max(8, current // 2))
                if next_batch >= current:
                    break
                current = next_batch
                continue
            if upper_batch - lower_batch <= 32:
                break
            current = _midpoint_batch_size(lower_batch, upper_batch)
            if current <= lower_batch or current >= upper_batch:
                break

    logger.info(
        "Selected %s batch_size=%d (target_vram_util=%.0f%% effective_ceiling=%.1f%% achieved=%.1f%% peak_allocated=%.2f GiB peak_reserved=%.2f GiB diagnostic_reserved_only=True)",
        mode,
        best_batch,
        target_util * 100.0,
        max_util * 100.0,
        best_util * 100.0,
        best_allocated / (1024**3),
        best_reserved / (1024**3),
    )
    _AUTO_BATCH_TUNE_CACHE[cache_key] = best_batch
    return best_batch


if TORCH_AVAILABLE:

    class MLP(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            output_dim: int = 1,
            dropout_rate: float = 0.3,
            hidden_dims: Optional[list[int]] = None,
        ) -> None:
            super().__init__()
            widths = list(hidden_dims) if hidden_dims else [hidden_dim]
            layers: list[nn.Module] = []
            prev_dim = input_dim
            for width in widths:
                layers.extend(
                    [
                        nn.Linear(prev_dim, width),
                        nn.LayerNorm(width),
                        nn.GELU(),
                        nn.Dropout(dropout_rate),
                    ]
                )
                prev_dim = width
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(prev_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.backbone(x))

    class LSTM(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 2,
            output_dim: int = 1,
            dropout_rate: float = 0.3,
            bidirectional: bool = False,
        ) -> None:
            super().__init__()
            self.bidirectional = bidirectional
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            out_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.dropout = nn.Dropout(dropout_rate)
            self.head = nn.Linear(out_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, (h_n, _) = self.lstm(x)
            if self.bidirectional:
                last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
            else:
                last_hidden = h_n[-1]
            return self.head(self.dropout(last_hidden))

    class GRU(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 2,
            output_dim: int = 1,
            dropout_rate: float = 0.3,
            bidirectional: bool = False,
        ) -> None:
            super().__init__()
            self.bidirectional = bidirectional
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            out_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.dropout = nn.Dropout(dropout_rate)
            self.head = nn.Linear(out_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, h_n = self.gru(x)
            if self.bidirectional:
                last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
            else:
                last_hidden = h_n[-1]
            return self.head(self.dropout(last_hidden))

    class CNN1D(nn.Module):
        def __init__(
            self,
            input_dim: int,
            seq_len: int,
            num_filters: int = 64,
            kernel_sizes: Optional[list[int]] = None,
            output_dim: int = 1,
            dropout_rate: float = 0.3,
        ) -> None:
            super().__init__()
            del seq_len
            kernel_sizes = kernel_sizes or [2, 3, 5]
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=input_dim,
                        out_channels=num_filters,
                        kernel_size=k,
                        padding=k // 2,
                    )
                    for k in kernel_sizes
                ]
            )
            self.dropout = nn.Dropout(dropout_rate)
            self.head = nn.Linear(num_filters * len(kernel_sizes), output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.transpose(1, 2)
            pooled = [torch.max(torch.relu(conv(x)), dim=2)[0] for conv in self.convs]
            return self.head(self.dropout(torch.cat(pooled, dim=1)))

    class PositionalEncoding(nn.Module):
        def __init__(
            self, d_model: int, dropout_rate: float = 0.1, max_seq_len: int = 5000
        ) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout_rate)
            pe = torch.zeros(max_seq_len, d_model)
            position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.dropout(x + self.pe[:, : x.size(1)])

    class Transformer(nn.Module):
        def __init__(
            self,
            input_dim: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout_rate: float = 0.1,
            output_dim: int = 1,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_encoder = PositionalEncoding(d_model, dropout_rate)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout_rate,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.dropout = nn.Dropout(dropout_rate)
            self.head = nn.Linear(d_model, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            x = self.encoder(x)
            x = self.dropout(x[:, -1, :])
            return self.head(x)

else:  # pragma: no cover

    class MLP:  # type: ignore[no-redef]
        pass

    class LSTM:  # type: ignore[no-redef]
        pass

    class GRU:  # type: ignore[no-redef]
        pass

    class CNN1D:  # type: ignore[no-redef]
        pass

    class Transformer:  # type: ignore[no-redef]
        pass


def _make_sequence_model(model_type: str, input_dim: int, config: Dict[str, Any]) -> nn.Module:
    hidden_dim = int(config.get("hidden_dim", 64))
    num_layers = int(config.get("num_layers", 2))
    dropout_rate = float(config.get("dropout_rate", 0.3))
    bidirectional = bool(config.get("bidirectional", False))
    seq_len = int(config.get("seq_len", 20))
    num_filters = int(config.get("num_filters", 64))
    kernel_sizes = config.get("kernel_sizes", [2, 3, 5])
    d_model = int(config.get("d_model", 64))
    nhead = int(config.get("nhead", 4))
    dim_feedforward = int(config.get("dim_feedforward", 128))
    task_type = str(config.get("task_type", "binary_classification"))
    output_dim = int(config.get("output_dim", get_output_dim(task_type, config.get("n_classes"))))

    if model_type == "lstm":
        return LSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            output_dim=output_dim,
        )
    if model_type == "gru":
        return GRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            output_dim=output_dim,
        )
    if model_type == "cnn":
        return CNN1D(
            input_dim=input_dim,
            seq_len=seq_len,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            dropout_rate=dropout_rate,
            output_dim=output_dim,
        )
    if model_type == "transformer":
        return Transformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            output_dim=output_dim,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def train_pytorch_model(
    X_train: Any,
    y_train: Any,
    config: Dict[str, Any],
    device: torch.device,
    X_val: Optional[Any] = None,
    y_val: Optional[Any] = None,
) -> nn.Module:
    _require_torch("train_pytorch_model")

    model_type = str(config.get("model_type", "mlp")).lower()
    task_type = str(config.get("task_type", "binary_classification"))
    n_classes = config.get("n_classes")
    learning_rate = float(config.get("learning_rate", 0.001))
    weight_decay = float(config.get("weight_decay", 1e-5))
    epochs = int(config.get("epochs", 20))
    requested_batch_size = int(config.get("batch_size", 64))
    seq_len = int(config.get("seq_len", 20))
    use_amp = bool(config.get("use_amp", False)) and device.type == "cuda"
    use_tf32 = bool(config.get("use_tf32", False))
    loader_settings = _build_sequence_loader_runtime_settings(device, config)
    loss_spec = resolve_loss_spec(task_type, {"n_classes": n_classes})

    def _train_once(current_batch_size: int) -> nn.Module:
        is_prebuilt_sequence = isinstance(X_train, np.ndarray) and X_train.ndim == 3
        is_sequence_store = isinstance(X_train, PanelSequenceStore)

        if model_type == "mlp":
            input_dim = int(X_train.shape[1])
            hidden_dim = int(config.get("hidden_dim", 64))
            hidden_dims = [int(v) for v in config.get("hidden_dims", [hidden_dim])]
            dropout_rate = float(config.get("dropout_rate", 0.3))
            model = MLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate,
                hidden_dims=hidden_dims,
                output_dim=loss_spec.output_dim,
            )
            X_train_tensor = torch.as_tensor(
                _to_numpy_features(X_train), dtype=torch.float32, device=device
            )
            y_train_np = _to_numpy_labels(y_train)
            if task_type == "multiclass_classification":
                y_train_tensor = torch.as_tensor(
                    y_train_np.reshape(-1), dtype=torch.long, device=device
                )
            else:
                y_train_tensor = torch.as_tensor(
                    y_train_np.reshape(-1, 1), dtype=torch.float32, device=device
                )
            X_val_tensor = None
            y_val_tensor = None
            if X_val is not None and y_val is not None:
                X_val_tensor = torch.as_tensor(
                    _to_numpy_features(X_val), dtype=torch.float32, device=device
                )
                y_val_np = _to_numpy_labels(y_val)
                if task_type == "multiclass_classification":
                    y_val_tensor = torch.as_tensor(
                        y_val_np.reshape(-1), dtype=torch.long, device=device
                    )
                else:
                    y_val_tensor = torch.as_tensor(
                        y_val_np.reshape(-1, 1), dtype=torch.float32, device=device
                    )
            train_loader = None
            val_loader = None
            input_dim = int(X_train.shape[1])
        else:
            if is_sequence_store:
                input_dim = int(next(iter(X_train.feature_by_ticker.values())).shape[1])
            else:
                input_dim = int(X_train.shape[-1] if is_prebuilt_sequence else X_train.shape[1])
            model = _make_sequence_model(model_type, input_dim, config)
            train_loader = _make_sequence_loader_for_data(
                X_train,
                y_train,
                seq_len=seq_len,
                batch_size=current_batch_size,
                shuffle=True,
                task_type=task_type,
                loader_settings=loader_settings,
            )
            val_loader = None
            if X_val is not None and (y_val is not None or isinstance(X_val, PanelSequenceStore)):
                val_loader = _make_sequence_loader_for_data(
                    X_val,
                    y_val,
                    seq_len=seq_len,
                    batch_size=current_batch_size,
                    shuffle=False,
                    task_type=task_type,
                    loader_settings=loader_settings,
                )
            X_train_tensor = None
            y_train_tensor = None
            X_val_tensor = None
            y_val_tensor = None

        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = get_loss_fn(task_type, {"n_classes": n_classes})
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        scaler = GradScaler(enabled=use_amp)
        best_monitor_loss = float("inf")
        best_state_dict = None
        patience_counter = 0
        max_patience = 10

        with _torch_precision_context(device, use_tf32):
            for _epoch in range(epochs):
                model.train()
                total_loss = 0.0
                num_batches = 0

                if model_type == "mlp":
                    assert X_train_tensor is not None and y_train_tensor is not None
                    n_train = int(X_train_tensor.shape[0])
                    effective_batch_size = min(max(1, current_batch_size), n_train)
                    permutation = torch.randperm(n_train, device=device)
                    for start in range(0, n_train, effective_batch_size):
                        batch_idx = permutation[start : start + effective_batch_size]
                        X_batch = X_train_tensor.index_select(0, batch_idx)
                        y_batch = y_train_tensor.index_select(0, batch_idx)
                        optimizer.zero_grad(set_to_none=True)
                        with autocast(device_type=device.type, enabled=use_amp):
                            logits = model(X_batch)
                            if task_type == "multiclass_classification":
                                y_batch = y_batch.reshape(-1)
                            loss = criterion(logits, y_batch)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        total_loss += float(loss.item())
                        num_batches += 1
                    train_loss = total_loss / max(num_batches, 1)

                    if X_val_tensor is not None and y_val_tensor is not None:
                        model.eval()
                        val_total_loss = 0.0
                        val_batches = 0
                        with torch.no_grad():
                            for start in range(0, int(X_val_tensor.shape[0]), effective_batch_size):
                                X_batch = X_val_tensor[start : start + effective_batch_size]
                                y_batch = y_val_tensor[start : start + effective_batch_size]
                                with autocast(device_type=device.type, enabled=use_amp):
                                    logits = model(X_batch)
                                    if task_type == "multiclass_classification":
                                        y_batch = y_batch.reshape(-1)
                                    loss = criterion(logits, y_batch)
                                val_total_loss += float(loss.item())
                                val_batches += 1
                        monitor_loss = val_total_loss / max(val_batches, 1)
                    else:
                        monitor_loss = train_loss
                else:
                    assert train_loader is not None
                    non_blocking = (
                        bool(loader_settings.get("pin_memory", False)) and device.type == "cuda"
                    )
                    for batch in train_loader:
                        if len(batch) < 2:
                            continue
                        X_batch, y_batch = batch
                        X_batch = X_batch.to(device, non_blocking=non_blocking)
                        y_batch = y_batch.to(device, non_blocking=non_blocking)
                        optimizer.zero_grad(set_to_none=True)
                        with autocast(device_type=device.type, enabled=use_amp):
                            logits = model(X_batch)
                            loss = criterion(logits, y_batch)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        total_loss += float(loss.item())
                        num_batches += 1
                    train_loss = total_loss / max(num_batches, 1)

                    if val_loader is not None:
                        model.eval()
                        val_total_loss = 0.0
                        val_batches = 0
                        with torch.no_grad():
                            for batch in val_loader:
                                if len(batch) < 2:
                                    continue
                                X_batch, y_batch = batch
                                X_batch = X_batch.to(device, non_blocking=non_blocking)
                                y_batch = y_batch.to(device, non_blocking=non_blocking)
                                with autocast(device_type=device.type, enabled=use_amp):
                                    logits = model(X_batch)
                                    loss = criterion(logits, y_batch)
                                val_total_loss += float(loss.item())
                                val_batches += 1
                        monitor_loss = val_total_loss / max(val_batches, 1)
                    else:
                        monitor_loss = train_loss

                scheduler.step(monitor_loss)
                if monitor_loss < best_monitor_loss:
                    best_monitor_loss = monitor_loss
                    best_state_dict = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        model._seq_len = seq_len
        model._trained_batch_size = int(current_batch_size)
        model._loader_settings = dict(loader_settings)
        model._auto_batch_tune = bool(config.get("auto_batch_tune", True))
        model._target_vram_util = float(config.get("target_vram_util", 0.88))
        model._train_target_vram_util = float(
            config.get("train_target_vram_util", config.get("target_vram_util", 0.88))
        )
        model._predict_target_vram_util = float(
            config.get("predict_target_vram_util", config.get("target_vram_util", 0.88))
        )
        model._use_amp = bool(config.get("use_amp", False))
        model._use_tf32 = bool(config.get("use_tf32", False))
        model._predict_batch_size = (
            int(config.get("predict_batch_size", 0))
            if int(config.get("predict_batch_size", 0)) > 0
            else int(current_batch_size)
        )
        model._task_type = task_type
        return model

    if model_type == "mlp" or device.type != "cuda":
        return _train_once(requested_batch_size)

    if isinstance(X_train, PanelSequenceStore):
        input_dim = int(next(iter(X_train.feature_by_ticker.values())).shape[1])
    elif isinstance(X_train, np.ndarray) and X_train.ndim == 3:
        input_dim = int(X_train.shape[-1])
    else:
        input_dim = int(X_train.shape[1])

    tuned_batch_size = _auto_tune_sequence_batch_size(
        mode="train",
        model_type=model_type,
        input_dim=input_dim,
        config=config,
        X=X_train,
        y=y_train,
        device=device,
        base_batch_size=requested_batch_size,
        loader_settings=loader_settings,
    )

    candidate_batch_sizes = _build_retry_batch_sizes(max(1, tuned_batch_size))

    last_exc: Optional[Exception] = None
    for attempt_batch_size in candidate_batch_sizes:
        if attempt_batch_size != tuned_batch_size:
            logger.warning(
                "Retrying %s training with reduced batch_size=%d after CUDA OOM",
                model_type,
                attempt_batch_size,
            )
        try:
            return _train_once(attempt_batch_size)
        except Exception as exc:
            if not _is_cuda_oom_error(exc):
                raise
            last_exc = exc
            logger.warning(
                "CUDA OOM while training %s with batch_size=%d: %s",
                model_type,
                attempt_batch_size,
                exc,
            )
            _clear_cuda_cache()

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to train torch model")


def _ensure_sequence_predict_batch_size(
    model: nn.Module,
    X_seq: Any,
    device: torch.device,
) -> int:
    base = int(getattr(model, "_predict_batch_size", 0) or 0)
    if base <= 0:
        base = max(int(getattr(model, "_trained_batch_size", 64)), 1)
    if device.type != "cuda" or not bool(getattr(model, "_auto_batch_tune", True)):
        return base
    if getattr(model, "_predict_batch_size_tuned", False):
        return base

    loader_settings = dict(getattr(model, "_loader_settings", {}))
    input_dim = (
        int(next(iter(X_seq.feature_by_ticker.values())).shape[1])
        if isinstance(X_seq, PanelSequenceStore)
        else int(np.asarray(X_seq).shape[-1])
    )
    predict_config = {
        "seq_len": int(getattr(model, "_seq_len", 20)),
        "auto_batch_tune": True,
        "target_vram_util": float(getattr(model, "_target_vram_util", 0.88)),
        "predict_target_vram_util": float(
            getattr(model, "_predict_target_vram_util", getattr(model, "_target_vram_util", 0.88))
        ),
        "use_amp": bool(getattr(model, "_use_amp", False)),
        "use_tf32": bool(getattr(model, "_use_tf32", False)),
    }
    tuned = _auto_tune_sequence_batch_size(
        mode="predict",
        model_type=str(model.__class__.__name__).lower(),
        input_dim=input_dim,
        config=predict_config,
        X=X_seq,
        y=None,
        device=device,
        base_batch_size=max(base, int(getattr(model, "_trained_batch_size", base)) * 2),
        model=model,
        loader_settings=loader_settings,
    )
    model._predict_batch_size = int(tuned)
    model._predict_batch_size_tuned = True
    return int(tuned)


def predict_pytorch_model_tabular(
    model: nn.Module,
    X: pd.DataFrame,
    device: torch.device,
) -> pd.Series:
    outputs = predict_pytorch_model_tabular_outputs(model, X, device)
    return pd.Series(outputs["score"], index=X.index, name="dpoint")


def predict_pytorch_model_tabular_outputs(
    model: nn.Module,
    X: pd.DataFrame,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    _require_torch("predict_pytorch_model_tabular")
    model.eval()
    X_tensor = torch.tensor(X.to_numpy(dtype=np.float32, copy=False), dtype=torch.float32)
    batch_size = max(
        1, int(getattr(model, "_predict_batch_size", 16384 if device.type == "cuda" else 1024))
    )
    use_amp = bool(getattr(model, "_use_amp", False)) and device.type == "cuda"
    use_tf32 = bool(getattr(model, "_use_tf32", False))
    task_type = str(getattr(model, "_task_type", "binary_classification"))
    all_outputs: Dict[str, list[torch.Tensor]] = {
        "score": [],
        "prediction": [],
        "raw_output": [],
        "proba_up": [],
        "proba": [],
    }
    with _torch_precision_context(device, use_tf32):
        with torch.no_grad():
            for start in range(0, X_tensor.shape[0], batch_size):
                batch = X_tensor[start : start + batch_size].to(device)
                with autocast(device_type=device.type, enabled=use_amp):
                    logits = model(batch)
                batch_outputs = _model_outputs_to_prediction_components(logits, task_type)
                for key, value in batch_outputs.items():
                    all_outputs[key].append(value.detach().cpu())
    return {
        key: torch.cat(values, dim=0).numpy() if values else np.array([], dtype=np.float32)
        for key, values in all_outputs.items()
    }


def predict_pytorch_model_sequence(
    model: nn.Module,
    X_seq: Any,
    meta_df: pd.DataFrame,
    device: torch.device,
) -> pd.Series:
    outputs = predict_pytorch_model_sequence_outputs(model, X_seq, meta_df, device)
    return pd.Series(outputs["score"], index=meta_df.index, name="dpoint")


def predict_pytorch_model_sequence_outputs(
    model: nn.Module,
    X_seq: Any,
    meta_df: pd.DataFrame,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    _require_torch("predict_pytorch_model_sequence")
    model.eval()
    batch_size = _ensure_sequence_predict_batch_size(model, X_seq, device)
    loader_settings = dict(getattr(model, "_loader_settings", {}))
    use_amp = bool(getattr(model, "_use_amp", False)) and device.type == "cuda"
    use_tf32 = bool(getattr(model, "_use_tf32", False))
    if isinstance(X_seq, PanelSequenceStore):
        loader = make_panel_sequence_dataloader(
            X_seq, batch_size=batch_size, shuffle=False, **loader_settings
        )
    else:
        loader = make_sequence_dataloader(
            np.asarray(X_seq, dtype=np.float32),
            None,
            batch_size=batch_size,
            shuffle=False,
            **loader_settings,
        )
    all_outputs: Dict[str, list[torch.Tensor]] = {
        "score": [],
        "prediction": [],
        "raw_output": [],
        "proba_up": [],
        "proba": [],
    }
    non_blocking = bool(loader_settings.get("pin_memory", False)) and device.type == "cuda"
    task_type = str(getattr(model, "_task_type", "binary_classification"))
    with _torch_precision_context(device, use_tf32):
        with torch.no_grad():
            for batch in loader:
                X_batch = batch[0] if isinstance(batch, (tuple, list)) else batch
                X_batch = X_batch.to(device, non_blocking=non_blocking)
                with autocast(device_type=device.type, enabled=use_amp):
                    logits = model(X_batch)
                batch_outputs = _model_outputs_to_prediction_components(logits, task_type)
                for key, value in batch_outputs.items():
                    all_outputs[key].append(value.detach().cpu())
    return {
        key: torch.cat(values, dim=0).numpy() if values else np.array([], dtype=np.float32)
        for key, values in all_outputs.items()
    }


def predict_pytorch_model(
    model: nn.Module,
    X: pd.DataFrame,
    device: torch.device,
    seq_len: int = 20,
) -> pd.Series:
    _require_torch("predict_pytorch_model")
    if isinstance(model, (LSTM, GRU, CNN1D, Transformer)):
        n_samples = len(X)
        if n_samples - seq_len + 1 <= 0:
            raise ValueError(f"seq_len ({seq_len}) exceeds sample count ({n_samples})")
        X_tensor = torch.tensor(X.to_numpy(dtype=np.float32, copy=False), dtype=torch.float32)
        X_seq_all = X_tensor.unfold(0, seq_len, 1).permute(0, 2, 1).contiguous()
        batch_size = _ensure_sequence_predict_batch_size(model, X_seq_all.numpy(), device)
        loader_settings = dict(getattr(model, "_loader_settings", {}))
        loader = make_sequence_dataloader(
            X_seq_all.numpy(),
            None,
            batch_size=batch_size,
            shuffle=False,
            **loader_settings,
        )
        use_amp = bool(getattr(model, "_use_amp", False)) and device.type == "cuda"
        use_tf32 = bool(getattr(model, "_use_tf32", False))
        non_blocking = bool(loader_settings.get("pin_memory", False)) and device.type == "cuda"
        all_probs = []
        with _torch_precision_context(device, use_tf32):
            with torch.no_grad():
                for batch in loader:
                    X_batch = batch[0] if isinstance(batch, (tuple, list)) else batch
                    X_batch = X_batch.to(device, non_blocking=non_blocking)
                    with autocast(device_type=device.type, enabled=use_amp):
                        logits = model(X_batch)
                    all_probs.append(_logits_to_probs(logits).detach().to(torch.float32).cpu())
        probs = torch.cat(all_probs, dim=0).to(torch.float32).numpy().flatten()
        return pd.Series(probs, index=X.index[seq_len - 1 :], name="dpoint")
    return predict_pytorch_model_tabular(model, X, device)


def _try_import_xgboost() -> Any:
    try:
        import xgboost as xgb

        return xgb
    except Exception:
        return None


def make_model(candidate: Dict[str, Any], seed: int) -> Any:
    model_config = candidate["model_config"]
    model_type = str(model_config["model_type"]).lower()
    task_type = str(model_config.get("task_type", "binary_classification"))
    params = dict(model_config.get("params", {}))

    if model_type == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=int(params.get("max_iter", 8000)),
                        C=float(params.get("C", 1.0)),
                        penalty=str(params.get("penalty", "l2")),
                        solver=str(params.get("solver", "lbfgs")),
                        class_weight=params.get("class_weight"),
                        random_state=seed,
                    ),
                ),
            ]
        )

    if model_type == "sgd":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        alpha=float(params.get("alpha", 1e-4)),
                        max_iter=int(params.get("max_iter", 3000)),
                        tol=float(params.get("tol", 1e-3)),
                        random_state=seed,
                    ),
                ),
            ]
        )

    if model_type == "xgb":
        xgb = _try_import_xgboost()
        if xgb is not None:
            xgb_params = dict(params)
            xgb_params.setdefault("random_state", seed)
            xgb_params.setdefault("tree_method", "hist")
            if task_type == "regression":
                xgb_params.setdefault("objective", "reg:squarederror")
                xgb_params.setdefault("eval_metric", "rmse")
                xgb_params.setdefault("verbosity", 0)
                return xgb.XGBRegressor(**xgb_params)
            if task_type == "multiclass_classification":
                n_classes = int(model_config.get("n_classes", params.get("num_class", 3)) or 3)
                xgb_params.setdefault("objective", "multi:softprob")
                xgb_params.setdefault("num_class", n_classes)
                xgb_params.setdefault("eval_metric", "mlogloss")
                xgb_params.setdefault("verbosity", 0)
                return xgb.XGBClassifier(**xgb_params)
            xgb_params.setdefault("objective", "binary:logistic")
            xgb_params.setdefault("eval_metric", "logloss")
            xgb_params.setdefault("verbosity", 0)
            return xgb.XGBClassifier(**xgb_params)
        fallback = (
            GradientBoostingRegressor(random_state=seed)
            if task_type == "regression"
            else GradientBoostingClassifier(random_state=seed)
        )
        return fallback

    raise ValueError(f"Unsupported model_type for make_model: {model_type}")


def predict_dpoint(model: Any, X: pd.DataFrame) -> pd.Series:
    X_values = (
        X.to_numpy(dtype=np.float32, copy=False)
        if isinstance(X, pd.DataFrame)
        else np.asarray(X, dtype=np.float32)
    )
    task_type = str(getattr(model, "_task_type", "binary_classification"))
    if is_torch_model_instance(model):
        device = resolve_torch_device(str(getattr(model, "_device_preference", "auto")))
        if getattr(model, "_is_panel_sequence_model", False):
            raise ValueError(
                "predict_dpoint does not support panel sequence models. Use predict_panel instead."
            )
        if isinstance(model, (LSTM, GRU, CNN1D, Transformer)):
            return predict_pytorch_model(
                model, pd.DataFrame(X_values), device, seq_len=int(getattr(model, "_seq_len", 20))
            )
        return predict_pytorch_model_tabular(model, pd.DataFrame(X_values), device)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_values)
        if task_type == "multiclass_classification":
            return pd.Series(
                multiclass_probabilities_to_score(proba),
                index=getattr(X, "index", None),
                name="dpoint",
            )
        return pd.Series(proba[:, 1], index=getattr(X, "index", None), name="dpoint")
    pred = model.predict(X_values)
    return pd.Series(pred, index=getattr(X, "index", None), name="dpoint")


def _torch_feature_meta(model: Any) -> Dict[str, Any]:
    return {
        "feature_names": list(getattr(model, "_feature_names", [])),
        "seq_len": int(getattr(model, "_seq_len", 20)),
        "is_panel_sequence_model": bool(getattr(model, "_is_panel_sequence_model", False)),
        "device_preference": str(getattr(model, "_device_preference", "auto")),
        "trained_batch_size": int(getattr(model, "_trained_batch_size", 64)),
        "predict_batch_size": int(getattr(model, "_predict_batch_size", 0)),
        "loader_settings": dict(getattr(model, "_loader_settings", {})),
        "auto_batch_tune": bool(getattr(model, "_auto_batch_tune", True)),
        "target_vram_util": float(getattr(model, "_target_vram_util", 0.88)),
        "train_target_vram_util": float(
            getattr(model, "_train_target_vram_util", getattr(model, "_target_vram_util", 0.88))
        ),
        "predict_target_vram_util": float(
            getattr(model, "_predict_target_vram_util", getattr(model, "_target_vram_util", 0.88))
        ),
        "use_amp": bool(getattr(model, "_use_amp", False)),
        "use_tf32": bool(getattr(model, "_use_tf32", False)),
        "task_type": str(getattr(model, "_task_type", "binary_classification")),
        "n_classes": getattr(model, "_n_classes", None),
        "has_preprocessor": getattr(model, "_preprocessor", None) is not None,
    }


def save_trained_model(
    model: Any,
    model_config: Dict[str, Any],
    destination: str,
    model_contract: Optional[Dict[str, Any]] = None,
) -> str:
    if is_torch_model_instance(model):
        os.makedirs(destination, exist_ok=True)
        state_path = os.path.join(destination, "model_state.pt")
        config_path = os.path.join(destination, "model_config.json")
        feature_meta_path = os.path.join(destination, "feature_meta.json")
        contract_path = os.path.join(destination, "model_contract.json")
        normalizer_path = os.path.join(destination, "normalizer.pkl")
        torch.save(model.state_dict(), state_path)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(model_config, f, ensure_ascii=False, indent=2)
        with open(feature_meta_path, "w", encoding="utf-8") as f:
            json.dump(_torch_feature_meta(model), f, ensure_ascii=False, indent=2)
        if model_contract is not None:
            with open(contract_path, "w", encoding="utf-8") as f:
                json.dump(model_contract, f, ensure_ascii=False, indent=2)
        if getattr(model, "_preprocessor", None) is not None:
            joblib.dump(model._preprocessor, normalizer_path)
        return destination

    if destination.endswith(".joblib"):
        model_path = destination
    else:
        model_path = f"{destination}.joblib"
    joblib.dump(model, model_path)
    if model_contract is not None:
        root, _ = os.path.splitext(model_path)
        with open(f"{root}.contract.json", "w", encoding="utf-8") as f:
            json.dump(model_contract, f, ensure_ascii=False, indent=2)
    return model_path


def _build_saved_torch_model(model_config: Dict[str, Any], feature_meta: Dict[str, Any]) -> Any:
    _require_torch("load_saved_model")
    model_type = str(model_config.get("model_type", "mlp")).lower()
    model_params = dict(model_config.get("model_params", {}))
    task_type = str(
        feature_meta.get("task_type", model_config.get("task_type", "binary_classification"))
    )
    n_classes = feature_meta.get("n_classes", model_config.get("n_classes"))
    feature_names = list(feature_meta.get("feature_names", []))
    input_dim = len(feature_names)
    if input_dim <= 0:
        raise ValueError("Saved torch model is missing feature_names/input_dim metadata")

    if model_type == "mlp":
        hidden_dim = int(model_params.get("hidden_dim", 64))
        hidden_dims = [int(v) for v in model_params.get("hidden_dims", [hidden_dim])]
        model = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout_rate=float(model_params.get("dropout_rate", 0.3)),
            hidden_dims=hidden_dims,
            output_dim=get_output_dim(task_type, n_classes),
        )
    else:
        runtime_config = {"task_type": task_type, "n_classes": n_classes, **model_params}
        model = _make_sequence_model(model_type, input_dim, runtime_config)

    state_path = feature_meta["_state_path"]
    state_dict = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model._feature_names = feature_names
    model._seq_len = int(feature_meta.get("seq_len", model_params.get("seq_len", 20)))
    model._is_panel_sequence_model = bool(feature_meta.get("is_panel_sequence_model", False))
    model._device_preference = str(
        feature_meta.get("device_preference", model_config.get("device", "auto"))
    )
    model._trained_batch_size = int(
        feature_meta.get("trained_batch_size", model_params.get("batch_size", 64))
    )
    model._predict_batch_size = int(
        feature_meta.get("predict_batch_size", model._trained_batch_size)
    )
    model._loader_settings = dict(feature_meta.get("loader_settings", {}))
    model._auto_batch_tune = bool(feature_meta.get("auto_batch_tune", True))
    model._target_vram_util = float(feature_meta.get("target_vram_util", 0.88))
    model._train_target_vram_util = float(
        feature_meta.get("train_target_vram_util", model._target_vram_util)
    )
    model._predict_target_vram_util = float(
        feature_meta.get("predict_target_vram_util", model._target_vram_util)
    )
    model._use_amp = bool(feature_meta.get("use_amp", False))
    model._use_tf32 = bool(feature_meta.get("use_tf32", False))
    model._task_type = task_type
    model._n_classes = n_classes
    normalizer_path = feature_meta.get("_normalizer_path")
    if normalizer_path and os.path.exists(normalizer_path):
        model._preprocessor = joblib.load(normalizer_path)
    return model


def load_saved_model(path: str) -> Any:
    if os.path.isdir(path):
        state_path = os.path.join(path, "model_state.pt")
        config_path = os.path.join(path, "model_config.json")
        feature_meta_path = os.path.join(path, "feature_meta.json")
        if (
            os.path.exists(state_path)
            and os.path.exists(config_path)
            and os.path.exists(feature_meta_path)
        ):
            with open(config_path, "r", encoding="utf-8") as f:
                model_config = json.load(f)
            with open(feature_meta_path, "r", encoding="utf-8") as f:
                feature_meta = json.load(f)
            feature_meta["_state_path"] = state_path
            feature_meta["_normalizer_path"] = os.path.join(path, "normalizer.pkl")
            return _build_saved_torch_model(model_config, feature_meta)
    return joblib.load(path)
