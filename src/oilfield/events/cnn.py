"""1D-CNN over the 10 s sensor sequence inside each window.

Input per window: for every sensor, the 60-step sequence with the window mean removed and a
global per-sensor scale applied, plus a 0/1 channel marking which steps were valid. Removing
the mean makes the network read the *shape* of the signal (ramps, steps, oscillations), which
transfers across wells far better than absolute pressures do; level and drift information
comes from the gradient-boosting model it is ensembled with.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import torch
from torch import nn


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def conv_block(c_in: int, c_out: int, k: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv1d(c_in, c_out, k, padding=k // 2, bias=False), nn.BatchNorm1d(c_out), nn.ReLU())


class WellEventCNN(nn.Module):
    def __init__(self, n_sensors: int, n_classes: int, width: int = 64, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(2 * n_sensors, width, 7),
            conv_block(width, width, 5),
            nn.MaxPool1d(2),
            conv_block(width, 2 * width, 5),
            conv_block(2 * width, 2 * width, 3),
            nn.MaxPool1d(2),
            conv_block(2 * width, 2 * width, 3),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * width, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.head(torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1))


def sensor_scale(train_std: np.ndarray) -> np.ndarray:
    """Per-sensor scale from the training windows' standard deviations (n_windows x n_sensors)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a sensor with no valid windows
        scale = np.nanpercentile(train_std, 90, axis=0)
    return np.where(np.isfinite(scale) & (scale > 1e-3), scale, 1.0).astype(np.float32)


def make_batch(values: np.ndarray, starts: np.ndarray, length: int, scale: np.ndarray) -> np.ndarray:
    x = values[starts[:, None] + np.arange(length)]           # (B, L, S)
    valid = np.isfinite(x)
    filled = np.where(valid, x, 0.0)
    mean = filled.sum(axis=1, keepdims=True) / np.maximum(valid.sum(axis=1, keepdims=True), 1)
    x = np.clip(np.where(valid, (x - mean) / scale, 0.0), -10.0, 10.0)
    return np.concatenate([x, valid], axis=2, dtype=np.float32).transpose(0, 2, 1)  # (B, 2S, L)


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
    return (w / w[counts > 0].mean()).astype(np.float32)


def train_cnn(
    values: np.ndarray,
    starts: np.ndarray,
    y: np.ndarray,
    *,
    length: int,
    scale: np.ndarray,
    n_classes: int,
    params: dict,
    seed: int = 0,
    device: torch.device | None = None,
    on_epoch: Callable[[int, float], None] | None = None,
) -> WellEventCNN:
    device = device or pick_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = WellEventCNN(values.shape[1], n_classes, params.get("width", 64), params.get("dropout", 0.3)).to(device)

    epochs, batch_size = params.get("epochs", 8), params.get("batch_size", 512)
    steps_per_epoch = int(np.ceil(len(starts) / batch_size))
    opt = torch.optim.AdamW(model.parameters(), lr=params.get("lr", 1e-3), weight_decay=params.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=params.get("lr", 1e-3), total_steps=epochs * steps_per_epoch)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights(y, n_classes)).to(device))

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(starts))
        total = 0.0
        for i in range(steps_per_epoch):
            idx = order[i * batch_size:(i + 1) * batch_size]
            xb = torch.from_numpy(make_batch(values, starts[idx], length, scale)).to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        if on_epoch:
            on_epoch(epoch, total / len(starts))
    return model


@torch.no_grad()
def predict_cnn(model: WellEventCNN, values: np.ndarray, starts: np.ndarray, *, length: int, scale: np.ndarray,
                batch_size: int = 2048, device: torch.device | None = None) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(starts), batch_size):
        xb = torch.from_numpy(make_batch(values, starts[i:i + batch_size], length, scale)).to(device)
        out.append(torch.softmax(model(xb), dim=1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.head[-1].out_features), dtype=np.float32)
