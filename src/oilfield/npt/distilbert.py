"""Fine-tune DistilBERT to tag the cause of an NPT remark.

A plain PyTorch loop rather than ``Trainer``: class-weighted loss for the long tail of causes,
model selection on validation macro-F1, and per-epoch MLflow metrics, in a few dozen lines.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from oilfield.events.cnn import pick_device
from oilfield.npt.labeling import CAUSES


def load_pretrained(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CAUSES),
        id2label=dict(enumerate(CAUSES)),
        label2id={c: i for i, c in enumerate(CAUSES)},
    )
    return model, tokenizer


def _encode(tokenizer, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    return tokenizer(list(texts), truncation=True, max_length=max_length, padding=True, return_tensors="pt")


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(CAUSES)).astype(np.float64)
    w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
    return torch.tensor(w / w[counts > 0].mean(), dtype=torch.float32)


@torch.no_grad()
def predict_proba(model, tokenizer, texts: list[str], max_length: int = 128, batch_size: int = 64,
                  device: torch.device | None = None) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = {k: v.to(device) for k, v in _encode(tokenizer, texts[i:i + batch_size], max_length).items()}
        out.append(torch.softmax(model(**enc).logits, dim=-1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CAUSES)), dtype=np.float32)


def fine_tune(
    model,
    tokenizer,
    train_texts: list[str],
    train_y: np.ndarray,
    val_texts: list[str],
    val_y: np.ndarray,
    params: dict,
    device: torch.device | None = None,
    on_epoch: Callable[[int, dict], None] | None = None,
):
    """Train and return the model state from the epoch with the best validation macro-F1."""
    device = device or pick_device()
    torch.manual_seed(params.get("seed", 13))
    rng = np.random.default_rng(params.get("seed", 13))
    model.to(device)

    epochs, batch_size, max_length = params.get("epochs", 4), params.get("batch_size", 32), params.get("max_length", 128)
    steps = epochs * int(np.ceil(len(train_texts) / batch_size))
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)], "weight_decay": params.get("weight_decay", 0.01)},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=params.get("learning_rate", 5e-5))
    sched = get_linear_schedule_with_warmup(opt, int(params.get("warmup_ratio", 0.06) * steps), steps)
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(train_y).to(device))

    best_f1, best_state = -1.0, None
    texts = np.asarray(train_texts, dtype=object)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(texts))
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            enc = {k: v.to(device) for k, v in _encode(tokenizer, texts[idx].tolist(), max_length).items()}
            yb = torch.as_tensor(train_y[idx], device=device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(**enc).logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)

        val_pred = predict_proba(model, tokenizer, list(val_texts), max_length, device=device).argmax(1)
        val_f1 = f1_score(val_y, val_pred, labels=sorted(set(val_y)), average="macro", zero_division=0)
        if on_epoch:
            on_epoch(epoch, {"train_loss": total / len(texts), "val_macro_f1": val_f1})
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


def save(model, tokenizer, out_dir: str | Path, extra: dict | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "npt_model.json").write_text(json.dumps({"kind": "distilbert", "causes": CAUSES, **(extra or {})}, indent=2))


def load(model_dir: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model.to(pick_device()), tokenizer
