"""Supervised fine-tuning loop.

AdamW with linear warm-up then cosine decay, fp16 autocast with gradient scaling (T4 has
no bf16), gradient clipping, model selection on the inner validation split and early
stopping. ``last.pt`` is written atomically after every epoch with everything needed to
resume after a Kaggle session ends; ``best.pt`` holds the selected weights.

Logged in the run directory, so nothing has to be re-run to inspect training later:
    train_steps.jsonl   every ``log_every_steps`` steps: window-mean loss, learning rate,
                        gradient norm, AMP-skipped steps, images/s, peak GPU memory
    history.jsonl       every epoch: mean train loss, validation metrics, time
    val_logits/         every epoch: validation logits, labels and index positions

Data order is a function of (seed, epoch) only, so a resumed run sees the same batches
as an uninterrupted one.
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from capsule_xai_audit.config import TrainConfig
from capsule_xai_audit.data.datasets import CachedImages, GeometricAugment, make_loader, to_unit_float
from capsule_xai_audit.eval.metrics import compute_metrics, probs_from_logits
from capsule_xai_audit.eval.predict import predict_logits
from capsule_xai_audit.slices.color import RandomLabShift
from capsule_xai_audit.utils.io import append_jsonl, atomic_write_bytes
from capsule_xai_audit.utils.logging import get_logger
from capsule_xai_audit.utils.runtime import autocast

log = get_logger(__name__)

VAL_METRICS = ("macro_f1", "nll", "ece", "accuracy")


@dataclass
class TrainState:
    epoch: int = 0  # epochs completed
    best_score: float = -math.inf
    best_epoch: int = -1
    bad_epochs: int = 0
    stopped_early: bool = False
    history: list[dict[str, float]] = field(default_factory=list)


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """No weight decay on biases, norm parameters and embeddings (1-D tensors)."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(("pos_embed", "cls_token", "reg_token")) else decay).append(
            p
        )
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def warmup_cosine(total_steps: int, warmup_steps: int) -> Any:
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return factor


def _save_torch(path: Path, obj: Any) -> None:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    atomic_write_bytes(path, buffer.getvalue())


def class_weights(labels: np.ndarray, n_classes: int, scheme: str) -> torch.Tensor | None:
    """Cross-entropy weights ``1/sqrt(n_c)``, scaled so the mean weight over frames is 1.

    Classes absent from ``labels`` get weight 0 (they never occur in the loss anyway).
    """
    if scheme == "none":
        return None
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1.0)), 0.0)
    w *= counts.sum() / (w * counts).sum()
    return torch.tensor(w, dtype=torch.float32)


def selection_score(metrics: dict[str, float], select_metric: str) -> float:
    return metrics["macro_f1"] if select_metric == "macro_f1" else -metrics["nll"]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        fit_data: CachedImages,
        val_data: CachedImages,
        cfg: TrainConfig,
        run_dir: Path,
        device: torch.device,
        num_classes: int | None = None,
    ) -> None:
        if fit_data.labels is None:
            raise ValueError("training data needs labels")
        self.model = model.to(device)
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = device
        self.amp = cfg.amp and device.type == "cuda"
        self.generator = torch.Generator()
        self.fit_loader = make_loader(
            fit_data,
            cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            generator=self.generator,
            drop_last=len(fit_data) > cfg.batch_size,
        )
        self.val_loader = make_loader(
            val_data, cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers
        )
        color = (
            RandomLabShift(cfg.color_aug_max_delta_e, cfg.color_aug_prob)
            if cfg.color_aug_max_delta_e > 0
            else None
        )
        self.augment = GeometricAugment(color=color)
        self.steps_per_epoch = len(self.fit_loader)
        if cfg.max_steps_per_epoch is not None:
            self.steps_per_epoch = min(self.steps_per_epoch, cfg.max_steps_per_epoch)
        total = self.steps_per_epoch * cfg.epochs
        self.optimizer = torch.optim.AdamW(param_groups(self.model, cfg.weight_decay), lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, warmup_cosine(total, round(cfg.warmup_epochs * self.steps_per_epoch))
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        weights = class_weights(
            fit_data.labels, num_classes or int(fit_data.labels.max()) + 1, cfg.class_weighting
        )
        self.loss_fn = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device))
        self.state = TrainState()

    # ----------------------------------------------------------------- checkpoints

    @property
    def last_path(self) -> Path:
        return self.run_dir / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.run_dir / "best.pt"

    def _save_last(self) -> None:
        _save_torch(
            self.last_path,
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "state": asdict(self.state),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        )

    def try_resume(self) -> bool:
        if not self.last_path.exists():
            return False
        ckpt = torch.load(self.last_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.state = TrainState(**ckpt["state"])
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        if ckpt["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng"]])
        log.info("resumed %s after epoch %d", self.run_dir, self.state.epoch)
        return True

    # ----------------------------------------------------------------- loop

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.generator.manual_seed(self.cfg.seed * 10_007 + epoch)
        total, count = 0.0, 0
        window: dict[str, float] = {"loss": 0.0, "grad_norm": 0.0, "steps": 0, "finite": 0, "images": 0}
        window_start = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        for step, (images, labels, _) in enumerate(self.fit_loader):
            if step >= self.steps_per_epoch:
                break
            x = self.augment(to_unit_float(images.to(self.device, non_blocking=True)))
            y = labels.to(self.device, non_blocking=True)
            with autocast(self.device, self.amp):
                loss = self.loss_fn(self.model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}")
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            loss_value = float(loss.detach())
            total += loss_value * len(y)
            count += len(y)
            window["loss"] += loss_value
            window["steps"] += 1
            window["images"] += len(y)
            if np.isfinite(grad_norm):  # non-finite: the scaler skipped this step
                window["grad_norm"] += grad_norm
                window["finite"] += 1
            last = step + 1 == self.steps_per_epoch
            if window["steps"] >= self.cfg.log_every_steps or last:
                self._log_steps(epoch, step, window, time.perf_counter() - window_start)
                window = {"loss": 0.0, "grad_norm": 0.0, "steps": 0, "finite": 0, "images": 0}
                window_start = time.perf_counter()
        return total / max(1, count)

    def _log_steps(self, epoch: int, step: int, window: dict[str, float], seconds: float) -> None:
        record = {
            "epoch": epoch,
            "step": step + 1,
            "global_step": epoch * self.steps_per_epoch + step + 1,
            "loss": window["loss"] / window["steps"],
            "lr": self.optimizer.param_groups[0]["lr"],
            "grad_norm": window["grad_norm"] / window["finite"] if window["finite"] else None,
            "amp_skipped_steps": int(window["steps"] - window["finite"]),
            "images_per_s": window["images"] / max(seconds, 1e-9),
            "gpu_mem_gib": torch.cuda.max_memory_allocated(self.device) / 2**30
            if self.device.type == "cuda"
            else None,
        }
        append_jsonl(self.run_dir / "train_steps.jsonl", record)
        log.info(
            "epoch %d step %d/%d loss %.4f lr %.2e %.0f img/s",
            epoch,
            record["step"],
            self.steps_per_epoch,
            record["loss"],
            record["lr"],
            record["images_per_s"],
        )

    def validate(self, epoch: int | None = None) -> dict[str, float]:
        logits, labels, positions = predict_logits(self.model, self.val_loader, self.device, self.amp)
        if epoch is not None and self.cfg.save_val_logits:
            out = self.run_dir / "val_logits" / f"epoch{epoch:02d}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, logits=logits.astype(np.float32), labels=labels, positions=positions)
        return compute_metrics(probs_from_logits(logits), labels, VAL_METRICS)

    def fit(self) -> TrainState:
        self.try_resume()
        while self.state.epoch < self.cfg.epochs and not self.state.stopped_early:
            epoch = self.state.epoch
            start = time.perf_counter()
            train_loss = self._train_epoch(epoch)
            val = self.validate(epoch)
            score = selection_score(val, self.cfg.select_metric)
            improved = score > self.state.best_score
            if improved:
                self.state.best_score, self.state.best_epoch, self.state.bad_epochs = score, epoch, 0
                _save_torch(self.best_path, {"model": self.model.state_dict(), "epoch": epoch, "val": val})
            else:
                self.state.bad_epochs += 1
            self.state.stopped_early = self.state.bad_epochs >= self.cfg.early_stopping_patience
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - start,
                **{f"val_{k}": v for k, v in val.items()},
            }
            self.state.history.append(record)
            self.state.epoch = epoch + 1
            self._save_last()
            append_jsonl(self.run_dir / "history.jsonl", record)
            log.info(
                "epoch %d loss %.4f val macro-F1 %.4f nll %.4f%s",
                epoch,
                train_loss,
                val["macro_f1"],
                val["nll"],
                " *" if improved else "",
            )
        return self.state

    def load_best(self) -> None:
        ckpt = torch.load(self.best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
