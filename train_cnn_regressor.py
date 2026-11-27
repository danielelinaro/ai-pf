import math
import os
import time
from pathlib import Path
from typing import Callable, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


__all__ = ["MeanAbsoluteRelativeError", "train_model", "load_checkpoint"]


class MeanAbsoluteRelativeError(nn.Module):
    """mean(|pred - target| / |target|). Targets must be bounded away from zero."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((pred - target).abs() / target.abs().clamp_min(self.eps)).mean()


def _run_epoch(model, loader, loss_fn, device, optimizer=None, grad_clip=None) -> float:
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x)
            y = y.to(device=device, dtype=pred.dtype, non_blocking=True).reshape(pred.shape)
            loss = loss_fn(pred, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss ({loss.item()}); check the data and the learning rate.")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            batch_size = x.shape[0]
            total += loss.item() * batch_size
            count += batch_size
    return total / count


def _save_checkpoint(path, state_dict, epoch, val_loss, model_config, history) -> None:
    """Write the checkpoint atomically, so an interrupted run never leaves a corrupt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model_state_dict": state_dict,
            "epoch": epoch,
            "val_loss": val_loss,
            "model_config": model_config,
            "history": history,
        },
        tmp,
    )
    os.replace(tmp, path)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_path: str = "best_model.pt",
    loss_fn: Optional[Callable] = None,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    grad_clip: Optional[float] = None,
    patience: Optional[int] = None,
    min_delta: float = 0.0,
    model_config: Optional[dict] = None,
    device: Optional[str] = None,
    logger: Optional[Any] = None,
    verbose: bool = True,
) -> dict:
    """
    Train `model` and keep the weights with the lowest validation loss.

    Whenever the validation loss improves, a checkpoint is written to `checkpoint_path`.
    When training ends, the best weights are loaded back into `model`.

    Args:
        model:           e.g. a DeepFilteringRegressor.
        train_loader:    yields (x, y) with x: (B, M, L) and y: (B,) or (B, 1).
        val_loader:      same format; used to select the best model.
        checkpoint_path: where the best model is saved.
        loss_fn:         defaults to nn.MSELoss(). MeanAbsoluteRelativeError() is also provided.
        epochs:          maximum number of epochs.
        lr, weight_decay: Adam settings (the paper uses Adam with lr=1e-3). Ignored if
                         `optimizer` is given.
        optimizer:       a ready-made optimizer, overriding the Adam defaults.
        scheduler:       optional LR scheduler. ReduceLROnPlateau is stepped with the
                         validation loss, any other scheduler is stepped once per epoch.
        grad_clip:       max gradient norm, or None for no clipping.
        patience:        stop if the validation loss has not improved for this many
                         epochs (None disables early stopping).
        min_delta:       minimum decrease of the validation loss that counts as an improvement.
        model_config:    keyword arguments needed to rebuild the model (e.g. the kwargs of
                         DeepFilteringRegressor). Stored in the checkpoint so that
                         `load_checkpoint` can recreate the model without you re-typing them.
                         It must contain only plain Python types (ints, lists, bools, ...).
        logger           Logging object to print to. If not provided, sys.stdout is used
        device:          "cuda", "cpu", ... (default: cuda if available, else cpu).

    Returns:
        dict with "history" (per-epoch train/val loss and lr), "best_epoch",
        "best_val_loss" and "checkpoint_path".
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print_fun = logger.info if logger is not None else print

    history = []
    best_val, best_epoch, best_state = math.inf, 0, None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = _run_epoch(model, train_loader, loss_fn, device, optimizer, grad_clip)
        val_loss = _run_epoch(model, val_loader, loss_fn, device)
        current_lr = optimizer.param_groups[0]["lr"]

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr})

        improved = val_loss < best_val - min_delta
        if improved:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _save_checkpoint(checkpoint_path, best_state, epoch, val_loss, model_config, history)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if verbose:
            print_fun(
                f"epoch {epoch:4d}/{epochs}  train {train_loss:.5f}  val {val_loss:.5f}  "
                f"lr {current_lr:.2e}  {time.time() - t0:.1f}s" + ("  *" if improved else "")
            )

        if patience is not None and epochs_without_improvement >= patience:
            if verbose:
                print_fun(f"Early stopping: no improvement for {patience} epochs.")
            break

    # The checkpoint written at the best epoch holds an older copy of `history`;
    # refresh it so that it contains the full training curve.
    if best_state is not None:
        _save_checkpoint(checkpoint_path, best_state, best_epoch, best_val, model_config, history)
        model.load_state_dict(best_state)

    if verbose:
        print_fun(f"Best validation loss {best_val:.5f} at epoch {best_epoch}; saved to {checkpoint_path}")

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "checkpoint_path": str(checkpoint_path),
    }


def load_checkpoint(path: str, model: Optional[nn.Module] = None, device: str = "cpu"):
    """
    Load a checkpoint written by `train_model`.

    If `model` is given, the weights are loaded into it. Otherwise the model is
    rebuilt as DeepFilteringRegressor(**checkpoint["model_config"]), which requires
    `model_config` to have been passed to `train_model`.

    Returns (model in eval mode, checkpoint dict).
    """
    ckpt = torch.load(path, map_location=device)
    if model is None:
        if ckpt.get("model_config") is None:
            raise ValueError("The checkpoint has no model_config; pass an already-built `model`.")
        from cnn_regressor import DeepFilteringRegressor

        model = DeepFilteringRegressor(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


if __name__ == "__main__":
    from torch.utils.data import TensorDataset, random_split

    from cnn_regressor import DeepFilteringRegressor

    torch.manual_seed(0)
    dtype = torch.float
    device = "cpu"

    # Toy problem: recover the frequency of a noisy sinusoid observed in M=2 dimensions.
    N, M, L = 3000, 2, 1024
    freq = torch.empty(N, dtype=dtype, device=device).uniform_(5.0, 25.0)  # target parameter (Hz)
    t = torch.arange(L, dtype=dtype, device=device) / L  # 1 second
    phase = torch.rand(N, M, 1, dtype=dtype, device=device) * 2 * math.pi
    x = torch.sin(2 * math.pi * freq.view(N, 1, 1) * t + phase) + \
        1.0 * torch.randn(N, M, L, dtype=dtype, device=device)
    x = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)  # standardize each series

    dataset = TensorDataset(x, freq)
    train_set, val_set = random_split(dataset, [2400, 600], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256)

    config = dict(
        num_channels=M, input_length=L, num_layers=2,
        kernel_size=[16, 8], pool_size=4, filters=[8, 16],
        hidden_units=[32],
    )
    model = DeepFilteringRegressor(**config)

    result = train_model(
        model, train_loader, val_loader,
        checkpoint_path="checkpoints/best_model.pt",
        loss_fn=MeanAbsoluteRelativeError(),
        epochs=3, lr=1e-3, patience=8,
        model_config=config,
        device=device,
    )

    # Reload the best model from disk, rebuilding it from the stored config
    best_model, ckpt = load_checkpoint("checkpoints/best_model.pt")
    print("Reloaded model from epoch", ckpt["epoch"], "with val loss", round(ckpt["val_loss"], 5))
    with torch.no_grad():
        print(best_model(x[:3]).squeeze(-1), freq[:3])
