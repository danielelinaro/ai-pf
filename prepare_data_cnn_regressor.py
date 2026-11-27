from typing import Optional

import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["make_dataloaders"]

def scale_per_dimension(scaler: StandardScaler, X: np.ndarray, fit: bool = False) -> np.ndarray:
    """
    Standardize an (N, M, L) array with one mean/std per dimension (M of them),
    computed over all trials and all time points.

    StandardScaler only handles 2D input, so the array is reshaped to (N*L, M),
    where each column is one dimension, scaled, and reshaped back to (N, M, L).
    """
    N, M, L = X.shape
    X2d = X.transpose(0, 2, 1).reshape(-1, M)
    X2d = scaler.fit_transform(X2d) if fit else scaler.transform(X2d)
    return X2d.reshape(N, L, M).transpose(0, 2, 1)


_TORCH_TO_NUMPY_DTYPES = {
    torch.float: np.float32,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int: np.int32,
    torch.int32: np.int32,
    torch.int64: np.int64,
}

def to_tensor(
    a: np.ndarray,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """numpy -> contiguous tensor of type dtype (StandardScaler returns float64)."""
    return torch.from_numpy(
        np.ascontiguousarray(
            a,
            dtype=_TORCH_TO_NUMPY_DTYPES[dtype]
        ),
    ).to(device)


def make_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    batch_size: int = 64,
    scale_target: bool = False,
    random_state: Optional[int] = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float,
):
    """
    Split, normalize and wrap the data for training.

    Args:
        X:            array of shape (N, M, L).
        y:            array of shape (N,), the parameter to predict.
        val_size, test_size: fractions of the N trials. Set test_size=0 for no test set.
        batch_size:   batch size of all loaders.
        scale_target: also standardize y. Predictions must then be mapped back with
                      scalers["y"].inverse_transform. Do not combine it with a relative-error
                      loss, since standardized targets can be zero or negative.
        random_state: seed of the split.

    Returns:
        loaders:  dict with "train", "val" (and "test" if test_size > 0) DataLoaders.
        scalers:  dict with the fitted StandardScalers ("x", and "y" if scale_target).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 3 or len(X) != len(y):
        raise ValueError(f"Expected X of shape (N, M, L) and y of shape (N,), got {X.shape} and {y.shape}")

    # 1) Split first, so the scalers never see validation/test data (no leakage).
    idx = np.arange(len(X))
    if test_size > 0:
        idx_trainval, idx_test = train_test_split(idx, test_size=test_size, random_state=random_state)
    else:
        idx_trainval, idx_test = idx, None
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_size / (1.0 - test_size), random_state=random_state
    )
    splits = {"train": idx_train, "val": idx_val}
    if idx_test is not None:
        splits["test"] = idx_test

    # 2) Fit the scalers on the training split only, then apply them to every split.
    x_scaler = StandardScaler()
    scale_per_dimension(x_scaler, X[idx_train], fit=True)
    scalers = {"x": x_scaler}
    if scale_target:
        scalers["y"] = StandardScaler().fit(y[idx_train])#.reshape(-1, 1))

    # 3) numpy -> tensors -> TensorDataset -> DataLoader.
    loaders = {}
    for name, ids in splits.items():
        xs = to_tensor(scale_per_dimension(x_scaler, X[ids]), device, dtype)
        ys = y[ids]
        if scale_target:
            ys = scalers["y"].transform(ys)
        dataset = TensorDataset(xs, to_tensor(ys, device, dtype))
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size if name == "train" else 4 * batch_size,
            shuffle=(name == "train"),  # only the training set is shuffled
            drop_last=False,
        )
    return loaders, scalers


if __name__ == "__main__":
    # Fake data with the same layout as yours: N trials, M dimensions, L time samples
    N, M, L = 1000, 5, 1024
    rng = np.random.default_rng(0)
    X = rng.normal(loc=[[3.0], [-1.0], [0.0], [10.0], [0.5]], scale=[[2.0], [0.1], [1.0], [5.0], [1.0]], size=(N, M, L))
    y = rng.uniform(5, 25, size=N)

    loaders, scalers = make_dataloaders(X, y, batch_size=64)
    xb, yb = next(iter(loaders["train"]))
    print({k: len(v.dataset) for k, v in loaders.items()}, tuple(xb.shape), tuple(yb.shape), xb.dtype)

    # Save the scaler: new data must be normalized with the *training* statistics
    joblib.dump(scalers["x"], "x_scaler.joblib")
    x_scaler = joblib.load("x_scaler.joblib")
    X_new = to_tensor(scale_per_dimension(x_scaler, X[:10]))

    # Then plug the loaders into the training loop:
    #   from depthwise_conv_stack import DeepFilteringRegressor
    #   from train import train_model
    #   model = DeepFilteringRegressor(num_channels=M, input_length=L, num_layers=3, ...)
    #   train_model(model, loaders["train"], loaders["val"], checkpoint_path="best_model.pt")
