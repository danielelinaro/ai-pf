from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

IntOrSeq = Union[int, Sequence[int]]

__all__ = ["DepthwiseConvStack", "DeepFilteringRegressor", "deep_filtering_predictor"]


def _expand(value, n: int, name: str) -> list:
    """Turn a scalar (or None) into a per-layer list, or validate a sequence."""
    if value is None or isinstance(value, int):
        return [value] * n
    value = list(value)
    if len(value) != n:
        raise ValueError(f"{name} must be an int or a sequence of length {n}, got {len(value)}")
    return value


class DepthwiseConvStack(nn.Module):
    """
    Stack of P blocks, each made of:
        Conv1d (grouped, groups=M)  ->  MaxPool1d  ->  [LeakyReLU]

    The M input dimensions are processed by M independent convolutional towers:
    nothing is shared or mixed between dimensions. Within a dimension, layer i has
    `filters[i]` filters, which mix the filters of layer i-1 as in a regular CNN.
    With M = 1 this is just an ordinary 1D CNN.

    Input shape:  (N, M, L)
    Output shape: (N, M * filters[-1], L_out)

    Args:
        num_channels:   M, the number of dimensions of the time series.
        num_layers:     P, the number of blocks.
        kernel_size:    Conv1d kernel size (int, or one value per layer).
        stride:         Conv1d stride (int, or one value per layer).
        pool_size:      MaxPool1d window size (int, or one value per layer).
        pool_stride:    MaxPool1d stride (int, one per layer, or None -> equal to pool_size).
        filters:        Filters per dimension in each layer (int, or one value per layer).
        dilation:       Conv1d dilation (int, or one value per layer).
        padding:        Conv1d zero padding (int, or one value per layer). 0 means "valid".
        use_activation: Whether to apply LeakyReLU after each pooling layer.
        negative_slope: Negative slope of the LeakyReLU (0.0 gives a plain ReLU).
        bias:           Whether the convolutions have a bias term.
    """

    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        kernel_size: IntOrSeq = 3,
        stride: IntOrSeq = 1,
        pool_size: IntOrSeq = 2,
        pool_stride: Optional[IntOrSeq] = None,
        filters: IntOrSeq = 1,
        dilation: IntOrSeq = 1,
        padding: IntOrSeq = 0,
        use_activation: bool = True,
        negative_slope: float = 0.01,
        bias: bool = True,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_channels = num_channels
        self.num_layers = num_layers
        self.kernel_sizes = _expand(kernel_size, num_layers, "kernel_size")
        self.strides = _expand(stride, num_layers, "stride")
        self.pool_sizes = _expand(pool_size, num_layers, "pool_size")
        self.pool_strides = _expand(pool_stride, num_layers, "pool_stride")
        self.filters = _expand(filters, num_layers, "filters")
        self.dilations = _expand(dilation, num_layers, "dilation")
        self.paddings = _expand(padding, num_layers, "padding")

        layers = []
        in_filters = 1  # one raw signal per dimension
        for i in range(num_layers):
            out_filters = self.filters[i]
            layers.append(
                nn.Conv1d(
                    in_channels=num_channels * in_filters,
                    out_channels=num_channels * out_filters,
                    kernel_size=self.kernel_sizes[i],
                    stride=self.strides[i],
                    padding=self.paddings[i],
                    dilation=self.dilations[i],
                    groups=num_channels,  # keeps each dimension independent
                    bias=bias,
                    device=device,
                    dtype=dtype
                )
            )
            layers.append(nn.MaxPool1d(kernel_size=self.pool_sizes[i], stride=self.pool_strides[i]))
            if use_activation:
                # LeakyReLU is monotonic, so it commutes with max pooling;
                # applying it after pooling just means fewer elements to process.
                layers.append(nn.LeakyReLU(negative_slope))
            in_filters = out_filters

        self.out_channels = num_channels * in_filters
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def output_length(self, input_length: int) -> int:
        """Length of the time axis after the whole stack, for a given input length."""
        length = input_length
        for i in range(self.num_layers):
            k, s, d, pad = self.kernel_sizes[i], self.strides[i], self.dilations[i], self.paddings[i]
            p = self.pool_sizes[i]
            ps = self.pool_strides[i] if self.pool_strides[i] is not None else p
            length = (length + 2 * pad - d * (k - 1) - 1) // s + 1  # conv
            length = (length - p) // ps + 1  # max pool
            if length < 1:
                raise ValueError(
                    f"input_length={input_length} is too short: the time axis vanishes at layer {i + 1}"
                )
        return length


class DeepFilteringRegressor(nn.Module):
    """
    DepthwiseConvStack followed by a stack of Linear layers, ending in a single
    output. Mirrors the "predictor" network of George & Huerta (2018), Phys. Rev. D
    97, 044039, generalized to M independent input dimensions.

        (N, M, L) -> conv stack -> flatten -> [Linear -> LeakyReLU] * H -> Linear

    Output shape: (N, num_outputs); note that this is (N, 1) by default, not (N,).

    Args:
        num_channels:   M, the number of dimensions of the time series.
        input_length:   L, length of the time axis (needed to size the first Linear layer).
        num_layers:     P, number of conv blocks.
        hidden_units:   Units of each hidden Linear layer (a sequence, possibly empty).
        num_outputs:    Units of the final Linear layer (1 -> a single scalar parameter).
        use_activation, negative_slope:
                        Apply to both the conv blocks and the hidden Linear layers.
                        The final Linear layer never has an activation.
        Remaining arguments are forwarded to DepthwiseConvStack.
    """

    def __init__(
        self,
        num_channels: int,
        input_length: int,
        num_layers: int,
        hidden_units: Sequence[int] = (64,),
        num_outputs: int = 1,
        kernel_size: IntOrSeq = 3,
        stride: IntOrSeq = 1,
        pool_size: IntOrSeq = 2,
        pool_stride: Optional[IntOrSeq] = None,
        filters: IntOrSeq = 1,
        dilation: IntOrSeq = 1,
        padding: IntOrSeq = 0,
        use_activation: bool = True,
        negative_slope: float = 0.01,
        bias: bool = True,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        self.features = DepthwiseConvStack(
            num_channels=num_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            stride=stride,
            pool_size=pool_size,
            pool_stride=pool_stride,
            filters=filters,
            dilation=dilation,
            padding=padding,
            use_activation=use_activation,
            negative_slope=negative_slope,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.flat_features = self.features.out_channels * self.features.output_length(input_length)

        layers = [nn.Flatten()]
        in_features = self.flat_features
        for units in hidden_units:
            layers.append(nn.Linear(in_features, units, device=device, dtype=dtype))
            if use_activation:
                layers.append(nn.LeakyReLU(negative_slope))
            in_features = units
        layers.append(nn.Linear(in_features, num_outputs, device=device, dtype=dtype))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# Hyperparameters of the two predictor networks in the paper (Figs. 5 and 6).
# Both use valid (unpadded) convolutions, conv stride 1, and max pooling with
# window 4 and stride 4, on inputs of length 8192.
_PAPER_CONFIGS = {
    "shallow": dict(  # Fig. 5: 3 conv layers + 2 Linear layers (~2 MB)
        num_layers=3,
        filters=[16, 32, 64],
        kernel_size=[16, 8, 8],
        dilation=[1, 4, 4],
        hidden_units=[64],
    ),
    "deeper": dict(  # Fig. 6: 4 conv layers + 3 Linear layers (~23 MB)
        num_layers=4,
        filters=[64, 128, 256, 512],
        kernel_size=[16, 16, 16, 32],
        dilation=[1, 2, 2, 2],
        hidden_units=[128, 64],
    ),
}


def deep_filtering_predictor(
    variant: str = "shallow",
    num_channels: int = 1,
    input_length: int = 8192,
    num_outputs: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float,
    **overrides,
) -> DeepFilteringRegressor:
    """Build one of the paper's predictor architectures ("shallow" or "deeper").

    Any keyword in `overrides` (e.g. negative_slope=0.0 for a plain ReLU, as in the
    paper) replaces the corresponding preset value.
    """
    if variant not in _PAPER_CONFIGS:
        raise ValueError(f"variant must be one of {list(_PAPER_CONFIGS)}")
    cfg = dict(_PAPER_CONFIGS[variant], pool_size=4, stride=1, padding=0)
    cfg.update(overrides)
    return DeepFilteringRegressor(
        num_channels=num_channels,
        input_length=input_length,
        num_outputs=num_outputs,
        device=device,
        dtype=dtype,
        **cfg
    )


if __name__ == "__main__":
    # Paper architectures, single time series of 8192 samples (1 s at 8192 Hz)
    for variant in ("shallow", "deeper"):
        model = deep_filtering_predictor(variant)
        y = model(torch.randn(4, 1, 8192))
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{variant:8s} flatten={model.flat_features:5d}  out={tuple(y.shape)}  params={n_params:,}")

    # Multi-dimensional data: each of the M=5 dimensions gets its own conv tower
    N, M, L = 8, 5, 8192
    model = deep_filtering_predictor("shallow", num_channels=M, input_length=L)
    print(model(torch.randn(N, M, L)).shape)

    # Fully custom model
    model = DeepFilteringRegressor(
        num_channels=M, input_length=1000, num_layers=3,
        kernel_size=[11, 7, 3], pool_size=[4, 2, 2], filters=[4, 8, 8], padding=[5, 3, 1],
        hidden_units=[128, 32],
    )
    print(model(torch.randn(N, M, 1000)).shape)
