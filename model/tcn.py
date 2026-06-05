"""Temporal Convolutional Network (TCN).

Implementation of the architecture from Bai et al., "An Empirical Evaluation of
Generic Convolutional and Recurrent Networks for Sequence Modeling" (2018),
built from causal, dilated 1D convolutions stacked in residual blocks.
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    """Trim the right-hand padding so the convolution stays causal."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Two dilated causal convolutions with a residual connection."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Initialize the raw conv weights *before* wrapping with weight_norm: the
        # parametrized API derives `weight` from a parametrization, so direct
        # `.weight.data` assignment must happen on the plain conv first.
        conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
        )
        conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
        )
        conv1.weight.data.normal_(0, 0.01)
        conv2.weight.data.normal_(0, 0.01)
        self.conv1 = weight_norm(conv1)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(conv2)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        # 1x1 conv to match channel counts on the residual path when needed.
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stack of temporal blocks with exponentially increasing dilation.

    Args:
        num_inputs: Number of input channels (features).
        num_channels: Output channels for each temporal block; its length sets depth.
        kernel_size: Convolution kernel size.
        dropout: Dropout probability inside each block.

    Input/Output shape: (batch, channels, sequence_length).
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCN(nn.Module):
    """TCN with a linear head for sequence-to-one or sequence-to-sequence tasks.

    Args:
        input_size: Number of input features per time step.
        output_size: Number of target outputs.
        num_channels: Output channels per temporal block; its length sets depth.
        kernel_size: Convolution kernel size.
        dropout: Dropout probability.
        seq_to_seq: If True, predict at every time step; otherwise use the last step.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_channels: list[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
        seq_to_seq: bool = False,
    ):
        super().__init__()
        self.tcn = TemporalConvNet(
            input_size, num_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.seq_to_seq = seq_to_seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Tensor of shape (batch, sequence_length, input_size).

        Returns:
            (batch, output_size) if seq_to_seq is False, else
            (batch, sequence_length, output_size).
        """
        # TemporalConvNet expects (batch, channels, sequence_length).
        y = self.tcn(x.transpose(1, 2))
        if self.seq_to_seq:
            return self.linear(y.transpose(1, 2))
        return self.linear(y[:, :, -1])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch, seq_len, features = 16, 100, 8
    model = TCN(
        input_size=features,
        output_size=1,
        num_channels=[32, 32, 32, 32],
        kernel_size=3,
        dropout=0.2,
    ).to(device)
    sample = torch.randn(batch, seq_len, features, device=device)
    out = model(sample)
    print(model)
    print(f"input:  {tuple(sample.shape)}")
    print(f"output: {tuple(out.shape)}")
