"""The small VGG model used by the analog-noise experiments."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


CHANNELS = ((3, 64), (64, 64), (64, 128), (128, 128), (128, 256), (256, 256))


class VGG7(nn.Module):
    """Six convolutions followed by global pooling and one linear layer."""

    layer_names = (*[f"conv{i}" for i in range(1, 7)], "classifier")

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.convs = nn.ModuleList(
            nn.Conv2d(a, b, kernel_size=3, padding=1, bias=False) for a, b in CHANNELS
        )
        self.bns = nn.ModuleList(nn.BatchNorm2d(b) for _, b in CHANNELS)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = F.relu(bn(conv(x)), inplace=False)
            if index in (1, 3, 5):
                x = F.max_pool2d(x, 2)
        return self.classifier(F.adaptive_avg_pool2d(x, 1).flatten(1))


def build_model(num_classes: int = 10) -> VGG7:
    return VGG7(num_classes=num_classes)


def fold_batch_norm(model: VGG7) -> VGG7:
    """Return an eval-mode copy with each BatchNorm folded into its convolution."""
    if model.training:
        raise ValueError("batch normalization can only be folded in eval mode")
    folded = copy.deepcopy(model)
    new_convs = nn.ModuleList()
    for conv, bn in zip(folded.convs, folded.bns):
        if isinstance(bn, nn.Identity):
            new_convs.append(conv)
            continue
        if not isinstance(bn, nn.BatchNorm2d):
            raise TypeError(f"unsupported normalization layer: {type(bn)!r}")
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        weight = conv.weight * scale[:, None, None, None]
        conv_bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        bias = bn.bias + (conv_bias - bn.running_mean) * scale
        replacement = nn.Conv2d(
            conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
            conv.padding, conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode,
        ).to(device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            replacement.weight.copy_(weight)
            replacement.bias.copy_(bias)
        new_convs.append(replacement)
    folded.convs = new_convs
    folded.bns = nn.ModuleList(nn.Identity() for _ in folded.bns)
    return folded.eval()


def load_checkpoint_model(
    path: str | Path, device: str | torch.device = "cpu", *, folded: bool = False
) -> VGG7:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = build_model(num_classes=int(config.get("num_classes", 10)))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return fold_batch_norm(model) if folded else model
