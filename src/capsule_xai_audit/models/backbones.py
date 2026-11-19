"""Backbone registry and a classifier wrapper that takes images in [0, 1].

Normalisation is part of the model, so attribution methods operate in pixel space and
every caller feeds the same thing the trainer did.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import timm
import torch
from torch import nn


@dataclass(frozen=True)
class BackboneSpec:
    timm_name: str
    kind: Literal["cnn", "vit"]
    # Module whose output Grad-CAM uses (None: not a CNN).
    cam_layer: str | None
    # Parameter-name prefixes from the output layer down to the input, for the cascading
    # randomisation sanity check. Each entry is one randomisation step.
    layer_groups: tuple[tuple[str, ...], ...]


BACKBONES: dict[str, BackboneSpec] = {
    "resnet50": BackboneSpec(
        timm_name="resnet50.a1_in1k",
        kind="cnn",
        cam_layer="layer4",
        layer_groups=(("fc",), ("layer4",), ("layer3",), ("layer2",), ("layer1",), ("conv1", "bn1")),
    ),
    "convnext_tiny": BackboneSpec(
        timm_name="convnext_tiny.fb_in1k",
        kind="cnn",
        cam_layer="stages.3",
        layer_groups=(("head",), ("stages.3",), ("stages.2",), ("stages.1",), ("stages.0",), ("stem",)),
    ),
    "dinov2_vits14": BackboneSpec(
        timm_name="vit_small_patch14_dinov2.lvd142m",
        kind="vit",
        cam_layer=None,
        layer_groups=(("head",), *((f"blocks.{i}",) for i in range(11, -1, -1)), ("patch_embed",)),
    ),
    # Small model for tests and smoke runs only.
    "resnet18": BackboneSpec(
        timm_name="resnet18.a1_in1k",
        kind="cnn",
        cam_layer="layer4",
        layer_groups=(("fc",), ("layer4",), ("layer3",), ("layer2",), ("layer1",), ("conv1", "bn1")),
    ),
}


def get_spec(name: str) -> BackboneSpec:
    try:
        return BACKBONES[name]
    except KeyError:
        raise KeyError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}") from None


class Classifier(nn.Module):
    """timm model + input normalisation. ``forward`` takes float images in [0, 1]."""

    # timm models are dynamically typed; buffers are declared for the type checker.
    net: Any
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, name: str, net: nn.Module, mean: tuple[float, ...], std: tuple[float, ...]):
        super().__init__()
        self.name = name
        self.spec = get_spec(name)
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1), persistent=False)
        self.channels_last = False

    def use_channels_last(self) -> Classifier:
        """Store conv weights as NHWC and feed NHWC inputs; numerics are unchanged."""
        if self.spec.kind == "cnn":
            self.channels_last = True
            self.net.to(memory_format=torch.channels_last)
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        # channels_last lets cuDNN use NHWC tensor-core kernels (about +30% for ResNet-50 in fp16).
        return x.contiguous(memory_format=torch.channels_last) if self.channels_last else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalize(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled pre-logit features.

        For ViTs this is the mean of the final-normed *patch* tokens (not the CLS token),
        so a linear head on it decomposes exactly into per-patch contributions (Slice A).
        """
        tokens = self.net.forward_features(self.normalize(x))
        if self.spec.kind == "vit":
            return self.patch_tokens_from(tokens).mean(dim=1)
        return self.net.forward_head(tokens, pre_logits=True)

    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N_patches, D) final-normed patch tokens of a ViT."""
        if self.spec.kind != "vit":
            raise TypeError(f"{self.name} has no patch tokens")
        return self.patch_tokens_from(self.net.forward_features(self.normalize(x)))

    def patch_tokens_from(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens[:, self.net.num_prefix_tokens :]

    @property
    def patch_grid(self) -> tuple[int, int]:
        grid = self.net.patch_embed.grid_size
        return int(grid[0]), int(grid[1])

    def submodule(self, dotted: str) -> nn.Module:
        return self.net.get_submodule(dotted)


def build_classifier(
    name: str, num_classes: int, pretrained: bool = True, img_size: int = 224, channels_last: bool = False
) -> Classifier:
    """``num_classes=0`` gives a feature extractor (timm removes the head)."""
    spec = get_spec(name)
    kwargs: dict[str, Any] = {"img_size": img_size} if spec.kind == "vit" else {}
    net: Any = timm.create_model(spec.timm_name, pretrained=pretrained, num_classes=num_classes, **kwargs)
    cfg = net.pretrained_cfg
    model = Classifier(name, net, tuple(cfg["mean"]), tuple(cfg["std"]))
    return model.use_channels_last() if channels_last else model


def feature_dim(model: Classifier) -> int:
    return int(model.net.num_features)
