"""Attribution methods behind one interface: ``attribute(method, model, x, target) -> (B, H, W)``.

Inputs are images in [0, 1] (the model normalises internally), so every map lives in
pixel space. Maps are non-negative importances:

    gradcam               ReLU(Grad-CAM) on the backbone's last stage, bilinear upsampled
    integrated_gradients  |sum over channels of IG|, black-image baseline
    saliency              max over channels of |d logit / d x|
    grad_x_input          |sum over channels of x * d logit / d x|
    random                U(0, 1) noise, seeded per image (model-independent control)
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from captum.attr import InputXGradient, IntegratedGradients, LayerAttribution, LayerGradCam, Saliency

from capsule_xai_audit.models.backbones import Classifier

METHODS: tuple[str, ...] = ("gradcam", "integrated_gradients", "saliency", "grad_x_input", "random")


def random_maps(size: tuple[int, int], seeds: Sequence[int], device: torch.device) -> torch.Tensor:
    """One uniform-noise map per seed; the same seed always gives the same map."""
    maps = [torch.rand(size, generator=torch.Generator().manual_seed(int(s))) for s in seeds]
    return torch.stack(maps).to(device) if maps else torch.zeros(0, *size, device=device)


def attribute(
    method: str,
    model: Classifier,
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    ig_steps: int = 32,
    seeds: Sequence[int] | None = None,
) -> torch.Tensor:
    """Attribution maps (B, H, W), float32, detached."""
    model.eval()
    x = x.float()
    h, w = x.shape[-2:]
    if method == "random":
        return random_maps((h, w), seeds if seeds is not None else range(len(x)), x.device)
    x = x.detach().requires_grad_(True)
    if method == "gradcam":
        if model.spec.cam_layer is None:
            raise ValueError(f"Grad-CAM needs a CNN; {model.name} has no CAM layer")
        cam = LayerGradCam(model, model.submodule(model.spec.cam_layer))
        maps = cam.attribute(x, target=target, relu_attributions=True)
        maps = LayerAttribution.interpolate(maps, (h, w), interpolate_mode="bilinear")[:, 0]
    elif method == "integrated_gradients":
        ig = IntegratedGradients(model)
        maps = ig.attribute(
            x, baselines=torch.zeros_like(x), target=target, n_steps=ig_steps, internal_batch_size=len(x)
        )
        maps = maps.sum(1).abs()
    elif method == "saliency":
        maps = Saliency(model).attribute(x, target=target, abs=True).amax(1)
    elif method == "grad_x_input":
        maps = InputXGradient(model).attribute(x, target=target).sum(1).abs()
    else:
        raise KeyError(f"unknown attribution method {method!r}; choose from {METHODS}")
    return maps.detach().float().clamp(min=0)
