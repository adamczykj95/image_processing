"""Extends Anomalib's Patchcore to expose the local feature-pooling kernel ("patch size")
as a hyperparameter.

Anomalib hardcodes this at 3x3 (`PatchcoreModel.__init__` sets
`self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)`) with no constructor argument for it
anywhere in the public API (verified against the installed 2.6.2 source) — so this
subclasses the internals directly rather than depending on an argument that doesn't exist.
This is a real, acknowledged risk: it depends on the non-public `feature_pooler` attribute
and `PatchcoreModel`'s exact constructor shape, and could break on a future Anomalib
upgrade. If it does, it will surface as an import/attribute error at startup, not silent
incorrect behavior.
"""
from collections.abc import Sequence

import torch
from anomalib import PrecisionType
from anomalib.models.image.patchcore.lightning_model import Patchcore
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


# ResNet-family stem+stage downsampling factors (verified empirically against wide_resnet50_2
# at multiple input sizes: 256px -> layer2 32x32 / layer3 16x16, etc. — matches 8x/16x exactly).
# Holds for every backbone currently offered in the app (wide_resnet50_2, resnet18,
# wide_resnet101_2), since they all share the same ResNet stem/stage structure. Would need
# updating if a non-ResNet-family backbone (EfficientNet/ConvNeXt/ViT, etc.) is ever added,
# since those downsample differently and may not even use these layer names.
RESNET_LAYER_DOWNSAMPLE = {"layer1": 4, "layer2": 8, "layer3": 16, "layer4": 32}


def feature_grid_size(image_size: int, layer: str) -> int:
    """Approximate per-side spatial grid size of a ResNet-family feature map at `layer`,
    for a square input of `image_size` pixels. Used to warn when a chosen patch_size is
    large relative to the actual feature grid it pools over (see module docstring above
    for why oversized kernels degrade silently rather than erroring)."""
    factor = RESNET_LAYER_DOWNSAMPLE.get(layer, 16)
    return max(1, image_size // factor)


class PatchSizePatchcoreModel(PatchcoreModel):
    def __init__(self, *args, patch_size: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if patch_size % 2 == 0:
            raise ValueError(f"patch_size must be odd (symmetric padding requires it), got {patch_size}")
        self.feature_pooler = torch.nn.AvgPool2d(patch_size, 1, patch_size // 2)


class PatchSizePatchcore(Patchcore):
    """Same as anomalib.models.Patchcore, plus a `patch_size` hyperparameter controlling
    the local-neighborhood pooling kernel used to build memory-bank patch embeddings.
    Smaller = more sensitive to small/fine defects, sharper heatmap localization, more
    prone to false positives on naturally-textured normal surfaces. Larger = more robust
    to that texture noise, at the cost of diluting small defects and blurrier localization.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3"),
        pre_trained: bool = True,
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        patch_size: int = 3,
        precision: str | PrecisionType = PrecisionType.FLOAT32,
        **kwargs,
    ) -> None:
        # Let the parent do its normal setup (pre_processor/evaluator/visualizer wiring,
        # precision handling) — including building its own default self.model, which we
        # then discard and replace below. Keeps us decoupled from that setup logic; only
        # the feature_pooler construction depends on the internal we're overriding.
        super().__init__(
            backbone=backbone,
            layers=layers,
            pre_trained=pre_trained,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
            precision=precision,
            **kwargs,
        )
        self.model = PatchSizePatchcoreModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            num_neighbors=num_neighbors,
            patch_size=patch_size,
        )
        if isinstance(precision, str):
            precision = PrecisionType(precision.lower())
        self.model = self.model.half() if precision == PrecisionType.FLOAT16 else self.model.float()
