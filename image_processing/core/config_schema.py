"""Pydantic config models shared across preprocessing, training, and sweeps."""
from typing import Literal

from pydantic import BaseModel, Field

StepType = Literal["align", "crop", "resize", "brightness_contrast", "color"]


class PipelineStep(BaseModel):
    id: str
    type: StepType
    params: dict = Field(default_factory=dict)


class PreprocessConfig(BaseModel):
    steps: list[PipelineStep] = Field(default_factory=list)


class ModelConfig(BaseModel):
    backbone: str = "wide_resnet50_2"
    layers: list[str] = Field(default_factory=lambda: ["layer2", "layer3"])
    coreset_sampling_ratio: float = 0.1
    num_neighbors: int = 9
    patch_size: int = 3  # local feature-pooling kernel; must be odd. See ml/patchcore_ext.py.


class RunConfig(BaseModel):
    preproc_hash: str
    preprocess: PreprocessConfig
    model: ModelConfig
    train_image_ids: list[str]
    val_good_image_ids: list[str]
    val_anomaly_image_ids: list[str]
    image_size: tuple[int, int]


class SweepGridConfig(BaseModel):
    preprocess_variants: list[PreprocessConfig] = Field(default_factory=list)
    backbone: list[str] = Field(default_factory=lambda: ["wide_resnet50_2"])
    layers: list[list[str]] = Field(default_factory=lambda: [["layer2", "layer3"]])
    coreset_sampling_ratio: list[float] = Field(default_factory=lambda: [0.1])
    num_neighbors: list[int] = Field(default_factory=lambda: [9])
    patch_size: list[int] = Field(default_factory=lambda: [3])
