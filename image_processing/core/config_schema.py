"""Pydantic config models shared across preprocessing, training, and sweeps."""
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_serializer, model_validator

StepType = Literal["align", "crop", "resize", "brightness_contrast", "color"]


class PipelineStep(BaseModel):
    id: str
    type: StepType
    params: dict = Field(default_factory=dict)


class ScopeRule(BaseModel):
    """One rule of a category scope: matches images carrying every category in `all_of`
    and none of the categories in `none_of`. A rule with both lists empty is meaningless
    (it would match everything) and is dropped by PreprocessConfig's validator."""

    all_of: list[str] = Field(default_factory=list)
    none_of: list[str] = Field(default_factory=list)


class PreprocessConfig(BaseModel):
    steps: list[PipelineStep] = Field(default_factory=list)
    # Which images this config applies to: an image is in scope if it matches ANY rule
    # (union of rules; each rule is an intersection minus exclusions — see ScopeRule).
    # Empty = applies to all images. See repository.categories_match_scope.
    scope: list[ScopeRule] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_categories(cls, data: Any) -> Any:
        # Configs written before `scope` existed carry a flat `categories` list, which
        # meant "has all of these" — exactly a single all_of rule.
        if isinstance(data, dict) and "categories" in data:
            data = dict(data)
            categories = data.pop("categories") or []
            if categories and not data.get("scope"):
                data["scope"] = [{"all_of": categories}]
        return data

    @field_validator("scope")
    @classmethod
    def _drop_empty_rules(cls, scope: list[ScopeRule]) -> list[ScopeRule]:
        return [rule for rule in scope if rule.all_of or rule.none_of]

    @model_serializer(mode="wrap")
    def _serialize(self, handler) -> dict:
        # preproc_hash is stable_hash(model_dump()), so any scope still expressible in the
        # legacy shape is dumped in it — unscoped and single-intersection configs keep the
        # exact hashes (and therefore caches and runs) they had before `scope` existed.
        data = handler(self)
        scope = data.pop("scope")
        if not scope:
            data["categories"] = []
        elif len(scope) == 1 and not scope[0]["none_of"]:
            data["categories"] = scope[0]["all_of"]
        else:
            data["scope"] = scope
        return data


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
