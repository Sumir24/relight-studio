"""Scene-domain names used for QA and optional inference hints, not renderer branches.

Every domain uses the same lighting engine and SceneBuffers contract. Profiles may
select additional scene-understanding passes or material priors in the future, but
must never change the meaning of the core light/geometry coordinates.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

SceneDomain = Literal[
    "auto", "portrait", "group_photo", "product", "food", "interior",
    "architecture", "landscape", "street", "vehicle", "animal",
    "macro", "still_life", "artwork", "other",
]

SUPPORTED_QA_DOMAINS: tuple[str, ...] = (
    "portrait", "group_photo", "product", "food", "interior",
    "architecture", "landscape", "street", "vehicle", "animal",
    "macro", "still_life", "artwork", "other",
)


@dataclass(frozen=True)
class SceneDomainHint:
    """Optional classification result; it is advisory, not a hard-coded gate."""
    domain: SceneDomain = "auto"
    confidence: float | None = None
    source: str = "unspecified"

    def validate(self) -> None:
        if self.domain not in ("auto", *SUPPORTED_QA_DOMAINS):
            raise ValueError(f"Unsupported scene domain: {self.domain}")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Scene-domain confidence must be in [0,1].")

    def to_metadata(self) -> dict:
        self.validate()
        return {"domain": self.domain, "confidence": self.confidence, "source": self.source,
                "affects_renderer": False}
