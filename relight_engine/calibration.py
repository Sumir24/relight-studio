from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from .contracts import Light, LightingRig


def kelvin_to_linear_rgb(kelvin: float) -> tuple[float, float, float]:
    """Approximate black-body colour as normalized linear RGB.

    This is a user-facing colour-temperature approximation, not a spectral renderer.
    It controls the tint of a calibrated light; intensity remains a separate parameter.
    """
    if not math.isfinite(kelvin) or not 1000.0 <= kelvin <= 40000.0:
        raise ValueError("Colour temperature must be between 1000 K and 40000 K.")
    t = kelvin / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
    else:
        r = 329.698727446 * ((t - 60) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60) ** -0.0755148492)
        b = 255.0
    srgb = np.clip(np.array([r, g, b], dtype=np.float32) / 255.0, 0.0, 1.0)
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    linear /= max(float(np.max(linear)), 1e-8)
    return tuple(float(x) for x in linear)


@dataclass(frozen=True)
class SourceLightCalibration:
    """Explicit approximation of the original illumination supplied by a user.

    `key_direction` points from a receiving surface toward the source light in camera
    coordinates (+X right, +Y up, +Z away from the camera). This does not estimate light
    from image pixels; it records a user calibration so source-to-target transfer has an
    explicit, auditable starting point.
    """
    key_direction: tuple[float, float, float]
    key_strength: float = 1.0
    key_temperature_kelvin: float = 5600.0
    ambient_strength: float = 0.15
    ambient_temperature_kelvin: float = 6500.0
    casts_shadows: bool = False
    name: str = "user_calibrated_source_key"

    def validate(self) -> None:
        direction = np.asarray(self.key_direction, dtype=np.float32)
        if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
            raise ValueError("key_direction must be a finite non-zero camera-space vector.")
        if not math.isfinite(self.key_strength) or self.key_strength < 0:
            raise ValueError("key_strength must be finite and non-negative.")
        if not math.isfinite(self.ambient_strength) or self.ambient_strength < 0:
            raise ValueError("ambient_strength must be finite and non-negative.")
        kelvin_to_linear_rgb(self.key_temperature_kelvin)
        kelvin_to_linear_rgb(self.ambient_temperature_kelvin)
        if not self.name.strip():
            raise ValueError("Calibration name cannot be empty.")

    def to_rig(self) -> LightingRig:
        self.validate()
        direction = np.asarray(self.key_direction, dtype=np.float32)
        direction /= np.linalg.norm(direction)
        light = Light(
            name=self.name,
            kind="directional",
            direction=tuple(float(x) for x in direction),
            color=kelvin_to_linear_rgb(self.key_temperature_kelvin),
            strength=float(self.key_strength),
            casts_shadows=bool(self.casts_shadows),
        )
        rig = LightingRig(
            ambient_color=kelvin_to_linear_rgb(self.ambient_temperature_kelvin),
            ambient_strength=float(self.ambient_strength),
            lights=(light,),
        )
        rig.validate()
        return rig

    def to_metadata(self) -> dict:
        self.validate()
        return {
            "source": "user_calibration_not_automatic_estimation",
            "name": self.name,
            "key_direction_camera_space": list(self.key_direction),
            "key_strength": float(self.key_strength),
            "key_temperature_kelvin": float(self.key_temperature_kelvin),
            "ambient_strength": float(self.ambient_strength),
            "ambient_temperature_kelvin": float(self.ambient_temperature_kelvin),
            "casts_shadows": bool(self.casts_shadows),
            "colour_temperature_conversion": "approximate black-body RGB tint, not spectral rendering",
        }


@dataclass(frozen=True)
class MaterialAssignment:
    """User-selected material values applied through a scene mask."""
    name: str
    mask: np.ndarray
    roughness: float
    metalness: float = 0.0
    specular_level: float = 0.04
    blend: float = 1.0
    # Optional linear-RGB conductor F0. None uses the renderer's neutral conductor prior.
    specular_color: tuple[float, float, float] | None = None
    # Caller-declared confidence in this manual assignment, separate from mask blend.
    confidence: float = 1.0

    def validate(self, shape: tuple[int, int]) -> None:
        if not self.name.strip():
            raise ValueError("Material assignment name cannot be empty.")
        if self.mask.shape != shape or self.mask.ndim != 2:
            raise ValueError(f"Material mask {self.name!r} must be aligned to image shape {shape}.")
        if not np.isfinite(self.mask).all() or np.any(self.mask < 0) or np.any(self.mask > 1):
            raise ValueError("Material masks must be finite and in [0,1].")
        for key, value in (("roughness", self.roughness), ("metalness", self.metalness),
                           ("specular_level", self.specular_level), ("blend", self.blend),
                           ("confidence", self.confidence)):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be finite and in [0,1].")
        if self.specular_color is not None:
            color = np.asarray(self.specular_color, dtype=np.float32)
            if color.shape != (3,) or not np.isfinite(color).all() or np.any(color < 0) or np.any(color > 1):
                raise ValueError("specular_color must be a linear-RGB triple in [0,1].")


@dataclass
class MaterialMaps:
    roughness: np.ndarray
    metalness: np.ndarray
    specular_level: np.ndarray
    coverage: np.ndarray
    confidence: np.ndarray
    specular_color: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


def build_material_maps(
    shape: tuple[int, int],
    assignments: list[MaterialAssignment] | tuple[MaterialAssignment, ...],
    *,
    base_roughness: float = 0.75,
    base_metalness: float = 0.0,
    base_specular_level: float = 0.04,
    base_specular_color: tuple[float, float, float] = (0.65, 0.65, 0.65),
) -> MaterialMaps:
    """Blend user-authored material overrides over explicit generic dielectric priors.

    Base values match the lighting core's current fallback values. Assignment values are
    user choices, not neural material estimates. Soft masks and blend strengths are
    normalized where they overlap, preventing parameter overshoot.
    """
    h, w = shape
    if h <= 0 or w <= 0:
        raise ValueError("Material map shape must be positive.")
    bases = (base_roughness, base_metalness, base_specular_level)
    base_color = np.asarray(base_specular_color, dtype=np.float32)
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in bases):
        raise ValueError("Base material values must be finite and in [0,1].")
    if base_color.shape != (3,) or not np.isfinite(base_color).all() or np.any(base_color < 0) or np.any(base_color > 1):
        raise ValueError("base_specular_color must be a linear-RGB triple in [0,1].")
    for assignment in assignments:
        assignment.validate(shape)

    if assignments:
        weights = np.stack([np.clip(a.mask.astype(np.float32), 0, 1) * a.blend for a in assignments], axis=0)
        weight_sum = np.sum(weights, axis=0)
        overflow = weight_sum > 1.0
        if np.any(overflow):
            weights[:, overflow] /= weight_sum[overflow]
            weight_sum[overflow] = 1.0
    else:
        weights = np.zeros((0, h, w), np.float32)
        weight_sum = np.zeros((h, w), np.float32)

    base_weight = 1.0 - weight_sum
    def compose(attribute: str, base_value: float) -> np.ndarray:
        out = np.full((h, w), base_value, np.float32) * base_weight
        for idx, assignment in enumerate(assignments):
            out += weights[idx] * float(getattr(assignment, attribute))
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    # Confidence is an independent user-declared reliability weight, not mask coverage.
    confidence_numerator = np.zeros((h, w), dtype=np.float32)
    for idx, assignment in enumerate(assignments):
        confidence_numerator += weights[idx] * float(assignment.confidence)
    material_confidence = np.divide(
        confidence_numerator,
        weight_sum,
        out=np.zeros((h, w), dtype=np.float32),
        where=weight_sum > 1e-8,
    )

    specular_color = None
    if any(assignment.specular_color is not None for assignment in assignments):
        specular_color = np.broadcast_to(base_color, (h, w, 3)).copy() * base_weight[..., None]
        for idx, assignment in enumerate(assignments):
            assignment_color = (
                base_color if assignment.specular_color is None
                else np.asarray(assignment.specular_color, dtype=np.float32)
            )
            specular_color += weights[idx][..., None] * assignment_color[None, None, :]
        specular_color = np.clip(specular_color, 0.0, 1.0).astype(np.float32)

    return MaterialMaps(
        roughness=compose("roughness", base_roughness),
        metalness=compose("metalness", base_metalness),
        specular_level=compose("specular_level", base_specular_level),
        coverage=np.clip(weight_sum, 0, 1).astype(np.float32),
        confidence=np.clip(material_confidence, 0, 1).astype(np.float32),
        specular_color=specular_color,
        metadata={
            "origin": "user_authored_mask_overrides_plus_explicit_generic_base_priors",
            "base_values": {
                "roughness": float(base_roughness),
                "metalness": float(base_metalness),
                "specular_level": float(base_specular_level),
                "specular_color": [float(value) for value in base_color],
            },
            "material_confidence_semantics": "caller-declared reliability of manual map assignments; separate from mask coverage/blend",
            "assignments": [
                {
                    "name": a.name,
                    "roughness": float(a.roughness),
                    "metalness": float(a.metalness),
                    "specular_level": float(a.specular_level),
                    "specular_color": (
                        None if a.specular_color is None
                        else [float(value) for value in a.specular_color]
                    ),
                    "blend": float(a.blend),
                    "confidence": float(a.confidence),
                }
                for a in assignments
            ],
            "neural_material_estimation_used": False,
        },
    )
