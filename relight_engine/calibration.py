from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from .contracts import Light, LightingRig
from .color import srgb_to_linear


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
    # Optional layered clear-coat and subsurface scattering parameters in [0,1].
    clearcoat: float = 0.0
    clearcoat_roughness: float = 0.15
    subsurface: float = 0.0
    anisotropy: float = 0.0
    anisotropy_angle: float = 0.0
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
                           ("clearcoat", self.clearcoat), ("clearcoat_roughness", self.clearcoat_roughness),
                           ("subsurface", self.subsurface), ("anisotropy", self.anisotropy),
                           ("confidence", self.confidence)):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be finite and in [0,1].")
        if not math.isfinite(self.anisotropy_angle):
            raise ValueError("anisotropy_angle must be finite.")
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
    clearcoat: np.ndarray | None = None
    clearcoat_roughness: np.ndarray | None = None
    subsurface: np.ndarray | None = None
    anisotropy: np.ndarray | None = None
    anisotropy_angle: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


def build_material_maps(
    shape: tuple[int, int],
    assignments: list[MaterialAssignment] | tuple[MaterialAssignment, ...],
    *,
    base_roughness: float = 0.75,
    base_metalness: float = 0.0,
    base_specular_level: float = 0.04,
    base_specular_color: tuple[float, float, float] = (0.65, 0.65, 0.65),
    base_clearcoat: float = 0.0,
    base_clearcoat_roughness: float = 0.15,
    base_subsurface: float = 0.0,
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
        clearcoat=compose("clearcoat", base_clearcoat) if any(a.clearcoat > 0.0 for a in assignments) else None,
        clearcoat_roughness=compose("clearcoat_roughness", base_clearcoat_roughness) if any(a.clearcoat > 0.0 for a in assignments) else None,
        subsurface=compose("subsurface", base_subsurface) if any(a.subsurface > 0.0 for a in assignments) else None,
        anisotropy=compose("anisotropy", 0.0) if any(a.anisotropy > 0.0 for a in assignments) else None,
        anisotropy_angle=(
            (sum(weights[i] * float(a.anisotropy_angle) for i, a in enumerate(assignments)) / np.maximum(weight_sum, 1e-8)).astype(np.float32) * (weight_sum > 1e-8)
            if any(a.anisotropy_angle != 0.0 for a in assignments)
            else None
        ),
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


def assign_estimated_material_priors(
    *,
    roughness: np.ndarray | None,
    metalness: np.ndarray | None,
    albedo_srgb: np.ndarray | None = None,
    source_rgb_srgb: np.ndarray | None = None,
    specular_level: np.ndarray | None = None,
    specular_color: np.ndarray | None = None,
    anisotropy: np.ndarray | None = None,
    material_confidence: np.ndarray | None = None,
    roughness_uncertainty: np.ndarray | None = None,
    metalness_uncertainty: np.ndarray | None = None,
    metalness_threshold_low: float = 0.80,
    metalness_threshold_high: float = 0.94,
    user_overrides: MaterialMaps | None = None,
) -> MaterialMaps:
    """Confidence-aware material prior assignment layer with user-override precedence.

    1. Preserves raw model predictions in metadata separately from calibrated outputs.
    2. Separates glossy dielectric/clearcoat response from true metallic conductors
       using ensemble uncertainty and roughness compatibility rather than brightness/color
       or category rules.
    3. Raises minimum dielectric roughness on high-frequency micro-textured surfaces
       (scale-invariant relative local contrast, preventing matte stone/asphalt from
       turning artificially glossy).
    4. Blends uncertain regions toward conservative fallbacks (roughness=0.75, metalness=0.0,
       specular_level=0.04) proportional to estimation confidence.
    5. Lets user-authored MaterialMaps override predictions wherever user coverage > 0.
    """
    ref = next(
        (arr for arr in (roughness, metalness, specular_level, material_confidence, anisotropy) if arr is not None),
        None,
    )
    if ref is None:
        raise ValueError("At least one model-estimated material map is required.")
    h, w = ref.shape[:2]

    conf = (
        np.clip(np.asarray(material_confidence, dtype=np.float32), 0.0, 1.0)
        if material_confidence is not None
        else np.full((h, w), 0.55, dtype=np.float32)
    )

    r_raw = np.clip(np.asarray(roughness, dtype=np.float32), 0.04, 0.98) if roughness is not None else None
    m_raw = np.clip(np.asarray(metalness, dtype=np.float32), 0.0, 1.0) if metalness is not None else None

    if r_raw is not None:
        eff_roughness = (0.75 * (1.0 - conf) + r_raw * conf).astype(np.float32)
    else:
        eff_roughness = np.full((h, w), 0.75, dtype=np.float32)

    if m_raw is not None:
        span = max(float(metalness_threshold_high - metalness_threshold_low), 1e-4)
        m_gate = np.clip((m_raw - float(metalness_threshold_low)) / span, 0.0, 1.0)
        m_cal = np.power(m_gate, 1.8).astype(np.float32)
        if metalness_uncertainty is not None:
            m_unc = np.clip(np.asarray(metalness_uncertainty, dtype=np.float32), 0.0, 1.0)
            # Glossy dark dielectrics (leather/dial) exhibit elevated metalness ensemble disagreement.
            m_cal = (m_cal * np.exp(-m_unc / 0.024)).astype(np.float32)
        if r_raw is not None:
            # Bare conductors with strong InteriorVerse metalness are low-to-moderate roughness;
            # uncertain high-roughness dielectrics (leather/shoes/suit) are tapered away from metalness.
            rough_metal_compat = np.clip((0.56 - r_raw) / (0.56 - 0.38), 0.0, 1.0)
            m_cal = (m_cal * rough_metal_compat).astype(np.float32)
        # Attenuate metalness cleanly in uncertain pixels toward dielectric 0.0.
        eff_metalness = (m_cal * np.clip(conf / 0.65, 0.0, 1.0)).astype(np.float32)
    else:
        eff_metalness = np.zeros((h, w), dtype=np.float32)

    # Scale-invariant high-frequency micro-texture roughness floor (when source RGB is supplied):
    # raises dielectric roughness on gritty/speckled matte textures (asphalt, stucco, foliage)
    # without touching smooth metals or using brightness/category thresholds.
    if source_rgb_srgb is not None and min(h, w) >= 4:
        from scipy.ndimage import gaussian_filter
        src_l = srgb_to_linear(np.clip(np.asarray(source_rgb_srgb, dtype=np.float32), 0.0, 1.0))
        y_tex = (0.2126 * src_l[..., 0] + 0.7152 * src_l[..., 1] + 0.0722 * src_l[..., 2]).astype(np.float32)
        y_blur1 = gaussian_filter(y_tex, 0.85, mode="reflect")
        y_mean = np.maximum(gaussian_filter(y_tex, 3.0, mode="reflect"), 0.06)
        # Clamp per-pixel high-pass at 0.25 so crisp edges/graphics don't dominate sustained fine-grain texture.
        rel_hp = np.clip(np.abs(y_tex - y_blur1) / y_mean, 0.0, 0.25)
        rel_var = gaussian_filter(rel_hp, 3.0, mode="reflect")
        r_tex_floor = (np.clip((rel_var - 0.065) / 0.08, 0.0, 1.0) * 0.76).astype(np.float32)
        non_metal_w = np.clip(1.0 - eff_metalness / 0.25, 0.0, 1.0)
        eff_roughness = np.maximum(eff_roughness, eff_roughness * (1.0 - non_metal_w) + np.maximum(eff_roughness, r_tex_floor) * non_metal_w).astype(np.float32)

    if specular_level is not None:
        eff_spec_level = np.clip(np.asarray(specular_level, dtype=np.float32), 0.0, 1.0)
    else:
        eff_spec_level = np.full((h, w), 0.04, dtype=np.float32)

    # Separate dielectric clearcoat response from metallic conductor behavior:
    # smooth confident non-metallic surfaces (eff_roughness < 0.46) receive an independent
    # dielectric clearcoat layer; gritty textured matte surfaces get strictly zero clearcoat.
    eff_clearcoat = None
    eff_cc_rough = None
    if r_raw is not None:
        coat_mask = np.clip((0.50 - eff_roughness) / 0.22, 0.0, 1.0) * conf * np.clip(1.0 - eff_metalness / 0.35, 0.0, 1.0) * 0.75
        if np.any(coat_mask > 1e-3):
            eff_clearcoat = np.clip(coat_mask, 0.0, 1.0).astype(np.float32)
            eff_cc_rough = np.clip(eff_roughness * 0.30, 0.06, 0.16).astype(np.float32)

    if specular_color is not None:
        eff_spec_color = np.clip(np.asarray(specular_color, dtype=np.float32), 0.04, 0.98).astype(np.float32)
    elif albedo_srgb is not None:
        eff_spec_color = np.clip(srgb_to_linear(np.asarray(albedo_srgb, dtype=np.float32)), 0.04, 0.98).astype(np.float32)
    else:
        eff_spec_color = np.full((h, w, 3), 0.65, dtype=np.float32)

    eff_aniso = (
        np.clip(np.asarray(anisotropy, dtype=np.float32), 0.0, 1.0) * conf
        if anisotropy is not None
        else None
    )

    eff_subsurface = None
    eff_aniso_angle = None
    cov = np.zeros((h, w), dtype=np.float32)

    if user_overrides is not None:
        cov = np.clip(user_overrides.coverage.astype(np.float32), 0.0, 1.0)
        inv = 1.0 - cov
        eff_roughness = np.clip(eff_roughness * inv + (user_overrides.roughness - 0.75 * inv), 0.0, 1.0).astype(np.float32)
        eff_metalness = np.clip(eff_metalness * inv + user_overrides.metalness, 0.0, 1.0).astype(np.float32)
        eff_spec_level = np.clip(eff_spec_level * inv + (user_overrides.specular_level - 0.04 * inv), 0.0, 1.0).astype(np.float32)
        conf = np.clip(conf * inv + user_overrides.confidence * cov, 0.0, 1.0).astype(np.float32)
        if user_overrides.specular_color is not None:
            sc_user = user_overrides.specular_color - inv[..., None] * 0.65
            eff_spec_color = np.clip(eff_spec_color * inv[..., None] + sc_user, 0.0, 1.0).astype(np.float32)
        if user_overrides.clearcoat is not None:
            base_cc = eff_clearcoat if eff_clearcoat is not None else np.zeros((h, w), dtype=np.float32)
            eff_clearcoat = np.clip(base_cc * inv + user_overrides.clearcoat, 0.0, 1.0).astype(np.float32)
        if user_overrides.clearcoat_roughness is not None:
            base_ccr = eff_cc_rough if eff_cc_rough is not None else np.full((h, w), 0.15, dtype=np.float32)
            eff_cc_rough = np.clip(base_ccr * inv + (user_overrides.clearcoat_roughness - 0.15 * inv), 0.0, 1.0).astype(np.float32)
        eff_subsurface = user_overrides.subsurface
        if user_overrides.anisotropy is not None:
            base_a = eff_aniso if eff_aniso is not None else np.zeros((h, w), dtype=np.float32)
            eff_aniso = np.clip(base_a * inv + user_overrides.anisotropy, 0.0, 1.0).astype(np.float32)
        eff_aniso_angle = user_overrides.anisotropy_angle

    raw_stats = {
        "roughness": (
            {"mean": float(np.mean(r_raw)), "p10": float(np.percentile(r_raw, 10)), "p90": float(np.percentile(r_raw, 90))}
            if r_raw is not None else None
        ),
        "metalness": (
            {"mean": float(np.mean(m_raw)), "p10": float(np.percentile(m_raw, 10)), "p90": float(np.percentile(m_raw, 90))}
            if m_raw is not None else None
        ),
        "roughness_uncertainty_mean": (float(np.mean(roughness_uncertainty)) if roughness_uncertainty is not None else None),
        "metalness_uncertainty_mean": (float(np.mean(metalness_uncertainty)) if metalness_uncertainty is not None else None),
    }

    return MaterialMaps(
        roughness=eff_roughness,
        metalness=eff_metalness,
        specular_level=eff_spec_level,
        coverage=cov,
        confidence=conf,
        specular_color=eff_spec_color,
        clearcoat=eff_clearcoat,
        clearcoat_roughness=eff_cc_rough,
        subsurface=eff_subsurface,
        anisotropy=None if eff_aniso is None else eff_aniso.astype(np.float32),
        anisotropy_angle=eff_aniso_angle,
        metadata={
            "origin": "confidence_aware_model_priors" if user_overrides is None else "user_override_over_model_priors",
            "raw_model_predictions": raw_stats,
            "calibrated_summary": {
                "roughness_mean": float(np.mean(eff_roughness)),
                "metalness_mean": float(np.mean(eff_metalness)),
                "clearcoat_mean": (float(np.mean(eff_clearcoat)) if eff_clearcoat is not None else 0.0),
                "confidence_mean": float(np.mean(conf)),
            },
            "metalness_calibration": {
                "method": "uncertainty_and_roughness_compatible_dielectric_gate",
                "threshold_low": float(metalness_threshold_low),
                "threshold_high": float(metalness_threshold_high),
                "brightness_heuristic_used": False,
                "category_rules_used": False,
            },
            "neural_material_estimation_used": True,
        },
    )
