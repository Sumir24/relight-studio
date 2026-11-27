from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from .contracts import EnvironmentLight, Light, LightingRig, SceneBuffers
from .color import luminance, srgb_to_linear


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
class SourceIllumination:
    """General multi-light + environment + SH source-illumination contract.

    Backward-compatible with `SourceLightCalibration` via `from_single_key()`.
    Automatic estimates carry `provenance["reviewed"] = False` and confidence in [0, 1].
    """
    lights: tuple[Light, ...] = ()
    ambient_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ambient_strength: float = 0.15
    ambient_ground_color: tuple[float, float, float] | None = None
    environment: EnvironmentLight | None = None
    sh_coefficients: np.ndarray | None = field(default=None, repr=False, compare=False)
    source_shading: np.ndarray | None = field(default=None, repr=False, compare=False)
    exposure_scale: float = 1.0
    white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0)
    confidence: float = 1.0
    provenance: dict = field(default_factory=dict, compare=False)

    def validate(self) -> None:
        if not math.isfinite(self.ambient_strength) or self.ambient_strength < 0.0:
            raise ValueError("SourceIllumination ambient_strength must be finite and non-negative.")
        if not math.isfinite(self.exposure_scale) or self.exposure_scale <= 0.0:
            raise ValueError("SourceIllumination exposure_scale must be finite and positive.")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("SourceIllumination confidence must be in [0, 1].")
        for cname, cval in (("ambient_color", self.ambient_color), ("white_balance", self.white_balance)):
            arr = np.asarray(cval, dtype=np.float64)
            if arr.shape != (3,) or not np.isfinite(arr).all() or np.any(arr < 0.0):
                raise ValueError(f"SourceIllumination {cname} must be a finite non-negative 3-vector.")
        if np.any(np.asarray(self.white_balance, dtype=np.float64) <= 0.0):
            raise ValueError("SourceIllumination white_balance channels must be strictly positive.")
        if self.ambient_ground_color is not None:
            g = np.asarray(self.ambient_ground_color, dtype=np.float64)
            if g.shape != (3,) or not np.isfinite(g).all() or np.any(g < 0.0):
                raise ValueError("SourceIllumination ambient_ground_color must be a finite non-negative 3-vector.")
        if self.environment is not None:
            self.environment.validate()
        for light in self.lights:
            light.validate()
        if self.sh_coefficients is not None:
            sh = np.asarray(self.sh_coefficients, dtype=np.float32)
            if sh.shape != (9, 3) or not np.isfinite(sh).all():
                raise ValueError("SourceIllumination sh_coefficients must have finite shape (9, 3).")

    @classmethod
    def from_single_key(cls, calib: SourceLightCalibration) -> SourceIllumination:
        calib.validate()
        rig = calib.to_rig()
        return cls(
            lights=rig.lights,
            ambient_color=rig.ambient_color,
            ambient_strength=rig.ambient_strength,
            confidence=1.0,
            provenance=calib.to_metadata(),
        )

    def to_rig(self) -> LightingRig:
        self.validate()
        wb = np.asarray(self.white_balance, dtype=np.float64)
        exp_s = float(self.exposure_scale)
        norm_lights = []
        for light in self.lights:
            c = tuple(float(x) for x in (np.asarray(light.color, dtype=np.float64) * wb))
            norm_lights.append(
                Light(
                    name=light.name,
                    kind=light.kind,
                    color=c,  # type: ignore[arg-type]
                    strength=float(light.strength) * exp_s,
                    direction=light.direction,
                    position=light.position,
                    area_size=light.area_size,
                    samples=light.samples,
                    casts_shadows=light.casts_shadows,
                    wrap=light.wrap,
                    radius=light.radius,
                    light_id=light.light_id,
                    spot_inner_angle_deg=light.spot_inner_angle_deg,
                    spot_outer_angle_deg=light.spot_outer_angle_deg,
                    intensity_mode=light.intensity_mode,
                )
            )
        amb_col = tuple(float(x) for x in (np.asarray(self.ambient_color, dtype=np.float64) * wb))
        g_col = (
            tuple(float(x) for x in (np.asarray(self.ambient_ground_color, dtype=np.float64) * wb))
            if self.ambient_ground_color is not None
            else None
        )
        rig = LightingRig(
            ambient_color=amb_col,  # type: ignore[arg-type]
            ambient_strength=float(self.ambient_strength) * exp_s,
            lights=tuple(norm_lights),
            ambient_ground_color=g_col,  # type: ignore[arg-type]
            environment=self.environment,
        )
        rig.validate()
        return rig

    def to_metadata(self) -> dict:
        self.validate()
        return {
            "light_count": len(self.lights),
            "ambient_strength": float(self.ambient_strength),
            "exposure_scale": float(self.exposure_scale),
            "white_balance": [float(x) for x in self.white_balance],
            "has_environment": self.environment is not None,
            "has_sh_coefficients": self.sh_coefficients is not None,
            "confidence": float(self.confidence),
            "provenance": dict(self.provenance),
        }


def compute_intrinsic_reconstruction_error(scene: SceneBuffers) -> dict | None:
    """Compute scale-normalized linear reconstruction error map and diagnostics when intrinsic buffers exist."""
    if scene.albedo is None or scene.source_shading is None:
        return None
    eps = 1e-4
    src_lin = srgb_to_linear(scene.rgb_srgb)
    alb_lin = srgb_to_linear(scene.albedo)
    sh = np.asarray(scene.source_shading, dtype=np.float32)
    if sh.ndim == 2:
        sh = np.repeat(sh[..., None], 3, axis=-1)
    pred = alb_lin * np.maximum(sh, 0.0)
    if scene.non_diffuse_residual is not None:
        pred = pred + np.maximum(scene.non_diffuse_residual, 0.0)

    valid = (scene.depth > 0) & (scene.lightable_mask > 0.05)
    if scene.alpha_mask is not None:
        valid = valid & (scene.alpha_mask > 0.05)
    if not np.any(valid):
        valid = np.ones(scene.shape, dtype=bool)

    y_src = np.maximum(luminance(src_lin), eps)
    y_pred = np.maximum(luminance(pred), eps)
    scale = float(np.median(y_src[valid]) / max(float(np.median(y_pred[valid])), eps))
    scale = float(np.clip(scale, 0.1, 10.0))
    err_map = np.mean(np.abs(pred * scale - src_lin), axis=-1).astype(np.float32)
    valid_err = err_map[valid]
    return {
        "error_map": err_map,
        "scale_factor": scale,
        "mean_error": float(np.mean(valid_err)),
        "rmse": float(np.sqrt(np.mean(valid_err * valid_err))),
        "p95_error": float(np.percentile(valid_err, 95)),
        "high_error_fraction": float(np.mean(valid_err > 0.08)),
    }


def estimate_source_illumination(
    scene: SceneBuffers,
    normals: np.ndarray,
    *,
    source_shading: np.ndarray | None = None,
) -> SourceIllumination:
    """Propose multi-light + SH source illumination from normals, albedo, and observed shading.

    Excludes/down-weights clipped highlights, transparent pixels, uncertain materials, and
    potential emissive regions during SH fitting. Returned object is an unreviewed proposal.
    """
    eps = 1e-4
    src_lin = srgb_to_linear(scene.rgb_srgb)
    y_src = np.maximum(luminance(src_lin), eps)

    sh_in = source_shading if source_shading is not None else scene.source_shading
    if sh_in is not None:
        obs_sh = np.asarray(sh_in, dtype=np.float32)
        if obs_sh.ndim == 2:
            obs_sh = np.repeat(obs_sh[..., None], 3, axis=-1)
    elif scene.albedo is not None:
        alb_lin = np.maximum(srgb_to_linear(scene.albedo), 0.04)
        obs_sh = np.clip(src_lin / alb_lin, 0.02, 4.0).astype(np.float32)
    else:
        obs_sh = np.broadcast_to(y_src[..., None], (*scene.shape, 3)).copy()

    # Build strict exclusion mask & soft fit weights (highlights, alpha, uncertain material, sky, emissive outliers).
    from .scene_policy import detect_sky_horizon_mask
    sky_mask = detect_sky_horizon_mask(scene)
    active = (scene.depth > 0) & (scene.lightable_mask > 0.05) & (sky_mask < 0.25)
    if scene.protected_mask is not None:
        active = active & (scene.protected_mask < 0.95)
    if scene.alpha_mask is not None:
        active = active & (scene.alpha_mask >= 0.05)
    not_clipped = np.max(scene.rgb_srgb, axis=-1) < 0.92

    weights = active.astype(np.float64) * not_clipped.astype(np.float64)
    if scene.alpha_mask is not None:
        weights *= np.clip(scene.alpha_mask.astype(np.float64), 0.0, 1.0)
    if scene.geometry_confidence is not None:
        weights *= np.clip(scene.geometry_confidence.astype(np.float64), 0.0, 1.0)
    if scene.material_confidence is not None:
        weights *= np.where(scene.material_confidence >= 0.25, np.clip(scene.material_confidence, 0.0, 1.0), 0.05)

    # Down-weight suspected emissive/glare outliers relative to albedo-implied brightness.
    obs_y = luminance(obs_sh)
    if np.any(weights > 0.05):
        med_obs = max(float(np.median(obs_y[weights > 0.05])), eps)
        emissive_outlier = obs_y > (3.5 * med_obs)
        weights = np.where(emissive_outlier, 0.0, weights)

    fit_mask = weights > 0.05
    if np.sum(fit_mask) < 16:
        fit_mask = scene.depth > 0
        weights = np.ones(scene.shape, dtype=np.float64)

    nx, ny, nz = normals[..., 0], normals[..., 1], normals[..., 2]
    # Inverse-density stratification over (nx, ny) bins prevents large flat walls/floors
    # from silencing diverse subject surface normals during source light estimation.
    bx = np.clip(((nx[fit_mask] + 1.0) * 4.0).astype(np.int32), 0, 7)
    by = np.clip(((ny[fit_mask] + 1.0) * 4.0).astype(np.int32), 0, 7)
    bin_ids = by * 8 + bx
    bin_counts = np.bincount(bin_ids, minlength=64).astype(np.float64)
    strat_w = 1.0 / np.sqrt(np.maximum(bin_counts[bin_ids], 1.0))

    basis = np.stack([
        np.ones_like(nx), ny, nz, nx,
        nx * ny, ny * nz, 3.0 * nz * nz - 1.0, nx * nz, nx * nx - ny * ny,
    ], axis=-1).astype(np.float64)

    w_pts = weights[fit_mask] * strat_w
    b_pts = basis[fit_mask]
    n_pts = normals[fit_mask].astype(np.float64)
    y_pts = obs_sh[fit_mask].astype(np.float64)
    if b_pts.shape[0] > 20000:
        step = max(1, b_pts.shape[0] // 20000)
        w_pts, b_pts, n_pts, y_pts = w_pts[::step], b_pts[::step], n_pts[::step], y_pts[::step]

    sw = np.sqrt(w_pts)[:, None]
    bw = b_pts * sw
    yw = y_pts * sw
    reg_diag = np.array([0.02, 0.25, 0.25, 0.25, 0.8, 0.8, 0.8, 0.8, 0.8], dtype=np.float64)
    gram = bw.T @ bw + np.diag(reg_diag * max(b_pts.shape[0] * 1e-3, 1.0))
    coeffs = np.linalg.solve(gram, bw.T @ yw)  # shape (9, 3)

    # Front-hemisphere constrained NNLS fit (L_z <= -0.18) prevents behind-wall source estimates.
    from scipy.optimize import nnls
    dome_dirs = [np.array([0.0, 0.0, -1.0], dtype=np.float64)]
    for az_deg in range(0, 360, 60):
        az = math.radians(az_deg)
        dome_dirs.append(np.array([0.55 * math.cos(az), 0.55 * math.sin(az), -math.sqrt(1.0 - 0.55 ** 2)], dtype=np.float64))
        dome_dirs.append(np.array([0.88 * math.cos(az + 0.52), 0.88 * math.sin(az + 0.52), -math.sqrt(1.0 - 0.88 ** 2)], dtype=np.float64))
    dome_mat = np.stack(dome_dirs, axis=0)  # (K, 3)
    lambert_cols = [np.ones((n_pts.shape[0],), dtype=np.float64)]
    for dvec in dome_mat:
        lambert_cols.append(np.maximum(n_pts @ dvec, 0.0))
    design = np.stack(lambert_cols, axis=1)  # (N, 1+K)
    design_w = design * sw

    dome_weights_rgb = np.zeros((1 + len(dome_mat), 3), dtype=np.float64)
    pred_nnls = np.zeros_like(y_pts)
    for ch in range(3):
        sol_c, _ = nnls(design_w, yw[:, ch])
        dome_weights_rgb[:, ch] = sol_c
        pred_nnls[:, ch] = np.maximum(design @ sol_c, 1e-4)

    ss_res = float(np.sum(w_pts[:, None] * np.square(y_pts - pred_nnls)))
    mean_y = np.sum(w_pts[:, None] * y_pts, axis=0, keepdims=True) / max(float(np.sum(w_pts)), eps)
    ss_tot = float(np.sum(w_pts[:, None] * np.square(y_pts - mean_y)))
    r2 = float(np.clip(1.0 - ss_res / max(ss_tot, 1e-6), 0.0, 1.0))
    rmse_rel = float(np.sqrt(ss_res / max(float(np.sum(w_pts)), eps)) / max(float(np.mean(mean_y)), eps))
    fit_conf = float(np.clip(0.55 * r2 + 0.45 * math.exp(-rmse_rel * 1.5), 0.05, 0.98))

    # Extract dominant front-hemisphere key light & ambient from NNLS lobe decomposition.
    dc = np.maximum(dome_weights_rgb[0], 0.08)
    lobe_w_lum = 0.2126 * dome_weights_rgb[1:, 0] + 0.7152 * dome_weights_rgb[1:, 1] + 0.0722 * dome_weights_rgb[1:, 2]
    total_dir_energy = float(np.sum(lobe_w_lum))
    if total_dir_energy > 1e-4:
        # Energy-weighted vector sum over dominant top lobes (stays in front hemisphere because all lobes have Z <= -0.18).
        weighted_vec = np.sum(dome_mat * lobe_w_lum[:, None], axis=0)
        key_dir = weighted_vec / max(float(np.linalg.norm(weighted_vec)), 1e-8)
        key_dir[2] = min(float(key_dir[2]), -0.15)
        key_dir = key_dir / max(float(np.linalg.norm(key_dir)), 1e-8)
        key_color_raw = np.maximum(np.sum(dome_weights_rgb[1:], axis=0), 0.02)
    else:
        key_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        key_color_raw = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    key_color_norm = key_color_raw / max(float(np.max(key_color_raw)), 1e-4)
    dc_norm = dc / max(float(np.max(dc)), 1e-4)
    dc_lum = max(float(0.2126 * dc[0] + 0.7152 * dc[1] + 0.0722 * dc[2]), 0.08)
    key_str = float(np.clip(total_dir_energy / max(dc_lum, 0.20), 0.15, 2.5))
    amb_str = float(np.clip(dc_lum * 0.55, 0.15, 1.5))

    return SourceIllumination(
        lights=(
            Light(
                name="estimated_source_key",
                kind="directional",
                direction=tuple(float(x) for x in key_dir),
                color=tuple(float(x) for x in key_color_norm),  # type: ignore[arg-type]
                strength=key_str,
                casts_shadows=False,
            ),
        ),
        ambient_color=tuple(float(x) for x in dc_norm),  # type: ignore[arg-type]
        ambient_strength=amb_str,
        sh_coefficients=coeffs.astype(np.float32),
        source_shading=obs_sh,
        exposure_scale=1.0,
        white_balance=(1.0, 1.0, 1.0),
        confidence=fit_conf,
        provenance={
            "source": "automatic_sh_source_proposal",
            "reviewed": False,
            "weighted_r2": r2,
            "relative_rmse": rmse_rel,
            "low_confidence_warning": fit_conf < 0.45,
        },
    )


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
    raw_maps: dict[str, np.ndarray] = field(default_factory=dict)


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
            "confidence_applied": False,
            "confidence_blending_stage": "deferred_to_renderer",
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
    raw_maps: dict[str, np.ndarray] = {}
    if r_raw is not None:
        raw_maps["roughness"] = r_raw.copy()
    if m_raw is not None:
        raw_maps["metalness"] = m_raw.copy()
    if specular_level is not None:
        raw_maps["specular_level"] = np.clip(np.asarray(specular_level, dtype=np.float32), 0.0, 1.0)
    if specular_color is not None:
        raw_maps["specular_color"] = np.clip(np.asarray(specular_color, dtype=np.float32), 0.0, 1.0)
    if anisotropy is not None:
        raw_maps["anisotropy"] = np.clip(np.asarray(anisotropy, dtype=np.float32), 0.0, 1.0)

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
            "confidence_applied": True,
            "confidence_blending_stage": "pre_blended_in_calibration",
        },
        raw_maps=raw_maps,
    )


def suggest_material_priors_from_semantics(scene: SceneBuffers) -> MaterialMaps | None:
    """Build conservative semantic PBR material priors when legacy scenes lack material maps.

    Never overrides explicit model maps or user-authored material maps; marks low-to-moderate
    confidence (`0.42`) and `origin="semantic_fallback_priors"`.
    """
    if not scene.semantic_masks:
        return None
    h, w = scene.shape
    # Collect dielectric / organic sub-masks first so overlapping parent metal masks
    # (e.g. a watch mask spanning crystal, black dial, painted markers, and surrounding fingers)
    # do not crush dial diffuse or bleed conductive metalness onto curved skin boundaries.
    non_metal_union = np.zeros((h, w), dtype=np.float32)
    for raw_label, raw_mask in scene.semantic_masks.items():
        kl = raw_label.lower()
        if any(
            t in kl
            for t in (
                "dial", "marker", "numeral", "text", "label", "screen",
                "glass", "crystal", "lens", "visor",
                "face", "skin", "hand", "arm", "leg", "neck", "ear", "finger",
                "hair", "fur", "leather", "bag", "handbag", "shoe", "boot",
                "suit", "trousers", "shirt", "tie", "coat", "jacket",
            )
        ):
            m = np.clip(np.asarray(raw_mask, dtype=np.float32), 0.0, 1.0)
            non_metal_union = np.maximum(non_metal_union, m)

    assignments: list[MaterialAssignment] = []
    for raw_label, raw_mask in scene.semantic_masks.items():
        kl = raw_label.lower()
        mask = np.clip(np.asarray(raw_mask, dtype=np.float32), 0.0, 1.0)
        if not np.any(mask > 0.05):
            continue
        if any(t in kl for t in ("dial", "screen")):
            assignments.append(MaterialAssignment(
                name=f"auto_{raw_label}",
                mask=mask,
                roughness=0.22,
                metalness=0.0,
                specular_level=0.06,
                clearcoat=0.65,
                clearcoat_roughness=0.06,
                blend=0.90,
                confidence=0.45,
            ))
        elif any(t in kl for t in ("marker", "numeral", "text", "label")):
            assignments.append(MaterialAssignment(
                name=f"auto_{raw_label}",
                mask=mask,
                roughness=0.28,
                metalness=0.20,
                specular_level=0.18,
                blend=0.90,
                confidence=0.45,
            ))
        elif any(t in kl for t in ("watch", "ring", "clasp", "buckle", "metal", "chrome", "steel", "gold", "jewelry", "silver")):
            # Suppress soft feathered halos beyond the crisp object boundary and subtract
            # overlapping dielectric/organic sub-parts so dials and adjacent fingers stay intact.
            crisp_mask = np.clip((mask - 0.30) / 0.55, 0.0, 1.0).astype(np.float32)
            crisp_mask = np.power(crisp_mask, 1.25) * np.clip(1.0 - 0.90 * non_metal_union, 0.0, 1.0)
            if not np.any(crisp_mask > 0.05):
                continue
            assignments.append(MaterialAssignment(
                name=f"auto_{raw_label}",
                mask=crisp_mask,
                roughness=0.24,
                metalness=0.85,
                specular_level=0.85,
                anisotropy=0.35,
                blend=0.85,
                confidence=0.45,
            ))
        elif any(t in kl for t in ("sunglasses", "glasses", "crystal", "glass", "lens", "visor")):
            assignments.append(MaterialAssignment(
                name=f"auto_{raw_label}",
                mask=mask,
                roughness=0.16,
                metalness=0.0,
                specular_level=0.06,
                clearcoat=0.85,
                clearcoat_roughness=0.06,
                blend=0.85,
                confidence=0.45,
            ))
        elif any(t in kl for t in ("face", "skin", "hand", "arm", "leg", "neck", "ear", "finger")):
            assignments.append(MaterialAssignment(
                name=f"auto_{raw_label}",
                mask=mask,
                roughness=0.52,
                metalness=0.0,
                specular_level=0.045,
                subsurface=0.35,
                blend=0.80,
                confidence=0.42,
            ))
        elif any(t in kl for t in ("hair", "his_hair", "fur")):
            assignments.append(MaterialAssignment(
                name=f"auto_{raw_label}",
                mask=mask,
                roughness=0.42,
                metalness=0.0,
                specular_level=0.05,
                anisotropy=0.45,
                blend=0.80,
                confidence=0.40,
            ))
        elif any(t in kl for t in ("bag", "handbag", "leather", "footwear", "shoe", "boot")):
            assignments.append(MaterialAssignment(
                name=f"auto_{raw_label}",
                mask=mask,
                roughness=0.44,
                metalness=0.0,
                specular_level=0.05,
                clearcoat=0.22,
                clearcoat_roughness=0.25,
                blend=0.75,
                confidence=0.40,
            ))
        elif any(t in kl for t in ("suit", "trousers", "shirt", "tie", "coat", "jacket", "wall", "floor", "asphalt", "stone", "mountain", "grass", "ground")):
            assignments.append(MaterialAssignment(
                name=f"auto_{raw_label}",
                mask=mask,
                roughness=0.84,
                metalness=0.0,
                specular_level=0.03,
                blend=0.70,
                confidence=0.40,
            ))
    if not assignments:
        return None
    if not any(any(t in k.lower() for t in ("face", "skin", "hand", "finger", "palm")) for k in scene.semantic_masks):
        skin_map = compute_skin_likelihood_map(
            scene.rgb_srgb,
            semantic_masks=scene.semantic_masks,
            depth=scene.depth,
            material_maps={"metalness": scene.metalness} if scene.metalness is not None else None,
        )
        if np.any(skin_map > 0.25):
            assignments.append(MaterialAssignment(
                name="auto_inferred_skin",
                mask=skin_map,
                roughness=0.54,
                metalness=0.0,
                specular_level=0.035,
                subsurface=0.30,
                blend=0.80,
                confidence=0.40,
            ))
    maps = build_material_maps(scene.shape, assignments)
    maps.metadata["origin"] = "semantic_fallback_priors"
    maps.metadata["confidence_applied"] = False
    return maps


def calibrate_conductor_from_appearance(scene: SceneBuffers) -> tuple[np.ndarray, np.ndarray] | None:
    """Model-agnostic specular-contrast & neutrality calibration for reflective conductors.

    Detects high-dynamic-range specular-to-trough contrast with low chroma saturation
    (e.g., polished/brushed metals, chrome, jewelry) while excluding organic/dielectric
    semantic masks, returning updated `(metalness, roughness)` maps.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

    rgb = np.asarray(scene.rgb_srgb, dtype=np.float32)
    c_max = np.max(rgb, axis=-1)
    c_min = np.min(rgb, axis=-1)
    y = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    sat = (c_max - c_min) / np.maximum(c_max, 1e-4)

    loc_max = maximum_filter(y, size=15)
    loc_min = minimum_filter(y, size=15)
    loc_contrast = loc_max - loc_min
    loc_sat = gaussian_filter(sat, sigma=3.0, mode="reflect")
    # Require true specular peak (>0.78) + deep trough (<0.24) + multi-pixel highlight support.
    peak_score = np.clip((loc_max - 0.76) / 0.14, 0.0, 1.0)
    trough_score = np.clip((0.26 - loc_min) / 0.16, 0.0, 1.0)
    spec_support = np.clip((gaussian_filter((y > 0.76).astype(np.float32), sigma=1.5, mode="reflect") - 0.10) / 0.25, 0.0, 1.0)

    # Require smooth midtone rolloff and low 1px halftone/stipple grain so 2D illustrations aren't flagged as chrome.
    midtone_support = np.clip((gaussian_filter(((y > 0.18) & (y < 0.74)).astype(np.float32), sigma=2.5, mode="reflect") - 0.12) / 0.25, 0.0, 1.0)
    hf_grain = gaussian_filter(np.abs(y - gaussian_filter(y, sigma=1.0, mode="reflect")), sigma=3.0, mode="reflect")
    smooth_rolloff = np.clip((0.085 - hf_grain) / 0.04, 0.0, 1.0)

    # Sharp specular-contrast cue: strong local dynamic range + neutral chroma + real highlight core.
    contrast_score = np.clip((loc_contrast - 0.50) / 0.30, 0.0, 1.0) * peak_score * trough_score
    neutral_score = np.clip((0.24 - loc_sat) / 0.16, 0.0, 1.0)
    conductor_score = gaussian_filter(
        contrast_score * neutral_score * spec_support * midtone_support * smooth_rolloff,
        sigma=2.5,
        mode="reflect",
    )

    if scene.normals is not None:
        n = np.asarray(scene.normals, dtype=np.float32)
        n_var = (
            (maximum_filter(n[..., 0], size=9) - minimum_filter(n[..., 0], size=9))
            + (maximum_filter(n[..., 1], size=9) - minimum_filter(n[..., 1], size=9))
        )
        # Flat planes with 2D high-contrast graphics/ink are not curved metallic conductors.
        curved_surface = np.clip((n_var - 0.03) / 0.08, 0.0, 1.0)
        conductor_score = conductor_score * curved_surface

    if scene.depth is not None:
        d = np.asarray(scene.depth, dtype=np.float32)
        d_max = maximum_filter(d, size=15)
        d_min = minimum_filter(np.where(d > 0, d, d_max), size=15)
        # Contrast must come from shading/reflection on a continuous surface, never a fg/bg depth cliff.
        same_surface = (d > 0) & ((d_max - d_min) < 0.22 * np.maximum(d, 1e-3))
        conductor_score = np.where(same_surface, conductor_score, 0.0)

    # Never override explicit dielectric/organic semantic regions.
    if scene.semantic_masks:
        non_metal = np.zeros(scene.shape, dtype=np.float32)
        for raw_label, raw_mask in scene.semantic_masks.items():
            kl = raw_label.lower()
            if any(
                t in kl
                for t in (
                    "dial", "marker", "numeral", "text", "label", "screen",
                    "face", "skin", "hand", "arm", "leg", "neck", "ear", "finger",
                    "hair", "fur", "suit", "trousers", "shirt", "tie", "coat", "jacket",
                    "wall", "floor", "sky", "mountain", "grass", "stone", "ground", "snow",
                )
            ):
                non_metal = np.maximum(non_metal, np.clip(np.asarray(raw_mask, dtype=np.float32), 0.0, 1.0))
        conductor_score = conductor_score * np.clip(1.0 - non_metal, 0.0, 1.0)

    if not np.any(conductor_score > 0.30):
        return None

    w_cond = np.clip((conductor_score - 0.25) / 0.45, 0.0, 1.0).astype(np.float32)
    base_m = (
        np.clip(np.asarray(scene.metalness, dtype=np.float32), 0.0, 1.0)
        if scene.metalness is not None
        else np.zeros(scene.shape, dtype=np.float32)
    )
    base_r = (
        np.clip(np.asarray(scene.roughness, dtype=np.float32), 0.04, 1.0)
        if scene.roughness is not None
        else np.full(scene.shape, 0.50, dtype=np.float32)
    )
    target_m = 0.90
    target_r = 0.14
    out_m = np.maximum(base_m, w_cond * target_m).astype(np.float32)
    out_r = (base_r * (1.0 - 0.75 * w_cond) + target_r * (0.75 * w_cond)).astype(np.float32)
    out_r = np.minimum(base_r, out_r).astype(np.float32)
    return out_m, np.clip(out_r, 0.04, 1.0).astype(np.float32)


def compute_skin_likelihood_map(
    image: np.ndarray,
    semantic_masks: dict[str, np.ndarray] | None = None,
    depth: np.ndarray | None = None,
    material_maps: object | None = None,
    *,
    min_skin_confidence: float = 0.35,
) -> np.ndarray:
    """Estimate a smooth [0, 1] human skin weight map, biased toward zero false positives on metals/non-organic surfaces.

    Signal priority:
    1. Explicit semantic skin/body masks (`face`, `skin`, `hand`, `finger`, `palm`, `arm`, `neck`, `ear`, `leg`).
    2. Chromatic/texture skin prior restricted to `person`/`man`/`woman` masks or elevated foreground
       regions outside non-skin material masks (`watch`, `ring`, `metal`, `suit`, `wall`, `sky`, etc.).
    Excludes hair, eyebrows, lashes, eyes, teeth, lips, jewelry, watch straps, high-texture wood/sand/leather,
    monochrome halftone illustrations, and metallic pixels (`metalness >= 0.15`).
    """
    from scipy.ndimage import binary_dilation, gaussian_filter, grey_closing

    rgb = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    h, w = rgb.shape[:2]
    r_c, g_c, b_c = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y_s = (0.299 * r_c + 0.587 * g_c + 0.114 * b_c).astype(np.float32)
    c_max = np.maximum(np.maximum(r_c, g_c), b_c)
    c_min = np.minimum(np.minimum(r_c, g_c), b_c)
    sat = ((c_max - c_min) / np.maximum(c_max, 1e-4)).astype(np.float32)

    # Smoothness & halftone rejection (slightly softer inside semantic person masks so chin stubble / nose highlights stay covered)
    y_blur = gaussian_filter(y_s, sigma=1.2, mode="reflect")
    y_var = np.maximum(gaussian_filter(y_s * y_s, sigma=1.2, mode="reflect") - y_blur * y_blur, 0.0)
    local_std = np.sqrt(y_var).astype(np.float32)
    hf_grain = gaussian_filter(np.abs(y_s - gaussian_filter(y_s, sigma=1.0, mode="reflect")), sigma=2.0, mode="reflect")
    smooth_tex_conf = np.clip((0.070 - local_std) / 0.035, 0.0, 1.0) * np.clip((0.065 - hf_grain) / 0.025, 0.0, 1.0)
    smooth_person_conf = np.clip((0.095 - local_std) / 0.045, 0.0, 1.0) * np.clip((0.075 - hf_grain) / 0.030, 0.0, 1.0)

    # Reject dark eyebrows/lashes/pupils, white eyes/teeth, and saturated red lips
    eye_teeth_white = np.clip((y_s - 0.65) / 0.20, 0.0, 1.0) * np.clip((0.12 - sat) / 0.08, 0.0, 1.0)
    dark_hair_feature = np.clip((0.08 - y_s) / 0.06, 0.0, 1.0)
    red_lip_chroma = np.clip(((r_c - g_c) - 2.2 * np.maximum(g_c - b_c, 0.0)) / 0.15, 0.0, 1.0) * np.clip((sat - 0.50) / 0.20, 0.0, 1.0)
    feature_keep = np.clip(1.0 - np.maximum(eye_teeth_white, np.maximum(dark_hair_feature, red_lip_chroma)), 0.0, 1.0)

    # Resolve metalness if supplied via MaterialMaps, SceneBuffers, or dict
    metal_arr = None
    if material_maps is not None:
        if isinstance(material_maps, dict):
            metal_arr = material_maps.get("metalness")
        else:
            metal_arr = getattr(material_maps, "metalness", None)
    non_metal_conf = (
        np.clip((0.15 - np.asarray(metal_arr, dtype=np.float32)) / 0.10, 0.0, 1.0)
        if metal_arr is not None
        else np.ones((h, w), dtype=np.float32)
    )

    explicit_skin = np.zeros((h, w), dtype=np.float32)
    person_union = np.zeros((h, w), dtype=np.float32)
    exclude_union = np.zeros((h, w), dtype=np.float32)
    accessory_edge_mask = np.zeros((h, w), dtype=bool)
    hand_accessory_mask = np.zeros((h, w), dtype=bool)

    skin_tokens = ("face", "skin", "hand", "finger", "palm", "arm", "neck", "ear", "leg")
    person_tokens = ("man", "woman", "person", "boy", "girl", "human", "hiker")
    exclude_tokens = (
        "watch", "ring", "jewelry", "bracelet", "necklace", "strap", "band", "clasp", "buckle",
        "metal", "chrome", "steel", "gold", "silver", "copper", "dial", "black_dial", "marker", "numeral",
        "suit", "trousers", "shirt", "tie", "coat", "jacket", "collar", "dress", "skirt",
        "hair", "his_hair", "fur", "eyebrow", "lash", "eye", "sunglasses", "glasses", "teeth", "lip",
        "bag", "handbag", "shoe", "footwear", "boot", "hat", "cap",
        "wall", "floor", "sky", "mountain", "grass", "stone", "ground", "snow", "background",
        "wood", "sand", "cardboard", "leather", "statue", "sculpture", "design", "rock",
    )
    accessory_tokens = ("watch", "ring", "jewelry", "bracelet", "strap", "band", "sunglasses", "glasses", "collar")
    hand_acc_tokens = ("watch", "ring", "jewelry", "bracelet", "strap", "band")

    if semantic_masks:
        for raw_label, raw_mask in semantic_masks.items():
            kl = raw_label.lower()
            m = np.clip(np.asarray(raw_mask, dtype=np.float32), 0.0, 1.0)
            if any(t in kl for t in exclude_tokens):
                exclude_union = np.maximum(exclude_union, m)
                if any(t in kl for t in accessory_tokens):
                    accessory_edge_mask |= (m > 0.20)
                if any(t in kl for t in hand_acc_tokens):
                    hand_accessory_mask |= (m > 0.20)
            elif any(t in kl for t in skin_tokens):
                explicit_skin = np.maximum(explicit_skin, m)
            elif any(t in kl for t in person_tokens):
                person_union = np.maximum(person_union, m)

    # Eroded safety border around watch/ring/jewelry straps where wrist vs strap is uncertain -> default low
    if np.any(accessory_edge_mask):
        dil = binary_dilation(accessory_edge_mask, iterations=2).astype(np.float32)
        exclude_union = np.maximum(exclude_union, 0.60 * dil)

    # Priority 1: Explicit semantic skin masks (still filtered against monochrome ink, metal, eyes/teeth/hair/accessories)
    p1_conf = explicit_skin * np.clip(1.0 - exclude_union, 0.0, 1.0) * non_metal_conf * feature_keep
    # Guard monochrome illustration 'face' masks (e.g. manga): require either slight skin chroma or smooth photo shading
    photo_skin_tone = np.clip((sat - 0.06) / 0.10, 0.0, 1.0) * np.clip((0.060 - hf_grain) / 0.025, 0.0, 1.0)
    p1_conf = p1_conf * np.maximum(photo_skin_tone, 0.0)

    # Priority 2: Chromatic + texture prior inside person masks OR elevated foreground outside exclusions
    chrom_skin_strict = (
        (r_c > g_c)
        & (g_c >= b_c * 0.93)
        & ((r_c - b_c) > 0.045)
        & (sat > 0.12)
        & (sat < 0.65)
        & (y_s > 0.08)
        & (y_s < 0.88)
    ).astype(np.float32)
    # Inside a semantic person mask (outside clothes/hair/accessories), pale highlight cheeks/noses can have sat >= 0.07
    chrom_skin_person = (
        (r_c > g_c)
        & (g_c >= b_c * 0.93)
        & ((r_c - b_c) > 0.035)
        & (sat > 0.07)
        & (sat < 0.68)
        & (y_s > 0.08)
        & (y_s < 0.92)
    ).astype(np.float32)
    chrom_skin = np.where(person_union > 0.20, chrom_skin_person, chrom_skin_strict)
    eff_tex_conf = np.where(person_union > 0.20, smooth_person_conf, smooth_tex_conf)
    # Tighter skin green ratio: exclude yellow sand/cardboard ((R-G)/(R-B) < 0.20) while keeping fair/pinkish palms (up to ~1.12)
    rg_frac = (r_c - g_c) / np.maximum(r_c - b_c, 1e-4)
    hue_win = np.clip((rg_frac - 0.18) / 0.15, 0.0, 1.0) * np.clip((1.16 - rg_frac) / 0.14, 0.0, 1.0)

    has_person_mask = bool(np.any(person_union > 0.15))
    fg_context = person_union.copy()
    if depth is not None:
        d = np.asarray(depth, dtype=np.float32)
        valid_d = d > 0
        if np.any(valid_d):
            z_lo, z_bg = np.percentile(d[valid_d], [5.0, 82.0])
            # Elevated foreground hand/product context when no person mask exists, or around hand-worn accessories.
            if z_bg > 1.35 * max(float(z_lo), 1e-4) and np.any(exclude_union > 0.25):
                fg_elev = np.clip((0.86 * z_bg - d) / max(0.25 * z_bg, 0.2), 0.0, 1.0)
                if has_person_mask:
                    if np.any(hand_accessory_mask):
                        r_px = max(12, int(round(0.30 * min(h, w))))
                        near_hand_acc = binary_dilation(hand_accessory_mask, iterations=r_px).astype(np.float32)
                        fg_context = np.maximum(fg_context, fg_elev * near_hand_acc)
                else:
                    fg_context = np.maximum(fg_context, fg_elev)

    p2_raw = chrom_skin * hue_win * eff_tex_conf * feature_keep * non_metal_conf * np.clip(1.0 - exclude_union, 0.0, 1.0) * fg_context
    raw_skin = np.maximum(p1_conf, p2_raw).astype(np.float32)
    # Close tiny 1-2px interior skin crease/stubble dropouts while strictly preserving metal/accessory exclusions
    raw_skin = grey_closing(raw_skin, size=(5, 5)) * non_metal_conf * np.clip(1.0 - exclude_union, 0.0, 1.0)

    # Feather edges gently, then enforce confidence floor so non-skin pixels stay exact 0.0
    # Feather edges gently, ramp smoothly from min_skin_confidence to avoid step edges, and drop tiny noise specks.
    import cv2
    feathered = gaussian_filter(raw_skin, sigma=2.0, mode="reflect")
    if float(min_skin_confidence) > 0.0:
        mc = float(min_skin_confidence)
        ramp = np.clip((feathered - mc) / 0.12, 0.0, 1.0)
        feathered = np.where(feathered >= mc, feathered * ramp, 0.0)
    nz_u8 = (feathered > 0.02).astype(np.uint8)
    n_cc, cc_lbl, cc_stats, _ = cv2.connectedComponentsWithStats(nz_u8, connectivity=8)
    min_cc_area = max(32, int(0.0005 * h * w))
    if n_cc > 1:
        keep_cc = np.zeros((h, w), dtype=bool)
        for idx_c in range(1, n_cc):
            if int(cc_stats[idx_c, cv2.CC_STAT_AREA]) >= min_cc_area:
                keep_cc |= (cc_lbl == idx_c)
        feathered = np.where(keep_cc, feathered, 0.0)
    return np.clip(feathered, 0.0, 1.0).astype(np.float32)


def get_skin_weight_for_scene(scene: SceneBuffers, settings: object | None = None) -> np.ndarray | None:
    """Resolve `skin_w` when `settings.skin_fidelity.enabled` is True; returns None otherwise."""
    sf = getattr(settings, "skin_fidelity", None) if settings is not None else None
    if sf is None or not bool(getattr(sf, "enabled", False)):
        return None
    skin_w = compute_skin_likelihood_map(
        scene.rgb_srgb,
        semantic_masks=scene.semantic_masks,
        depth=scene.depth,
        material_maps={"metalness": scene.metalness} if scene.metalness is not None else None,
        min_skin_confidence=float(getattr(sf, "min_skin_confidence", 0.35)),
    )
    if not np.any(skin_w > 0.0):
        return None
    return skin_w


def refine_source_shading_with_albedo(
    scene: SceneBuffers,
    source_shading: np.ndarray | None = None,
) -> np.ndarray | None:
    """Reconcile smooth estimated source_shading with albedo-guided `src / albedo` to recover sharp gobo/blind shadows.

    Uses `log_luminance(albedo)` as structural guide so sharp painted/textural albedo borders do not blur,
    while sharp shadow bars across uniform walls/floors (e.g. window blinds) are faithfully preserved in shading.
    """
    if scene.albedo is None:
        return source_shading
    from scipy.ndimage import gaussian_filter

    eps = 1e-4
    src_lin = srgb_to_linear(scene.rgb_srgb)
    alb_lin = np.maximum(srgb_to_linear(scene.albedo), 0.04)
    implied = np.clip(src_lin / alb_lin, 0.08, 3.0).astype(np.float32)

    guide = np.log(np.maximum(luminance(alb_lin), eps)).astype(np.float32)
    sigma = 6.0
    eps_guide = 0.008
    mean_i = gaussian_filter(guide, sigma, mode="reflect")
    mean_ii = gaussian_filter(guide * guide, sigma, mode="reflect")
    var_i = np.maximum(mean_ii - mean_i * mean_i, 0.0)

    def _gf2d(p: np.ndarray) -> np.ndarray:
        mean_p = gaussian_filter(p, sigma, mode="reflect")
        mean_ip = gaussian_filter(guide * p, sigma, mode="reflect")
        cov_ip = mean_ip - mean_i * mean_p
        a = cov_ip / (var_i + eps_guide)
        b = mean_p - a * mean_i
        return gaussian_filter(a, sigma, mode="reflect") * guide + gaussian_filter(b, sigma, mode="reflect")

    imp_refined = np.stack([_gf2d(implied[..., c]) for c in range(3)], axis=-1)
    imp_refined = np.clip(imp_refined, 0.08, 3.0).astype(np.float32)

    # Suppress blending where albedo itself has strong texture edges (let base source_shading govern there)
    # and weight higher on smooth albedo walls/surfaces where gobo/blind shadows exist.
    alb_tex = np.sqrt(var_i)
    flat_alb_w = np.clip((0.18 - alb_tex) / 0.14, 0.15, 0.85).astype(np.float32)[..., None]

    sh_in = source_shading if source_shading is not None else scene.source_shading
    if sh_in is None:
        return imp_refined
    sh_arr = np.asarray(sh_in, dtype=np.float32)
    if sh_arr.ndim == 2:
        sh_arr = np.repeat(sh_arr[..., None], 3, axis=-1)
    active = (scene.depth > 0) & (scene.lightable_mask > 0.05)
    if not np.any(active):
        active = scene.depth > 0
    sh_med = max(float(np.median(luminance(sh_arr)[active])), eps)
    imp_med = max(float(np.median(luminance(imp_refined)[active])), eps)
    imp_scaled = np.clip(imp_refined * (sh_med / imp_med), 0.05, 3.0).astype(np.float32)
    return np.clip((1.0 - flat_alb_w) * sh_arr + flat_alb_w * imp_scaled, 0.02, 3.0).astype(np.float32)
