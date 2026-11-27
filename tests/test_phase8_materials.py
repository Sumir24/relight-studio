from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from relight_engine.adapters.diffusers_intrinsics import DiffusersIntrinsicAdapter
from relight_engine.calibration import (
    MaterialAssignment,
    assign_estimated_material_priors,
    build_material_maps,
)
from relight_engine.contracts import CameraModel, EngineSettings, EnvironmentLight, Light, LightingRig, SceneBuffers
from relight_engine.engine import relight
from relight_engine.lighting import evaluate_environment_lighting, evaluate_rig
from relight_engine.preview import downsample_scene_buffers_for_preview
from relight_engine.workflows import apply_user_material_assignments, target_rig_from_dict


def _make_scene(h: int = 24, w: int = 32, *, roughness=None, metalness=None, material_confidence=None, anisotropy=None, anisotropy_angle=None) -> tuple[SceneBuffers, CameraModel]:
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    scene = SceneBuffers(
        rgb_srgb=np.full((h, w, 3), 0.45, dtype=np.float32),
        depth=np.full((h, w), 2.5, dtype=np.float32),
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
        roughness=roughness,
        metalness=metalness,
        material_confidence=material_confidence,
        anisotropy=anisotropy,
        anisotropy_angle=anisotropy_angle,
        albedo=np.full((h, w, 3), 0.50, dtype=np.float32),
        source_shading=np.ones((h, w, 3), dtype=np.float32) * 0.5,
    )
    camera = CameraModel(fx=60.0, fy=60.0, cx=w / 2.0, cy=h / 2.0)
    return scene, camera


def test_confidence_aware_material_assignment_gates_uncertain_dielectrics_and_preserves_user_overrides():
    h, w = 16, 16
    rough = np.full((h, w), 0.30, dtype=np.float32)
    metal = np.full((h, w), 0.55, dtype=np.float32)  # typical InteriorVerse dielectric bias (below 0.68)
    metal[8:, 8:] = 0.90  # high-confidence metal patch
    conf = np.full((h, w), 0.70, dtype=np.float32)
    conf[:8, :8] = 0.10  # highly uncertain corner -> falls back toward prior (rough=0.75, metal=0.0)

    mask_user = np.zeros((h, w), dtype=np.float32)
    mask_user[:8, 8:] = 1.0
    user = build_material_maps((h, w), [MaterialAssignment("lacquer", mask_user, roughness=0.12, metalness=0.0, specular_level=0.06)])

    assigned = assign_estimated_material_priors(
        roughness=rough,
        metalness=metal,
        albedo_srgb=np.full((h, w, 3), 0.75, dtype=np.float32),
        material_confidence=conf,
        user_overrides=user,
    )
    # Uncertain dielectric region falls back close to conservative roughness 0.75 and metalness 0.0
    assert float(assigned.roughness[2, 2]) > 0.65
    assert float(assigned.metalness[2, 2]) == pytest.approx(0.0)
    # Mid-range raw metalness 0.55 on confident dielectric stays 0.0 (no chrome foliage/cloth)
    assert float(assigned.metalness[10, 2]) == pytest.approx(0.0)
    # Strong conductor prediction stays clearly metallic (tightened [0.80, 0.94] gate)
    assert float(assigned.metalness[10, 10]) > 0.50
    # Smooth non-metallic zone receives independent dielectric clearcoat response
    assert assigned.clearcoat is not None and float(assigned.clearcoat[10, 2]) > 0.10
    assert float(assigned.clearcoat[10, 10]) < 0.01
    # Raw predictions retained separately from calibrated values
    assert assigned.metadata["raw_model_predictions"]["metalness"]["mean"] > float(np.mean(assigned.metalness))
    # User override wins cleanly inside its mask without wiping unmasked predictions
    assert float(assigned.roughness[2, 10]) == pytest.approx(0.12, abs=1e-4)
    assert float(assigned.roughness[10, 10]) == pytest.approx(0.75 * 0.30 + 0.30 * 0.70, abs=1e-4)
    assert assigned.metadata["metalness_calibration"]["brightness_heuristic_used"] is False
    assert assigned.metadata["metalness_calibration"]["category_rules_used"] is False


def test_phase81_glossy_dielectric_vs_conductor_texture_floor_and_hdri_balance():
    h, w = 24, 24
    # Noisy textured surface where Marigold predicted low roughness 0.30 vs smooth steel at 0.30
    rng = np.random.default_rng(7)
    src = np.full((h, w, 3), 0.35, dtype=np.float32)
    src[:, :12, :] = np.clip(0.35 + rng.normal(0.0, 0.09, size=(h, 12, 3)), 0.05, 0.95).astype(np.float32)
    rough = np.full((h, w), 0.35, dtype=np.float32)
    # Left half: high metal unc / higher rough-like dielectric; Right half: smooth confident metal
    metal = np.full((h, w), 0.91, dtype=np.float32)
    m_unc = np.full((h, w), 0.005, dtype=np.float32)
    m_unc[:, :12] = 0.06  # uncertain glossy dark dielectric (e.g. leather / dial)
    conf = np.full((h, w), 0.80, dtype=np.float32)
    assigned = assign_estimated_material_priors(
        roughness=rough,
        metalness=metal,
        albedo_srgb=np.full((h, w, 3), 0.40, dtype=np.float32),
        source_rgb_srgb=src,
        material_confidence=conf,
        metalness_uncertainty=m_unc,
    )
    # Textured left half has elevated roughness floor & suppressed false metalness
    assert float(np.mean(assigned.roughness[4:-4, 2:10])) > float(np.mean(assigned.roughness[4:-4, 14:-2])) + 0.10
    assert float(np.mean(assigned.metalness[:, :12])) < 0.12
    assert float(np.mean(assigned.metalness[:, 12:])) > 0.50

    # HDRI diffuse non-double-counting & independent diffuse/specular controls
    from relight_engine.geometry import backproject_depth
    sc, cam = _make_scene(h, w, roughness=np.full((h, w), 0.75, np.float32), metalness=np.zeros((h, w), np.float32))
    pts = backproject_depth(sc.depth, cam)
    hdri = np.full((16, 32, 3), 0.25, dtype=np.float32)
    rig_amb_only = LightingRig(ambient_strength=0.25, lights=())
    rig_spec_only_hdri = LightingRig(
        ambient_strength=0.25,
        lights=(),
        environment=EnvironmentLight(env_map=hdri, env_map_strength=1.0, diffuse_strength=0.0, specular_strength=1.0),
    )
    d_amb, s_amb, _ = evaluate_rig(sc, cam, pts, sc.normals, rig_amb_only, EngineSettings())
    d_hdr, s_hdr, _ = evaluate_rig(sc, cam, pts, sc.normals, rig_spec_only_hdri, EngineSettings())
    # Disabling environment diffuse_strength keeps diffuse shadow floor identical to analytic ambient_strength
    assert np.allclose(d_amb, d_hdr, atol=1e-5)
    assert float(np.mean(s_hdr)) > float(np.mean(s_amb))


def test_apply_user_material_assignments_preserves_underlying_model_priors():
    h, w = 8, 8
    rough_prior = np.full((h, w), 0.28, dtype=np.float32)
    metal_prior = np.full((h, w), 0.85, dtype=np.float32)
    conf_prior = np.full((h, w), 0.65, dtype=np.float32)
    scene, _ = _make_scene(h, w, roughness=rough_prior, metalness=metal_prior, material_confidence=conf_prior)
    scene.metadata["material_maps_estimated"] = True
    scene.material_metadata = {"roughness": {"source": "model_estimated"}}

    patch = np.zeros((h, w), dtype=np.float32)
    patch[:4, :4] = 1.0
    meta = apply_user_material_assignments(
        scene,
        {"mask_strap": patch},
        [{"name": "strap", "mask_key": "strap", "roughness": 0.90, "metalness": 0.0}],
    )
    assert float(scene.roughness[1, 1]) == pytest.approx(0.90, abs=1e-5)
    assert float(scene.metalness[1, 1]) == pytest.approx(0.0, abs=1e-5)
    # Unmasked region retains model priors instead of being flattened to 0.75/0.0
    assert float(scene.roughness[6, 6]) == pytest.approx(0.28, abs=1e-5)
    assert float(scene.metalness[6, 6]) == pytest.approx(0.85, abs=1e-5)
    assert meta["neural_material_estimation_used"] is True


def test_diffusers_appearance_adapter_extracts_ensemble_uncertainty_and_sanity_gates(monkeypatch):
    from contextlib import nullcontext
    from types import ModuleType, SimpleNamespace

    fake_diffusers = ModuleType("diffusers")

    class _UncPipe:
        target_properties = {
            "target_names": ["albedo", "material"],
            "albedo": {"prediction_space": "srgb"},
        }

        @classmethod
        def from_pretrained(cls, *_a, **_kw):
            return cls()

        def set_progress_bar_config(self, **_kw):
            return None

        def to(self, _d):
            return self

        def __call__(self, *_a, **kwargs):
            assert kwargs.get("output_uncertainty") is True
            albedo = np.full((8, 10, 3), 0.65, dtype=np.float32)
            mat = np.zeros((8, 10, 3), dtype=np.float32)
            mat[..., 0] = 0.40
            mat[..., 1] = 0.85
            unc = np.zeros((2, 8, 10, 3), dtype=np.float32)
            unc[1, :4, :, 0] = 0.005  # top half low uncertainty
            unc[1, 4:, :, 0] = 0.250  # bottom half high ensemble disagreement
            unc[1, 4:, :, 1] = 0.200
            return SimpleNamespace(prediction=np.stack([albedo, mat], axis=0), uncertainty=unc)

    fake_diffusers.MarigoldIntrinsicsPipeline = _UncPipe
    fake_torch = ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    fake_torch.inference_mode = lambda: nullcontext()
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    rgb = np.full((8, 10, 3), 0.60, dtype=np.float32)
    adapter = DiffusersIntrinsicAdapter(variant_kind="appearance", device="cpu", ensemble_size=3)
    pred = adapter.estimate(rgb)
    assert pred.material_confidence is not None
    assert float(pred.material_confidence[1, 1]) > 0.55
    assert float(pred.material_confidence[6, 1]) < 0.15
    assert pred.metadata["material_target_semantics"]["uncertainty_source"] == "ensemble_dispersion"
    assert pred.specular_color is not None and pred.specular_level is not None


def test_metalness_diffuse_conservation_and_anisotropic_ggx():
    h, w = 16, 16
    from relight_engine.geometry import backproject_depth

    cam = CameraModel(60.0, 60.0, w / 2.0, h / 2.0)
    sc_die, _ = _make_scene(h, w, roughness=np.full((h, w), 0.3, np.float32), metalness=np.zeros((h, w), np.float32))
    sc_met, _ = _make_scene(h, w, roughness=np.full((h, w), 0.3, np.float32), metalness=np.ones((h, w), np.float32))
    pts = backproject_depth(sc_die.depth, cam)
    rig = LightingRig(ambient_strength=0.2, lights=(Light("k", "directional", direction=(0.4, 0.2, -0.9), strength=1.2, casts_shadows=False),))
    d_die, s_die, _ = evaluate_rig(sc_die, cam, pts, sc_die.normals, rig, EngineSettings())
    d_met, s_met, _ = evaluate_rig(sc_met, cam, pts, sc_met.normals, rig, EngineSettings())
    assert float(np.mean(d_met)) < float(np.mean(d_die)) * 0.20
    assert float(np.mean(s_met)) > float(np.mean(s_die)) * 5.0

    # Anisotropic GGX at 0 angle vs pi/2 shifts response orthogonally around off-axis half vector
    sc_an0, _ = _make_scene(
        h, w,
        roughness=np.full((h, w), 0.35, np.float32),
        metalness=np.ones((h, w), np.float32),
        anisotropy=np.full((h, w), 0.85, np.float32),
        anisotropy_angle=np.zeros((h, w), np.float32),
    )
    sc_an90, _ = _make_scene(
        h, w,
        roughness=np.full((h, w), 0.35, np.float32),
        metalness=np.ones((h, w), np.float32),
        anisotropy=np.full((h, w), 0.85, np.float32),
        anisotropy_angle=np.full((h, w), np.pi / 2.0, np.float32),
    )
    rig_x = LightingRig(ambient_strength=0.0, lights=(Light("xkey", "directional", direction=(0.65, 0.0, -0.76), strength=1.5, casts_shadows=False),))
    _, s_an0, _ = evaluate_rig(sc_an0, cam, pts, sc_an0.normals, rig_x, EngineSettings())
    _, s_an90, _ = evaluate_rig(sc_an90, cam, pts, sc_an90.normals, rig_x, EngineSettings())
    assert not np.allclose(s_an0, s_an90, atol=1e-3)

    # Preview downsampling preserves anisotropy fields cleanly
    ds_scene, _ = downsample_scene_buffers_for_preview(sc_an0, cam, 8)
    ds_scene.validate()
    assert ds_scene.anisotropy is not None and max(ds_scene.shape) == 8


def test_equirectangular_hdri_ibl_roughness_and_rotation(tmp_path):
    # Create an equirectangular HDRI with a bright warm hotspot at front-right (u ~ 0.65, v ~ 0.35)
    he, we = 32, 64
    yy, xx = np.mgrid[0:he, 0:we].astype(np.float32)
    hdri = np.full((he, we, 3), 0.08, dtype=np.float32)
    blob = np.exp(-((xx - 0.65 * we) ** 2 / 8.0 + (yy - 0.35 * he) ** 2 / 6.0))
    hdri += blob[..., None] * np.array([8.0, 4.0, 1.0], dtype=np.float32)[None, None, :]

    env_a = EnvironmentLight(env_map=hdri, env_map_strength=1.0, rotation_deg=0.0)
    env_b = EnvironmentLight(env_map=hdri, env_map_strength=1.0, rotation_deg=180.0)
    env_a.validate()
    env_b.validate()

    h, w = 16, 16
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 0] = 0.45
    normals[..., 1] = 0.25
    normals[..., 2] = -0.85
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    view = np.broadcast_to(np.array([0.0, 0.0, -1.0], dtype=np.float32), (h, w, 3))

    ed_smooth, es_smooth = evaluate_environment_lighting(
        normals, view,
        np.full((h, w), 0.05, np.float32),
        np.ones((h, w), np.float32),
        np.full((h, w), 0.04, np.float32),
        np.full((h, w, 3), 0.90, np.float32),
        env_a,
    )
    ed_rough, es_rough = evaluate_environment_lighting(
        normals, view,
        np.full((h, w), 0.90, np.float32),
        np.ones((h, w), np.float32),
        np.full((h, w), 0.04, np.float32),
        np.full((h, w, 3), 0.90, np.float32),
        env_a,
    )
    _, es_rot = evaluate_environment_lighting(
        normals, view,
        np.full((h, w), 0.05, np.float32),
        np.ones((h, w), np.float32),
        np.full((h, w), 0.04, np.float32),
        np.full((h, w, 3), 0.90, np.float32),
        env_b,
    )
    # Smooth metal captures intense HDRI peak; rough surface diffuses it out; 180-deg rotation moves peak away
    assert float(np.mean(es_smooth)) > float(np.mean(es_rough)) * 1.3
    assert float(np.mean(es_smooth)) > float(np.mean(es_rot)) * 1.5
    # Metal has suppressed diffuse irradiance compared to dielectric under identical HDRI
    ed_diel, _ = evaluate_environment_lighting(
        normals, view,
        np.full((h, w), 0.50, np.float32),
        np.zeros((h, w), np.float32),
        np.full((h, w), 0.04, np.float32),
        np.full((h, w, 3), 0.65, np.float32),
        env_a,
    )
    assert float(np.mean(ed_diel)) > float(np.mean(ed_smooth)) * 5.0

    # Check loading env_map via .npy file in workflows.target_rig_from_dict
    npy_path = tmp_path / "studio.npy"
    np.save(npy_path, hdri)
    rig = target_rig_from_dict({
        "ambient_strength": 0.0,
        "lights": [],
        "environment": {"env_map_path": str(npy_path), "env_map_strength": 0.8, "rotation_deg": 15.0},
    })
    assert rig.environment is not None and rig.environment.env_map is not None
    scene, cam = _make_scene(h, w, roughness=np.full((h, w), 0.25, np.float32), metalness=np.ones((h, w), np.float32))
    res = relight(scene, cam, rig, rig, EngineSettings())
    assert "environment_ibl" in res.metrics["active_specular_path"]
