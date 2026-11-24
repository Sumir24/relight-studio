# Core v1 status — superseded by the universal scene-pipeline update

The deterministic renderer remains the shared lighting core. The earlier scene adapter was landscape-specific and is no longer the active scene-understanding design.

The current product architecture is documented in `ENGINE_V1_PLAN.md` and `SCENE_PIPELINE_STATUS.md`.

Key correction: all semantic labels are dynamic; no landscape-specific class list or hardcoded sky/water protection is used. All image types use the same core scene-buffer and lighting contracts.
