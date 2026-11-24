# Relight Studio benchmarks

The domain list in `domain_matrix.json` is a QA coverage plan, not a renderer taxonomy. Case labels passed to benchmark scripts are for reporting only and never switch rendering behaviour.

## Available tools

- `../scripts/benchmark_image_inputs.py`: input decoding and color/alpha preflight for arbitrary image files.
- `run_multidomain_scene_benchmark.py`: runs the same neural scene-analysis CLI on a list of images and collects scene manifests plus structural QA reports.
- `../scripts/validate_scene_buffers.py`: audits the saved geometry and mask buffers for structural and degeneracy problems.

## Interpretation

Input preflight is not scene understanding. Structural scene QA is not ground-truth depth/mask accuracy. Neither is a perceptual relighting score. Real model inference, annotated geometry/masks and human visual evaluation remain separate validation stages.
