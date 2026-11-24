# Relight Studio — Material Response Roadmap

## Why this is separate from scene categories

A scene category describes *where an object appears* (portrait, food, interior, vehicle). A material family describes *how a surface responds to light* (diffuse dielectric, rough dielectric, conductor, translucent, transparent, emissive, layered/subsurface). The renderer must not use a different physical engine for each scene category. It should use reusable surface-response models selected from local material evidence and user correction.

## Current baseline

The current shared renderer provides:

- Lambertian diffuse response.
- GGX microfacet specular response using roughness, metalness and dielectric specular-level maps.
- Directional, point and sampled rectangular-area target lights.
- Metric inverse-square falloff only when metric depth is provided; relative depth uses explicitly non-metric normalized falloff.
- Approximate screen-space 2.5D visibility and source-preserving image reconstruction.

The current implementation does **not** infer the correct material maps from a photograph. Material maps can be user-authored or supplied by a future validated estimator. Missing material information must remain marked as unknown or use a clearly declared conservative prior.

## Reusable material families to validate

1. **Diffuse dielectric** — paper, plaster, skin diffuse component, matte plastics, unglazed ceramics.
2. **Rough dielectric** — fabric, wood finishes, leather, painted walls, matte coatings.
3. **Conductor / metallic** — bare metal, chrome, metallic packaging and vehicle trim; needs coloured conductor Fresnel rather than the current scalar metalness approximation.
4. **Layered surface** — varnished wood, coated packaging, painted metal and clear-coated products; needs base and coat lobes.
5. **Subsurface / translucent** — skin, wax, leaves, food flesh and thin materials; requires a controlled approximation and confidence estimate.
6. **Transparent / refractive** — glass, liquids and clear plastics; needs transmission, background-aware refraction and proper object/background separation.
7. **Emissive** — screens, lamps, LEDs and bright signs; emitted radiance must not be treated as ordinary diffuse reflectance.
8. **Reflective environment-dependent** — mirrors, polished objects and glossy scenes; needs environment information or a clearly limited 2.5D approximation.

## Universal selection rule

- Use open-vocabulary object/region labels to locate regions; never equate a semantic name with a material.
- Estimate material properties at the pixel/region level with explicit confidence and provenance.
- Allow user correction and per-instance overrides.
- Select a reusable response family from material evidence, not from scene category.
- Fall back conservatively when evidence is weak; report the limitation.

## Validation requirements

For every material family, test multiple objects from at least three scene categories. Measure direction response, highlight width/energy, texture retention, edge leakage, shadow behaviour, no-op invariance and stability under small mask/depth errors. Do not promote a family to production based only on a synthetic plane or one photograph.

## Non-goals for the next milestone

Do not implement full physically based path tracing or attempt to reconstruct unseen geometry before the shared single-image pipeline is reliable. Prioritize calibrated visible-surface response and an honest quality gate.
