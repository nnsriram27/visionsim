# Task 2 Report: Texel sim assembly — bpy atlas build + `_combine` TEXEL mode

## Status: DONE

Commit: `f89be73` — `feat(heatsim): TEXEL sim domain — texel point clouds with per-face materials and per-texel albedo/irradiance`

## Files

- `visionsim/simulate/heatsim/adapter.py` — `AtlasPlan`, `build_atlas_plan`, `_extract_face_uv_and_slots`,
  `_prepare_bake_uv`, `_write_atlas_uv_layer`, `_sample_bilinear`, `_texel_albedo`,
  `_compute_texel_irradiance`, `_combine_texel_object`, `_combine(..., atlas_plan=None)` TEXEL branch,
  `solve_scene(..., atlas_plan=None)` cache-key `"atlas"` field + texel-flux wiring. `layout` tuples
  widened to `(name, offset, n, kind)`; `_split_history` / `solve_scene_animated` updated to match
  (output unchanged, confirmed via the unmodified regression suites).
- `visionsim/simulate/heatsim/materials.py` — `resolve_face_materials` (per-face slot lookup, no
  averaging; reuses `_slot_tables`).
- `visionsim/simulate/heatsim/irradiance_kernel.py` (vendored, excluded from ruff/mypy) —
  `_sky_irradiance_plain` extracted from `compute_per_vertex_irradiance`'s per-object sky branch
  (used in place of its two identical inline computations); new `compute_irradiance_at_points`
  entry point for arbitrary point clouds.
- `visionsim/simulate/heatsim/constants.py` — added `ATLAS_UV_LAYER_NAME = "HeatSim_Atlas_UV"`.
- `tests/test_heatsim_texel_mode.py` (new, 12 tests: the 6 named in the brief + 5 supporting
  edge-case tests + 1 real-bpy end-to-end smoke test).

## Environment verification

```
$ PYTHONPATH=<worktree> .venv/bin/python -c "import visionsim; print(visionsim.__file__)"
/home/sriram/research/visionsim/.claude/worktrees/thermal-atlas/visionsim/__init__.py
```

## TDD evidence

Given the size of this task, tests and implementation for each function were written and iterated
together rather than in one big RED pass; the first real run of the fake-bpy suite genuinely failed
(`TypeError: unhashable type: 'types.SimpleNamespace'` — my own test fixture used a bare
`SimpleNamespace` as a `flux_by_obj` dict key, which `_combine` legitimately requires to be
hashable like a real `bpy.types.Object`) and was fixed by adding a small hashable `_NS` stand-in,
not by loosening the implementation.

### GREEN (fake-bpy suite, no Blender)

```
$ PYTHONPATH=<worktree> LD_PRELOAD=... .venv/bin/python -m pytest tests/test_heatsim_texel_mode.py -v
test_resolve_face_materials_exact_per_face PASSED
test_resolve_face_materials_returns_none_without_slots PASSED
test_resolve_face_materials_out_of_range_index_is_clamped PASSED
test_combine_texel_mode_mixes_texel_and_vertex_objects PASSED
test_combine_texel_mode_object_level_materials_when_no_assignment PASSED
test_combine_texel_dirichlet_pinning_per_texel PASSED
test_combine_vertex_mode_unchanged_when_no_plan PASSED
test_uv_failure_demotes_to_vertex_path_with_warning PASSED
test_build_atlas_plan_topology_mismatch_demotes_with_warning PASSED
test_build_atlas_plan_excludes_dense_objects PASSED
test_cache_key_gains_atlas_digest PASSED
======= 11 passed =======
```

### Mutation checks (proves the tests bite, not just pass)

1. `materials.resolve_face_materials`: mutated `slots` to always index slot 0 (defeating the
   per-face lookup). Result: `test_resolve_face_materials_exact_per_face` and
   `test_resolve_face_materials_out_of_range_index_is_clamped` both failed with a concrete wrong
   value (`0.082` wood instead of `4.2` steel). Reverted; `git diff --stat` on `materials.py`
   showed only the intended new function afterward.
2. `adapter._combine_texel_object`: mutated the Dirichlet-pinning branch to skip zeroing `alpha`
   on pinned texels. Result: `test_combine_texel_dirichlet_pinning_per_texel` failed
   (`assert np.allclose(combined.alpha[pinned], 0.0)` → got `[4.2, 4.2]`). Reverted; `adapter.py`
   diff afterward was exactly my implementation (485 insertions / 11 deletions, matched against the
   pre-mutation backup).

### Real-bpy end-to-end smoke test (not one of the 6 named tests, added for extra confidence)

`test_texel_pipeline_solves_end_to_end_under_bpy` drives `build_atlas_plan` → `solve_scene(...,
atlas_plan=...)` on a real 4-vertex Blender plane under Cycles/light, and asserts: the plane joined
the atlas (texel count > vertex count), `HeatSim_Atlas_UV` was written to `mesh.uv_layers`, the
solve returns a `(timesteps, n_texels)` history, and every temperature is finite and physical
(200 K < T < 2000 K). This is the only check that exercises the real `bpy` UV-layer / `foreach_get`
plumbing (`_extract_face_uv_and_slots`, `_write_atlas_uv_layer`) that the fake-bpy tests
necessarily stub out.

## Regression suite (existing files, unmodified — verified via empty `git diff`)

```
$ .venv/bin/python -m pytest tests/test_heatsim_adapter.py tests/test_heatsim_assignments_integration.py \
    tests/test_heatsim_materials.py tests/test_heatsim_irradiance.py tests/test_heatsim_texel_mode.py \
    --executable=/home/sriram/softwares/blender-5.1.0-linux-x64/blender -v
======= 69 passed in 26.24s =======
```
Also ran (not required, but touched by the `layout` tuple widening / `_split_history` change):
```
$ .venv/bin/python -m pytest tests/test_heatsim_animated.py tests/test_heatsim_atlas.py \
    tests/test_heatsim_cache.py tests/test_heatsim_config.py tests/test_heatsim_dispatch.py \
    tests/test_heatsim_properties.py tests/test_heatsim_shader.py tests/test_heatsim_solver.py \
    --executable=... -v
======= 35 passed =======
```
`git diff` on the four brief-mandated files (`test_heatsim_adapter.py`,
`test_heatsim_assignments_integration.py`, `test_heatsim_materials.py`, `test_heatsim_irradiance.py`)
is empty.

## Lint / type-check

```
$ .venv/bin/ruff check visionsim/simulate/heatsim/adapter.py visionsim/simulate/heatsim/materials.py \
    visionsim/simulate/heatsim/constants.py tests/test_heatsim_texel_mode.py
All checks passed!

$ LD_PRELOAD=... .venv/bin/python -m mypy visionsim/simulate/heatsim/materials.py
Success: no issues found in 1 source file
```
(`adapter.py` is ruff-only per the brief; `irradiance_kernel.py`/`constants.py` are vendored,
excluded from both — confirmed against `pyproject.toml`'s exclude lists.)

## Design decisions not fully pinned down by the brief

- **`layout` tuple widened to 4-tuple** (`name, offset, n, kind`) rather than adding a parallel
  `kinds` list, since the brief explicitly calls for "layout entries gain a kind tag." This forced
  updating three internal unpacking sites (`_split_history`, `solve_scene`'s meta dict,
  `solve_scene_animated`'s two loops) — all mechanical (`for name, off, n, _kind in ...`), verified
  behavior-preserving by the still-green `test_heatsim_animated.py` (not in the mandated regression
  set, run anyway since it's the only suite that exercises `combined.layout` end-to-end).
- **`solve_scene` gained `atlas_plan: Optional[AtlasPlan] = None`** and wires
  `_compute_texel_irradiance` + the cache-key `"atlas"` field. The brief's "Produces" list only
  names `_combine`'s TEXEL branch and the cache-key *shape*, not explicitly `solve_scene`'s
  signature — but the cache-key test (`test_cache_key_gains_atlas_digest`) can only be written
  against something that builds `key_cfg`, and `_combine` alone doesn't touch the cache. I judged
  this the natural, low-risk home (guarded entirely behind `atlas_plan is not None`, default
  reproduces today's key/behavior exactly) rather than leaving it undone or inventing a separate
  function Task 3 would have to discover and wire itself.
- **`_compute_texel_irradiance` still runs `_compute_irradiance` (the per-vertex path)
  unconditionally on TEXEL objects too**, then overwrites their entry in `flux_by_obj` with the
  correctly-sized per-texel result. Simpler and lower-risk than threading an exclusion list through
  `_compute_irradiance`, and cheap in practice: objects that join the atlas do so *because* they
  have few vertices (that's the selection criterion), so the discarded per-vertex pass is
  inexpensive by construction.
- **Per-texel UV for albedo sampling** (`tex["uv"]`) is derived from `rasterize_tile`'s `xy` +
  tile size (`(xy+0.5)/tile_size`), which is mathematically identical to the object's own raw bake
  UV at the texel center (since `rasterize_tile`'s `loop_uv` input for atlas objects *is* the
  object's raw `HeatSim_Bake_UV` layer, unscaled — one tile per object, no rescale into tile-local
  space needed pre-atlas-placement). Verified this reasoning against `irradiance.bake_albedo_map`'s
  own vertex-reduction code path, which samples the identical image with the identical raw UV.
- **`_write_atlas_uv_layer`** (remap into the shared atlas image, write `HeatSim_Atlas_UV`) is
  implemented per spec §4.1 and exercised by the real-bpy end-to-end test (layer exists after the
  solve), but its exact placement *values* aren't asserted — the shader that consumes this layer is
  out of scope for Task 2 (§4.6, a later task). It's wrapped in try/except and never blocks the
  solve on failure.
- **Atlas config defaults** (`atlas_texel_density=50.0` etc.) inside `build_atlas_plan` are
  placeholders for when `cfg` doesn't supply a key — real defaults come from `ThermalConfig`
  (Task 3/config plumbing, out of scope here; brief's file list for Task 2 doesn't include
  `config.py`).

## Concerns

None blocking. One thing worth flagging for whoever wires Task 3 (config/service plumbing):

1. `_augment_interior_points` (POINTS-domain interior volume sampling, off by default) silently
   skips TEXEL objects — it keys off `geom_by_obj`, which `_combine`'s TEXEL branch never
   populates (texel objects don't go through `_extract_geometry` in `_combine` at all; that
   extraction already happened once, inside `build_atlas_plan`). This is a graceful no-op, not a
   crash, and matches the existing "known gap" pattern already documented on that function
   (per-vertex materials aren't reflected in interior points either) — but it means
   `interior_point_ratio > 0` currently has zero effect for atlas-participating objects. Not
   required by any named test; flagging for visibility rather than fixing speculatively.

## Self-review

Read the full diff top to bottom after implementation; ran the two mutation checks above; confirmed
`git diff` on the four mandated test files is empty; confirmed the editable install points at the
worktree before any test run; ran the full regression set (mandated 4 suites + texel_mode) plus the
non-mandated but layout-touching suites (animated, atlas, cache, config, dispatch, properties,
shader, solver) all green with `--executable`.
