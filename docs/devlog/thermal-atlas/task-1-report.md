# Task 1 Report: `atlas.py` — selection, allocation, packing, rasterization

## Status: DONE

Commit: `b6e30fb` — `feat(heatsim): atlas allocation, packing and texel rasterization (pure numpy)`

## Files

- `visionsim/simulate/heatsim/atlas.py` (new, ~330 lines) — pure numpy + stdlib, no `bpy` import.
- `tests/test_heatsim_atlas.py` (new, 18 tests).

## Environment verification

```
$ PYTHONPATH=/home/sriram/research/visionsim/.claude/worktrees/thermal-atlas .venv/bin/python -c \
    "import visionsim, os; print(visionsim.__file__)"
/home/sriram/research/visionsim/.claude/worktrees/thermal-atlas/visionsim/__init__.py
```
Confirmed pointing at the worktree before running anything.

## TDD evidence

### RED (module missing)

Temporarily moved `atlas.py` aside, ran the test file:

```
$ PYTHONPATH=<worktree> LD_PRELOAD=... .venv/bin/python -m pytest tests/test_heatsim_atlas.py -v
ERROR collecting tests/test_heatsim_atlas.py
ImportError: cannot import name 'atlas' from 'visionsim.simulate.heatsim'
=========================== 1 error in 1.72s ===============================
```
Correct failure reason (import error, not a logic bug) — moved the file back and implemented.

### GREEN

```
$ PYTHONPATH=<worktree> LD_PRELOAD=... .venv/bin/python -m pytest tests/test_heatsim_atlas.py -v
...
tests/test_heatsim_atlas.py::test_surface_area_of_unit_quad PASSED
tests/test_heatsim_atlas.py::test_selection_excludes_dense_objects PASSED
tests/test_heatsim_atlas.py::test_selection_area_zero_uses_eps_guard PASSED
tests/test_heatsim_atlas.py::test_tile_size_scales_with_area_and_respects_min_max_and_multiple_of_4 PASSED
tests/test_heatsim_atlas.py::test_big_object_not_power_of_two_overshot PASSED
tests/test_heatsim_atlas.py::test_soft_max_rescales_uniformly_and_warns PASSED
tests/test_heatsim_atlas.py::test_soft_max_counts_retained_vertices PASSED
tests/test_heatsim_atlas.py::test_shelf_pack_no_overlap_and_padding PASSED
tests/test_heatsim_atlas.py::test_allocate_empty_areas PASSED
tests/test_heatsim_atlas.py::test_rasterize_full_cover_quad PASSED
tests/test_heatsim_atlas.py::test_rasterize_half_tile_triangle PASSED
tests/test_heatsim_atlas.py::test_rasterize_no_double_claim PASSED
tests/test_heatsim_atlas.py::test_rasterize_degenerate_triangle_skipped PASSED
tests/test_heatsim_atlas.py::test_rasterize_degenerate_triangle_mixed_with_valid_no_nan PASSED
tests/test_heatsim_atlas.py::test_dilate_fills_margin_and_preserves_valid PASSED
tests/test_heatsim_atlas.py::test_dilate_preserves_valid_and_is_noop_when_all_valid PASSED
tests/test_heatsim_atlas.py::test_dilate_zero_iterations_is_noop PASSED
tests/test_heatsim_atlas.py::test_rasterize_performance_sanity PASSED
============================== 18 passed in 1.50-1.60s ==============================
```
All 18 pass on first implementation attempt (one bug caught pre-test-run, see below).

Bug caught during self-check before considering this done: the half-open edge-test threshold
logic (`thresh - _EDGE_EPS` algebra) initially collapsed both the "top-left" and "not top-left"
branches to the same `w >= 0` condition, silently losing the half-open exclusivity needed for
`test_rasterize_no_double_claim`. Fixed by writing the two branches explicitly
(`w >= -eps` vs `w >= eps`) rather than via a single subtracted threshold. All rasterize tests
(including full-cover and no-double-claim, which exercise the shared diagonal edge) still passed
afterward, so this was caught and fixed before it could hide behind a passing-by-luck test.

### Lint + type-check

```
$ .venv/bin/ruff check visionsim/simulate/heatsim/atlas.py tests/test_heatsim_atlas.py
All checks passed!

$ LD_PRELOAD=... .venv/bin/python -m mypy visionsim/simulate/heatsim/atlas.py --ignore-missing-imports
Success: no issues found in 1 source file
```
(One `var-annotated` error fixed along the way: `covered: np.ndarray = np.zeros(...)`.)

## Design notes / decisions not fully pinned down by the brief

- **Soft-max rescale ratio**: computed as `(soft_max - retained_vertex_count) / tiles_total`
  (tiles-only budget), not `soft_max / total`, per the brief's "counts retained vertices" test
  name — retained vertices don't shrink with density, so excluding them from the numerator is
  what makes `test_soft_max_counts_retained_vertices` meaningfully different from a plain
  soft-max-exceeded check. It's a single corrective pass (not iterative), consistent with "warn
  loudly + rescale" rather than a solver — clamping can still leave the post-rescale total above
  `soft_max`; the brief calls this a "warn-only safety valve," not a guarantee.
- **Shelf-pack bin width**: not specified by the brief (only "shelf-pack sorted by height desc;
  atlas_size grows to fit"). Chose `width = round_up4(max(max_tile_side, ceil(sqrt(total_padded_area))))`
  — a squarish target width computed from the tile set, with height growing shelf-by-shelf. Tiles
  are square (side×side) by construction from `allocate`, so "sorted by height desc" reduces to
  "sorted by side desc."
- **Half-open edge rule**: standard top-left fill rule, normalizing each triangle to CCW in
  UV-texel space per-triangle before testing (an original-CW triangle's UV vertex order is
  swapped only for the edge test; 3D position/normal interpolation always uses the original
  `faces` index order, so no vertex-order bugs leak into geometry output).
- **Selection eps**: hardcoded `_AREA_EPS = 1e-9` (not a parameter — the brief's signature for
  `select_for_atlas` has no eps argument).

## Self-review against brief's edge cases (all exercised manually + via tests)

- **Empty `areas` dict**: `allocate({}, density)` → `atlas_size=(0,0)`, `tiles={}`,
  `rescaled=False`, `effective_density==density`. Covered by `test_allocate_empty_areas`.
- **area=0 object**: `_tile_side` clamps `K=max(area,0)*density` at 0, side rounds to `tile_min`.
  Verified manually: `allocate({"zero": 0.0}, 500.0).tiles["zero"].size == (16, 16)`.
- **tile_min > needed**: covered by `test_tile_size_scales_with_area_and_respects_min_max_and_multiple_of_4`
  (`"tiny"` case, K=15 clamped up to 16).
- **All-degenerate mesh**: verified manually — a single colinear-UV triangle rasterizes to empty
  `(0,2)`/`(0,3)` arrays, no exception, no NaN. Also covered by
  `test_rasterize_degenerate_triangle_skipped` and the mixed-with-valid variant.
- **Negative area (defensive)**: `max(area_m2, 0.0)` inside `_tile_side` prevents a negative K /
  NaN `sqrt`; manually verified it clamps to `tile_min`, not required by the brief but cheap
  insurance against a malformed caller.
- **Pathological `retained_vertex_count >= soft_max`** (retained alone exceeds the ceiling, so no
  rescale of tiles can possibly satisfy it): `ratio` would be non-positive; guarded by falling
  back to `_AREA_EPS` so `effective_density` stays a tiny positive number instead of zero/negative
  (which would break `sqrt` in `_tile_side`). Still warns, still sets `rescaled=True`. Manually
  verified no crash and a sane (tile_min-clamped) result.
- **Multi-channel `dilate`** (e.g. `(H,W,3)` for positions/normals, not just the required 2D
  temperature case): manually smoke-tested — center preserved, neighbor correctly filled with the
  full 3-vector mean, no NaNs. Not one of the brief's named tests but a real downstream use case
  (Task 2/adapter dilates position and normal arrays, not just scalar temperature).

## Concerns

None blocking. Two things worth flagging for whoever wires Task 2:

1. The shelf-pack bin-width heuristic is my own choice (brief didn't specify one) — reasonable
   and tested for correctness (no-overlap + padding), but not necessarily bit-for-bit what a
   different implementer would produce. Since `AtlasLayout.atlas_size`/`offset` are only consumed
   internally (Task 2 renders whatever this returns), this shouldn't matter functionally.
   Numbers are hand-verified for K/side/rescale math, not eyeballed off tool output.
2. `rasterize_tile`'s Python-level loop is per-face as instructed; the included performance test
   (2000 triangles / 352×352 tile) ran in ~well under a second alongside the rest of the suite,
   comfortably inside the brief's target.

## Fix pass (mirrored-UV test)

Added `test_rasterize_mirrored_uv_triangle_interpolates_correctly` to
`tests/test_heatsim_atlas.py`, closing the review gap: none of the 18 shipped tests exercised
`rasterize_tile`'s `area2 < 0` vertex-swap branch (atlas.py, the CW-UV / mirrored-UV-island case).

Fixture: a quad split into two triangles (same base mesh as `_quad_mesh`). Face 0's `loop_uv`
corners are permuted (uv1/uv2 swapped relative to the "natural" orientation) so the signed UV area
is negative, forcing the swap branch; face 1 keeps ordinary CCW UV as a control. Assertions:
- full coverage (64/64 texels), no double-claimed texels, both faces contributed.
- normals are unit +Z for every texel (normal comes from `faces` winding, not `loop_uv`, so it's
  provably unaffected by the mirror).
- `position_mm` for every face-0 texel matches a hand-derived skewed barycentric formula
  (`position = (size*px/w, size*(px/w - py/h), 0)`), *not* the naive `(x+0.5)/W*size` linear
  shortcut used for face 1 — the derivation (post-swap `idx=(i0,i2,i1)`, `pts=(uv0,uv2,uv1)`,
  weights `wa=1-px/w, wb=px/w-py/h, wc=py/h`) is spelled out in the test docstring/comments, plus
  three explicitly named probe texels ((0,0)->(5,0,0), (7,0)->(75,70,0), (4,2)->(45,20,0)) for
  diagnosability.
- Formula was cross-checked numerically against the real (unmodified) implementation before being
  hardcoded into the test, to rule out a self-consistent-but-wrong hand derivation.

**Mutation check** (proves the test bites): temporarily flipped the swap condition in atlas.py
from `if area2 < 0:` to `if area2 >= 0:`, ran the suite:
```
6 failed, 13 passed in 1.21s
FAILED tests/test_heatsim_atlas.py::test_rasterize_full_cover_quad
FAILED tests/test_heatsim_atlas.py::test_rasterize_half_tile_triangle
FAILED tests/test_heatsim_atlas.py::test_rasterize_no_double_claim
FAILED tests/test_heatsim_atlas.py::test_rasterize_degenerate_triangle_mixed_with_valid_no_nan
FAILED tests/test_heatsim_atlas.py::test_rasterize_mirrored_uv_triangle_interpolates_correctly
FAILED tests/test_heatsim_atlas.py::test_rasterize_performance_sanity
```
New test's specific failure (its own assertion, not a pre-existing one):
```
    assert len(xy_rows) == w * h
E   assert 0 == (8 * 8)
E    +  where 0 = len([])
```
(with the mutation, face 0's now-un-swapped triangle has zero UV-space coverage in this fixture,
so it drops out entirely.) `atlas.py` was then restored exactly via `git checkout --
visionsim/simulate/heatsim/atlas.py` (confirmed `git diff` on atlas.py is empty), and the full
suite re-run clean: `19 passed`.

Lint: `ruff check tests/test_heatsim_atlas.py` -> All checks passed.
