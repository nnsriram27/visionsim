# Thermal Atlas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. Implementers follow TDD: the test cases are named per task; write them first, watch them fail, then implement.

**Goal:** Decouple thermal simulation and rendering resolution from mesh vertex count: simulate on area-proportional texel sample points and render by sampling a scene-wide temperature atlas texture, so a 4-vertex floor shows real spatial thermal detail.

**Architecture:** One new pure-numpy module (`atlas.py`: selection, tile allocation, shelf packing, UV-space rasterization) + a TEXEL mode threaded through the existing adapter/solver/shader pipeline. The POINTS-domain solver is untouched. `render_domain="VERTEX"` (default) stays byte-identical.

**Spec:** `docs/superpowers/specs/2026-08-04-thermal-atlas-design.md` (approved, density-driven allocator revision)

## Global Constraints

- Python 3.9 floor; `from __future__ import annotations`; no `match`; no PEP-604 unions outside annotations; no backslash/outer-quote-reuse inside f-string expressions.
- `atlas.py` and all `materials.py`/`config.py` changes are ruff+mypy gated (`inv lint`, `inv type-check`). `adapter.py`, `blender.py` ruff-only. Vendored modules (`irradiance*.py`, `solver.py`, …) excluded from both — behavior changes there must be guard-style only, never altering results for currently-working inputs.
- **Backward compatibility is the hard gate:** with `render_domain="VERTEX"`, all existing tests pass unmodified. New cache-key fields may invalidate old caches (acceptable, one recompute) but must not change solved values.
- Units: geometry mm (world scale, `_extract_geometry` output = matrix_world-applied × 1000); areas for density in **m²** (mm² / 1e6); α mm²/s, ρ kg/m³ (→ kg/mm³ only at the solver boundary), c J/(kg·K), ε ∈ [0,1], T in K.
- No new third-party dependencies.
- Environment: interpreter `/home/sriram/research/visionsim/.venv/bin/python` with `PYTHONPATH` pointing at THIS worktree (the venv's editable install points at the main checkout — never import without the override). Prefix `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libsqlite3.so.0`. Blender: `/home/sriram/softwares/blender-5.1.0-linux-x64/blender`; Blender-fixture tests need `--executable=<that path>`. Pre-existing unrelated failure: `tests/test_cli.py::test_completions`.
- Work dir: `/home/sriram/research/visionsim/.claude/worktrees/thermal-atlas` (branch `thermal-atlas`). All commits here.

## Solver benchmark (Task 0 — controller-run, informs `atlas_texel_density` default)

500-step POINTS solve, synthetic 6-plane room, RTX 2080: results recorded in the
progress ledger when complete. Density default chosen so kitchen1's expected total
(texels + retained verts) lands in the 150–250k band with solve time acceptable
(≤ ~5 min). Placeholder default until then: `atlas_texel_density = 1500.0` texels/m²
(~2.6 cm texels).

---

### Task 1: `atlas.py` — selection, allocation, packing, rasterization (pure numpy)

**Files:** Create `visionsim/simulate/heatsim/atlas.py`; Test `tests/test_heatsim_atlas.py`.

**Produces (exact signatures):**
```python
@dataclass(frozen=True)
class TileSpec:
    obj_name: str
    size: tuple[int, int]        # (w, h) texels, multiples of 4, tile_min<=side<=tile_max
    offset: tuple[int, int]      # (x, y) in atlas pixels
@dataclass(frozen=True)
class AtlasLayout:
    atlas_size: tuple[int, int]
    tiles: dict[str, TileSpec]
    effective_density: float     # after any soft-max rescale
    rescaled: bool

def surface_area_m2(verts_mm: np.ndarray, faces: np.ndarray) -> float
def select_for_atlas(n_verts: int, area_m2: float, density: float) -> bool
    # True iff n_verts / max(area_m2, eps) < density
def allocate(areas: dict[str, float], density: float, *, tile_min: int = 16,
             tile_max: int = 512, soft_max: int = 500_000,
             retained_vertex_count: int = 0, padding: int = 2) -> AtlasLayout
    # texel count = area*density; side = ceil(sqrt(K)) rounded UP to multiple of 4,
    # clamped [tile_min, tile_max]; if sum(sides^2)+retained > soft_max: rescale
    # density uniformly (sqrt factor on sides), set rescaled=True, warnings.warn.
    # Shelf-pack sorted by height desc; atlas_size grows to fit (multiple of 4).
def rasterize_tile(verts_mm, faces, loop_uv, tile_size) -> dict[str, np.ndarray]
    # loop_uv: (n_faces, 3, 2) triangle UVs in [0,1] tile-local space.
    # Rasterize each triangle over texel centers ((x+.5)/W, (y+.5)/H); half-open
    # edge rule; later triangles do NOT overwrite earlier ones (first hit wins).
    # Returns {"xy": (K,2) int, "position_mm": (K,3), "normal": (K,3) unit face
    # normals, "face": (K,) int} for covered texels only.
def dilate(image: np.ndarray, valid: np.ndarray, iterations: int = 4) -> np.ndarray
    # Push-out: each invalid texel takes the mean of valid 8-neighbors; repeat.
```

**Test cases to write first (names are the contract):**
- `test_surface_area_of_unit_quad` (two triangles, 1 m² from mm coords)
- `test_selection_excludes_dense_objects` (orchid-like: high verts/area → False; floor-like 16 verts/80 m² → True)
- `test_tile_size_scales_with_area_and_respects_min_max_and_multiple_of_4`
- `test_big_object_not_power_of_two_overshot` (81 m² @1500/m² → side ≈ 352, NOT 512)
- `test_soft_max_rescales_uniformly_and_warns` (pytest.warns; effective_density < requested; rescaled=True) and `test_soft_max_counts_retained_vertices`
- `test_shelf_pack_no_overlap_and_padding` (pairwise tile rect disjointness incl. padding)
- `test_rasterize_full_cover_quad` (quad covering whole tile → every texel covered, positions interpolate linearly, normals unit +Z)
- `test_rasterize_half_tile_triangle` (~half texels covered, none outside)
- `test_rasterize_no_double_claim` (two abutting triangles: each texel claimed exactly once)
- `test_rasterize_degenerate_triangle_skipped` (zero-area UV triangle → no texels, no NaN)
- `test_dilate_fills_margin_and_preserves_valid`

**Steps:** write tests → RED → implement → GREEN → `inv lint && inv type-check` clean for the new files → commit `feat(heatsim): atlas allocation, packing and texel rasterization (pure numpy)`.

---

### Task 2: Texel sim assembly — bpy atlas build + `_combine` TEXEL mode

**Files:** Modify `visionsim/simulate/heatsim/adapter.py`, `visionsim/simulate/heatsim/materials.py` (small helper), `visionsim/simulate/heatsim/irradiance_kernel.py` (guard-style entry point only); Test `tests/test_heatsim_texel_mode.py`.

**Produces:**
- `materials.resolve_face_materials(obj, assignment, fallback, face_slots: np.ndarray) -> dict[str, np.ndarray]` — per-FACE α/ρ/c/ε/role/dirichlet_K arrays (slot lookup, no averaging); reuses the existing slot-table logic.
- `adapter.build_atlas_plan(scene, sim_objects, cfg) -> AtlasPlan` — world-space areas via `_extract_geometry`, selection, `atlas.allocate`, per-object `rasterize_tile` using the object's **bake UV** (created by the existing `irradiance.prepare_object_bake_uv` machinery, then remapped into the tile and written as mesh UV layer `"HeatSim_Atlas_UV"`). Objects whose UV prep fails are demoted to the vertex path (warn, never raise) — reuse the hardened helpers.
- `adapter._combine(..., atlas_plan=None)` — TEXEL entries contribute texel points (position_mm) with per-face-resolved materials + per-texel albedo/irradiance; vertex-path objects contribute exactly as today. `layout` entries gain a kind tag `("VERTEX"|"TEXEL")`; `_split_history` unchanged (slices still contiguous).
- Per-texel **albedo**: sample the object's existing albedo bake image bilinearly at texel centers (same bake-UV space; the bake already exists per object via `get_or_bake_vertex_albedo`'s underlying `bake_albedo_map` — reuse the image before it is reduced to vertices; if absent → albedo 0 = full absorption, consistent with kernel fallback).
- Per-texel **irradiance**: add `irradiance_kernel.compute_irradiance_at_points(scene, positions_mm, normals, albedo) -> (K,) W/m²` factored from the existing per-vertex path WITHOUT changing per-vertex results (pure extraction refactor; existing irradiance tests must stay green).
- Cache key: `key_cfg["atlas"] = None | {density, tile_min, tile_max, soft_max, layout_digest}`.

**Test cases (fake-bpy style, mirroring `tests/test_heatsim_assignments_integration.py` fixtures):**
- `test_resolve_face_materials_exact_per_face` (2-slot mesh → per-face values with no blending; Dirichlet slot faces flagged)
- `test_combine_texel_mode_mixes_texel_and_vertex_objects` (one sparse fake object with a stubbed rasterization + one dense object → combined arrays sized K+n; layout kinds correct)
- `test_combine_texel_dirichlet_pinning_per_texel` (α=0, irr=0, mask False, t0=reservoir on source-slot texels)
- `test_combine_vertex_mode_unchanged_when_no_plan` (atlas_plan=None → arrays identical to before — the backward-compat gate)
- `test_uv_failure_demotes_to_vertex_path_with_warning`
- `test_cache_key_gains_atlas_digest`
- Existing suites `test_heatsim_adapter.py`, `test_heatsim_assignments_integration.py`, `test_heatsim_irradiance.py` (with `--executable`) must pass unmodified.

**Commit:** `feat(heatsim): TEXEL sim domain — texel point clouds with per-face materials and per-texel albedo/irradiance`.

---

### Task 3: Atlas write, shader sampling, config plumbing

**Files:** Modify `visionsim/simulate/heatsim/adapter.py` (atlas writer), `visionsim/simulate/nodes/thermal.py`, `visionsim/simulate/config.py`, `visionsim/simulate/blender.py`, regen `blender.pyi`; Test `tests/test_heatsim_atlas_render.py` + extend `tests/test_thermal_preview.py`-style node checks.

**Produces:**
- `adapter.write_atlas(history, atlas_plan, cache_root) -> Path` — final-timestep texel temps scattered into the atlas array, `atlas.dilate` margin, alpha channel = validity; saved as 32-bit EXR in the cache dir; loaded/packed as a Blender image `HeatSim_Temperature_Atlas` at render time. `write_frame_attributes` in TEXEL mode: vertex-path objects exactly as today; atlas objects get the fallback object property only (their per-pixel signal comes from the atlas), and `global_temperature_range` pools texel temps too (P1–P99 unchanged).
- `nodes/thermal.py`: the temperature source group gains
  `UVMap("HeatSim_Atlas_UV") → ImageTexture(atlas, Non-Color, linear, extension=CLIP)`
  and `Mix(vertex_temperature_path, atlas_R, atlas_A)` — one shared group; objects
  without the UV layer/atlas coverage sample A=0 and follow today's chain. Both the
  AOV output and the gray-body radiance emission consume the mixed value.
- `ThermalConfig`: `render_domain: Literal["VERTEX","TEXEL"]="VERTEX"`,
  `atlas_texel_density: float = <benchmark default>`, `atlas_tile_min: int = 16`,
  `atlas_tile_max: int = 512`, `atlas_texel_soft_max: int = 500_000`. Threaded through
  `_thermal_config`/`_thermal_solve` and **all three** exposed methods
  (`prepare_thermal`, `heatsim_solve`, `include_thermal` — the asdict-dispatch parity
  tests bind them); stubs regenerated via `inv generate-stubs`, `inv test-stubs` green.

**Test cases:** `test_write_atlas_scatters_dilates_and_marks_alpha`; `test_atlas_shader_group_samples_atlas_and_mixes_by_alpha` (Blender `--python-expr`, node-graph assertions like the existing preview-group test); `test_thermal_config_atlas_fields_dispatch_parity` (extend the existing asdict test); `test_global_temperature_range_includes_texels`.

**Commit:** `feat(heatsim): temperature atlas rendering — EXR atlas write + shader sampling + config`.

---

### Task 4: kitchen1 vertical-slice validation (controller-run)

1. `render_domain="VERTEX"` regression: full suite green (`--executable`), plus a 2-frame kitchen1 VERTEX render to confirm unchanged behavior.
2. kitchen1 TEXEL render: 500 s / 1 s steps, 50 frames, 512×512, GPU, sidecar assignments. Build contact sheet TEXEL vs VERTEX.
3. Acceptance: floor/walls show spatial temperature structure (not 4-corner gradients); no visible tile seams; dense objects (orchid) unchanged; solve time and point count logged against the benchmark expectation.
4. Ledger + summary for user review of the visuals.

---

## Follow-ups (out of scope)

classroom + bulk scenes; animated TEXEL mode; texel-space Dirichlet in animated path; atlas-driven ground-truth temperature EXR variant (current per-pixel AOV output already benefits automatically).
