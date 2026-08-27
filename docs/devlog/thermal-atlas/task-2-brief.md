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

