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

