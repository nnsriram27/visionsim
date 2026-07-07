# Thermal Modality — Migration Guide (heat-sim-blender → visionsim)

**Status:** design approved, pre-implementation.
**Branch:** `heatsim` (off `main`, currently a clean slate — no thermal code yet).
**Scope of this document:** the *what* and *where* — architecture, file-by-file plan, conventions. The ordered *how* (build steps + checkpoints) lives in `02_IMPLEMENTATION_PLAN.md`.
**Do not commit to visionsim. Local commits only if asked; never push.**

---

## 0. Goal

Add a `thermal` output modality to visionsim so that:

```bash
vsim blender.render-animation cup_pour.blend out/ --config.include-thermal
```

produces, alongside the usual passes:

- `out/temperature/` — per-pixel **temperature in Kelvin** (1-channel float EXR) — ground truth, rendered as a Cycles **AOV**, co-rendered with rgb/depth/normals in the main render. (`+ previews/temperature/` turbo PNGs.)
- `out/thermal_radiance/` — the **gray-body radiance image a thermal camera sees** (`L = ε·σT⁴ + (1−ε)·L_in`), produced by a **second render** per frame with thermal materials/world.

The temperature field is produced by a **lazy, cached FEM heat solve** that runs the first time a scene is rendered and is reused on every subsequent render (e.g. new camera poses).

This mirrors visionsim's own two-layer philosophy: **simulate** = ground truth (`temperature/`), and a future **emulate.thermal** sensor (microbolometer noise/optics) will consume `temperature/` to produce realistic IR frames (milestone M3).

### Design decisions (locked with the user)
| Decision | Choice |
|---|---|
| Outputs | `temperature/` (Kelvin AOV) + `thermal_radiance/` (gray-body render). No separate emissivity output. |
| Solve↔render | Lazy auto-solve **with caching**: render solves once if no cache, else loads+renders. Optional explicit `heatsim-solve` command, never required. |
| Packaging | **Vendor the pure solver** into `visionsim/simulate/heatsim/`; self-contained, no addon dependency; structured for easy future re-porting. |
| Per-object materials | Port heat-sim-blender's editable per-object material model into visionsim (schema-compatible). Global `ThermalConfig` defaults + per-object overrides. |
| torch | **GPU/CUDA** torch installed into Blender's Python; solver default device `cuda` with automatic CPU fallback. |
| Irradiance | **Direct-Kernel only** (analytic direct light + SH9 sky). Do **not** port the Cycles-bake path. |
| Milestone 1 | Static single-frame solve → `temperature/` AOV **and** `thermal_radiance/` render on `cup_pour.blend`, with tests for both. |

---

## 1. The one new architectural concept

Every existing visionsim modality is a **stateless single-`bpy.ops.render.render()` Cycles AOV pass** derived from the scene as-is. There is **no precompute / scene-mutation / pre-render hook** in the pipeline (verified across `blender.py`). Thermal needs exactly one new idea: a **cached "solve → prepare" step before the render**, plus (for radiance) **one extra render pass**. We add only that, and route the outputs through the existing modality plumbing.

```
render_job (job.py)
  load_addons / initialize / set_resolution / use_animations / ...
  if config.include_thermal:
      client.prepare_thermal(**asdict(config.thermal))   # ① NEW: lazy solve + cache + write sim_temperature + arm AOV
      client.include_thermal(**asdict(config.thermal))    # ② register temperature/ AOV output (+ arm radiance 2nd render)
  render loop (exposed_render_current_frame):
      main render  ─────────────────────────►  temperature/ (AOV)  [+ rgb/depth/normals/...]
      if thermal.radiance:
          enter thermal scene (swap materials→emission, lights off, gray world)
          2nd render ───────────────────────►  thermal_radiance/
          restore scene state
```

Why these two seams:
- **`temperature/`** is a pure additive **AOV** (`view_layer.aovs`, value type) fed by an `Attribute("sim_temperature") → ShaderNodeOutputAOV` node appended to each material. It does **not** alter the look of the main render, so it co-renders with everything else. (No existing modality uses AOVs — this is the net-new render mechanism, and it's the clean one.)
- **`thermal_radiance/`** needs materials swapped to gray-body emission + lights off + grayscale world, which is global scene state — so it cannot co-render with rgb. It is a **second render** with save/restore, the only place we touch the render loop.

---

## 2. Source inventory — what is ported from heat-sim-blender

Reference repo: `/home/sriram/research/heat-sim-blender` (commit pinned in `VENDOR.md` at port time).

### Tier A — pure numerics (no `bpy`), vendored **near-verbatim** → pristine, trivially re-syncable
| visionsim file | source | role |
|---|---|---|
| `heatsim/solver.py` | `addon/lib/heatsim_fem.py` | `HeatSimFEM`: implicit backward-Euler + matrix-free CG on torch sparse tensors; `perform_gt_heat_simulation()` (heatsim_fem.py:847) is the public driver. |
| `heatsim/laplacian.py` | `addon/lib/robust_laplacian_backend.py` | robust_laplacian point-cloud / mesh Laplacian+mass wrappers. |
| `heatsim/constants.py` | `addon/lib/constants.py` | mm-scaled `SIGMA` (constants.py:14), `AMBIENT_TEMP` (:15), material library, name constants. |

Allowed edits to Tier A: import paths only, and route the unconditional debug `print()`s through a logger (gated off by default). Nothing else.

### Tier B — Direct-Kernel irradiance (reads `bpy`/`mathutils`), vendored near-verbatim → "Blender-coupled vendored"
| visionsim file | source | role |
|---|---|---|
| `heatsim/irradiance_kernel.py` | `addon/lib/irradiance_kernel.py` | `compute_per_vertex_irradiance()` (:385): analytic per-vertex direct-light + SH9 sky irradiance. One cached albedo bake (irradiance_kernel.py:193-200) — see risk R8. |
| `heatsim/light_models.py` | `addon/lib/light_models.py` | analytic per-light diffuse form-factors + shadow-ray generation. |
| `heatsim/sh9_sky.py` | `addon/lib/sh9_sky.py` | SH9 sky prefilter + per-vertex sky evaluation (numpy only). |
| `heatsim/sky_visibility.py` | `addon/lib/sky_visibility.py` | per-vertex bent-normal + AO bake for sky occlusion. |
| `heatsim/bvh_backend.py` | `addon/lib/bvh_backend.py` | ray/shadow visibility via Embree if present, else `mathutils.bvhtree`. |

**Not ported** (Cycles-bake path, explicitly excluded by the user): `addon/lib/irradiance.py`, `addon/lib/uv_utils.py`.
**Not ported** (alternative solver / unused): `addon/lib/zombie_adapter.py`, `addon/lib/mesh_utils.py`.

### Tier C — visionsim-native glue (new code, written in visionsim style)
| visionsim file | derived from | role |
|---|---|---|
| `heatsim/adapter.py` | rewrite of `addon/lib/fem_adapter.py` | `bpy` scene ⇄ numpy arrays; gather geometry (mm), per-object material params, run solver, write `sim_temperature` + cache. Keeps fem_adapter's data-shaping logic (triangulation, Bridson point sampling, unit conversions) but driven by a config object, not `scene.heat_sim_settings`. |
| `heatsim/thermal_shader.py` | port of `addon/lib/visualization.py` + `temperature_viz.py` | build the `temperature` AOV node + register the view-layer AOV; build the gray-body emission material for `thermal_radiance`; default-temperature stamping. |
| `heatsim/cache.py` | de-Blenderized `addon/lib/temperature_io.py` | read/write the `.heatsim` temperature archive (npz) keyed by a config hash; no dependency on a saved `.blend` path. |
| `heatsim/properties.py` | trim of `addon/properties.py` | register schema-compatible per-object `heat_sim_material` PropertyGroup + `heat_simulation_enabled` on `bpy.types.Object`. |
| `heatsim/__init__.py` | — | package exports + a single `register()`/`unregister()` for the PropertyGroup. |
| `heatsim/VENDOR.md` | — | provenance + per-file modification list for re-porting (§10). |

**Not ported** (addon UI/registration machinery — replaced by visionsim CLI/RPC): `addon/__init__.py`, `operators.py`, `panels.py`, `animation_sync.py`.

---

## 3. File-by-file plan in visionsim

### 3.1 New package: `visionsim/simulate/heatsim/`
As listed in §2 Tier A/B/C. The package's only `bpy`-registration is `properties.py`'s PropertyGroup; everything else is functions called from `blender.py`'s service methods.

### 3.2 `visionsim/simulate/config.py` — add the modality config (tyro auto-generates the flags)
- Add `ThermalConfig` dataclass next to `DepthsConfig` (config.py:40-51) / `NormalsConfig` (config.py:54-63). Full field set in §4.
- In `RenderConfig` (fields block config.py:156-246), add — mirroring `include_depths`/`depths` at config.py:174-176:
  ```python
  include_thermal: bool = False
  """If true, enable thermal outputs (temperature map + thermal-camera radiance)"""
  thermal: ThermalConfig = field(default_factory=ThermalConfig)
  """Thermal modality configuration options"""
  ```
- In `RenderConfig.__post_init__` (config.py:247-268): add `self.include_thermal = True` to the `include_all` cascade (config.py:251-261); add `self.thermal.preview &= self.previews` to the preview-gating block (config.py:263-268).

### 3.3 `visionsim/simulate/job.py` — dispatch (manual, like every include_*)
In the include-dispatch block (job.py:60-79), add:
```python
if config.include_thermal:
    client.prepare_thermal(**asdict(config.thermal))   # solve (cached) + write sim_temperature + register AOV node on materials
    client.include_thermal(**asdict(config.thermal))    # register temperature/ file-output (+ previews) and arm radiance
```
`prepare_thermal` runs before the render loop; it is the new pre-render hook. (Both client methods auto-exist via RPyC `exposed_` forwarding — no client-side edit.)

### 3.4 `visionsim/simulate/blender.py` — the service methods
Add to `BlenderService`:

- **`exposed_prepare_thermal(self, **thermal_cfg)`** — the solve/prepare step:
  1. Resolve per-object material params: `obj.heat_sim_material` → `ThermalConfig.overrides[name]` → globals (§5).
  2. Compute the cache key (hash of blend identity + solver-relevant config); check `cache.py`.
  3. Cache miss → gather geometry (mm) + irradiance (Direct-Kernel) via `adapter.py`, run `solver.HeatSimFEM.perform_gt_heat_simulation`, write npz via `cache.py`. Cache hit → load.
  4. Write the current frame's `sim_temperature` (and per-vertex `emissivity`) attributes onto each mesh; stamp OBJECT-domain default-temperature fallbacks for un-simulated/hidden meshes (`thermal_shader.py`).
  5. Append the `Attribute("sim_temperature") → ShaderNodeOutputAOV("temperature")` node to each material and register the matching value AOV on the view layer.
- **`exposed_include_thermal(self, radiance=True, preview=True, initial_temperature_K=295.0, ...)`** — the output wiring (template: `exposed_include_depths`, blender.py:940-995). **Explicit named params mirroring every `ThermalConfig` field with identical defaults** (parity test — `**kwargs` would fail it). Uses `preview`/`radiance`/`exr_codec`/`bit_depth`; ignores the solver fields.
  - `_include_output("temperature", <temperature AOV socket>, file_format="OPEN_EXR", color_mode="BW", c=1, exr_codec=..., bit_depth=...)` (uses `_include_output` at blender.py:541-614 / `register_output_type` at :503-539).
  - If `preview`: a turbo-colormap node group (new `nodes/thermal.py`, modeled on `colorize.py`) → `_include_output("previews/temperature", ..., preview=True, color_mode="RGB", c=3)`.
  - If `radiance`: record that a second render is needed (store thermal scene params on `self`); the actual extra render is in the render loop below.
- **Render-loop hook in `exposed_render_current_frame`** (blender.py:1853-1916): after the main `bpy.ops.render.render()` (blender.py:1899), if radiance is armed: enter thermal scene (swap materials→gray-body emission, disable lights, gray world via `thermal_shader.py`), set the `thermal_radiance` output node's path (reuse the same per-frame path logic at blender.py:1880-1890), render again, then restore. Must restore on exception (try/finally).
- **`exposed_heatsim_solve(self, **thermal_cfg)`** *(optional, for the standalone CLI command)* — just the solve+cache half of `prepare_thermal`.

### 3.5 `visionsim/simulate/install.py` — dependencies (GPU torch)
Extend the Blender-python install list (install.py:53-60) to add **CUDA torch + scipy + robust_laplacian**:
```python
[sys.executable, "-m", "pip", "install", "torch", "scipy", "robust_laplacian"]   # default (CUDA) torch
```
- `ThermalConfig.device` defaults to `"cuda"`; solver auto-falls back to CPU when `torch.cuda.is_available()` is False (so GPU-less CI still runs).
- `robust_laplacian` is a soft dependency — the solver degrades to a scipy-kNN Laplacian if it is absent.
- Validated: torch + scipy + robust_laplacian install cleanly into Blender 5.1.0's bundled CPython 3.13 on this machine.

### 3.6 `visionsim/cli/blender.py` — optional standalone solve command
Add a public `heatsim_solve(blend_file, /, config: RenderConfig, ...)` (auto-registers as `vsim blender.heatsim-solve`) that spawns one client and calls `client.heatsim_solve(...)`. Also set `probe_config.include_thermal = False` in `optimize_rate`'s probe config (cli/blender.py:179-188) for completeness.

### 3.7 `visionsim/simulate/blender.pyi` — regenerate
Run `inv generate-stubs` after the `blender.py` API is settled (adds `prepare_thermal`, `include_thermal`, `heatsim_solve` to the client stub). Gated by the `inv test-stubs` CI check.

### 3.8 Tests — `tests/test_files/scenes/` + `tests/test_simulate.py`
- Add a small thermal fixture (trimmed `cup_pour`-style Cycles scene; keep it tiny for CPU CI). `cup_pour.blend` (583 KB) is the manual/integration target.
- Extend the layout parametrize lists in `test_simulate.py` (the lists at ~:15-40, :57-74, :93-110) with `temperature`, `previews/temperature`, `thermal_radiance`.
- Add a focused render test (template: `test_render_layout` at test_simulate.py:42-54 + the self-contained `test_database_threading` at :131-149): spawn Blender, `include_thermal()`, render 1–2 frames at 50×50, assert `temperature/*.exr` is `(50,50,1)`, `thermal_radiance/*.exr` exists, and `Metadata.load(.../transforms.db)` round-trips.
- Note the parity test `test_output_configs` (test_docstrings.py:33-53): **`ThermalConfig` field defaults must exactly equal `exposed_include_thermal`'s signature defaults**, or CI fails. And `test_docstrings` requires every public param documented.

### 3.9 Docs
A short narrative page under `docs/source/sections/` for the thermal modality; Sphinx auto-docs the API from the docstrings (the `*Config` field docstrings + the `exposed_include_thermal` docstring).

---

## 4. CLI / config surface

```python
@dataclass
class ThermalConfig:
    # --- outputs ---
    radiance: bool = True
    """If true, also render the gray-body thermal-camera radiance image (second render pass)"""
    preview: bool = True
    """Also save a turbo-colormap PNG preview of the temperature map"""
    # --- per-object override hook (else globals below) ---
    # overrides: dict[str, ...]  # (M2: per-object params by object name; M1 uses globals + obj.heat_sim_material)
    # --- global material defaults (used where no per-object value is set) ---
    initial_temperature_K: float = 295.0
    """Default initial temperature for meshes without a per-object value"""
    thermal_diffusivity_mm2_s: float = 0.17
    """Default thermal diffusivity (mm^2/s)"""
    density_kg_m3: float = 1330.0
    """Default material density (kg/m^3)"""
    specific_heat_J_kgK: float = 880.0
    """Default specific heat (J/kg*K)"""
    emissivity: float = 0.9
    """Default surface emissivity in [0, 1]"""
    # --- solver ---
    irradiance_scale: float = 100.0
    """Scale factor applied to computed irradiance (heating input)"""
    sim_time_s: float = 1.0
    """Total simulated time in seconds (static scene mode)"""
    timestep_s: float = 0.05
    """Solver timestep in seconds"""
    domain: Literal["POINTS", "MESH"] = "POINTS"
    """FEM domain: surface point cloud (recommended) or mesh"""
    laplacian_backend: Literal["ROBUST", "IGL"] = "ROBUST"
    """Laplacian backend"""
    device: Literal["cuda", "cpu"] = "cuda"
    """Torch device for the solve; falls back to cpu if cuda is unavailable"""
    # --- radiance render ---
    radiance_scale: float = 1.0
    """Gray-body emission magnitude knob for the thermal_radiance render"""
    # --- file formats (mirror DepthsConfig) ---
    exr_codec: EXR_CODECS = "DWAA"
    """Encoding used to compress EXRs"""
    bit_depth: Literal[16, 32] = 32
    """Bit depth for temperature/radiance EXRs"""
```

`--config.include-thermal` is the single switch the user asked for; `--config.thermal.radiance False`, `--config.thermal.device cpu`, `--config.thermal.emissivity 0.95`, etc. tune it. Every field carries a `"""docstring"""` (tyro `--help` + Sphinx + the docstring test).

**Parity-test constraint (verified in `tests/test_docstrings.py:48-53`):** `test_output_configs` does an **exact dict equality** — `{field: default}` of `ThermalConfig` must equal `{param: default}` of `exposed_include_thermal` (minus `self`), for the new parametrize entry `(exposed_include_thermal, ThermalConfig)`. Therefore `exposed_include_thermal` must declare **every** `ThermalConfig` field with identical defaults — including the solver fields it does not itself use (it wires outputs; it simply ignores `device`/`domain`/`timestep_s`/… which `prepare_thermal` consumes). `prepare_thermal`/`heatsim_solve` are **not** `include_*` methods, so they are exempt from this test and may take `**asdict(config.thermal)` freely.

---

## 5. Per-object thermal materials

Port heat-sim-blender's per-object model into `heatsim/properties.py` as a **schema-compatible** PropertyGroup registered on `bpy.types.Object`:

- `obj.heat_sim_material`: `initial_temperature_K`, `thermal_diffusivity_mm2_s`, `density_kg_m3`, `specific_heat_J_kgK`, `emissivity`, `thermal_role`, `dirichlet_temperature_K`.
- `obj.heat_simulation_enabled`: participation flag.

Two payoffs: (1) the properties are genuinely **editable** (Blender UI + Python), and (2) **addon-authored blends like `cup_pour.blend` work as-is** (their per-object thermal data is already present and schema-identical).

The adapter resolves each object's params in priority order:
```
obj.heat_sim_material (if set in the blend)
  ▸ else ThermalConfig.overrides[obj.name]      (M2: headless per-object control)
  ▸ else ThermalConfig global defaults
Objects not enabled / not simulated / hidden:
  ▸ OBJECT-domain default-temperature stamp (ported from heat-sim-blender) so they render at a sane T, not 0 K.
```
M1 may run the test purely on globals + defaults; the per-object path is wired and exercised by reading `cup_pour`'s existing values.

---

## 6. Data flow detail

1. **Geometry** (`adapter.py`): evaluated depsgraph mesh × `matrix_world`, converted to **millimeters** (×1000). Quad→tri triangulation. POINTS domain optionally augments interior points via Bridson sampling (uses `mathutils.bvhtree`).
2. **Irradiance** (Direct-Kernel): `compute_per_vertex_irradiance()` → W/m², converted to W/mm² (÷1e6) × `irradiance_scale`.
3. **Solve** (`solver.py`): assemble Laplacian (sign-flipped to negative-semidefinite, see R3) + mass matrix; implicit backward-Euler; matrix-free CG per step; produces `(T, N)` temperature history (Kelvin).
4. **Cache** (`cache.py`): write `<cache_root>/<key>/temperatures.npz`. Key = hash of (blend path + mtime, solver-relevant config). `cache_root` defaults to `<blend>.heatsim/` (addon convention), overridable.
5. **Attribute write**: for the rendered frame, write per-vertex `sim_temperature` (raw K) + `emissivity`; stamp defaults on the rest.
6. **temperature AOV**: `Attribute("sim_temperature") → ShaderNodeOutputAOV("temperature")` appended to each material; view-layer AOV registered. Main render writes `temperature/`.
7. **thermal_radiance** (if enabled): enter thermal scene (gray-body emission materials, lights off, gray world), second render → `thermal_radiance/`, restore.

---

## 7. Units & conventions to preserve (from heat-sim-blender)

| Convention | Value / rule | Source |
|---|---|---|
| Length unit | **millimeters** (geometry ×1000) | fem_adapter.py:899 |
| `σ` (solver) | mm-scaled `constants.SIGMA` (W/mm²·K⁴) | constants.py:14 |
| `σ` (radiance shader) | SI `5.670374419e-8`, used as a magnitude knob × `radiance_scale` — **do not** unify with the solver σ | visualization.py:510 |
| Irradiance | W/m² → W/mm² (÷1e6) | fem_adapter.py:907 |
| Density | kg/m³ → kg/mm³ (÷1e9) | fem_adapter.py:1372 |
| Laplacian sign | solver wants **negative-semidefinite**; robust_laplacian is PSD → flip (`L = -L_psd`) | heatsim_fem.py:598 |
| dt encoding | `NUM_FRAME_DELTA = timestep_s × 60`, solver divides by 60 — hard-coded 60, preserve | fem_adapter.py:677, heatsim_fem.py:931 |
| Result contract | per-vertex `sim_temperature` FLOAT/POINT attribute + `.heatsim` npz archive | fem_adapter.py:1781 |
| Render fallback | missing `sim_temperature` → OBJECT-domain `heatsim_default_temperature` (must be stamped first) | visualization.py:169-244 |

---

## 8. Testing plan (M1)

- **Unit (host, no Blender):** the pure solver on a tiny synthetic mesh — assert finite, physical temperatures (matches the heat-sim-blender harness checks: no NaN, `T_min > 200 K`, `T_max < 2000 K`). Runs without Blender.
- **Integration (real Blender, CPU in CI):** spawn on the thermal fixture, `include_thermal()`, render 1–2 frames @ 50×50; assert `temperature/*.exr` shape `(50,50,1)`, `thermal_radiance/*.exr` exists, `transforms.db` round-trips. Structural, not pixel-snapshot (matches visionsim's existing render tests).
- **Manual:** full `cup_pour.blend` render at real resolution on GPU — eyeball the turbo preview + radiance image.
- **Parity / docstring:** `inv test` covers `test_output_configs` (config↔signature parity) and `test_docstrings`.

---

## 9. Dependencies

- Add **CUDA torch + scipy + robust_laplacian** to `install.py`'s Blender-python install (§3.5). Already proven installable into Blender 5.1.0 / cp313 here.
- Host env (pyproject) is unchanged — these run inside Blender.
- CI (GPU-less): CUDA torch imports and runs CPU-side; solver device falls back to cpu. Heavier download is the only cost.

---

## 10. Re-portability strategy (`heatsim/VENDOR.md`)

- Tier A/B files are **near-verbatim** copies with a header: `# Vendored from heat-sim-blender:<path> @ <commit>`.
- `VENDOR.md` lists, per vendored file: upstream path, source commit, and the exact local modifications applied (e.g. "import paths; debug prints → logger").
- **All** visionsim-specific logic lives in Tier C (`adapter.py`, `thermal_shader.py`, `cache.py`, `properties.py`, the `blender.py` methods) so the vendored core stays pristine.
- **Re-porting a future upstream change** = re-copy the Tier A/B file(s) + re-apply the short modification list. Optional `tools/sync_heatsim.py` to automate the copy+transform (deferred unless wanted).

---

## 11. Milestones

- **M1 (this guide):** static single-frame solve → `temperature/` AOV + `thermal_radiance/` render on `cup_pour`; vendored solver + Direct-Kernel irradiance; per-object property plumbing; GPU torch; tests green.
- **M2:** animated per-frame solve (heat-sim-blender's ANIMATE mode), per-frame cache, `ThermalConfig.overrides` for headless per-object control.
- **M3:** `emulate.thermal` microbolometer sensor (consumes `temperature/`), richer material defaults.

---

## 12. Risks & mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | `thermal_radiance` second render cost + scene-state corruption | Contained to one render-loop hook; save/restore in `try/finally`; default radiance on but a single sub-toggle to disable. |
| R2 | GPU torch + Cycles GPU contention | Solve and render are sequential, not concurrent; document; allow `device=cpu`. |
| R3 | Laplacian sign / unit-system mismatches | Codified in §7; covered by the unit solver test. |
| R4 | `robust_laplacian` wheel unavailable on a target Blender ABI | Soft dependency (scipy-kNN fallback); validated for cp313 here. |
| R5 | Material-swap AOV interferes with rgb pass | `temperature` AOV is additive (extra OutputAOV node), not a material swap — main render is untouched. |
| R6 | Solver debug `print()` spam in batch runs | Route through a logger, off by default (Tier A allowed edit). |
| R7 | Cache key staleness | Hash includes blend mtime + solver config; explicit `heatsim-solve` can force a rebuild. |
| R8 | Direct-Kernel still does one Cycles albedo bake | Cache it; allow a "fully absorbing" fallback (irradiance_kernel.py:503-505) if Cycles bake is undesired. |

---

## 13. Definition of done (visionsim conventions)

`from __future__ import annotations` in every new module · full type hints (py3.9+) · pathlib + `.resolve()` · `_log`/`self.log` not `print` · specific exceptions with value-rich messages · Google docstrings (every public param documented; literals in ``double backticks``; `:meth:` cross-refs) · `@dataclass` with per-field `"""docstrings"""` · all green: `inv lint` / `inv type-check` / `inv test-stubs` / `inv test` / `inv build-docs`.
