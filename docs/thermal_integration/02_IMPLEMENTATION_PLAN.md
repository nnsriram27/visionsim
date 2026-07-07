# Thermal Modality (M1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `thermal` output modality to visionsim that renders a per-pixel temperature map (`temperature/`, Kelvin AOV) and a gray-body thermal-camera image (`thermal_radiance/`) from a single-frame FEM heat solve, driven by `--config.include-thermal`.

**Architecture:** Vendor the pure heat-sim-blender FEM solver + Direct-Kernel irradiance into `visionsim/simulate/heatsim/`; add a cached "solve → prepare" RPC step (`exposed_prepare_thermal`) that runs before the render loop and writes a per-vertex `sim_temperature` attribute; surface temperature as an additive Cycles **AOV** in the main render, and render `thermal_radiance` as a second per-frame render with swapped emission materials.

**Tech Stack:** Python 3.9+ (`from __future__ import annotations`), tyro CLI, RPyC (Blender service), Blender 5.1 / Cycles, numpy/scipy/torch(CUDA)/robust_laplacian inside Blender, peewee (metadata, unchanged), pytest + real Blender.

## Global Constraints

- Target Python **>= 3.9**; every new module starts with `from __future__ import annotations`.
- ruff line-length **121**; ruff `--extend-select I`; mypy (`visionsim` + `tasks.py`); all green via `inv lint` / `inv type-check` / `inv test-stubs` / `inv test` / `inv build-docs`.
- Google docstrings; **every public param documented**; literals in ``double backticks``; `:meth:`/`:class:` cross-refs (enforced by `tests/test_docstrings.py`).
- `@dataclass` config with a `"""docstring"""` under **each** field.
- **Parity (exact dict equality, `tests/test_docstrings.py:48-53`):** `exposed_include_thermal`'s explicit named params + defaults must equal `ThermalConfig`'s fields + defaults exactly. `prepare_thermal`/`heatsim_solve` are exempt.
- Logging via `_log` / `self.log` — **no `print`** (route vendored solver prints through a logger).
- Length unit is **millimeters** (geometry ×1000). Preserve the unit/sign conventions in `01_MIGRATION_GUIDE.md §7` verbatim.
- Vendored Tier-A/B files: **near-verbatim copies**, header `# Vendored from heat-sim-blender:<path> @ <commit>`, modifications logged in `heatsim/VENDOR.md`.
- **Do not commit to visionsim unless asked; local commits only; never push.** (The `git commit` steps below are local-only; skip them if the user prefers an uncommitted working tree.)
- Branch: `heatsim`. Source repo: `/home/sriram/research/heat-sim-blender` (record the commit in `VENDOR.md`).

---

### Task 1: Dependencies — GPU torch + scipy + robust_laplacian into Blender's Python

**Files:**
- Modify: `visionsim/simulate/install.py:53-60` (the pip install command list)

**Interfaces:**
- Consumes: nothing.
- Produces: `torch`, `scipy`, `robust_laplacian` importable inside Blender's bundled Python.

- [ ] **Step 1: Add the deps to the install list.** In `install.py`, after the existing `pip install rpyc peewee typing-extensions` command, add a new command (keep `--no-warn-script-location`; do **not** pass `--no-dependencies` here — torch/scipy need their deps):

```python
# inside install_dependencies(), after the rpyc/peewee/typing-extensions install:
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "torch", "scipy", "robust_laplacian"]
)  # default (CUDA) torch; GPU-less hosts import it and run CPU-side
```

- [ ] **Step 2: Run post-install into the target Blender.**

Run: `vsim post-install --editable` (or `python -m visionsim.cli post-install --path /home/sriram/softwares/blender-5.1.0-linux-x64/blender`)
Expected: pip installs complete; no errors.

- [ ] **Step 3: Verify imports inside Blender.**

Run:
```bash
/home/sriram/softwares/blender-5.1.0-linux-x64/blender -b --python-expr \
"import torch, scipy, robust_laplacian, numpy; print('THERMAL_DEPS_OK', torch.__version__, scipy.__version__)"
```
Expected: prints `THERMAL_DEPS_OK <torch-ver> <scipy-ver>` with no ImportError.

- [ ] **Step 4: Commit.**

```bash
git add visionsim/simulate/install.py
git commit -m "add torch/scipy/robust_laplacian to blender post-install"
```

---

### Task 2: Vendor the pure solver tier (`solver.py`, `laplacian.py`, `constants.py`)

**Files:**
- Create: `visionsim/simulate/heatsim/__init__.py`
- Create: `visionsim/simulate/heatsim/solver.py` (from `addon/lib/heatsim_fem.py`)
- Create: `visionsim/simulate/heatsim/laplacian.py` (from `addon/lib/robust_laplacian_backend.py`)
- Create: `visionsim/simulate/heatsim/constants.py` (from `addon/lib/constants.py`)
- Create: `visionsim/simulate/heatsim/VENDOR.md`
- Test: `tests/test_heatsim_solver.py`

**Interfaces:**
- Consumes: numpy, scipy, torch, robust_laplacian (Task 1).
- Produces:
  - `heatsim.solver.HeatSimFEM` — keep the upstream class + `perform_gt_heat_simulation(...) -> numpy.ndarray` of shape `(num_timesteps, num_vertices)` (Kelvin). **Do not change its signature.**
  - `heatsim.laplacian.point_cloud_laplacian_and_mass(points: np.ndarray, n_neighbors: int, mollify_factor: float) -> tuple[sparse, sparse]` and `mesh_laplacian_and_mass(verts, faces) -> tuple[sparse, sparse]` (upstream signatures).
  - `heatsim.constants` — `SIGMA`, `AMBIENT_TEMP`, material dicts.

- [ ] **Step 1: Copy the three files verbatim** into `visionsim/simulate/heatsim/`, renaming `heatsim_fem.py→solver.py`, `robust_laplacian_backend.py→laplacian.py`, `constants.py→constants.py`. Add the provenance header line to the top of each (below any module docstring):

```python
# Vendored from heat-sim-blender:addon/lib/heatsim_fem.py @ <commit-sha>
```

- [ ] **Step 2: Fix intra-package imports.** In `solver.py`, change `from .constants import ...`/`from . import constants` and `from .robust_laplacian_backend import ...` to `from visionsim.simulate.heatsim import constants` and `from visionsim.simulate.heatsim.laplacian import ...`. Same idea in `laplacian.py`. Grep to confirm no `addon`/`..lib` imports remain:

Run: `grep -rnE "addon|lib\.|bpy" visionsim/simulate/heatsim/solver.py visionsim/simulate/heatsim/laplacian.py visionsim/simulate/heatsim/constants.py`
Expected: no matches (these three are pure, no `bpy`).

- [ ] **Step 3: Route debug prints through a logger.** At the top of `solver.py` add `import logging; _log = logging.getLogger("rich")`, and replace each unconditional `print(...)` (e.g. heatsim_fem.py:204-206, :296-297, :360) with `_log.debug(...)`. Record this transform in `VENDOR.md`.

- [ ] **Step 4: Write `__init__.py`** exporting the public surface:

```python
from __future__ import annotations

from visionsim.simulate.heatsim import constants, laplacian, solver

__all__ = ["constants", "laplacian", "solver"]
```

- [ ] **Step 5: Write `VENDOR.md`** listing each vendored file: upstream path, source commit, and the exact modifications ("import paths rewritten to visionsim.simulate.heatsim.*; unconditional `print()` → `_log.debug()`").

- [ ] **Step 6: Write the failing host-side solver test** (`tests/test_heatsim_solver.py`) — runs WITHOUT Blender, on a tiny synthetic point cloud:

```python
from __future__ import annotations

import numpy as np

from visionsim.simulate.heatsim.solver import HeatSimFEM


def test_solver_produces_finite_physical_temperatures():
    # tiny grid of surface points (mm), uniform irradiance, 5 steps
    n = 64
    rng = np.random.default_rng(0)
    points = rng.uniform(-10.0, 10.0, size=(n, 3)).astype(np.float64)  # mm
    irradiance = np.full(n, 1e-4, dtype=np.float64)                    # W/mm^2
    fem = HeatSimFEM(device="cpu")  # adjust to upstream constructor kwargs
    history = fem.perform_gt_heat_simulation(
        points=points,
        faces=None,                # POINTS domain
        irradiance_map=irradiance,
        initial_temperature=295.0,
        thermal_diffusivity=0.17,
        density=1330.0 / 1e9,      # kg/mm^3
        specific_heat=880.0,
        emissivity=0.9,
        num_steps=5,
        dt=0.05,
        laplacian_domain="POINTS",
        laplacian_backend="ROBUST",
    )
    arr = np.asarray(history, dtype=np.float64)
    assert arr.ndim == 2 and arr.shape[1] == n
    assert not np.isnan(arr).any()
    assert arr.min() > 200.0 and arr.max() < 2000.0
```

> NOTE for implementer: the kwarg names above are illustrative — open `solver.py` and match `HeatSimFEM.__init__` / `perform_gt_heat_simulation` exactly. Adjust the test to the real signature; do not change the solver.

- [ ] **Step 7: Run the test, expect FAIL** (signature mismatch or import error first):

Run: `python -m pytest tests/test_heatsim_solver.py -v`
Expected: FAIL (then iterate Step 6's kwargs until it imports and runs).

- [ ] **Step 8: Make it pass** by aligning the test to the real solver signature (no solver edits). Re-run:

Run: `python -m pytest tests/test_heatsim_solver.py -v`
Expected: PASS.

- [ ] **Step 9: Commit.**

```bash
git add visionsim/simulate/heatsim/ tests/test_heatsim_solver.py
git commit -m "vendor pure heat-sim FEM solver + laplacian + constants"
```

---

### Task 3: De-Blenderized temperature cache (`cache.py`)

**Files:**
- Create: `visionsim/simulate/heatsim/cache.py` (derived from `addon/lib/temperature_io.py`)
- Test: `tests/test_heatsim_cache.py`

**Interfaces:**
- Consumes: numpy.
- Produces:
  - `cache_key(blend_path: Path, solver_cfg: dict) -> str` — stable hash of `(blend_path, blend mtime, solver_cfg)`.
  - `write_temperatures(cache_root: Path, key: str, per_object: dict[str, np.ndarray], meta: dict) -> Path` — writes `<cache_root>/<key>/temperatures.npz`.
  - `read_temperatures(cache_root: Path, key: str) -> dict[str, np.ndarray] | None` — returns the per-object history dict or `None` on miss.

- [ ] **Step 1: Write the failing test** (`tests/test_heatsim_cache.py`):

```python
from __future__ import annotations

import numpy as np

from visionsim.simulate.heatsim import cache


def test_cache_roundtrip_and_miss(tmp_path):
    key = cache.cache_key(tmp_path / "scene.blend", {"dt": 0.05, "domain": "POINTS"})
    assert isinstance(key, str) and key

    assert cache.read_temperatures(tmp_path, key) is None  # miss before write

    per_object = {"cup": np.full((4, 10), 295.0), "plate": np.full((4, 7), 296.0)}
    out = cache.write_temperatures(tmp_path, key, per_object, {"num_timesteps": 4})
    assert out.exists()

    back = cache.read_temperatures(tmp_path, key)
    assert back is not None
    assert set(back) == {"cup", "plate"}
    assert np.allclose(back["cup"], 295.0) and back["plate"].shape == (4, 7)
```

- [ ] **Step 2: Run, expect FAIL.** `python -m pytest tests/test_heatsim_cache.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `cache.py`:**

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def cache_key(blend_path: Path, solver_cfg: dict) -> str:
    """Stable cache key from the blend identity and solver-relevant config.

    Args:
        blend_path: Path to the source blend file.
        solver_cfg: Solver-relevant config values that affect the result.

    Returns:
        A short hex digest used as the cache subdirectory name.
    """
    blend_path = Path(blend_path)
    try:
        mtime = blend_path.stat().st_mtime_ns
    except OSError:
        mtime = 0
    payload = json.dumps({"p": str(blend_path), "m": mtime, "c": solver_cfg}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def write_temperatures(cache_root: Path, key: str, per_object: dict[str, np.ndarray], meta: dict) -> Path:
    """Write per-object temperature histories to ``<cache_root>/<key>/temperatures.npz``.

    Args:
        cache_root: Root directory for thermal caches.
        key: Cache key from :func:`cache_key`.
        per_object: Mapping of object name to a ``(timesteps, vertices)`` array.
        meta: JSON-serializable metadata stored alongside the arrays.

    Returns:
        The path to the written ``.npz`` archive.
    """
    out_dir = Path(cache_root) / key
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "temperatures.npz"
    np.savez_compressed(out, __meta__=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **per_object)
    return out


def read_temperatures(cache_root: Path, key: str) -> dict[str, np.ndarray] | None:
    """Read per-object temperature histories, or return ``None`` on a cache miss.

    Args:
        cache_root: Root directory for thermal caches.
        key: Cache key from :func:`cache_key`.

    Returns:
        Mapping of object name to its history array, or ``None`` if absent.
    """
    path = Path(cache_root) / key / "temperatures.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {k: data[k] for k in data.files if k != "__meta__"}
```

- [ ] **Step 4: Run, expect PASS.** `python -m pytest tests/test_heatsim_cache.py -v` → PASS.

- [ ] **Step 5: Commit.** `git add visionsim/simulate/heatsim/cache.py tests/test_heatsim_cache.py && git commit -m "add de-blenderized thermal temperature cache"`

---

### Task 4: Vendor Direct-Kernel irradiance + per-object properties

**Files:**
- Create: `visionsim/simulate/heatsim/irradiance_kernel.py`, `light_models.py`, `sh9_sky.py`, `sky_visibility.py`, `bvh_backend.py` (from the matching `addon/lib/` files)
- Create: `visionsim/simulate/heatsim/properties.py` (trim of `addon/properties.py`)
- Modify: `visionsim/simulate/heatsim/__init__.py` (add `register()`/`unregister()`)
- Test: `tests/test_heatsim_properties.py` (real Blender)

**Interfaces:**
- Consumes: numpy, `bpy`/`mathutils` (these run inside Blender), Task 2.
- Produces:
  - `heatsim.irradiance_kernel.compute_per_vertex_irradiance(...) -> np.ndarray` (W/m², upstream signature).
  - `heatsim.properties.register()` / `unregister()`; `bpy.types.Object.heat_sim_material` PropertyGroup (`initial_temperature_K`, `thermal_diffusivity_mm2_s`, `density_kg_m3`, `specific_heat_J_kgK`, `emissivity`, `thermal_role`, `dirichlet_temperature_K`) and `bpy.types.Object.heat_simulation_enabled: BoolProperty`.

- [ ] **Step 1: Copy the five irradiance modules** with provenance headers; rewrite imports to `visionsim.simulate.heatsim.*`. Record in `VENDOR.md`.

Run: `grep -rn "addon\|from \.\.lib\|from \.lib" visionsim/simulate/heatsim/*.py`
Expected: no matches.

- [ ] **Step 2: Create `properties.py`** by trimming `addon/properties.py` down to the per-object material PropertyGroup + the `heat_simulation_enabled` registration (drop the scene-level `HeatSimSettings` and all UI). Provide `register()`/`unregister()` that add/remove `bpy.types.Object.heat_sim_material` and `bpy.types.Object.heat_simulation_enabled`. Header comment notes it is schema-compatible with heat-sim-blender so addon-authored blends load their values.

- [ ] **Step 3: Wire `register()`/`unregister()`** into `heatsim/__init__.py`:

```python
from visionsim.simulate.heatsim import properties


def register() -> None:
    """Register the per-object thermal material PropertyGroup on ``bpy.types.Object``."""
    properties.register()


def unregister() -> None:
    """Unregister the thermal material PropertyGroup."""
    properties.unregister()
```

- [ ] **Step 4: Write the failing Blender test** (`tests/test_heatsim_properties.py`) — runs inside the spawned Blender via the existing `executable` fixture pattern, OR as a `--python-expr`. Minimal version using a direct Blender invocation:

```python
from __future__ import annotations

import subprocess


def test_object_thermal_props_register(executable):
    code = (
        "import bpy;"
        "from visionsim.simulate.heatsim import register;"
        "register();"
        "o=bpy.data.objects.new('o', bpy.data.meshes.new('m'));"
        "o.heat_sim_material.emissivity=0.7;"
        "assert abs(o.heat_sim_material.emissivity-0.7)<1e-6;"
        "assert hasattr(o,'heat_simulation_enabled');"
        "print('THERMAL_PROPS_OK')"
    )
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True)
    assert "THERMAL_PROPS_OK" in out.stdout, out.stderr
```

- [ ] **Step 5: Run, expect FAIL** (register missing / prop missing). `python -m pytest tests/test_heatsim_properties.py --executable=/home/sriram/softwares/blender-5.1.0-linux-x64/blender -v`

- [ ] **Step 6: Implement until PASS** (fix `properties.py`). Re-run the same command → PASS (`THERMAL_PROPS_OK`).

- [ ] **Step 7: Commit.** `git add visionsim/simulate/heatsim/ tests/test_heatsim_properties.py && git commit -m "vendor direct-kernel irradiance + per-object thermal properties"`

---

### Task 5: Scene adapter — geometry/materials → solver → `sim_temperature` (`adapter.py`)

**Files:**
- Create: `visionsim/simulate/heatsim/adapter.py` (rewrite of `addon/lib/fem_adapter.py` core)
- Test: `tests/test_heatsim_adapter.py` (real Blender, tiny scene)

**Interfaces:**
- Consumes: `heatsim.solver`, `heatsim.laplacian`, `heatsim.constants`, `heatsim.irradiance_kernel`, `heatsim.cache`, `heatsim.properties`; `bpy`.
- Produces:
  - `gather_meshes(scene) -> list[bpy.types.Object]` — visible, renderable, `heat_simulation_enabled` meshes (port the filter at `fem_adapter.py:2867,2875`).
  - `resolve_material(obj, defaults: dict) -> dict` — per-object params with priority `obj.heat_sim_material` → `defaults`.
  - `solve_scene(scene, *, defaults: dict, solver_cfg: dict, cache_root: Path) -> dict[str, np.ndarray]` — returns/loads per-object `(T, N)` history, writing the cache.
  - `write_frame_attributes(scene, history: dict[str, np.ndarray], timestep: int, defaults: dict) -> None` — writes per-vertex `sim_temperature` + `emissivity`, stamps OBJECT-domain `heatsim_default_temperature` fallbacks on the rest.

- [ ] **Step 1: Implement `adapter.py`** by porting the data-shaping core of `fem_adapter.py` (geometry to mm via `matrix_world` ×1000; quad→tri; per-object material resolution; optional POINTS Bridson interior sampling; irradiance via `compute_per_vertex_irradiance` → ÷1e6 × `irradiance_scale`; combine objects; call `HeatSimFEM.perform_gt_heat_simulation`; split the combined `(T, N_total)` back to per-object). Drive everything from the passed `defaults`/`solver_cfg` dicts — **no `scene.heat_sim_settings`**. Preserve the unit/sign/dt conventions (`01_MIGRATION_GUIDE.md §7`). `solve_scene` checks `cache.read_temperatures` first and writes on miss.

- [ ] **Step 2: Write the failing Blender test** (`tests/test_heatsim_adapter.py`) on a tiny generated scene (a subdivided plane + a point light), asserting `sim_temperature` is written and finite:

```python
from __future__ import annotations

import subprocess


def test_solve_writes_finite_sim_temperature(executable, tmp_path):
    code = f"""
import bpy, numpy as np
from pathlib import Path
from visionsim.simulate.heatsim import register, adapter
register()
bpy.ops.mesh.primitive_plane_add(size=2.0)
plane = bpy.context.active_object
plane.heat_simulation_enabled = True
bpy.ops.object.light_add(type='SUN'); 
defaults = dict(initial_temperature_K=295.0, thermal_diffusivity_mm2_s=0.17,
                density_kg_m3=1330.0, specific_heat_J_kgK=880.0, emissivity=0.9,
                irradiance_scale=100.0)
solver_cfg = dict(sim_time_s=0.15, timestep_s=0.05, domain='POINTS',
                  laplacian_backend='ROBUST', device='cpu')
hist = adapter.solve_scene(bpy.context.scene, defaults=defaults, solver_cfg=solver_cfg,
                           cache_root=Path(r'{tmp_path}'))
adapter.write_frame_attributes(bpy.context.scene, hist, timestep=-1, defaults=defaults)
attr = plane.data.attributes['sim_temperature'].data
vals = np.array([d.value for d in attr])
assert np.isfinite(vals).all() and vals.min() > 200 and vals.max() < 2000
print('THERMAL_ADAPTER_OK')
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True)
    assert "THERMAL_ADAPTER_OK" in out.stdout, out.stderr
```

- [ ] **Step 3: Run, expect FAIL**, iterate on `adapter.py` until it imports and runs:

Run: `python -m pytest tests/test_heatsim_adapter.py --executable=/home/sriram/softwares/blender-5.1.0-linux-x64/blender -v`
Expected: eventually PASS (`THERMAL_ADAPTER_OK`).

- [ ] **Step 4: Verify the cache is reused** (second `solve_scene` call returns without re-solving) — add a sub-assertion or a `_log.debug("cache hit")` check. Keep it light.

- [ ] **Step 5: Commit.** `git add visionsim/simulate/heatsim/adapter.py tests/test_heatsim_adapter.py && git commit -m "add heatsim scene adapter (geometry->solve->sim_temperature, cached)"`

---

### Task 6: Thermal shader — temperature AOV + gray-body radiance material (`thermal_shader.py`)

**Files:**
- Create: `visionsim/simulate/heatsim/thermal_shader.py` (port of `addon/lib/visualization.py` essentials)
- Create: `visionsim/simulate/nodes/thermal.py` (turbo preview node group; modeled on `nodes/colorize.py`)
- Modify: `visionsim/simulate/nodes/__init__.py` (export `thermal_preview_node_group`)
- Test: `tests/test_heatsim_shader.py` (real Blender)

**Interfaces:**
- Consumes: `bpy`; `heatsim.constants`.
- Produces:
  - `setup_temperature_aov(scene, view_layer) -> None` — registers a value AOV named `temperature` on `view_layer.aovs` and appends an `Attribute("sim_temperature") → ShaderNodeOutputAOV("temperature")` to every material (port of `visualization._add_aov_value` :138 + `ensure_view_layer_thermal_aovs` :582). Returns the compositor socket name to wire (`"temperature"`).
  - `enter_thermal_scene(scene, *, radiance_scale: float) -> dict` — swap each mesh's materials to the gray-body emission material (`ensure_fem_thermal_material` :258), disable lights, gray world; returns a state dict for restore.
  - `restore_scene(scene, state: dict) -> None` — undo `enter_thermal_scene`.
  - `stamp_default_temperatures(scene, *, default_K: float) -> None` — OBJECT-domain `heatsim_default_temperature` stamp (`visualization.stamp_default_temperatures` :169).
  - `nodes.thermal.thermal_preview_node_group() -> bpy.types.NodeTree` — turbo colormap (scalar→RGB), structure mirroring `colorize_indices_node_group`.

- [ ] **Step 1: Implement `thermal_shader.py`** (port the four functions above; the gray-body material uses SI σ `5.670374419e-8` × `radiance_scale`, per `01_MIGRATION_GUIDE.md §7`).

- [ ] **Step 2: Implement `nodes/thermal.py`** — a `CompositorNodeTree` taking a `Temperature` float input, `MapRange`-normalizing `[Tmin,Tmax]→[0,1]`, into a turbo `CompositorNodeValToRGB` color ramp, `Image` color output. Use `nodes/common.py`'s `new_socket` + node-type constants for version compat. Export it from `nodes/__init__.py`.

- [ ] **Step 3: Write the failing Blender test** (`tests/test_heatsim_shader.py`):

```python
from __future__ import annotations

import subprocess


def test_temperature_aov_registered(executable):
    code = (
        "import bpy;"
        "from visionsim.simulate.heatsim import thermal_shader as ts;"
        "bpy.ops.mesh.primitive_cube_add();"
        "vl=bpy.context.view_layer;"
        "ts.setup_temperature_aov(bpy.context.scene, vl);"
        "assert any(a.name=='temperature' for a in vl.aovs);"
        "print('THERMAL_AOV_OK')"
    )
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True)
    assert "THERMAL_AOV_OK" in out.stdout, out.stderr
```

- [ ] **Step 4: Run → FAIL → implement → PASS.**

Run: `python -m pytest tests/test_heatsim_shader.py --executable=/home/sriram/softwares/blender-5.1.0-linux-x64/blender -v`
Expected: PASS (`THERMAL_AOV_OK`).

- [ ] **Step 5: Commit.** `git add visionsim/simulate/heatsim/thermal_shader.py visionsim/simulate/nodes/thermal.py visionsim/simulate/nodes/__init__.py tests/test_heatsim_shader.py && git commit -m "add thermal AOV + gray-body radiance shader + turbo preview"`

---

### Task 7: `ThermalConfig` + RenderConfig wiring (config.py)

**Files:**
- Modify: `visionsim/simulate/config.py` (add `ThermalConfig`; add `include_thermal`/`thermal` to `RenderConfig`; `__post_init__`)
- Test: (covered by Task 8's parity entry; this task's gate is `inv lint`/`inv type-check`)

**Interfaces:**
- Consumes: nothing.
- Produces: `config.ThermalConfig` (exact field set in `01_MIGRATION_GUIDE.md §4`); `RenderConfig.include_thermal: bool`, `RenderConfig.thermal: ThermalConfig`.

- [ ] **Step 1: Add `ThermalConfig`** to `config.py` next to `DepthsConfig`, with **every field carrying a `"""docstring"""`**, exactly as in `01_MIGRATION_GUIDE.md §4` (radiance, preview, initial_temperature_K, thermal_diffusivity_mm2_s, density_kg_m3, specific_heat_J_kgK, emissivity, irradiance_scale, sim_time_s, timestep_s, domain, laplacian_backend, device, radiance_scale, exr_codec, bit_depth).

- [ ] **Step 2: Add to `RenderConfig`** (after `points`/`include_points`):

```python
include_thermal: bool = False
"""If true, enable thermal outputs (temperature map + thermal-camera radiance)"""
thermal: ThermalConfig = field(default_factory=ThermalConfig)
"""Thermal modality configuration options"""
```

- [ ] **Step 3: Update `__post_init__`:** add `self.include_thermal = True` to the `include_all` block, and `self.thermal.preview &= self.previews` to the preview-gating block.

- [ ] **Step 4: Lint + type-check.**

Run: `inv lint && inv type-check`
Expected: clean.

- [ ] **Step 5: Commit.** `git add visionsim/simulate/config.py && git commit -m "add ThermalConfig + include_thermal to RenderConfig"`

---

### Task 8: Service methods + render-loop radiance hook (blender.py)

**Files:**
- Modify: `visionsim/simulate/blender.py` (add `exposed_prepare_thermal`, `exposed_include_thermal`, `exposed_heatsim_solve`; render-loop hook in `exposed_render_current_frame`)
- Modify: `tests/test_docstrings.py:35-46` (add the parity parametrize entry)

**Interfaces:**
- Consumes: `heatsim.adapter`, `heatsim.thermal_shader`, `heatsim.cache`, `heatsim.properties`; `_include_output`/`register_output_type`/`exposed_render_current_frame` (existing).
- Produces (client-visible via RPyC): `prepare_thermal(...)`, `include_thermal(...)`, `heatsim_solve(...)`.

- [ ] **Step 1: Ensure properties are registered** when the service initializes (call `heatsim.register()` in `BlenderService` init/initialize alongside other setup), so `obj.heat_sim_material` exists for any loaded blend.

- [ ] **Step 2: Implement `exposed_prepare_thermal`** with explicit named params (a superset is fine here — it is exempt from parity): build `defaults`/`solver_cfg` dicts from the params, call `adapter.solve_scene(...)` (cache-aware), `adapter.write_frame_attributes(...)` for the active frame, `thermal_shader.stamp_default_temperatures(...)`, then `thermal_shader.setup_temperature_aov(scene, self.view_layer)`.

- [ ] **Step 3: Implement `exposed_include_thermal`** — explicit named params **mirroring `ThermalConfig` exactly** (parity). Wire the temperature output (template `exposed_include_depths` :940-995):

```python
self._include_output("temperature", self.render_layers.outputs["temperature"],
                     label="Temperature Output", file_format="OPEN_EXR",
                     color_mode="BW", exr_codec=exr_codec, bit_depth=bit_depth, c=1)
```
If `preview`: build `thermal_preview_node_group()`, wire it, `_include_output("previews/temperature", ..., preview=True, color_mode="RGB", c=3)`. If `radiance`: store `self._thermal_radiance = {"radiance_scale": radiance_scale, "exr_codec": exr_codec, "bit_depth": bit_depth}` and `register_output_type("thermal_radiance", ...)` so it gets a `transforms.db` + per-frame path.

- [ ] **Step 4: Add the radiance second-render hook** in `exposed_render_current_frame` (after the main render at :1899), guarded by `getattr(self, "_thermal_radiance", None)`:

```python
if getattr(self, "_thermal_radiance", None):
    state = thermal_shader.enter_thermal_scene(self.scene, radiance_scale=self._thermal_radiance["radiance_scale"])
    try:
        # point the thermal_radiance output node at this frame's path (reuse the same indexing as above)
        # then render again:
        bpy.ops.render.render(animation=False, write_still=False)
    finally:
        thermal_shader.restore_scene(self.scene, state)
```
(Reuse the per-frame `folder_index`/`frame_index` path logic at :1867-1890 for the `thermal_radiance` output node.)

- [ ] **Step 5: Implement `exposed_heatsim_solve`** = the solve+cache half of `prepare_thermal` (no AOV setup), for the optional CLI command.

- [ ] **Step 6: Add the parity test entry** in `tests/test_docstrings.py` parametrize list:

```python
(blender.BlenderService.exposed_include_thermal, config.ThermalConfig),
```

- [ ] **Step 7: Run parity + docstring tests** (host-side, no Blender needed for these two):

Run: `python -m pytest "tests/test_docstrings.py::test_output_configs" tests/test_docstrings.py::test_docstrings -v`
Expected: PASS (this proves `exposed_include_thermal` ↔ `ThermalConfig` exact parity + all params documented).

- [ ] **Step 8: Commit.** `git add visionsim/simulate/blender.py tests/test_docstrings.py && git commit -m "add prepare_thermal/include_thermal service methods + radiance render hook"`

---

### Task 9: Job dispatch + optional CLI command (job.py, cli/blender.py)

**Files:**
- Modify: `visionsim/simulate/job.py:60-79` (dispatch)
- Modify: `visionsim/cli/blender.py` (`heatsim_solve` command; `optimize_rate` probe)

**Interfaces:**
- Consumes: `client.prepare_thermal`, `client.include_thermal`, `client.heatsim_solve`.
- Produces: `vsim blender.heatsim-solve` CLI command.

- [ ] **Step 1: Add the dispatch** to `render_job` (in the include block):

```python
if config.include_thermal:
    client.prepare_thermal(**asdict(config.thermal))
    client.include_thermal(**asdict(config.thermal))
```

- [ ] **Step 2: Add `heatsim_solve`** public function to `cli/blender.py` (auto-registers as `vsim blender.heatsim-solve`), mirroring `render_animation`'s spawn/guard structure but calling `client.heatsim_solve(**asdict(config.thermal))`. Full Google docstring (all params).

- [ ] **Step 3: Set `probe_config.include_thermal = False`** in `optimize_rate` (cli/blender.py:179-188).

- [ ] **Step 4: Lint/type/docstring.**

Run: `inv lint && inv type-check && python -m pytest tests/test_docstrings.py -v`
Expected: clean / PASS.

- [ ] **Step 5: Commit.** `git add visionsim/simulate/job.py visionsim/cli/blender.py && git commit -m "wire thermal into render_job + add heatsim-solve command"`

---

### Task 10: End-to-end render test + fixture (the M1 gate)

**Files:**
- Create: `tests/test_files/scenes/thermal_cup.blend` (small Cycles fixture — trimmed `cup_pour`, or `cup_pour.blend` copied if size is acceptable)
- Modify: `tests/test_simulate.py` (extend parametrize lists; add the focused thermal test)

**Interfaces:**
- Consumes: `BlenderClient`, the fixture, `Dataset.load_data`, `Metadata.load`.
- Produces: the M1 acceptance test.

- [ ] **Step 1: Add the fixture.** Prefer a trimmed scene; if copying `cup_pour.blend` (583 KB), place it at `tests/test_files/scenes/thermal_cup.blend`. Ensure engine is `CYCLES`.

- [ ] **Step 2: Add `temperature`, `previews/temperature`, `thermal_radiance`** to the layout parametrize lists in `test_simulate.py`.

- [ ] **Step 3: Write the focused thermal render test:**

```python
from pathlib import Path

from visionsim.dataset import Dataset, Metadata
from visionsim.simulate.blender import BlenderClient


def test_render_thermal(tmp_path_factory, executable):
    out = tmp_path_factory.mktemp("renders")
    log_dir = tmp_path_factory.mktemp("logs")
    scene = Path(__file__).parent / "test_files" / "scenes" / "thermal_cup.blend"

    with BlenderClient.spawn(executable=executable, timeout=60, log=log_dir) as client:
        client.initialize(scene.resolve(), out.resolve())
        client.set_resolution(50, 50)
        client.set_animation_range(1, 3)  # frames 1,2 (stop exclusive)
        client.prepare_thermal(**{"radiance": True, "device": "cpu", "domain": "POINTS"})
        client.include_thermal(**{"radiance": True})  # defaults fill the rest
        client.render_animation()

    temp = out / "temperature"
    rad = out / "thermal_radiance"
    assert temp.exists() and (temp / "transforms.db").exists()
    assert len(list(temp.glob("**/*.exr"))) == 2
    assert Dataset.load_data(next(temp.glob("**/*.exr"))).shape == (50, 50, 1)
    assert rad.exists() and len(list(rad.glob("**/*.exr"))) == 2
    Metadata.load(temp / "transforms.db")
```

> NOTE: keyword args to `prepare_thermal`/`include_thermal` must match the implemented signatures; pass only what you need (defaults fill the rest) for `include_thermal`, but pass the full set if parity made them required-with-defaults (they have defaults, so partial is fine).

- [ ] **Step 4: Run the M1 gate.**

Run: `python -m pytest tests/test_simulate.py::test_render_thermal --executable=/home/sriram/softwares/blender-5.1.0-linux-x64/blender -v -s`
Expected: PASS — `temperature/*.exr` `(50,50,1)` and `thermal_radiance/*.exr` present.

- [ ] **Step 5: Commit.** `git add tests/test_files/scenes/thermal_cup.blend tests/test_simulate.py && git commit -m "add end-to-end thermal render test + fixture"`

---

### Task 11: Stubs, docs, full green

**Files:**
- Modify: `visionsim/simulate/blender.pyi` (regenerate)
- Create: `docs/source/sections/thermal.rst` (narrative page)

- [ ] **Step 1: Regenerate stubs.** `inv generate-stubs` → adds `prepare_thermal`/`include_thermal`/`heatsim_solve` to the client stub.
- [ ] **Step 2: Stub test.** `inv test-stubs` → PASS.
- [ ] **Step 3: Add the docs page** (`thermal.rst`): what `--config.include-thermal` produces, the two outputs, the per-object material model, the cache behavior; cross-ref `:meth:`exposed_include_thermal``.
- [ ] **Step 4: Full local gate.**

Run: `inv lint && inv type-check && inv test-stubs && python -m pytest tests/ --executable=/home/sriram/softwares/blender-5.1.0-linux-x64/blender -v && inv build-docs`
Expected: all green.

- [ ] **Step 5: Commit.** `git add visionsim/simulate/blender.pyi docs/source/sections/thermal.rst && git commit -m "regenerate stubs + add thermal docs page"`

---

## Self-Review

**Spec coverage** (against `01_MIGRATION_GUIDE.md`): outputs `temperature/`+`thermal_radiance/` (Tasks 6, 8, 10) ✓; lazy cached solve (Tasks 3, 5, 8) ✓; vendor pure solver (Task 2) + Direct-Kernel irradiance (Task 4) ✓; per-object materials (Task 4, 5) ✓; GPU torch (Task 1) ✓; config/flags + parity (Tasks 7, 8) ✓; job dispatch + CLI (Task 9) ✓; tests on cup_pour (Task 10) ✓; stubs/docs/DoD (Task 11) ✓; default-temperature fallback (Tasks 5, 6) ✓; units/sign/dt conventions referenced (Global Constraints + §7) ✓; VENDOR.md re-portability (Task 2, 4) ✓.

**Placeholders:** the large ported files (`solver.py`, `adapter.py`, `thermal_shader.py`, irradiance modules) are specified by *exact source file + transform + gating test* rather than reproduced line-for-line — appropriate for a vendor-and-adapt port; the tractable new code (cache, config, dispatch, tests) is given in full. No `TBD`/`add error handling`-style gaps.

**Type/name consistency:** `solve_scene`/`write_frame_attributes`/`resolve_material`/`gather_meshes` (Task 5), `setup_temperature_aov`/`enter_thermal_scene`/`restore_scene`/`stamp_default_temperatures` (Task 6), `cache_key`/`write_temperatures`/`read_temperatures` (Task 3) are used consistently in Tasks 8–10. `exposed_include_thermal` mirrors `ThermalConfig` (Tasks 7, 8) — parity gate in Task 8 Step 7.

**Known soft spots to confirm during execution:** (a) the real `HeatSimFEM.perform_gt_heat_simulation` signature (Task 2 Step 6 note); (b) whether the combined-mesh split back to per-object histories matches upstream `_combine_meshes` ordering (Task 5); (c) fixture size/runtime on CPU CI (Task 10 — trim if slow).
