# VisionSim ↔ heat-sim Thermal Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make VisionSim's M1 thermal solve of `bunny_textured.blend` reproduce heat-sim-blender's temperature field — both the checkerboard texture imprint and the absolute magnitude — by porting the per-vertex albedo bake and honoring the blend's authored `irradiance_scale`.

**Architecture:** Two independent, complementary fixes in `visionsim/simulate/heatsim/` (+ one `blender.py` wiring point). (1) Port heat-sim's `bake_albedo_map` into a new `irradiance.py` and stop the adapter from pre-stamping a constant `albedo=0` attribute, so the already-vendored albedo consumer (`get_or_bake_vertex_albedo` → `_bake_vertex_albedo_via_cycles`) actually bakes the checkerboard. (2) Read the blend's authored scene-level `irradiance_scale` (1000) via the raw ID-property and override the `ThermalConfig` default (100), closing the 10× magnitude gap. The FEM solver, unit scaling, and direct-kernel irradiance are already byte-identical to heat-sim and are NOT touched.

**Tech Stack:** Python 3.13, Blender 5.1 `bpy` (Cycles bake), numpy, pytest (pure-python + `blender --background` subprocess tests). VisionSim repo on branch `heatsim`.

## Global Constraints

- Work in the VisionSim repo `/home/sriram/research/visionsim` on branch `heatsim`. Do NOT push to any remote; local commits only.
- Do NOT modify the FEM solver, unit scaling, or the direct irradiance kernel physics — they are verified identical to heat-sim. Only the albedo bake module, the adapter's albedo pre-stamp, and the `irradiance_scale` plumbing change.
- Do NOT port the Cycles *irradiance* bake. Only the *albedo* bake.
- The ported `bake_albedo_map` must return an object exposing `.pixels` as a float `numpy` array of shape `(H, W, 3)` — the exact contract `_bake_vertex_albedo_via_cycles` (`irradiance_kernel.py:206-239`) consumes. Do not change that consumer.
- Blender executable for tests: `/net/acadia2a/data/sriram/blender-fem-research/blender` (has torch + robust_laplacian). Pass it as `--executable`.
- Determinism for any comparison: `device="cpu"`, seeds 0, `temperature_follow_timeline=False`.
- Constant string values are exact: `ALBEDO_LAYER_NAME = "HeatSim_Albedo"`, `BAKE_UV_LAYER_NAME = "HeatSim_Bake_UV"`.
- Parity acceptance: VisionSim bunny peak ΔT and mean ΔT within **±15 %** of heat-sim's (heat-sim ≈ +1.86 K peak, +0.286 K mean), and the checkerboard present + spatially correlated.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `visionsim/simulate/heatsim/irradiance.py` | Cycles COLOR albedo bake → `(H,W,3)` pixels (`bake_albedo_map`) | **Create** (port) |
| `visionsim/simulate/heatsim/constants.py` | shared constant names | **Modify** — add `ALBEDO_LAYER_NAME` |
| `visionsim/simulate/heatsim/adapter.py` | blend→solver adapter | **Modify** — drop constant-albedo pre-stamp; add `read_authored_irradiance_scale` |
| `visionsim/simulate/blender.py` | thermal solve entry (`_thermal_solve`) | **Modify** — override `defaults["irradiance_scale"]` from blend |
| `tests/test_heatsim_irradiance.py` | albedo-bake + pre-stamp tests | **Create** |
| `tests/test_heatsim_adapter.py` | `read_authored_irradiance_scale` unit test | **Modify** (append) |

---

## Task 1: Port `bake_albedo_map` into a new `irradiance.py`

**Files:**
- Create: `visionsim/simulate/heatsim/irradiance.py`
- Modify: `visionsim/simulate/heatsim/constants.py` (add one constant)
- Create: `tests/test_heatsim_irradiance.py`

**Interfaces:**
- Consumes: VisionSim `constants.ALBEDO_LAYER_NAME`, `constants.BAKE_UV_LAYER_NAME`; UV helpers `snapshot_uv_states`, `restore_uv_states` (already defined in `adapter.py`).
- Produces: `bake_albedo_map(scene, obj, texture_size: int) -> Optional[BakedFluxMap]` in module `visionsim.simulate.heatsim.irradiance`; `BakedFluxMap` with attribute `.pixels` of shape `(H, W, 3)` float64. This is exactly what `_bake_vertex_albedo_via_cycles` imports at `irradiance_kernel.py:204` (`from visionsim.simulate.heatsim import irradiance`) and reads at `:206-207` (`baked = irradiance.bake_albedo_map(...); baked.pixels`).

**Context:** VisionSim already vendored the entire albedo *consumer* chain (`get_or_bake_vertex_albedo` → `_bake_vertex_albedo_via_cycles`). The only missing piece is the module they import. It does not exist (`ls visionsim/simulate/heatsim/irradiance.py` → not found). This task creates it by porting the bake machinery verbatim from the heat-sim addon.

- [ ] **Step 1: Add the missing constant**

The port's `_ensure_albedo_image` uses `ALBEDO_LAYER_NAME`, which VisionSim's `constants.py` lacks (it only has `BAKE_UV_LAYER_NAME`). Add it next to `BAKE_UV_LAYER_NAME` in `visionsim/simulate/heatsim/constants.py`:

```python
ALBEDO_LAYER_NAME = "HeatSim_Albedo"
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_heatsim_irradiance.py`. This is a `blender --background` subprocess test (matching the pattern in `tests/test_heatsim_adapter.py`). It bakes a procedural checker material and asserts the returned `.pixels` are `(H,W,3)` and spatially varying.

```python
import subprocess


def test_bake_albedo_map_returns_varying_pixels(executable):
    code = r"""
import bpy, numpy as np
from visionsim.simulate.heatsim import register, irradiance

register()
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.mesh.primitive_grid_add(x_subdivisions=12, y_subdivisions=12, size=2.0)
obj = bpy.context.active_object

mat = bpy.data.materials.new('checker_mat')
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes.get('Principled BSDF')
checker = nt.nodes.new('ShaderNodeTexChecker')
checker.inputs['Scale'].default_value = 6.0
nt.links.new(checker.outputs['Color'], bsdf.inputs['Base Color'])
obj.data.materials.append(mat)

bpy.context.scene.render.engine = 'CYCLES'
try:
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.cycles.samples = 4
except Exception:
    pass

baked = irradiance.bake_albedo_map(bpy.context.scene, obj, 128)
assert baked is not None, 'bake_albedo_map returned None'
px = baked.pixels
assert px.ndim == 3 and px.shape[2] == 3, f'bad pixel shape {px.shape}'
assert float(px.std()) > 1e-3, f'expected spatial variation, got std={px.std()}'
mean = float(px.mean())
assert 0.0 <= mean <= 1.0, f'albedo mean out of range: {mean}'
print('ALBEDO_BAKE_OK', px.shape, round(mean, 3), round(float(px.std()), 3))
"""
    out = subprocess.run(
        [str(executable), "-b", "--python-expr", code],
        capture_output=True, text=True,
    )
    assert "ALBEDO_BAKE_OK" in out.stdout, out.stdout + "\n" + out.stderr
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /home/sriram/research/visionsim && pytest tests/test_heatsim_irradiance.py -v --executable /net/acadia2a/data/sriram/blender-fem-research/blender`
Expected: FAIL — the subprocess prints a `ModuleNotFoundError`/`ImportError` for `visionsim.simulate.heatsim.irradiance` (module does not exist yet), so `ALBEDO_BAKE_OK` is absent from stdout.

- [ ] **Step 4: Create `irradiance.py` by porting the bake machinery**

Create `visionsim/simulate/heatsim/irradiance.py`. Port the following functions **verbatim** from `/home/sriram/research/heat-sim-blender/addon/lib/irradiance.py`, with only the import changes below. Copy the exact bodies from the source file (do not paraphrase):

- `BakedFluxMap` dataclass (source lines 17–26)
- `_ensure_uv_layer` (29–67)
- `prepare_object_bake_uv` (70–141)
- `_ensure_bake_image` and `_ensure_albedo_image` (166–189)
- `_prepare_image_nodes_for_bake` (192–217)
- `_BakeMaterialUVOverride` dataclass + `_pick_source_uv_for_object` + `_apply_uv_override_to_material` + `_install_bake_uv_material_overrides` + `_restore_bake_uv_material_overrides` (220–416)
- `_image_pixels_to_rgb` (419–428)
- `_bilinear_sample` (431–454) and `_image_to_vertex_irradiance` (457–476)
- `bake_albedo_map` (992–1118)

Module header and imports (this REPLACES heat-sim's import block — repoint `constants` to VisionSim's and take the UV helpers from `adapter`):

```python
"""Cycles COLOR albedo bake, ported from the heat-sim-blender addon.

Vendored so VisionSim's Direct-Kernel albedo path (``irradiance_kernel.
_bake_vertex_albedo_via_cycles`` → this module's ``bake_albedo_map``) can
resolve per-vertex reflectivity without depending on the installed addon.

Only the albedo (DIFFUSE/COLOR) bake is ported; the Cycles *irradiance* bake
is intentionally not vendored (VisionSim uses the analytic Direct-Kernel for
irradiance). ``bake_albedo_map`` returns a ``BakedFluxMap`` whose ``.pixels``
is an ``(H, W, 3)`` float64 array — the contract consumed by
``irradiance_kernel._bake_vertex_albedo_via_cycles``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Set

import bpy
import numpy as np

from .constants import ALBEDO_LAYER_NAME, BAKE_UV_LAYER_NAME
```

Inside `bake_albedo_map`, replace the heat-sim `snapshot_uv_states`/`restore_uv_states` module-level import with a **lazy import from the adapter** (VisionSim's copies live there; importing lazily avoids any import cycle since `adapter` is already loaded by the time the bake runs). Add these two lines at the very top of `bake_albedo_map`'s body, and use the imported names unchanged in the function:

```python
    from .adapter import snapshot_uv_states, restore_uv_states
```

Everything else in the copied functions stays identical. Do not port heat-sim's other functions (irradiance bake, etc.).

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd /home/sriram/research/visionsim && pytest tests/test_heatsim_irradiance.py -v --executable /net/acadia2a/data/sriram/blender-fem-research/blender`
Expected: PASS — stdout contains `ALBEDO_BAKE_OK (128, 128, 3) <mean> <std>` with std > 0.001.

- [ ] **Step 6: Verify no import cycle / no leftover heat-sim imports**

Run: `cd /home/sriram/research/visionsim && grep -nE "from .uv_utils|from .constants import|import irradiance" visionsim/simulate/heatsim/irradiance.py`
Expected: only `from .constants import ALBEDO_LAYER_NAME, BAKE_UV_LAYER_NAME` (no `.uv_utils`, no `IRRADIANCE_LAYER_NAME`, no `CYCLES_LOUT_TO_IRRADIANCE`). The `from .adapter import snapshot_uv_states, restore_uv_states` line lives inside `bake_albedo_map`, not at module top.

- [ ] **Step 7: Add `irradiance.py` to the vendored ruff + mypy exclude lists**

The other vendored bpy-heavy modules (`irradiance_kernel.py`, `light_models.py`, etc.) are excluded from ruff and mypy in `pyproject.toml`. This port is the same kind of file, so add it to both lists to keep `inv lint` / `inv type-check` clean.

In `pyproject.toml`, in the `[tool.ruff]` `exclude = [` block (alongside the existing `"visionsim/simulate/heatsim/temperature_io.py",` entry), add:

```
    "visionsim/simulate/heatsim/irradiance.py",
```

In the `[tool.mypy]` `exclude = [` block (alongside `"visionsim/simulate/heatsim/temperature_io\\.py",`), add (note the escaped dot):

```
    "visionsim/simulate/heatsim/irradiance\\.py",
```

- [ ] **Step 8: Run the lint + type-check gates**

Run: `cd /home/sriram/research/visionsim && inv lint && inv type-check`
Expected: `inv lint` clean (no new findings). `inv type-check` shows only the 6 pre-existing errors in `emulate/dvs/v2e/emulator.py` + `dataset/dataset.py` (NOT in heatsim) — no new errors from the port. If new heatsim errors appear, the exclude entries are missing/misspelled — fix Step 7.

- [ ] **Step 9: Commit**

```bash
cd /home/sriram/research/visionsim
git add visionsim/simulate/heatsim/irradiance.py visionsim/simulate/heatsim/constants.py pyproject.toml tests/test_heatsim_irradiance.py
git commit -m "feat(heatsim): port Cycles albedo bake (bake_albedo_map) into VisionSim"
```

---

## Task 2: Stop pre-stamping constant albedo so the bake runs end-to-end

**Files:**
- Modify: `visionsim/simulate/heatsim/adapter.py` (`_compute_irradiance` ~lines 227-233; remove `_ensure_albedo_attr` function ~196-224)
- Modify: `tests/test_heatsim_irradiance.py` (append end-to-end test)

**Interfaces:**
- Consumes: `bake_albedo_map` from Task 1 (via the unchanged `get_or_bake_vertex_albedo` → `_bake_vertex_albedo_via_cycles` chain).
- Produces: after this task, solving a textured object leaves a per-vertex POINT/FLOAT `albedo` mesh attribute with spatial variation (mean in (0,1), std > 0), and `absorbed = E·(1 − albedo)` at `irradiance_kernel.py:506` applies the checkerboard.

**Context:** `_compute_irradiance` currently pre-stamps a constant `albedo` attribute on every sim object (`_ensure_albedo_attr(obj, 0.0)`), which makes `get_or_bake_vertex_albedo` short-circuit at tier 1 (read existing attribute → 0.0) and never bake. That pre-stamp existed only to dodge the missing `irradiance` module (now supplied by Task 1). Removing it lets the bake run. Bake failure is still handled safely: `get_or_bake_vertex_albedo` omits the object from the map on failure, and `irradiance_kernel.py:507-509` falls back to `absorbed = e_total` (full absorption) — the same behavior the constant-0 stamp produced.

- [ ] **Step 1: Write the failing end-to-end test**

Append to `tests/test_heatsim_irradiance.py`. It solves a checker-textured grid through the adapter and reads back the `albedo` attribute, asserting it varies (proving the bake ran and was consumed, not the old constant 0).

```python
def test_solve_produces_varying_albedo_attribute(executable, tmp_path):
    code = r"""
import bpy, numpy as np
from pathlib import Path
from visionsim.simulate.heatsim import register, adapter

register()
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.mesh.primitive_grid_add(x_subdivisions=20, y_subdivisions=20, size=2.0)
obj = bpy.context.active_object
obj.name = 'ThermalPlane'
obj.heat_simulation_enabled = True

mat = bpy.data.materials.new('checker_mat')
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes.get('Principled BSDF')
checker = nt.nodes.new('ShaderNodeTexChecker')
checker.inputs['Scale'].default_value = 6.0
nt.links.new(checker.outputs['Color'], bsdf.inputs['Base Color'])
obj.data.materials.append(mat)

bpy.ops.object.light_add(type='SUN')
bpy.context.active_object.data.energy = 10.0
world = bpy.context.scene.world
world.use_nodes = True
bg = world.node_tree.nodes.get('Background')
bg.inputs['Strength'].default_value = 1.0

defaults = dict(initial_temperature_K=295.0, thermal_diffusivity_mm2_s=0.17,
                density_kg_m3=1330.0, specific_heat_J_kgK=880.0, emissivity=0.9,
                irradiance_scale=100.0)
solver_cfg = dict(sim_time_s=0.1, timestep_s=0.05, domain='POINTS',
                  laplacian_backend='ROBUST', device='cpu')
adapter.solve_scene(bpy.context.scene, defaults=defaults,
                    solver_cfg=solver_cfg, cache_root=Path(r'{tmp}'))

mesh = obj.data
attr = mesh.attributes.get('albedo')
assert attr is not None, 'no albedo attribute after solve'
vals = np.zeros(len(mesh.vertices), dtype=np.float32)
attr.data.foreach_get('value', vals)
assert float(vals.std()) > 1e-3, f'albedo not varying (std={{vals.std()}}) - bake did not run'
assert 0.0 < float(vals.mean()) < 1.0, f'albedo mean out of range: {{vals.mean()}}'
print('VARYING_ALBEDO_OK', round(float(vals.mean()), 3), round(float(vals.std()), 3))
""".replace("{tmp}", str(tmp_path))
    out = subprocess.run(
        [str(executable), "-b", "--python-expr", code],
        capture_output=True, text=True,
    )
    assert "VARYING_ALBEDO_OK" in out.stdout, out.stdout + "\n" + out.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sriram/research/visionsim && pytest tests/test_heatsim_irradiance.py::test_solve_produces_varying_albedo_attribute -v --executable /net/acadia2a/data/sriram/blender-fem-research/blender`
Expected: FAIL — the pre-stamped constant `albedo=0` gives `vals.std() == 0`, so the `std > 1e-3` assertion fails and `VARYING_ALBEDO_OK` is absent.

- [ ] **Step 3: Remove the constant-albedo pre-stamp**

In `visionsim/simulate/heatsim/adapter.py`, `_compute_irradiance`, delete these three lines (the current text is exactly this):

```python
    default_albedo = float(defaults.get("albedo", 0.0))
    for obj in sim_objects:
        _ensure_albedo_attr(obj, default_albedo)
```

Then delete the now-unused `_ensure_albedo_attr` function definition (its docstring block + body, ~`adapter.py:196-224`). The surrounding `_compute_irradiance` keeps building `settings` and calling `irradiance_kernel.compute_per_vertex_irradiance(...)` unchanged.

- [ ] **Step 4: Verify `_ensure_albedo_attr` has no other references**

Run: `cd /home/sriram/research/visionsim && grep -rn "_ensure_albedo_attr" visionsim/ tests/`
Expected: no matches (function and its only caller are gone). If any match remains, it is a real reference — stop and reconcile before proceeding.

- [ ] **Step 5: Run the end-to-end test to verify it passes**

Run: `cd /home/sriram/research/visionsim && pytest tests/test_heatsim_irradiance.py -v --executable /net/acadia2a/data/sriram/blender-fem-research/blender`
Expected: PASS — both tests green; stdout contains `VARYING_ALBEDO_OK <mean> <std>` with std > 0.001.

- [ ] **Step 6: Run the existing adapter tests to confirm no regression**

Run: `cd /home/sriram/research/visionsim && pytest tests/test_heatsim_adapter.py -v --executable /net/acadia2a/data/sriram/blender-fem-research/blender`
Expected: PASS — untextured objects (e.g. the grid in `test_solve_writes_finite_sim_temperature`) still solve; the bake either produces a uniform albedo or returns None → full absorption, so `sim_irradiance.max() > 0` still holds.

- [ ] **Step 7: Lint gate**

`adapter.py` is a linted (non-excluded) glue module. Run: `cd /home/sriram/research/visionsim && inv lint`
Expected: clean (no unused-import or import-order findings introduced by removing the pre-stamp / function).

- [ ] **Step 8: Commit**

```bash
cd /home/sriram/research/visionsim
git add visionsim/simulate/heatsim/adapter.py tests/test_heatsim_irradiance.py
git commit -m "fix(heatsim): drop constant-albedo pre-stamp so the ported bake runs"
```

---

## Task 3: Honor the blend's authored `irradiance_scale`

**Files:**
- Modify: `visionsim/simulate/heatsim/adapter.py` (add `read_authored_irradiance_scale`)
- Modify: `visionsim/simulate/blender.py` (`_thermal_solve`, override `defaults["irradiance_scale"]`)
- Modify: `tests/test_heatsim_adapter.py` (append pure-python unit test)

**Interfaces:**
- Produces: `adapter.read_authored_irradiance_scale(scene) -> Optional[float]` — returns the heat-sim addon's authored scene-level `irradiance_scale`, read from the raw ID-property `scene.get("heat_sim_settings")["irradiance_scale"]` (VisionSim does not register the addon's scene PropertyGroup, so attribute access `scene.heat_sim_settings` raises `AttributeError`; the raw ID-property read is confirmed to return `1000.0` for `bunny_textured.blend`). Returns `None` when the blend has no authored `heat_sim_settings`, so callers keep their own default.
- Consumes: nothing new. `_combine` already reads `defaults.get("irradiance_scale", ...)` at `adapter.py:265`; this task just changes what value lands in `defaults`.

**Context:** heat-sim's `bunny_textured.blend` authored `scene.heat_sim_settings.irradiance_scale = 1000`. VisionSim hardcodes `ThermalConfig.irradiance_scale = 100` (`config.py:177`) and never reads the blend — a 10× magnitude deficit. The blend's authored value is the source of truth for a heat-sim scene (the same precedence VisionSim already uses for per-object material: authored-in-blend wins over the config default).

- [ ] **Step 1: Write the failing unit test**

Append to `tests/test_heatsim_adapter.py`. This is a pure-python test (no Blender) using a fake scene that mimics `bpy` `.get()` semantics.

```python
def test_read_authored_irradiance_scale():
    from visionsim.simulate.heatsim.adapter import read_authored_irradiance_scale

    class _FakeScene:
        def __init__(self, data):
            self._data = data
        def get(self, key, default=None):
            return self._data.get(key, default)

    # Authored heat_sim_settings with an irradiance_scale -> returns it.
    authored = _FakeScene({"heat_sim_settings": {"irradiance_scale": 1000.0}})
    assert read_authored_irradiance_scale(authored) == 1000.0

    # No heat_sim_settings at all -> None (caller keeps its default).
    assert read_authored_irradiance_scale(_FakeScene({})) is None

    # heat_sim_settings present but no irradiance_scale key -> None.
    partial = _FakeScene({"heat_sim_settings": {"fem_domain": "POINTS"}})
    assert read_authored_irradiance_scale(partial) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sriram/research/visionsim && pytest tests/test_heatsim_adapter.py::test_read_authored_irradiance_scale -v`
Expected: FAIL with `ImportError`/`AttributeError` — `read_authored_irradiance_scale` does not exist yet.

- [ ] **Step 3: Implement `read_authored_irradiance_scale` in adapter.py**

Add this function to `visionsim/simulate/heatsim/adapter.py` (near `resolve_material`). If `Optional` is not already imported at the top of the file, add it to the existing `typing` import.

```python
def read_authored_irradiance_scale(scene) -> "Optional[float]":
    """Return the heat-sim addon's authored scene-level ``irradiance_scale``.

    The addon stores it under ``scene.heat_sim_settings.irradiance_scale``.
    VisionSim does not register that scene-level PropertyGroup, so the value is
    read from the raw ID-property (``scene.get("heat_sim_settings")`` returns an
    ``IDPropertyGroup``). Returns ``None`` when the blend has no authored
    heat-sim scene settings, so the caller keeps its own default.
    """
    try:
        raw = scene.get("heat_sim_settings")
        if raw is None:
            return None
        val = raw.get("irradiance_scale")
        return None if val is None else float(val)
    except Exception:
        return None
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `cd /home/sriram/research/visionsim && pytest tests/test_heatsim_adapter.py::test_read_authored_irradiance_scale -v`
Expected: PASS.

- [ ] **Step 5: Wire the override into `_thermal_solve`**

In `visionsim/simulate/blender.py`, `_thermal_solve`, immediately after the `defaults = { ... }` dict literal is built (the block ending with the `"irradiance_scale": irradiance_scale,` line and its closing `}`), insert:

```python
        # A heat-sim-authored .blend carries its own scene-level irradiance_scale
        # (e.g. 1000). Let it override the ThermalConfig default so the scene
        # renders identically to the addon without manual CLI tuning.
        _authored_scale = adapter.read_authored_irradiance_scale(self.scene)
        if _authored_scale is not None:
            defaults["irradiance_scale"] = _authored_scale
```

(`adapter` is already imported at the top of `_thermal_solve` via `from visionsim.simulate.heatsim import adapter`; `self.scene` is `bpy.context.scene`.)

- [ ] **Step 6: Write the integration test for the override**

Append to `tests/test_heatsim_irradiance.py`. It sets a raw `heat_sim_settings` ID-property on the scene and confirms the reader picks it up (proving the raw-property path works under a real `bpy` scene).

```python
def test_authored_irradiance_scale_read_under_bpy(executable):
    code = r"""
import bpy
from visionsim.simulate.heatsim import register, adapter
register()
sc = bpy.context.scene
sc['heat_sim_settings'] = {'irradiance_scale': 1000.0}
val = adapter.read_authored_irradiance_scale(sc)
assert val == 1000.0, f'expected 1000.0, got {val!r}'
del sc['heat_sim_settings']
assert adapter.read_authored_irradiance_scale(sc) is None
print('AUTHORED_SCALE_OK')
"""
    out = subprocess.run(
        [str(executable), "-b", "--python-expr", code],
        capture_output=True, text=True,
    )
    assert "AUTHORED_SCALE_OK" in out.stdout, out.stdout + "\n" + out.stderr
```

- [ ] **Step 7: Run the integration test**

Run: `cd /home/sriram/research/visionsim && pytest tests/test_heatsim_irradiance.py::test_authored_irradiance_scale_read_under_bpy -v --executable /net/acadia2a/data/sriram/blender-fem-research/blender`
Expected: PASS — stdout contains `AUTHORED_SCALE_OK`.

- [ ] **Step 8: Lint + type-check gate**

Both `adapter.py` and `blender.py` are linted, non-excluded modules; the new function has a type annotation. Run: `cd /home/sriram/research/visionsim && inv lint && inv type-check`
Expected: `inv lint` clean; `inv type-check` shows only the 6 pre-existing errors (none new from `adapter.py`/`blender.py`).

- [ ] **Step 9: Commit**

```bash
cd /home/sriram/research/visionsim
git add visionsim/simulate/heatsim/adapter.py visionsim/simulate/blender.py tests/test_heatsim_adapter.py tests/test_heatsim_irradiance.py
git commit -m "feat(heatsim): honor blend-authored irradiance_scale (parity with addon)"
```

---

## Task 4: Parity acceptance — re-render bunny in both tools and compare

**Files:**
- Create: `tests/parity/compare_bunny_thermal.py` (host-side comparison script; not a pytest — a runnable acceptance harness)

**Interfaces:**
- Consumes: the completed Tasks 1-3 (VisionSim solves with baked albedo + irradiance_scale=1000).
- Produces: a printed parity report (peak/mean ΔT for both tools, ratio, checkerboard correlation) and a pass/fail against the ±15 % tolerance.

**Context:** This is the end-to-end acceptance from the spec (§6). It solves `bunny_textured.blend` fresh in both tools and compares the per-vertex temperature rise. heat-sim reference (measured this session): peak ΔT ≈ +1.86 K, mean ΔT ≈ +0.286 K. After the fixes VisionSim's inputs equal heat-sim's, so ΔT should land within a few percent.

- [ ] **Step 1: Produce a fresh heat-sim reference**

Delete any stale heat-sim cache and re-solve+render the bunny (aluminum, as authored). Run:

```bash
cd /home/sriram/research/heat-sim-blender/blender_files
rm -rf bunny_textured.heatsim bunny_cmp_heatsim_ref
/net/acadia2a/data/sriram/blender-fem-research/blender --background bunny_textured.blend \
  --python /home/sriram/research/heat-sim-blender/.claude/worktrees/cup-pour-render/scripts/bunny_textured/render_bunny_alum_then_pvc.py \
  -- out=/home/sriram/research/heat-sim-blender/blender_files/bunny_cmp_heatsim_ref material=alum frames=2
```
Expected: writes `bunny_textured.heatsim/latest/temperatures.npz` (the per-vertex temperature history) and To/ PNGs. Confirm the npz exists:
`ls /home/sriram/research/heat-sim-blender/blender_files/bunny_textured.heatsim/latest/temperatures.npz`

- [ ] **Step 2: Produce a fresh VisionSim solve**

Delete VisionSim's cache and solve the same blend:

```bash
cd /home/sriram/research/heat-sim-blender/blender_files
rm -rf bunny_textured.blend.heatsim
cd /home/sriram/research/visionsim
.venv/bin/vsim blender.render-animation \
  /home/sriram/research/heat-sim-blender/blender_files/bunny_textured.blend \
  out/bunny_parity --config.include-thermal \
  --config.executable /net/acadia2a/data/sriram/blender-fem-research/blender \
  --frame-start 1 --frame-end 1
```
Expected: writes `bunny_textured.blend.heatsim/<hash>/temperatures.npz`. Confirm:
`find /home/sriram/research/heat-sim-blender/blender_files/bunny_textured.blend.heatsim -name temperatures.npz`

- [ ] **Step 3: Write the comparison script**

Create `tests/parity/compare_bunny_thermal.py`:

```python
"""Compare per-vertex bunny temperature between heat-sim and VisionSim caches.

Run with the FEM blender (has numpy):
  BL=/net/acadia2a/data/sriram/blender-fem-research/blender
  "$BL" --background --python tests/parity/compare_bunny_thermal.py
"""
import glob
import numpy as np

HS = "/home/sriram/research/heat-sim-blender/blender_files/bunny_textured.heatsim/latest/temperatures.npz"
VS_GLOB = "/home/sriram/research/heat-sim-blender/blender_files/bunny_textured.blend.heatsim/**/temperatures.npz"


def load(path):
    d = np.load(path)
    key = "bunny" if "bunny" in d.files else d.files[0]
    t = d[key]  # (steps, N)
    return t[-1] - t[0]  # per-vertex ΔT over the run


def main():
    hs = load(HS)
    vs_path = sorted(glob.glob(VS_GLOB, recursive=True))[-1]
    vs = load(vs_path)
    n = min(hs.shape[0], vs.shape[0])
    hs, vs = hs[:n], vs[:n]

    hs_peak, hs_mean = float(hs.max()), float(hs.mean())
    vs_peak, vs_mean = float(vs.max()), float(vs.mean())
    peak_ratio = vs_peak / hs_peak if hs_peak else float("nan")
    mean_ratio = vs_mean / hs_mean if hs_mean else float("nan")
    corr = float(np.corrcoef(hs, vs)[0, 1]) if n > 2 else float("nan")

    print(f"heat-sim  ΔT: peak={hs_peak:.4f} mean={hs_mean:.4f}")
    print(f"visionsim ΔT: peak={vs_peak:.4f} mean={vs_mean:.4f}")
    print(f"ratio (vs/hs): peak={peak_ratio:.3f} mean={mean_ratio:.3f}")
    print(f"per-vertex ΔT correlation (checkerboard imprint): {corr:.3f}")

    peak_ok = 0.85 <= peak_ratio <= 1.15
    mean_ok = 0.85 <= mean_ratio <= 1.15
    corr_ok = corr >= 0.8
    verdict = "PASS" if (peak_ok and mean_ok and corr_ok) else "FAIL"
    print(f"PARITY_{verdict} peak_ok={peak_ok} mean_ok={mean_ok} corr_ok={corr_ok}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the comparison**

Run:
```bash
cd /home/sriram/research/visionsim
/net/acadia2a/data/sriram/blender-fem-research/blender --background \
  --python tests/parity/compare_bunny_thermal.py 2>&1 | grep -aE "heat-sim|visionsim|ratio|correlation|PARITY"
```
Expected: `PARITY_PASS` — peak and mean ratios within [0.85, 1.15] and per-vertex ΔT correlation ≥ 0.8 (checkerboard imprint present and aligned).

- [ ] **Step 5: If FAIL, diagnose (do not tune to hide a gap)**

If `PARITY_FAIL`: check which sub-check failed. `corr < 0.8` → the checkerboard is missing or misaligned (revisit Task 1/2: did the bake run? is the `albedo` attribute varying?). `peak_ratio`/`mean_ratio` far from 1 → check the effective `irradiance_scale` used (Task 3: was 1000 read?) and confirm both tools resolved the same material. Report the numbers; do not adjust tolerances or add compensating factors.

- [ ] **Step 6: Commit the acceptance harness + record results**

```bash
cd /home/sriram/research/visionsim
git add tests/parity/compare_bunny_thermal.py
git commit -m "test(heatsim): bunny thermal parity acceptance harness"
```
Record the printed parity numbers in the task ledger / final report.

---

## Task 5: Robustness — ignore a degenerate (all-zero) cached albedo

**Files:**
- Modify: `visionsim/simulate/heatsim/irradiance_kernel.py` (`get_or_bake_vertex_albedo`, the tier-1 attribute read and tier-2 disk-cache read)
- Test: `tests/test_heatsim_irradiance.py` (append)

**Interfaces:**
- Consumes: the ported bake (Task 1) via the unchanged tier-3 path.
- Produces: `get_or_bake_vertex_albedo` treats an all-zero stored/cached albedo as absent and falls through to a fresh bake, so a stale zeros cache can never shadow a real material.

**Context (discovered during Task 4 parity):** the kernel albedo disk cache lives at `<stem>.heatsim/latest/albedo_cache.npz` — a stem-based, non-content-keyed path that collides with heat-sim's cache dir. Pre-Task-2 VisionSim runs (constant albedo=0) and heat-sim's own bake wrote **all-zeros** there. After the bake was unblocked, the tier-2 disk-cache hit still served those zeros → albedo 0 → full absorption → parity failed on mean ΔT (1.8×) until the stale cache was manually cleared. An all-zero albedo is the degenerate "fully absorbing" fallback, never a meaningful bake result — so treat it as a re-bake trigger. This is a cache-correctness fix in the albedo-resolution logic, NOT a change to irradiance physics (the Global Constraint bars physics changes, which this is not).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_heatsim_irradiance.py`:

```python
def test_all_zero_cached_albedo_is_ignored_and_rebaked(executable):
    code = r"""
import bpy, numpy as np
from visionsim.simulate.heatsim import register, irradiance_kernel
register()
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()
bpy.ops.mesh.primitive_grid_add(x_subdivisions=12, y_subdivisions=12, size=2.0)
obj = bpy.context.active_object
mat = bpy.data.materials.new('checker'); mat.use_nodes = True
nt = mat.node_tree; bsdf = nt.nodes.get('Principled BSDF')
ck = nt.nodes.new('ShaderNodeTexChecker'); ck.inputs['Scale'].default_value = 6.0
nt.links.new(ck.outputs['Color'], bsdf.inputs['Base Color'])
obj.data.materials.append(mat)
bpy.context.scene.render.engine = 'CYCLES'
try: bpy.context.scene.cycles.device = 'CPU'; bpy.context.scene.cycles.samples = 4
except Exception: pass
nv = len(obj.data.vertices)
# Stale all-zeros disk cache for this object must NOT be served; must re-bake.
amap = irradiance_kernel.get_or_bake_vertex_albedo(
    bpy.context.scene, [obj], texture_size=128,
    disk_cache={obj.name: np.zeros(nv, dtype=np.float64)})
alb = amap.get(obj.name)
assert alb is not None, 'albedo absent'
assert float(alb.std()) > 0.05, f'zeros cache was served instead of re-baking (std={alb.std()})'
print('ZERO_CACHE_IGNORED_OK', round(float(alb.mean()),3), round(float(alb.std()),3))
"""
    import subprocess
    out = subprocess.run([str(executable), "-b", "--python-expr", code],
                         capture_output=True, text=True)
    assert "ZERO_CACHE_IGNORED_OK" in out.stdout, out.stdout + "\n" + out.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sriram/research/visionsim && pytest tests/test_heatsim_irradiance.py::test_all_zero_cached_albedo_is_ignored_and_rebaked -v --executable /net/acadia2a/data/sriram/blender-fem-research/blender`
Expected: FAIL — the all-zeros disk cache is served (`std == 0`), so the assertion fails and `ZERO_CACHE_IGNORED_OK` is absent.

- [ ] **Step 3: Add the all-zero guard to `get_or_bake_vertex_albedo`**

In `visionsim/simulate/heatsim/irradiance_kernel.py`, in the per-object loop of `get_or_bake_vertex_albedo`:

Tier 1 (existing mesh attribute) — change:
```python
        vals = _read_vertex_albedo_attr(obj, attr_name)
        if vals is not None:
            out[obj.name] = np.clip(vals, 0.0, 1.0)
            continue
```
to:
```python
        vals = _read_vertex_albedo_attr(obj, attr_name)
        # An all-zero albedo is the degenerate "fully absorbing" fallback (and the
        # sentinel a stale/cross-tool cache leaves behind), never a real bake — so
        # ignore it and fall through to a fresh bake.
        if vals is not None and float(np.max(vals)) > 0.0:
            out[obj.name] = np.clip(vals, 0.0, 1.0)
            continue
```

Tier 2 (disk cache) — change:
```python
        cached = disk_cache.get(obj.name)
        if cached is not None and int(cached.shape[0]) == len(obj.data.vertices):
```
to:
```python
        cached = disk_cache.get(obj.name)
        if (cached is not None
                and int(cached.shape[0]) == len(obj.data.vertices)
                and float(np.max(np.asarray(cached))) > 0.0):
```

Leave the tier-3 bake path unchanged.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/sriram/research/visionsim && pytest tests/test_heatsim_irradiance.py::test_all_zero_cached_albedo_is_ignored_and_rebaked -v --executable /net/acadia2a/data/sriram/blender-fem-research/blender`
Expected: PASS — stdout contains `ZERO_CACHE_IGNORED_OK <mean> <std>` with std > 0.05.

- [ ] **Step 5: Run the full heatsim-irradiance test file (no regression)**

Run: `cd /home/sriram/research/visionsim && pytest tests/test_heatsim_irradiance.py -v --executable /net/acadia2a/data/sriram/blender-fem-research/blender`
Expected: all tests PASS.

- [ ] **Step 6: Lint + type-check gate**

`irradiance_kernel.py` is in the mypy/ruff exclude list (vendored), but run the whole-tree gate to confirm nothing regressed. Run: `cd /home/sriram/research/visionsim && PATH="$PWD/.venv/bin:$PATH" .venv/bin/inv lint && PATH="$PWD/.venv/bin:$PATH" .venv/bin/inv type-check`
Expected: lint clean; type-check only the 6 pre-existing errors.

- [ ] **Step 7: Commit**

```bash
cd /home/sriram/research/visionsim
git add visionsim/simulate/heatsim/irradiance_kernel.py tests/test_heatsim_irradiance.py
git commit -m "fix(heatsim): ignore degenerate all-zero cached albedo, re-bake instead"
```

## Out of Scope (per spec §5)

- The optional ambient/initial-temperature alignment (heat-sim 295.372 K vs VisionSim 295.0 K, a 0.372 K constant offset) is **not** implemented — it does not affect ΔT parity. Only revisit if absolute-Kelvin parity is later required.
- No changes to the FEM solver, unit scaling, direct irradiance kernel, or the Cycles irradiance bake.

---

## Self-Review

**Spec coverage:**
- §4.1 (port `bake_albedo_map` + unblock pre-stamp) → Tasks 1 & 2. ✓
- §4.2 (honor authored `irradiance_scale`) → Task 3 (adjusted: raw ID-property read, since `scene.heat_sim_settings` is not registered in VisionSim — verified empirically). ✓
- §4.3 (optional ambient alignment) → explicitly out of scope. ✓
- §6 (parity verification: ±15 % magnitude + checkerboard correlation) → Task 4. ✓

**Placeholder scan:** No TBD/TODO. All test code and edit targets are concrete with exact line references and exact strings. Port bodies are cited by exact source file + line ranges (verbatim copy, not paraphrase). ✓

**Type/name consistency:** `bake_albedo_map(scene, obj, texture_size) -> Optional[BakedFluxMap]` with `.pixels (H,W,3)` used consistently in Tasks 1/2. `read_authored_irradiance_scale(scene) -> Optional[float]` consistent in Task 3. `ALBEDO_LAYER_NAME = "HeatSim_Albedo"` / `BAKE_UV_LAYER_NAME = "HeatSim_Bake_UV"` exact. ✓
