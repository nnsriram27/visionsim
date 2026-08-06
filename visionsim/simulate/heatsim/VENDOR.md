# Vendored files

Source repository: `heat-sim-blender` (sibling repo at `/home/sriram/research/heat-sim-blender`)
Source commit: `543ee81` (`543ee814488742eb6147e2296d0d29ce385f97d2`)

## Files

| Destination | Upstream path | Modifications |
|---|---|---|
| `visionsim/simulate/heatsim/solver.py` | `addon/lib/heatsim_fem.py` | See below |
| `visionsim/simulate/heatsim/laplacian.py` | `addon/lib/robust_laplacian_backend.py` | See below |
| `visionsim/simulate/heatsim/constants.py` | `addon/lib/constants.py` | See below |
| `visionsim/simulate/heatsim/irradiance_kernel.py` | `addon/lib/irradiance_kernel.py` | See below |
| `visionsim/simulate/heatsim/light_models.py` | `addon/lib/light_models.py` | See below |
| `visionsim/simulate/heatsim/sh9_sky.py` | `addon/lib/sh9_sky.py` | See below |
| `visionsim/simulate/heatsim/sky_visibility.py` | `addon/lib/sky_visibility.py` | See below |
| `visionsim/simulate/heatsim/bvh_backend.py` | `addon/lib/bvh_backend.py` | See below |
| `visionsim/simulate/heatsim/temperature_io.py` | `addon/lib/temperature_io.py` | See below |
| `visionsim/simulate/heatsim/irradiance.py` | `addon/lib/irradiance.py` | See below |
| `visionsim/simulate/heatsim/uv_utils.py` | `addon/lib/irradiance.py` (UV helpers) | See below |

## Lint / type-check exclusions

The rule is provenance-based: a file keeps the exemption only while it stays close to
upstream. Once we have substantially rewritten one, it is ours to maintain and gets linted
like the rest of the codebase.

**Excluded from ruff and mypy** (near-verbatim upstream; linting them would force edits that
diverge from the source): `solver.py`, `laplacian.py`, `constants.py`, `light_models.py`,
`sh9_sky.py`, `sky_visibility.py`, `bvh_backend.py`, `temperature_io.py`, `uv_utils.py`.

**Linted by ruff, excluded from mypy only** (substantially modified here; mypy cannot check
them because they import `bpy`/`mathutils` unconditionally and only run inside Blender):

| File | Drift since vendoring |
|---|---|
| `irradiance_kernel.py` | +263 / -18 over 7 commits |
| `irradiance.py` | +71 / -19 over 3 commits |

`properties.py` is hand-written glue (NOT vendored); it is ruff-linted but excluded from mypy
because bpy-dynamic PropertyGroup annotations cannot be type-checked outside Blender even with
the host-safe try/except import pattern.

### Original code must not accumulate in vendored files

Logic that is ours rather than upstream's belongs in a non-vendored module, so the provenance
claim above stays true and the code gets linted. Precedent: the shortwave-occluder material
classification (`casts_shadow`, `material_is_clear`, `surface_shader_nodes`) was written here,
briefly lived in `irradiance_kernel.py`, and now lives in `occluders.py`, which the kernel
imports. Follow that pattern for anything new.

## Modifications applied to each file

### solver.py (from heatsim_fem.py)

1. **Provenance header** added as first line: `# Vendored from heat-sim-blender:addon/lib/heatsim_fem.py @ 543ee81`
2. **Import rewrite** — intra-package relative imports replaced with absolute visionsim paths:
   - `from . import constants` → `from visionsim.simulate.heatsim import constants`
   - `from .robust_laplacian_backend import (HAS_ROBUST_LAPLACIAN, ROBUST_IMPORT_ERROR, mesh_laplacian_and_mass, point_cloud_laplacian_and_mass,)` → `from visionsim.simulate.heatsim.laplacian import (...)`
3. **Logger added**: `import logging` added to imports; `_log = logging.getLogger("rich")` added after imports.
4. **All unconditional `print(...)` calls replaced with `_log.debug(...)`** — including debug ranges in `_build_boundary_and_sources`, simulation step progress in `_simulate_heat_torch`, steady-state convergence messages, Laplacian/mass matrix build messages in `_build_matrices`, and diagnostic prints in `perform_gt_heat_simulation` and `run_heat_simulation`.

### laplacian.py (from robust_laplacian_backend.py)

1. **Provenance header** added as first line: `# Vendored from heat-sim-blender:addon/lib/robust_laplacian_backend.py @ 543ee81`
2. No import rewrites required (file has no local imports).
3. No print statements to route.
4. **Clamp `n_neighbors` to `len(points)-1`** in `point_cloud_laplacian_and_mass` (defensive, matches scipy fallback; prevents robust_laplacian crash on small point clouds).

### constants.py (from constants.py)

1. **Provenance header** added as first line: `# Vendored from heat-sim-blender:addon/lib/constants.py @ 543ee81`
2. No import rewrites required (file has no local imports).
3. No print statements to route.

### irradiance_kernel.py (from irradiance_kernel.py)

1. **Provenance header** added as first line.
2. **Import rewrite**: `from . import bvh_backend, light_models, sh9_sky, sky_visibility` → `from visionsim.simulate.heatsim import bvh_backend, light_models, sh9_sky, sky_visibility`; lazy imports of `irradiance` and `temperature_io` inside functions rewritten to `from visionsim.simulate.heatsim import irradiance` / `temperature_io` (runtime dependencies — see NEEDS_CONTEXT note below).
3. **Logger added**: `import logging`, `_log = logging.getLogger("rich")`; all unconditional `print(...)` → `_log.debug(...)`.

### light_models.py (from light_models.py)

1. **Provenance header** added as first line.
2. No intra-package imports to rewrite.
3. No print statements to route.

### sh9_sky.py (from sh9_sky.py)

1. **Provenance header** added as first line.
2. No intra-package imports to rewrite.
3. **Logger added**: `import logging`, `_log = logging.getLogger("rich")`; all unconditional `print(...)` → `_log.debug(...)`.

### sky_visibility.py (from sky_visibility.py)

1. **Provenance header** added as first line.
2. **Import rewrite**: lazy `from . import irradiance_kernel` → `from visionsim.simulate.heatsim import irradiance_kernel`; lazy `from . import temperature_io` → `from visionsim.simulate.heatsim import temperature_io` (runtime dependency — see NEEDS_CONTEXT note below).
3. **Logger added**: `import logging`, `_log = logging.getLogger("rich")`; all unconditional `print(...)` → `_log.debug(...)`.

### bvh_backend.py (from bvh_backend.py)

1. **Provenance header** added as first line.
2. No intra-package imports to rewrite (embreex and mathutils are external).
3. **Logger added**: `import logging`, `_log = logging.getLogger("rich")`; single `print(...)` → `_log.debug(...)`.

### temperature_io.py (from temperature_io.py)

1. **Provenance header** added as first line: `# Vendored from heat-sim-blender:addon/lib/temperature_io.py @ 543ee81`
2. No import rewrites required (file has no intra-package local imports; only stdlib, `bpy`, `numpy`).
3. **Logger added**: `import logging`, `_log = logging.getLogger("rich")`; all unconditional `print(...)` → `_log.debug(...)`.

## NEEDS_CONTEXT runtime dependencies

`irradiance_kernel.py` and `sky_visibility.py` contain lazy function-scoped imports of two addon modules NOT in the vendor set:

- `temperature_io` — disk caching of albedo and sky-visibility data. Used in `compute_per_vertex_irradiance()` and `get_or_bake_for_objects()`. **Vendored** (this task).
- `irradiance` — Cycles albedo bake. Used in `_bake_vertex_albedo_via_cycles()`. Explicitly called out as acceptable ("Their albedo step may call Cycles once") in the task brief — this is intentional design.

Both are lazy imports (inside functions, not at module top-level) so they do not prevent the modules from being imported; they only fail at runtime when those specific code paths are exercised.

### irradiance.py (from irradiance.py)

1. **Provenance header** in the module docstring.
2. **Logger added**; prints routed to `_log`.
3. **Headless-bake robustness** (ours, not upstream): zero-geometry meshes are skipped, an
   OBJECT-mode context is forced before selection operators, and the UV/mode restore is
   guarded so one object's unwrap failure cannot abort a whole-scene bake.
4. UV snapshot/restore helpers extracted to `uv_utils.py`.

### uv_utils.py (from irradiance.py)

1. **Provenance header** in the module docstring; UV state snapshot/restore helpers lifted
   verbatim out of the upstream `irradiance.py` so both bake paths can share them.
