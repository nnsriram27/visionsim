# Vendored files

Source repository: `heat-sim-blender` (sibling repo at `/home/sriram/research/heat-sim-blender`)
Source commit: `543ee81` (`543ee814488742eb6147e2296d0d29ce385f97d2`)

## Files

| Destination | Upstream path | Modifications |
|---|---|---|
| `visionsim/simulate/heatsim/solver.py` | `addon/lib/heatsim_fem.py` | See below |
| `visionsim/simulate/heatsim/laplacian.py` | `addon/lib/robust_laplacian_backend.py` | See below |
| `visionsim/simulate/heatsim/constants.py` | `addon/lib/constants.py` | See below |

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

### constants.py (from constants.py)

1. **Provenance header** added as first line: `# Vendored from heat-sim-blender:addon/lib/constants.py @ 543ee81`
2. No import rewrites required (file has no local imports).
3. No print statements to route.
