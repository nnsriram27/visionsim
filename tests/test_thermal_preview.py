"""Tests for the thermal preview colormap: the global temperature range helper
and the inferno-stop compositor node group (see
docs/superpowers/specs/2026-07-07-visionsim-thermal-parity-design.md §9).

Both run inside ``blender --background`` (adapter imports bpy; the node group
builds a compositor tree), matching the pattern in ``test_heatsim_irradiance``.
"""

import subprocess


def test_global_temperature_range_spans_final_timestep_and_floors(executable):
    code = r"""
import numpy as np
from visionsim.simulate.heatsim import adapter

# final rows: bunny [296, 297.5], plane [295.2, 295.1]
hist = {
    'bunny': np.array([[295.0, 295.0], [296.0, 297.5]]),
    'plane': np.array([[295.0, 295.0], [295.2, 295.1]]),
}
tmin, tmax = adapter.global_temperature_range(hist, default_K=295.0)
assert abs(tmin - 295.0) < 1e-9, tmin       # floored at default (finals min 295.1 > 295)
assert abs(tmax - 297.5) < 1e-9, tmax       # hottest final-timestep vertex, not 295-400

# empty history -> (default, default + 1)
lo, hi = adapter.global_temperature_range({}, 295.0)
assert (lo, hi) == (295.0, 296.0), (lo, hi)

# near-uniform field -> span widened to >= 1 K (no colour collapse)
lo, hi = adapter.global_temperature_range({'o': np.full((2, 4), 300.0)}, 300.0)
assert lo == 300.0 and (hi - lo) >= 1.0, (lo, hi)

print('RANGE_OK', round(tmin, 3), round(tmax, 3))
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True)
    assert "RANGE_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_thermal_preview_node_group_uses_inferno_stops_and_range(executable):
    code = r"""
import bpy  # noqa: F401
from visionsim.simulate.nodes import thermal_preview_node_group

ng = thermal_preview_node_group(tmin=295.0, tmax=297.0)

# MapRange wired to [tmin, tmax] -> [0, 1]
mr = ng.nodes['TempNormalize']
assert abs(mr.inputs[1].default_value - 295.0) < 1e-6, mr.inputs[1].default_value
assert abs(mr.inputs[2].default_value - 297.0) < 1e-6, mr.inputs[2].default_value

# Inferno ramp: 11 stops from near-black to pale yellow (heat-sim _INFERNO_STOPS)
ramp = ng.nodes['InfernoRamp'].color_ramp
assert len(ramp.elements) == 11, len(ramp.elements)
lo, hi = ramp.elements[0], ramp.elements[10]
assert abs(lo.position - 0.0) < 1e-6 and abs(hi.position - 1.0) < 1e-6
assert lo.color[0] < 0.02 and lo.color[2] < 0.02, tuple(lo.color)   # near-black low end
assert hi.color[0] > 0.9 and hi.color[1] > 0.9, tuple(hi.color)     # pale-yellow high end

# Regression guard: the old turbo low stop was blue-ish (0.190, 0.072, 0.232)
assert abs(lo.color[0] - 0.18995) > 1e-3, 'colormap is still turbo, not inferno'

print('INFERNO_NODEGROUP_OK')
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True)
    assert "INFERNO_NODEGROUP_OK" in out.stdout, out.stdout + "\n" + out.stderr
