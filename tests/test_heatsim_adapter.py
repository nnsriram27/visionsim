from __future__ import annotations

import subprocess


def test_solve_writes_finite_sim_temperature(executable, tmp_path):
    """End-to-end adapter smoke test inside a real Blender process.

    Builds a tiny lit scene (subdivided plane + overhead sun + a world with
    some background light), runs the cached FEM solve via the adapter, writes
    the last-timestep ``sim_temperature`` attribute, and asserts the result is
    finite and physical. Also checks that the Direct-Kernel irradiance actually
    produced non-zero per-vertex flux (so the test cannot pass with a silent
    zero-flux fallback) and that a second ``solve_scene`` reuses the cache.
    """
    code = f"""
import bpy, numpy as np
from pathlib import Path
from visionsim.simulate.heatsim import register, adapter

register()

# --- start from a clean scene -------------------------------------------------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Subdivided plane (grid) -> enough points for the ROBUST point-cloud Laplacian.
bpy.ops.mesh.primitive_grid_add(x_subdivisions=15, y_subdivisions=15, size=2.0)
plane = bpy.context.active_object
plane.name = 'ThermalPlane'
plane.heat_simulation_enabled = True

# Overhead sun (default rotation emits along -Z, i.e. straight down).
bpy.ops.object.light_add(type='SUN')
sun = bpy.context.active_object
sun.data.energy = 10.0

# Give the world some light so the sky term is non-zero too.
world = bpy.context.scene.world
if world is None:
    world = bpy.data.worlds.new('World')
    bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get('Background')
if bg is not None:
    bg.inputs['Color'].default_value = (0.2, 0.2, 0.2, 1.0)
    bg.inputs['Strength'].default_value = 1.0

defaults = dict(initial_temperature_K=295.0, thermal_diffusivity_mm2_s=0.17,
                density_kg_m3=1330.0, specific_heat_J_kgK=880.0, emissivity=0.9,
                irradiance_scale=100.0)
solver_cfg = dict(sim_time_s=0.15, timestep_s=0.05, domain='POINTS',
                  laplacian_backend='ROBUST', device='cpu')
cache_root = Path(r'{tmp_path}')

hist = adapter.solve_scene(bpy.context.scene, defaults=defaults,
                           solver_cfg=solver_cfg, cache_root=cache_root)
assert 'ThermalPlane' in hist, list(hist.keys())
T_hist = np.asarray(hist['ThermalPlane'])
assert T_hist.ndim == 2 and T_hist.shape[0] >= 2, T_hist.shape

# Second call must come straight from the cache (no re-solve).
hist2 = adapter.solve_scene(bpy.context.scene, defaults=defaults,
                            solver_cfg=solver_cfg, cache_root=cache_root)
assert np.array_equal(T_hist, np.asarray(hist2['ThermalPlane']))

adapter.write_frame_attributes(bpy.context.scene, hist, timestep=-1, defaults=defaults)

# sim_temperature: finite + physical.
attr = plane.data.attributes['sim_temperature'].data
vals = np.array([d.value for d in attr])
assert np.isfinite(vals).all(), 'non-finite temperatures'
assert vals.min() > 200 and vals.max() < 2000, (float(vals.min()), float(vals.max()))

# emissivity attribute written.
eps = np.array([d.value for d in plane.data.attributes['emissivity'].data])
assert np.allclose(eps, 0.9), float(eps.mean())

# The Direct-Kernel irradiance pass produced real (non-zero) flux.
irr = np.array([d.value for d in plane.data.attributes['sim_irradiance'].data])
assert np.isfinite(irr).all() and irr.max() > 0.0, float(irr.max())

print('THERMAL_ADAPTER_OK')
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True)
    assert "THERMAL_ADAPTER_OK" in out.stdout, out.stderr
    assert "cache hit" in out.stdout, "second solve_scene did not reuse the cache:\n" + out.stdout
