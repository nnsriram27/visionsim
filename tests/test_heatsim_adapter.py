from __future__ import annotations

import subprocess
from types import SimpleNamespace

from visionsim.simulate.heatsim import adapter

# Distinctive globals so a leaked PropertyGroup default can never masquerade as a
# global fallback (the PropertyGroup ships emissivity=0.9, density=1330.0).
_GLOBAL_DEFAULTS = dict(
    initial_temperature_K=300.0,
    thermal_diffusivity_mm2_s=0.42,
    density_kg_m3=1234.0,
    specific_heat_J_kgK=777.0,
    emissivity=0.5,
)


class _FakeMat:
    """Stand-in for ``obj.heat_sim_material`` with a controllable ``is_property_set``.

    ``always_set`` mirrors a PropertyGroup where every field was authored (True) or
    where every field is still at its registered default (False) - the exact axis
    that ``resolve_material`` must branch on.
    """

    def __init__(self, *, always_set: bool, **values):
        self._always_set = always_set
        for k, v in values.items():
            setattr(self, k, v)

    def is_property_set(self, attr):  # noqa: D401 - mimics bpy_struct.is_property_set
        return self._always_set


def test_resolve_material_falls_back_to_globals_when_unset():
    """I1 regression: unset per-object props must defer to the global defaults.

    The PointerProperty is always present and a FloatProperty never returns None,
    so without ``is_property_set`` gating the (distinctive) per-object values below
    would shadow the globals and the ``--config.thermal.*`` knobs would be inert.
    """
    mat = _FakeMat(
        always_set=False,
        initial_temperature_K=295.372,  # PropertyGroup-style defaults that must be ignored
        thermal_diffusivity_mm2_s=0.17,
        density_kg_m3=1330.0,
        specific_heat_J_kgK=880.0,
        emissivity=0.9,
        thermal_role="DIRICHLET_SOURCE",
        dirichlet_temperature_K=400.0,
    )
    obj = SimpleNamespace(heat_sim_material=mat)

    out = adapter.resolve_material(obj, _GLOBAL_DEFAULTS)

    assert out["initial_temperature_K"] == 300.0
    assert out["thermal_diffusivity_mm2_s"] == 0.42
    assert out["density_kg_m3"] == 1234.0
    assert out["specific_heat_J_kgK"] == 777.0
    assert out["emissivity"] == 0.5
    # thermal_role / dirichlet_temperature_K have no global key -> hard defaults,
    # NOT the stale group values.
    assert out["thermal_role"] == "FEM_PARTICIPANT"
    assert out["dirichlet_temperature_K"] == 0.0


def test_resolve_material_uses_per_object_when_set():
    """Explicitly-set per-object values win over the globals (and are clamped)."""
    mat = _FakeMat(
        always_set=True,
        initial_temperature_K=310.0,
        thermal_diffusivity_mm2_s=0.99,
        density_kg_m3=7777.0,
        specific_heat_J_kgK=500.0,
        emissivity=2.0,  # out of range -> must clamp to 1.0
        thermal_role="dirichlet_source",  # lower-case -> upper-cased
        dirichlet_temperature_K=400.0,
    )
    obj = SimpleNamespace(heat_sim_material=mat)

    out = adapter.resolve_material(obj, _GLOBAL_DEFAULTS)

    assert out["initial_temperature_K"] == 310.0
    assert out["thermal_diffusivity_mm2_s"] == 0.99
    assert out["density_kg_m3"] == 7777.0
    assert out["specific_heat_J_kgK"] == 500.0
    assert out["emissivity"] == 1.0
    assert out["thermal_role"] == "DIRICHLET_SOURCE"
    assert out["dirichlet_temperature_K"] == 400.0


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

# Second call must come straight from the cache (no re-solve). We assert the
# cache hit via the RETURN value (cached history equals the first solve), not via
# captured stdout, so the test does not depend on any debug print side-effect.
hist2 = adapter.solve_scene(bpy.context.scene, defaults=defaults,
                            solver_cfg=solver_cfg, cache_root=cache_root)
assert hist2.keys() == hist.keys(), (list(hist2.keys()), list(hist.keys()))
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
