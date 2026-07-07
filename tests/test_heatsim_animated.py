from __future__ import annotations

import subprocess

# ---------------------------------------------------------------------------
# Shared synthetic-scene builder: a POINTS-domain FEM plate sitting just under
# a DIRICHLET_SOURCE box. The box's bottom face footprint matches the plate's
# extent and sits a few mm above it, so cross-object point pairs are *closer*
# than same-object neighbor spacing -- guaranteeing the point-cloud kNN
# Laplacian links plate points to the hot reservoir (see adapter._combine /
# solver._build_matrices POINTS branch: the Laplacian is a spatial graph over
# ALL combined points, irrespective of which mesh they came from).
# ---------------------------------------------------------------------------
_SCENE_SETUP = r"""
import bpy
from visionsim.simulate.heatsim import register
register()

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Plate: FEM participant, stable topology, starts at (just under) ambient.
bpy.ops.mesh.primitive_grid_add(x_subdivisions=6, y_subdivisions=6, size=0.4)
plate = bpy.context.active_object
plate.name = 'Plate'
plate.heat_simulation_enabled = True
plate.heat_sim_material.initial_temperature_K = 295.0
plate.heat_sim_material.thermal_diffusivity_mm2_s = 50.0

# Box: Dirichlet reservoir at 369 K, bottom face directly above the plate.
bpy.ops.mesh.primitive_cube_add(size=0.4, location=(0.0, 0.0, 0.205))
box = bpy.context.active_object
box.name = 'Box'
box.heat_simulation_enabled = True
box.heat_sim_material.thermal_role = 'DIRICHLET_SOURCE'
box.heat_sim_material.dirichlet_temperature_K = 369.0
"""

_DEFAULTS = """
defaults = dict(initial_temperature_K=295.0, thermal_diffusivity_mm2_s=0.17,
                density_kg_m3=1330.0, specific_heat_J_kgK=880.0, emissivity=0.9,
                irradiance_scale=100.0)
solver_cfg = dict(domain='POINTS', laplacian_backend='ROBUST', device='cpu')
"""


def test_animated_solve_heats_plate_toward_dirichlet_source(executable, tmp_path):
    code = (
        _SCENE_SETUP
        + _DEFAULTS
        + """
import numpy as np
from pathlib import Path
from visionsim.simulate.heatsim import adapter

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 5

history, frames = adapter.solve_scene_animated(
    bpy.context.scene, defaults=defaults, solver_cfg=solver_cfg,
    cache_root=Path(r'{tmp_path}'),
    frame_start=1, frame_end=5, every_n=1, substeps_per_frame=4,
)

assert frames == [1, 2, 3, 4, 5], frames
assert 'Plate' in history, list(history.keys())
assert 'Box' not in history, 'Dirichlet source must not be recorded'

plate_hist = np.asarray(history['Plate'])
n_plate_verts = len(plate.data.vertices)
assert plate_hist.shape == (5, n_plate_verts), plate_hist.shape

mean_frame1 = float(plate_hist[0].mean())
mean_frame5 = float(plate_hist[-1].mean())
assert np.isfinite(plate_hist).all(), 'non-finite temperatures in plate history'
assert mean_frame5 > mean_frame1, (mean_frame1, mean_frame5)

print('ANIMATED_SOLVE_OK', round(mean_frame1, 4), round(mean_frame5, 4))
"""
    ).replace("{tmp_path}", str(tmp_path))
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True)
    assert "ANIMATED_SOLVE_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_animated_solve_survives_dirichlet_vertex_count_change(executable, tmp_path):
    """The box gains a Subdivision Surface modifier whose level is keyframed to
    step up mid-run, so its *evaluated* vertex count changes between frames.
    This must not crash, and the plate's (FEM-participant, stable-topology)
    history must stay a consistent ``(n_frames, n_plate_verts)`` shape."""
    code = (
        _SCENE_SETUP
        + """
mod = box.modifiers.new('Subsurf', 'SUBSURF')
mod.levels = 0
mod.keyframe_insert(data_path='levels', frame=1)
mod.levels = 3
mod.keyframe_insert(data_path='levels', frame=3)
"""
        + _DEFAULTS
        + """
import numpy as np
from pathlib import Path
from visionsim.simulate.heatsim import adapter

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 5

history, frames = adapter.solve_scene_animated(
    bpy.context.scene, defaults=defaults, solver_cfg=solver_cfg,
    cache_root=Path(r'{tmp_path}'),
    frame_start=1, frame_end=5, every_n=1, substeps_per_frame=4,
)

n_plate_verts = len(plate.data.vertices)
plate_hist = np.asarray(history['Plate'])
assert plate_hist.shape == (5, n_plate_verts), plate_hist.shape
assert np.isfinite(plate_hist).all(), 'non-finite temperatures after a topology change'

print('ANIMATED_RESIZE_OK', plate_hist.shape)
"""
    ).replace("{tmp_path}", str(tmp_path))
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True)
    assert "ANIMATED_RESIZE_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_write_frame_attributes_dirichlet_fallback_uses_reservoir_temperature(executable, tmp_path):
    """When ``write_frame_attributes`` hits its fallback branch (object absent
    from ``history``, e.g. a topology-changing Dirichlet liquid whose evaluated
    vertex count no longer matches), a ``DIRICHLET_SOURCE`` object must stamp
    its reservoir temperature (``dirichlet_temperature_K``), not the ambient
    ``initial_temperature_K`` default. A ``FEM_PARTICIPANT`` hitting the same
    fallback keeps stamping the ambient default (unchanged behavior)."""
    code = (
        _SCENE_SETUP
        + _DEFAULTS
        + """
from visionsim.simulate.heatsim import adapter

# Empty history => both Plate (FEM_PARTICIPANT) and Box (DIRICHLET_SOURCE)
# hit the fallback branch.
adapter.write_frame_attributes(bpy.context.scene, {}, -1, defaults)

assert box['heatsim_default_temperature'] == 369.0, box['heatsim_default_temperature']
assert plate['heatsim_default_temperature'] == 295.0, plate['heatsim_default_temperature']

print('DIRICHLET_FALLBACK_OK')
"""
    )
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True)
    assert "DIRICHLET_FALLBACK_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_thermal_write_frame_advances_animated_field(executable, tmp_path):
    """Task 5 integration test: drive the real ``BlenderService`` methods end-to-end.

    After an animated ``exposed_prepare_thermal`` solve, ``exposed_set_current_frame`` +
    ``_thermal_write_frame`` (the per-frame render hook used by ``exposed_render_frame``)
    must write an increasingly hot ``sim_temperature`` onto the plate as later frames are
    requested, since the transient field keeps evolving toward the Dirichlet reservoir.
    The Dirichlet box itself is absent from the animated history, so it must still hit
    the Task 4 reservoir-temperature fallback rather than ambient.
    """
    code = (
        _SCENE_SETUP
        + r"""
import bpy
import numpy as np
from visionsim.simulate.blender import BlenderService

blend_path = r'{tmp_path}/animated_test.blend'
root_path = r'{tmp_path}'
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

service = BlenderService()
service.exposed_initialize(blend_path, root_path)

service.exposed_prepare_thermal(
    animated=True,
    domain='POINTS',
    laplacian_backend='ROBUST',
    device='cpu',
    frame_start=1,
    frame_end=5,
    substeps_per_frame=4,
    every_n_frames=1,
    initial_temperature_K=295.0,
    thermal_diffusivity_mm2_s=0.17,
    density_kg_m3=1330.0,
    specific_heat_J_kgK=880.0,
    emissivity=0.9,
    irradiance_scale=100.0,
)

assert service._thermal_animated_history is not None, 'animated history was not stored on the service'
assert service._thermal_animated_frames == [1, 2, 3, 4, 5], service._thermal_animated_frames
assert 'Plate' in service._thermal_animated_history, list(service._thermal_animated_history.keys())

plate_obj = bpy.data.objects['Plate']


def plate_mean():
    attr = plate_obj.data.attributes['sim_temperature']
    vals = np.zeros(len(attr.data), dtype=np.float32)
    attr.data.foreach_get('value', vals)
    return float(vals.mean())


service.exposed_set_current_frame(1)
service._thermal_write_frame(1)
mean_early = plate_mean()

service.exposed_set_current_frame(5)
service._thermal_write_frame(5)
mean_late = plate_mean()

assert mean_late > mean_early, (mean_early, mean_late)

# The Dirichlet box has no per-frame history entry, so the write must fall
# through to the reservoir-temperature fallback (Task 4), not ambient.
box_obj = bpy.data.objects['Box']
assert box_obj['heatsim_default_temperature'] == 369.0, box_obj['heatsim_default_temperature']

print('ANIMATED_RENDER_WRITE_OK', round(mean_early, 4), round(mean_late, 4))
"""
    ).replace("{tmp_path}", str(tmp_path))
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True)
    assert "ANIMATED_RENDER_WRITE_OK" in out.stdout, out.stdout + "\n" + out.stderr
