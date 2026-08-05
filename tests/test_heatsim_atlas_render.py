"""Tests for Task 3: the atlas EXR writer (``adapter.write_atlas``) and the render-time
plumbing that loads/packs it. Both node-graph shader tests and the
``render_domain="TEXEL"`` config-dispatch parity tests live in sibling files
(``tests/test_heatsim_shader.py``, ``tests/test_heatsim_config.py``); this file covers
the writer itself and the ``write_frame_attributes``/``global_temperature_range`` TEXEL
behavior specified in the Task 3 brief.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from visionsim.simulate.heatsim import adapter, atlas


def _tile_layout(atlas_size, tiles):
    return atlas.AtlasLayout(atlas_size=atlas_size, tiles=tiles, effective_density=500.0, rescaled=False)


def _plan(atlas_size, tiles, texels, digest="testdigest"):
    return adapter.AtlasPlan(layout=_tile_layout(atlas_size, tiles), texels=texels, digest=digest)


# ---------------------------------------------------------------------------
# write_atlas: needs bpy (image creation/save), run inside Blender.
# ---------------------------------------------------------------------------


def test_write_atlas_scatters_dilates_and_marks_alpha(executable, tmp_path):
    code = f"""
import numpy as np
from pathlib import Path
from visionsim.simulate.heatsim import adapter, atlas

# One object, one 6x6 tile at atlas offset (0,0). Two solved texels at (1,1) and (4,4)
# (far apart so their dilation margins don't overlap and mask each other's zeros).
tile = atlas.TileSpec(obj_name='obj', size=(6, 6), offset=(0, 0))
layout = atlas.AtlasLayout(atlas_size=(6, 6), tiles={{'obj': tile}}, effective_density=500.0, rescaled=False)
texels = {{'obj': {{'xy': np.array([[1, 1], [4, 4]], dtype=np.int64)}}}}
plan = adapter.AtlasPlan(layout=layout, texels=texels, digest='rt')

history = {{'obj': np.array([[300.0, 300.0], [310.0, 320.0]])}}  # (T=2, K=2); final row used

out_path = adapter.write_atlas(history, plan, Path(r'{tmp_path}'))
assert out_path.exists(), out_path
assert out_path.name == 'atlas_temperature.exr'
assert 'rt' in str(out_path.parent.name)

import bpy
img = bpy.data.images.load(str(out_path))
w, h = img.size
assert (w, h) == (6, 6), (w, h)
px = np.array(img.pixels[:], dtype=np.float64).reshape(h, w, 4)

# Scattered texels: match (within EXR write/load round-trip precision -- Blender's
# Image.save() has no exposed lossless-compression knob for a plain generated
# image, so a few mK of drift is expected and harmless for a Kelvin-scale field)
# at (x=1,y=1) -> 310.0 and (x=4,y=4) -> 320.0, alpha=1.
assert abs(px[1, 1, 0] - 310.0) < 0.05, px[1, 1]
assert abs(px[4, 4, 0] - 320.0) < 0.05, px[4, 4]
assert px[1, 1, 3] == 1.0
assert px[4, 4, 3] == 1.0

# A direct 8-neighbor of a solved texel is inside the dilation margin: alpha=1 (valid
# coverage per the dilated mask) and a nonzero temperature pulled from that neighbor
# (never the raw zero an un-dilated scatter would leave).
assert px[1, 2, 3] == 1.0, px[1, 2]
assert px[1, 2, 0] > 0.0, px[1, 2]

# Far corner, well outside the (small, capped) dilation margin from either solved texel:
# still unwritten -> alpha=0, temperature left at the initialized zero.
assert px[5, 0, 3] == 0.0, px[5, 0]
assert abs(px[5, 0, 0] - 0.0) < 1e-6, px[5, 0]

print('WRITE_ATLAS_OK')
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True)
    assert "WRITE_ATLAS_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_write_atlas_no_texels_writes_empty_placeholder(executable, tmp_path):
    """No atlas objects this solve (render_domain=TEXEL requested but nothing qualified,
    or atlas_plan built from an empty scene) -> write_atlas must still return a stable,
    loadable, all-invalid path rather than raising."""
    code = f"""
from pathlib import Path
from visionsim.simulate.heatsim import adapter, atlas

layout = atlas.AtlasLayout(atlas_size=(0, 0), tiles={{}}, effective_density=500.0, rescaled=False)
plan = adapter.AtlasPlan(layout=layout, texels={{}}, digest='empty')

out_path = adapter.write_atlas({{}}, plan, Path(r'{tmp_path}'))
assert out_path.exists(), out_path

import bpy
img = bpy.data.images.load(str(out_path))
import numpy as np
px = np.array(img.pixels[:], dtype=np.float64)
assert np.all(px[3::4] == 0.0)  # every alpha channel is 0 (nothing valid)

print('WRITE_ATLAS_EMPTY_OK')
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True)
    assert "WRITE_ATLAS_EMPTY_OK" in out.stdout, out.stdout + "\n" + out.stderr


# ---------------------------------------------------------------------------
# write_frame_attributes TEXEL behavior (no bpy needed via a minimal fake scene/mesh).
# ---------------------------------------------------------------------------


class _FakeAttrData(list):
    def foreach_set(self, prop, values):
        for elem, value in zip(self, values):
            setattr(elem, prop, float(value))


class _FakeAttr:
    def __init__(self, n):
        self.data = _FakeAttrData(type("D", (), {"value": 0.0})() for _ in range(n))


class _FakeAttrs(dict):
    def __init__(self, n):
        super().__init__()
        self._n = n

    def new(self, name, type, domain):  # noqa: A002 - mirrors bpy signature
        self[name] = _FakeAttr(self._n)
        return self[name]


class _FakeMesh:
    def __init__(self, n_verts):
        self.vertices = [object()] * n_verts
        self.attributes = _FakeAttrs(n_verts)

    def update(self):
        pass


class _FakeObj:
    def __init__(self, name, n_verts):
        self.name = name
        self.type = "MESH"
        self.data = _FakeMesh(n_verts)
        self.heat_sim_material = None
        self._props = {}

    def __setitem__(self, key, value):
        self._props[key] = value

    def __getitem__(self, key):
        return self._props[key]


class _FakeScene:
    def __init__(self, objects):
        self.objects = objects


_DEFAULTS = dict(
    initial_temperature_K=295.0,
    thermal_diffusivity_mm2_s=0.17,
    density_kg_m3=1330.0,
    specific_heat_J_kgK=880.0,
    emissivity=0.9,
)


def test_write_frame_attributes_texel_objects_get_fallback_only():
    vertex_obj = _FakeObj("vertex_mesh", n_verts=3)
    atlas_obj = _FakeObj("atlas_mesh", n_verts=3)  # same vertex count as its texel count on purpose

    # atlas_mesh's "history" has K=3 texels -- deliberately the SAME as its vertex count, so a
    # shape-coincidence could otherwise slip through the old (implicit) shape-mismatch fallback.
    history = {
        "vertex_mesh": np.array([[295.0, 295.0, 295.0], [300.0, 301.0, 302.0]]),
        "atlas_mesh": np.array([[295.0, 295.0, 295.0], [350.0, 360.0, 370.0]]),
    }
    plan = _plan((8, 8), {"atlas_mesh": atlas.TileSpec("atlas_mesh", (8, 8), (0, 0))}, {"atlas_mesh": {"xy": np.zeros((3, 2), dtype=np.int64)}})

    scene = _FakeScene([vertex_obj, atlas_obj])
    adapter.write_frame_attributes(scene, history, -1, _DEFAULTS, atlas_plan=plan)

    # Vertex-path object: unchanged, per-vertex sim_temperature written from the final row.
    vals = [d.value for d in vertex_obj.data.attributes["sim_temperature"].data]
    assert vals == pytest.approx([300.0, 301.0, 302.0])
    assert "heatsim_default_temperature" not in vertex_obj._props

    # Atlas object: NO per-vertex sim_temperature attribute written (even though the shapes
    # would have "matched" under the old implicit-mismatch fallback) -- only the OBJECT-level
    # fallback, at the ambient default (FEM participant).
    assert "sim_temperature" not in atlas_obj.data.attributes
    assert "emissivity" not in atlas_obj.data.attributes
    assert atlas_obj["heatsim_default_temperature"] == pytest.approx(295.0)

    # Coverage gate: 1.0 for the atlas participant, 0.0 for the vertex-path object.
    assert atlas_obj["heatsim_atlas_coverage"] == pytest.approx(1.0)
    assert vertex_obj["heatsim_atlas_coverage"] == pytest.approx(0.0)


def test_write_frame_attributes_vertex_mode_unaffected_by_atlas_plan_none():
    """atlas_plan=None (or omitted) reproduces exactly today's VERTEX-only behavior --
    no coverage-gate property is stamped anywhere."""
    obj = _FakeObj("mesh", n_verts=2)
    history = {"mesh": np.array([[295.0, 295.0], [305.0, 306.0]])}

    scene = _FakeScene([obj])
    adapter.write_frame_attributes(scene, history, -1, _DEFAULTS)

    vals = [d.value for d in obj.data.attributes["sim_temperature"].data]
    assert vals == pytest.approx([305.0, 306.0])
    assert "heatsim_atlas_coverage" not in obj._props


# ---------------------------------------------------------------------------
# global_temperature_range already pools whatever is in `history`, TEXEL entries included
# (they arrive there via the same _split_history/solve_scene path as VERTEX entries) --
# this locks that behavior explicitly rather than relying on it being incidental.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# End-to-end integration: drive the real BlenderService.exposed_prepare_thermal /
# exposed_include_thermal with render_domain="TEXEL". Not one of the brief's named
# fake-bpy tests, but the unit tests above (write_atlas, write_frame_attributes, the
# shader node graph) each exercise one piece in isolation -- this closes the gap the
# same way test_heatsim_animated.py's BlenderService-driven test does for the animated
# path, catching a real orchestration mismatch between _thermal_solve/write_atlas/the
# atlas-image load+pack/setup_temperature_aov that no single-function test would see.
# ---------------------------------------------------------------------------


def test_prepare_and_include_thermal_texel_mode_end_to_end(executable, tmp_path):
    code = f"""
import bpy
from visionsim.simulate.heatsim import register
register()

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# A coarse (4-vertex) plane spanning 4 m^2: well under any reasonable
# atlas_texel_density, so it must join the atlas (mirrors test_heatsim_texel_mode.py's
# end-to-end atlas smoke test).
bpy.ops.mesh.primitive_plane_add(size=2.0)
plane = bpy.context.active_object
plane.name = 'CoarsePlane'
plane.heat_simulation_enabled = True

mat = bpy.data.materials.new('plane_mat')
mat.use_nodes = True
plane.data.materials.append(mat)

bpy.ops.object.light_add(type='SUN')
bpy.context.active_object.data.energy = 10.0
world = bpy.context.scene.world
world.use_nodes = True
bg = world.node_tree.nodes.get('Background')
bg.inputs['Strength'].default_value = 1.0

blend_path = r'{tmp_path}/texel_test.blend'
root_path = r'{tmp_path}'
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

from visionsim.simulate.blender import BlenderService
from visionsim.simulate.heatsim.constants import ATLAS_COVERAGE_PROP, ATLAS_IMAGE_NAME

service = BlenderService()
service.exposed_initialize(blend_path, root_path)

service.exposed_prepare_thermal(
    render_domain='TEXEL',
    atlas_texel_density=64.0,
    atlas_tile_min=16,
    atlas_tile_max=64,
    atlas_texel_soft_max=500_000,
    domain='POINTS',
    laplacian_backend='ROBUST',
    device='cpu',
    sim_time_s=0.1,
    timestep_s=0.05,
    initial_temperature_K=295.0,
    thermal_diffusivity_mm2_s=0.17,
    density_kg_m3=1330.0,
    specific_heat_J_kgK=880.0,
    emissivity=0.9,
    irradiance_scale=100.0,
)

assert service._thermal_atlas_plan is not None
assert 'CoarsePlane' in service._thermal_atlas_plan.texels

plane_obj = bpy.data.objects['CoarsePlane']
assert 'sim_temperature' not in plane_obj.data.attributes, 'atlas object must not get a per-vertex write'
assert plane_obj[ATLAS_COVERAGE_PROP] == 1.0

atlas_img = bpy.data.images.get(ATLAS_IMAGE_NAME)
assert atlas_img is not None, 'atlas image was not loaded/packed'
assert atlas_img.packed_file is not None, 'atlas image was not packed'

# Re-fetch the material via the object rather than the captured Python `mat` handle --
# the albedo bake path (triggered by texel irradiance for the atlas) may swap/copy
# material slots, invalidating the original `bpy.types.Material` reference. That same
# albedo bake also leaves its OWN ShaderNodeTexImage node (the bake target) on the
# material, so look for the atlas-specific one by its image rather than assuming
# there is exactly one Image Texture node.
live_mat = plane_obj.material_slots[0].material
assert live_mat is not None
mat_nodes = live_mat.node_tree.nodes
aov_nodes = [n for n in mat_nodes if n.type == 'OUTPUT_AOV']
assert len(aov_nodes) == 1
atlas_tex_nodes = [n for n in mat_nodes if n.bl_idname == 'ShaderNodeTexImage' and n.image is atlas_img]
assert len(atlas_tex_nodes) == 1, [n.name for n in mat_nodes if n.bl_idname == 'ShaderNodeTexImage']

service.exposed_include_thermal(render_domain='TEXEL')
assert 'temperature' in service.render_layers.outputs

print('TEXEL_E2E_OK')
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True)
    assert "TEXEL_E2E_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_global_temperature_range_includes_texels():
    # A "vertex" object near ambient plus an "atlas" (texel) object running much hotter --
    # both keyed into `history` exactly the same way (solve_scene/​_split_history don't
    # distinguish TEXEL from VERTEX entries), so the pooled range must span both.
    history = {
        "vertex_mesh": np.stack([np.full(50, 295.0), np.full(50, 295.5)]),
        "atlas_mesh": np.stack([np.full(400, 295.0), np.full(400, 340.0)]),
    }

    tmin, tmax = adapter.global_temperature_range(history, default_K=295.0)

    # The texel object's 340 K dominates the pool (400 texels vs 50 vertices), so P99 must
    # land near it, not be capped near the near-ambient vertex object's ~295.5 K.
    assert tmax > 330.0, tmax
    assert tmin <= 295.5, tmin
