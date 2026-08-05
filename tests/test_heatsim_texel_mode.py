from __future__ import annotations

import json
import logging
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from visionsim.simulate.heatsim import adapter, materials

_DEFAULTS = dict(
    initial_temperature_K=300.0,
    thermal_diffusivity_mm2_s=0.42,
    density_kg_m3=1234.0,
    specific_heat_J_kgK=777.0,
    emissivity=0.5,
    irradiance_scale=1.0,
)
_SOLVER_CFG = {"domain": "POINTS", "interior_points": False}


# ---------------------------------------------------------------------------
# Fake-bpy fixtures (mirrors tests/test_heatsim_assignments_integration.py)
# ---------------------------------------------------------------------------


class _Poly:
    def __init__(self, vertices, material_index, area):
        self.vertices = list(vertices)
        self.material_index = material_index
        self.area = area


class _Mesh:
    def __init__(self, verts_xyz, polygons):
        self._xyz = np.asarray(verts_xyz, dtype=np.float64)
        self.vertices = [object()] * len(self._xyz)
        self.polygons = list(polygons)


class _Obj:
    __hash__ = object.__hash__
    __eq__ = object.__eq__

    def __init__(self, name, mesh, slot_names):
        self.name = name
        self.type = "MESH"
        self.data = mesh
        self.material_slots = [type("S", (), {"material": type("M", (), {"name": n})()})() for n in slot_names]
        self.heat_sim_material = None


class _NS:
    """Like ``SimpleNamespace`` but hashable by identity (mirrors real ``bpy`` objects,
    which ``_combine``'s ``flux_by_obj`` dict keys on)."""

    __hash__ = object.__hash__
    __eq__ = object.__eq__

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _square(name="square"):
    """4 verts, 2 coplanar tris; tri 0 uses slot 0 (WOODY), tri 1 uses slot 1 (STEELY)."""
    xyz = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    return _Obj(name, _Mesh(xyz, [_Poly([0, 1, 2], 0, 0.5), _Poly([0, 2, 3], 1, 0.5)]), ["WOODY", "STEELY"])


def _sidecar(tmp_path, block, name="s.thermal.json"):
    path = tmp_path / name
    path.write_text(json.dumps({
        "schema_version": 1, "scene": "s.blend", "defaults": {"preset": None}, "materials": block,
    }), encoding="utf-8")
    return materials.load_assignments(path)


@pytest.fixture(autouse=True)
def _stub_geometry(monkeypatch):
    """_extract_geometry needs bpy/mathutils; feed it straight from the fake mesh."""
    def fake(obj):
        verts = np.asarray(obj.data._xyz, dtype=np.float64) * 1000.0  # m -> mm
        faces = np.asarray([list(p.vertices) for p in obj.data.polygons], dtype=np.int32)
        return verts, faces, len(verts)

    monkeypatch.setattr(adapter, "_extract_geometry", fake)


# ---------------------------------------------------------------------------
# materials.resolve_face_materials
# ---------------------------------------------------------------------------


def test_resolve_face_materials_exact_per_face(tmp_path):
    sa = _sidecar(tmp_path, {
        "WOODY": {"preset": "wood"},
        "STEELY": {"preset": "steel", "role": "DIRICHLET_SOURCE", "dirichlet_K": 350.0},
    })
    obj = _square()
    face_slots = np.array([0, 1], dtype=np.int32)  # face 0 -> WOODY, face 1 -> STEELY

    out = materials.resolve_face_materials(obj, sa, _DEFAULTS, face_slots)

    assert out is not None
    for key in ("t0", "alpha", "rho", "c", "eps"):
        assert out[key].shape == (2,)
    assert out["alpha"][0] == pytest.approx(materials.PRESETS["wood"].alpha_mm2_s)
    assert out["alpha"][1] == pytest.approx(materials.PRESETS["steel"].alpha_mm2_s)
    # No averaging: the two faces' values are exactly the two distinct presets, not a blend.
    assert out["alpha"][0] != pytest.approx(out["alpha"][1])
    assert not (materials.PRESETS["wood"].alpha_mm2_s < out["alpha"][0] < materials.PRESETS["steel"].alpha_mm2_s)

    # Dirichlet slot faces are flagged; the FEM slot's face is not.
    assert not out["dirichlet_mask"][0]
    assert out["dirichlet_mask"][1]
    assert out["t0"][1] == pytest.approx(350.0)


def test_resolve_face_materials_returns_none_without_slots(tmp_path):
    sa = _sidecar(tmp_path, {"WOODY": {"preset": "wood"}})
    obj = _square()
    obj.material_slots = []
    assert materials.resolve_face_materials(obj, sa, _DEFAULTS, np.array([0], dtype=np.int32)) is None


def test_resolve_face_materials_out_of_range_index_is_clamped(tmp_path):
    sa = _sidecar(tmp_path, {"WOODY": {"preset": "wood"}, "STEELY": {"preset": "steel"}})
    obj = _square()
    out = materials.resolve_face_materials(obj, sa, _DEFAULTS, np.array([7], dtype=np.int32))
    assert out is not None
    assert out["alpha"][0] == pytest.approx(materials.PRESETS["steel"].alpha_mm2_s)


# ---------------------------------------------------------------------------
# adapter._combine TEXEL mode
# ---------------------------------------------------------------------------


def _texel_table(k, face, face_material_index):
    rng = np.random.default_rng(0)
    return {
        "position_mm": rng.uniform(0.0, 100.0, size=(k, 3)),
        "normal": np.tile(np.array([0.0, 0.0, 1.0]), (k, 1)),
        "uv": np.zeros((k, 2), dtype=np.float64),
        "face": np.asarray(face, dtype=np.int64),
        "face_material_index": np.asarray(face_material_index, dtype=np.int32),
    }


def test_combine_texel_mode_mixes_texel_and_vertex_objects():
    k = 5
    sparse_obj = _NS(name="sparse", heat_sim_material=None)
    dense_obj = _square("dense")

    atlas_plan = SimpleNamespace(texels={
        "sparse": _texel_table(k, face=np.zeros(k, dtype=np.int64), face_material_index=[0]),
    })

    combined = adapter._combine([sparse_obj, dense_obj], {}, _DEFAULTS, _SOLVER_CFG, atlas_plan=atlas_plan)

    assert combined is not None
    n_dense = 4
    assert combined.verts.shape[0] == k + n_dense
    assert combined.alpha.shape[0] == k + n_dense
    assert combined.layout == [
        ("sparse", 0, k, "TEXEL"),
        ("dense", k, n_dense, "VERTEX"),
    ]
    # TEXEL object contributes no faces (POINTS-only); VERTEX object's faces are offset by k.
    assert combined.faces.shape[0] == 2  # only "dense"'s two triangles
    assert combined.faces.min() >= k


def test_combine_texel_mode_object_level_materials_when_no_assignment():
    """No sidecar: a TEXEL object still gets the object-level constant, broadcast per-texel."""
    k = 3
    obj = _NS(name="sparse", heat_sim_material=None)
    atlas_plan = SimpleNamespace(texels={
        "sparse": _texel_table(k, face=np.zeros(k, dtype=np.int64), face_material_index=[0]),
    })

    combined = adapter._combine([obj], {}, _DEFAULTS, _SOLVER_CFG, atlas_plan=atlas_plan)

    assert combined is not None
    assert np.allclose(combined.alpha, _DEFAULTS["thermal_diffusivity_mm2_s"])
    assert np.allclose(combined.eps, _DEFAULTS["emissivity"])
    assert np.allclose(combined.t0, _DEFAULTS["initial_temperature_K"])
    assert np.all(combined.boundary_mask)


def test_combine_texel_dirichlet_pinning_per_texel(tmp_path):
    sa = _sidecar(tmp_path, {
        "WOODY": {"preset": "wood"},
        "STEELY": {"preset": "steel", "role": "DIRICHLET_SOURCE", "dirichlet_K": 350.0},
    })
    obj = _square()  # name "square", slots WOODY (face 0), STEELY (face 1)
    tex = _texel_table(4, face=[0, 0, 1, 1], face_material_index=[0, 1])
    atlas_plan = SimpleNamespace(texels={"square": tex})
    flux_by_obj = {obj: np.array([10.0, 10.0, 10.0, 10.0])}

    combined = adapter._combine([obj], flux_by_obj, _DEFAULTS, _SOLVER_CFG, assignment=sa, atlas_plan=atlas_plan)

    assert combined is not None
    pinned = ~combined.boundary_mask
    assert pinned.tolist() == [False, False, True, True]
    assert np.allclose(combined.alpha[pinned], 0.0)
    assert np.allclose(combined.irradiance[pinned], 0.0)
    assert np.all(combined.irradiance[~pinned] > 0.0)
    assert np.allclose(combined.t0[pinned], 350.0)
    assert np.allclose(combined.t0[~pinned], _DEFAULTS["initial_temperature_K"])


def test_combine_vertex_mode_unchanged_when_no_plan():
    """Backward-compat gate: atlas_plan=None (or omitted) reproduces today's behaviour."""
    obj = _square()
    combined_default = adapter._combine([obj], {}, _DEFAULTS, _SOLVER_CFG)
    combined_explicit = adapter._combine([obj], {}, _DEFAULTS, _SOLVER_CFG, atlas_plan=None)

    for combined in (combined_default, combined_explicit):
        assert combined is not None
        assert np.allclose(combined.alpha, _DEFAULTS["thermal_diffusivity_mm2_s"])
        assert np.allclose(combined.eps, _DEFAULTS["emissivity"])
        assert np.allclose(combined.t0, _DEFAULTS["initial_temperature_K"])
        assert np.all(combined.boundary_mask)
        assert combined.layout == [("square", 0, 4, "VERTEX")]

    assert np.array_equal(combined_default.alpha, combined_explicit.alpha)
    assert np.array_equal(combined_default.verts, combined_explicit.verts)
    assert np.array_equal(combined_default.faces, combined_explicit.faces)


# ---------------------------------------------------------------------------
# adapter.build_atlas_plan
# ---------------------------------------------------------------------------


class _AtlasMesh:
    def __init__(self, n_verts):
        self.vertices = [object()] * n_verts


class _AtlasObj:
    def __init__(self, name, n_verts):
        self.name = name
        self.data = _AtlasMesh(n_verts)
        self.heat_sim_material = None


def _big_plane_geom(side_mm=10_000.0):
    """4 verts spanning a 10m x 10m plane => 100 m^2 for ~0.04 verts/m^2 (well under any
    reasonable atlas_texel_density, so select_for_atlas admits it)."""
    verts = np.array([
        [0.0, 0.0, 0.0], [side_mm, 0.0, 0.0], [side_mm, side_mm, 0.0], [0.0, side_mm, 0.0],
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return verts, faces, 4


_ATLAS_CFG = dict(atlas_texel_density=50.0, atlas_tile_min=16, atlas_tile_max=512, atlas_texel_soft_max=500_000)


def test_uv_failure_demotes_to_vertex_path_with_warning(monkeypatch, caplog):
    good = _AtlasObj("good", 4)
    bad = _AtlasObj("bad", 4)
    geoms = {"good": _big_plane_geom(), "bad": _big_plane_geom()}

    monkeypatch.setattr(adapter, "_extract_geometry", lambda obj: geoms[obj.name])
    monkeypatch.setattr(adapter, "_prepare_bake_uv", lambda obj: None)

    def fake_uv(obj, layer_name):
        if obj.name == "bad":
            return None  # simulates an unwrap that never produced a usable UV layer
        loop_uv = np.array([
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ])
        face_material_index = np.array([0, 0], dtype=np.int32)
        return loop_uv, face_material_index

    monkeypatch.setattr(adapter, "_extract_face_uv_and_slots", fake_uv)
    monkeypatch.setattr(adapter, "_write_atlas_uv_layer", lambda *a, **kw: None)

    with caplog.at_level(logging.WARNING):
        plan = adapter.build_atlas_plan(scene=None, sim_objects=[good, bad], cfg=_ATLAS_CFG)

    assert "good" in plan.texels
    assert "bad" not in plan.texels
    assert plan.texels["good"]["position_mm"].shape[0] > 0

    messages = [rec.getMessage() for rec in caplog.records]
    assert any("bad" in m and "demoted" in m for m in messages)


def test_build_atlas_plan_topology_mismatch_demotes_with_warning(monkeypatch, caplog):
    """Base mesh vertex count != evaluated vertex count -> demoted before UV is even tried."""
    obj = _AtlasObj("mod_obj", n_verts=6)  # base mesh reports 6 verts
    evaluated = _big_plane_geom()  # evaluated geometry has 4 verts

    monkeypatch.setattr(adapter, "_extract_geometry", lambda o: evaluated)
    calls = []
    monkeypatch.setattr(adapter, "_prepare_bake_uv", lambda o: calls.append(o.name))

    with caplog.at_level(logging.WARNING):
        plan = adapter.build_atlas_plan(scene=None, sim_objects=[obj], cfg=_ATLAS_CFG)

    assert "mod_obj" not in plan.texels
    assert calls == []  # never even attempted the UV prep
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("mod_obj" in m and "modifier" in m for m in messages)


def test_build_atlas_plan_excludes_dense_objects(monkeypatch):
    """An object whose native vertex density already exceeds the target keeps the vertex path."""
    dense = _AtlasObj("dense", n_verts=50_000)
    dense_geom = (np.random.default_rng(0).uniform(0, 10, size=(50_000, 3)),
                  np.array([[0, 1, 2]], dtype=np.int32), 50_000)

    monkeypatch.setattr(adapter, "_extract_geometry", lambda o: dense_geom)

    plan = adapter.build_atlas_plan(scene=None, sim_objects=[dense], cfg=_ATLAS_CFG)
    assert plan.texels == {}


# ---------------------------------------------------------------------------
# solve_scene cache key
# ---------------------------------------------------------------------------


class _FakeScene:
    def __init__(self):
        self.objects = []


def test_cache_key_gains_atlas_digest(tmp_path, monkeypatch):
    captured = []
    from visionsim.simulate.heatsim import cache as cache_mod

    real_cache_key = cache_mod.cache_key

    def spy(blend_path, key_cfg):
        captured.append(key_cfg)
        return real_cache_key(blend_path, key_cfg)

    monkeypatch.setattr(adapter, "gather_meshes", lambda scene: [])
    monkeypatch.setattr(adapter.cache, "cache_key", spy)

    scene = _FakeScene()
    adapter.solve_scene(scene, defaults=_DEFAULTS, solver_cfg=_SOLVER_CFG, cache_root=tmp_path)

    fake_plan = SimpleNamespace(
        texels={}, density=50.0, tile_min=16, tile_max=512, soft_max=500_000, digest="deadbeef",
    )
    adapter.solve_scene(
        scene, defaults=_DEFAULTS, solver_cfg=_SOLVER_CFG, cache_root=tmp_path, atlas_plan=fake_plan,
    )

    assert len(captured) == 2
    assert captured[0]["atlas"] is None
    assert captured[1]["atlas"] == {
        "density": 50.0, "tile_min": 16, "tile_max": 512, "soft_max": 500_000, "layout_digest": "deadbeef",
    }


# ---------------------------------------------------------------------------
# End-to-end smoke test (real bpy): build_atlas_plan -> solve_scene(atlas_plan=...)
# -> write_frame_attributes, on a genuine low-vertex-density plane. Not one of the
# brief's named fake-bpy tests, but the fake fixtures above cannot catch a real bpy
# API mismatch (foreach_get signatures, UV layer creation, ...), so this closes that
# gap the same way test_heatsim_irradiance.py's executable tests do for the bake path.
# ---------------------------------------------------------------------------


def test_texel_pipeline_solves_end_to_end_under_bpy(executable, tmp_path):
    code = f"""
import bpy, numpy as np
from pathlib import Path
from visionsim.simulate.heatsim import register, adapter
from visionsim.simulate.heatsim.constants import ATLAS_UV_LAYER_NAME

register()
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# A coarse (4-vertex) plane spanning 4 m^2: at any reasonable density this is
# well under the atlas_texel_density target, so it must join the atlas.
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

defaults = dict(initial_temperature_K=295.0, thermal_diffusivity_mm2_s=0.17,
                density_kg_m3=1330.0, specific_heat_J_kgK=880.0, emissivity=0.9,
                irradiance_scale=100.0)
solver_cfg = dict(sim_time_s=0.1, timestep_s=0.05, domain='POINTS',
                  laplacian_backend='ROBUST', device='cpu')
atlas_cfg = dict(atlas_texel_density=64.0, atlas_tile_min=16, atlas_tile_max=64,
                  atlas_texel_soft_max=500_000)

sim_objects = adapter.gather_meshes(bpy.context.scene)
plan = adapter.build_atlas_plan(bpy.context.scene, sim_objects, atlas_cfg)
assert 'CoarsePlane' in plan.texels, 'coarse plane should have joined the atlas'
n_texels = plan.texels['CoarsePlane']['position_mm'].shape[0]
assert n_texels > len(plane.data.vertices), (n_texels, len(plane.data.vertices))

# HeatSim_Atlas_UV must have been written for the render-time shader lookup.
assert ATLAS_UV_LAYER_NAME in plane.data.uv_layers

hist = adapter.solve_scene(bpy.context.scene, defaults=defaults, solver_cfg=solver_cfg,
                           cache_root=Path(r'{tmp_path}'), atlas_plan=plan)
assert 'CoarsePlane' in hist, list(hist.keys())
T_hist = np.asarray(hist['CoarsePlane'])
assert T_hist.ndim == 2 and T_hist.shape[1] == n_texels, T_hist.shape
assert np.isfinite(T_hist).all()
assert T_hist.min() > 200 and T_hist.max() < 2000, (float(T_hist.min()), float(T_hist.max()))

print('TEXEL_PIPELINE_OK', n_texels, len(plane.data.vertices))
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True)
    assert "TEXEL_PIPELINE_OK" in out.stdout, out.stdout + "\n" + out.stderr
