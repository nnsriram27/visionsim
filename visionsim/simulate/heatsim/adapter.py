"""Scene adapter: Blender scene -> solver inputs -> per-vertex temperatures.

Hand-written glue that distils the data-shaping core of heat-sim-blender's
``addon/lib/fem_adapter.py`` (@ 543ee81) into four functions:

* :func:`gather_meshes`         - select the meshes that participate in the solve.
* :func:`resolve_material`      - per-object thermal params (``heat_sim_material`` -> ``defaults``).
* :func:`solve_scene`           - cache-aware geometry -> irradiance -> FEM solve -> per-object history.
* :func:`write_frame_attributes`- stamp ``sim_temperature`` / ``emissivity`` (and a fallback temp).

Unit / sign / dt conventions are preserved verbatim from upstream because the
wrong units silently corrupt the physics:

* geometry in **millimetres** (world coords x 1000),
* irradiance W/m^2 -> W/mm^2 (/1e6) then x ``irradiance_scale``,
* density kg/m^3 -> kg/mm^3 (/1e9) wherever the solver consumes it,
* the FEM solver is driven exactly as ``tests/test_heatsim_solver.py`` drives it
  (``NUM_FRAME_DELTA = timestep_s * 60`` so ``dt = NUM_FRAME_DELTA / 60``;
  ``record_time == sim_time`` records every step).

The module imports ``bpy``/``mathutils`` defensively so it stays importable (for
linting / type-checking) outside Blender; the bpy-coupled solver and Direct-Kernel
irradiance modules are imported lazily inside :func:`solve_scene`.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from visionsim.simulate.heatsim import cache

try:
    import bpy  # type: ignore
    import mathutils  # type: ignore
except ImportError:  # pragma: no cover - only hit outside Blender
    bpy = None  # type: ignore
    mathutils = None  # type: ignore

_log = logging.getLogger("rich")

# Unit conversions (everything the solver sees is in mm-based units).
_M_TO_MM = 1000.0
_KGM3_TO_KGMM3 = 1.0e9  # divide: kg/m^3 -> kg/mm^3  (1000**3)
_WM2_TO_WMM2 = 1.0e6    # divide: W/m^2 -> W/mm^2     (1000**2)


# ---------------------------------------------------------------------------
# Object selection + material resolution
# ---------------------------------------------------------------------------


def gather_meshes(scene: Any) -> list:
    """Return the MESH objects that take part in the heat solve.

    A mesh participates when it is visible, renderable and (per-object) heat
    simulation is enabled. Mirrors the filter at ``fem_adapter.py:2867,2875``;
    a missing ``heat_simulation_enabled`` attribute defaults to ``True``.
    """
    out: list = []
    for obj in scene.objects:
        if getattr(obj, "type", None) != "MESH":
            continue
        if not obj.visible_get() or obj.hide_render:
            continue
        if not bool(getattr(obj, "heat_simulation_enabled", True)):
            continue
        out.append(obj)
    return out


def _is_set(mat: Any, attr: str) -> bool:
    """True iff *attr* was explicitly set on the per-object PropertyGroup *mat*.

    ``obj.heat_sim_material`` is a ``PointerProperty`` registered on every object,
    so ``mat`` is never ``None`` and a ``FloatProperty`` always returns its group
    default - the global ``defaults`` (``--config.thermal.*``) would otherwise be
    unreachable.  Blender's ``bpy_struct.is_property_set`` distinguishes an
    explicitly-authored value from the registered default, restoring the locked
    "per-object overrides, globals as fallback" contract.  Non-Blender fakes that
    expose ``is_property_set`` are honoured too; anything else is treated as unset.
    """
    if mat is None:
        return False
    checker = getattr(mat, "is_property_set", None)
    if checker is None:
        return False
    try:
        return bool(checker(attr))
    except Exception:  # pragma: no cover - defensive
        return False


def resolve_material(obj: Any, defaults: dict) -> dict:
    """Resolve per-object thermal parameters.

    Priority: an *explicitly set* per-object value on ``obj.heat_sim_material``
    (detected via :func:`_is_set`), else the global ``defaults`` dict.  Because the
    PropertyGroup is registered on every object, only ``is_property_set`` can tell a
    user-authored override from the registered group default; without that gate the
    global ``--config.thermal.*`` knobs would be silently inert.  SI units throughout
    (``thermal_diffusivity`` in mm^2/s, ``density`` in kg/m^3, ``specific_heat`` in
    J/(kg.K)).
    """
    mat = getattr(obj, "heat_sim_material", None)

    def _pick(attr: str, key: str) -> float:
        if _is_set(mat, attr):
            return float(getattr(mat, attr))
        return float(defaults[key])

    role = "FEM_PARTICIPANT"
    dirichlet_T = 0.0
    if _is_set(mat, "thermal_role"):
        role = str(getattr(mat, "thermal_role") or "FEM_PARTICIPANT").upper()
    if _is_set(mat, "dirichlet_temperature_K"):
        dirichlet_T = float(getattr(mat, "dirichlet_temperature_K") or 0.0)

    return {
        "initial_temperature_K": _pick("initial_temperature_K", "initial_temperature_K"),
        "thermal_diffusivity_mm2_s": _pick("thermal_diffusivity_mm2_s", "thermal_diffusivity_mm2_s"),
        "density_kg_m3": _pick("density_kg_m3", "density_kg_m3"),
        "specific_heat_J_kgK": _pick("specific_heat_J_kgK", "specific_heat_J_kgK"),
        "emissivity": float(np.clip(_pick("emissivity", "emissivity"), 0.0, 1.0)),
        "thermal_role": role,
        "dirichlet_temperature_K": dirichlet_T,
    }


# ---------------------------------------------------------------------------
# Geometry extraction (evaluated mesh -> world mm + triangulated faces)
# ---------------------------------------------------------------------------


def _extract_geometry(obj: Any) -> Optional[tuple]:
    """``(verts_mm (N,3) float64, faces (M,3) int32, n_verts)`` for the evaluated
    mesh, or ``None`` if it has no geometry. Verts are world-space x 1000 (mm)
    and quads are triangulated, matching ``fem_adapter._extract_mesh_data``.

    The evaluated mesh's vertex order/count matches the Direct-Kernel irradiance
    extraction (it uses the same ``foreach_get('co')`` path), so the returned
    flux aligns index-for-index with these vertices.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh = obj.evaluated_get(depsgraph).data
    n_verts = len(mesh.vertices)
    if n_verts == 0 or len(mesh.polygons) == 0:
        return None

    flat: np.ndarray = np.zeros(n_verts * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", flat)
    verts: np.ndarray = flat.reshape(n_verts, 3)
    mw = np.array(obj.matrix_world, dtype=np.float64)
    verts = (verts @ mw[:3, :3].T) + mw[:3, 3]
    verts = verts * _M_TO_MM  # world metres -> mm

    poly_count = len(mesh.polygons)
    loop_starts: np.ndarray = np.zeros(poly_count, dtype=np.int32)
    loop_totals: np.ndarray = np.zeros(poly_count, dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", loop_starts)
    mesh.polygons.foreach_get("loop_total", loop_totals)
    loop_vidx: np.ndarray = np.zeros(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vidx)

    face_arrays = []
    tri_starts = loop_starts[loop_totals == 3]
    if tri_starts.size:
        face_arrays.append(
            np.column_stack([loop_vidx[tri_starts], loop_vidx[tri_starts + 1], loop_vidx[tri_starts + 2]])
        )
    quad_starts = loop_starts[loop_totals == 4]
    if quad_starts.size:
        v0 = loop_vidx[quad_starts]
        v1 = loop_vidx[quad_starts + 1]
        v2 = loop_vidx[quad_starts + 2]
        v3 = loop_vidx[quad_starts + 3]
        face_arrays.append(np.column_stack([v0, v1, v2]))
        face_arrays.append(np.column_stack([v0, v2, v3]))
    if not face_arrays:
        return None

    faces = np.vstack(face_arrays).astype(np.int32)
    return verts, faces, n_verts


# ---------------------------------------------------------------------------
# Direct-Kernel irradiance
# ---------------------------------------------------------------------------


def _ensure_albedo_attr(obj: Any, value: float) -> None:
    """Stamp a constant ``albedo`` POINT/FLOAT attribute when one is absent.

    The Direct Kernel's albedo step would otherwise fall through to a one-shot
    Cycles bake (``irradiance.bake_albedo_map``) - a module that is intentionally
    NOT vendored here. Pre-seeding the attribute makes the kernel read it instead
    of baking; ``value == 0`` reproduces the kernel's own "missing albedo => fully
    absorbing" fallback exactly. A pre-existing valid ``albedo`` attribute is kept.
    """
    mesh = getattr(obj, "data", None)
    if mesh is None:
        return
    n = len(mesh.vertices)
    existing = mesh.attributes.get("albedo")
    if (
        existing is not None
        and getattr(existing, "domain", None) == "POINT"
        and getattr(existing, "data_type", None) == "FLOAT"
        and len(existing.data) == n
    ):
        return
    if existing is not None:
        try:
            mesh.attributes.remove(existing)
        except Exception:
            pass
    attr = mesh.attributes.new(name="albedo", type="FLOAT", domain="POINT")
    attr.data.foreach_set("value", np.full(n, float(value), dtype=np.float32))
    mesh.update()


def _compute_irradiance(scene: Any, sim_objects: list, solver_cfg: dict, defaults: dict) -> dict:
    """Run the Direct-Kernel and return ``{obj: (N,) float64 W/m^2 absorbed}``."""
    from visionsim.simulate.heatsim import irradiance_kernel

    default_albedo = float(defaults.get("albedo", 0.0))
    for obj in sim_objects:
        _ensure_albedo_attr(obj, default_albedo)

    # SimpleNamespace surrogate for the addon's scene PropertyGroup (the kernel
    # only reads these via getattr(..., default)). Sky occlusion defaults off so
    # the kernel takes the unoccluded SH9 sky path (no per-vertex AO bake).
    settings = SimpleNamespace(
        irradiance_texture_size=int(solver_cfg.get("irradiance_texture_size", 512)),
        enable_sky_occlusion=bool(solver_cfg.get("enable_sky_occlusion", False)),
        sky_ao_min_for_bent=float(solver_cfg.get("sky_ao_min_for_bent", 0.02)),
        direct_kernel_soft_shadow_rays=int(solver_cfg.get("direct_kernel_soft_shadow_rays", 8)),
    )
    raw = irradiance_kernel.compute_per_vertex_irradiance(scene, list(sim_objects), settings)
    out: dict = {}
    for obj, payload in raw.items():
        flux = payload.get("vertex_flux") if isinstance(payload, dict) else payload
        if flux is not None:
            out[obj] = np.asarray(flux, dtype=np.float64).reshape(-1)
    return out


# ---------------------------------------------------------------------------
# Combine objects into one FEM system
# ---------------------------------------------------------------------------


def _combine(sim_objects: list, flux_by_obj: dict, defaults: dict, solver_cfg: dict) -> Optional[SimpleNamespace]:
    """Stack per-object geometry + material vectors into one solver-ready system.

    Surface vertices come first (layout records each object's slice); optional
    interior POINTS-mode samples are appended afterwards by
    :func:`_augment_interior_points` so the surface slices stay valid.
    """
    irradiance_scale = float(defaults.get("irradiance_scale", 1.0))

    verts_l, faces_l = [], []
    irr_l, t0_l, alpha_l, rho_l, c_l, eps_l, bmask_l = [], [], [], [], [], [], []
    layout: list = []  # (name, offset, n_verts) over the surface vertices
    geom_by_obj: dict = {}
    offset = 0

    for obj in sim_objects:
        geom = _extract_geometry(obj)
        if geom is None:
            continue
        verts, faces, n = geom
        geom_by_obj[obj] = geom
        mat = resolve_material(obj, defaults)
        is_dirichlet = mat["thermal_role"] == "DIRICHLET_SOURCE"

        flux = flux_by_obj.get(obj)
        if flux is not None and int(np.asarray(flux).reshape(-1).shape[0]) == n:
            irr = (np.asarray(flux, dtype=np.float64).reshape(-1) / _WM2_TO_WMM2) * irradiance_scale
        else:
            irr = np.zeros(n, dtype=np.float64)

        if is_dirichlet:
            # Pinned reservoir: no diffusion, no incident flux, excluded from the
            # radiation/convection boundary (mirrors fem_adapter Dirichlet setup).
            t_dir = mat["dirichlet_temperature_K"] or mat["initial_temperature_K"]
            t0 = np.full(n, float(t_dir), dtype=np.float64)
            alpha = np.zeros(n, dtype=np.float64)
            irr = np.zeros(n, dtype=np.float64)
            bmask = np.zeros(n, dtype=bool)
        else:
            t0 = np.full(n, mat["initial_temperature_K"], dtype=np.float64)
            alpha = np.full(n, mat["thermal_diffusivity_mm2_s"], dtype=np.float64)
            bmask = np.ones(n, dtype=bool)

        verts_l.append(verts)
        faces_l.append(faces + offset)
        irr_l.append(irr)
        t0_l.append(t0)
        alpha_l.append(alpha)
        rho_l.append(np.full(n, mat["density_kg_m3"], dtype=np.float64))
        c_l.append(np.full(n, mat["specific_heat_J_kgK"], dtype=np.float64))
        eps_l.append(np.full(n, mat["emissivity"], dtype=np.float64))
        bmask_l.append(bmask)
        layout.append((obj.name, offset, n))
        offset += n

    if not verts_l:
        return None

    combined = SimpleNamespace(
        verts=np.vstack(verts_l),
        faces=np.vstack(faces_l).astype(np.int32),
        irradiance=np.concatenate(irr_l),
        t0=np.concatenate(t0_l),
        alpha=np.concatenate(alpha_l),
        density=np.concatenate(rho_l) / _KGM3_TO_KGMM3,  # kg/m^3 -> kg/mm^3
        c=np.concatenate(c_l),
        eps=np.concatenate(eps_l),
        boundary_mask=np.concatenate(bmask_l),
        surface_count=offset,
        layout=layout,
    )

    ratio = float(solver_cfg.get("interior_point_ratio", 0.0))
    if str(solver_cfg.get("domain", "POINTS")).upper() == "POINTS" and ratio > 0.0:
        _augment_interior_points(combined, sim_objects, geom_by_obj, defaults, ratio)

    return combined


def _augment_interior_points(
    combined: SimpleNamespace, sim_objects: list, geom_by_obj: dict, defaults: dict, ratio: float
) -> None:
    """Append optional interior volume samples (POINTS mode) per object.

    Off by default (the vendored ``heat_sim_material`` schema has no point-volume
    fields). When enabled via ``solver_cfg['interior_point_ratio']`` we draw
    rejection samples inside each closed mesh using a mathutils BVH ray-parity
    test - a compact stand-in for upstream's Bridson sampler (we only need extra
    interior nodes, not blue-noise spacing). Interior nodes carry the object's
    default material, zero incident flux, and are excluded from the boundary.
    Best-effort: any failure simply adds no interior points.
    """
    if mathutils is None:
        return
    extra_v: list = []
    extra_irr: list = []
    extra_t0: list = []
    extra_alpha: list = []
    extra_rho: list = []
    extra_c: list = []
    extra_eps: list = []
    for obj in sim_objects:
        geom = geom_by_obj.get(obj)
        if geom is None:
            continue
        verts, faces, n = geom
        mat = resolve_material(obj, defaults)
        if mat["thermal_role"] == "DIRICHLET_SOURCE":
            continue
        target = int(round(n * ratio))
        if target <= 0:
            continue
        try:
            pts = _sample_interior(verts, faces, target, _stable_seed(obj.name))
        except Exception as exc:  # pragma: no cover - defensive
            _log.debug("[heatsim.adapter] interior sampling failed for %s: %s", obj.name, exc)
            pts = np.zeros((0, 3), dtype=np.float64)
        if pts.shape[0] == 0:
            continue
        k = int(pts.shape[0])
        extra_v.append(pts)
        extra_irr.append(np.zeros(k, dtype=np.float64))
        extra_t0.append(np.full(k, mat["initial_temperature_K"], dtype=np.float64))
        extra_alpha.append(np.full(k, mat["thermal_diffusivity_mm2_s"], dtype=np.float64))
        extra_rho.append(np.full(k, mat["density_kg_m3"], dtype=np.float64))
        extra_c.append(np.full(k, mat["specific_heat_J_kgK"], dtype=np.float64))
        extra_eps.append(np.full(k, mat["emissivity"], dtype=np.float64))

    if not extra_v:
        return
    combined.verts = np.vstack([combined.verts, np.vstack(extra_v)])
    combined.irradiance = np.concatenate([combined.irradiance, np.concatenate(extra_irr)])
    combined.t0 = np.concatenate([combined.t0, np.concatenate(extra_t0)])
    combined.alpha = np.concatenate([combined.alpha, np.concatenate(extra_alpha)])
    combined.density = np.concatenate([combined.density, np.concatenate(extra_rho) / _KGM3_TO_KGMM3])
    combined.c = np.concatenate([combined.c, np.concatenate(extra_c)])
    combined.eps = np.concatenate([combined.eps, np.concatenate(extra_eps)])
    combined.boundary_mask = np.concatenate(
        [combined.boundary_mask, np.zeros(int(np.vstack(extra_v).shape[0]), dtype=bool)]
    )


def _stable_seed(name: str) -> int:
    # Use a cryptographic hash so the seed is stable across processes regardless
    # of PYTHONHASHSEED (Python's built-in hash() is randomised per process).
    return int(hashlib.sha256(name.encode()).hexdigest(), 16) % (2**31)


def _sample_interior(verts: np.ndarray, faces: np.ndarray, target: int, seed: int) -> np.ndarray:
    """Rejection-sample up to ``target`` points inside a closed mesh (mm)."""
    bb_min = verts.min(axis=0)
    bb_max = verts.max(axis=0)
    bb_diag = float(np.linalg.norm(bb_max - bb_min))
    if bb_diag <= 1e-9:
        return np.zeros((0, 3), dtype=np.float64)
    verts_v = [mathutils.Vector((float(v[0]), float(v[1]), float(v[2]))) for v in verts]
    faces_t = [tuple(int(i) for i in f) for f in np.asarray(faces, dtype=np.int32)]
    bvh = mathutils.bvhtree.BVHTree.FromPolygons(verts_v, faces_t, all_triangles=True)
    ray_dir = mathutils.Vector((0.873, 0.417, 0.254)).normalized()
    rng = np.random.default_rng(seed)
    out: list = []
    for _ in range(target * 40):
        if len(out) >= target:
            break
        p = bb_min + rng.random(3) * (bb_max - bb_min)
        origin = mathutils.Vector((float(p[0]), float(p[1]), float(p[2])))
        hits = 0
        cur = origin
        for _bounce in range(256):
            loc, _nrm, _idx, _dist = bvh.ray_cast(cur, ray_dir, bb_diag * 3.0)
            if loc is None:
                break
            hits += 1
            cur = loc + ray_dir * (bb_diag * 1e-6)
        if hits % 2 == 1:
            out.append(p)
    if not out:
        return np.zeros((0, 3), dtype=np.float64)
    return np.ascontiguousarray(np.array(out, dtype=np.float64))


# ---------------------------------------------------------------------------
# Solver drive + history split
# ---------------------------------------------------------------------------


def _run_solver(combined: SimpleNamespace, solver_cfg: dict, defaults: dict) -> np.ndarray:
    """Drive ``HeatSimFEM`` exactly like ``tests/test_heatsim_solver.py``.

    ``NUM_FRAME_DELTA = timestep_s * 60`` so ``dt = NUM_FRAME_DELTA / 60 == timestep_s``;
    ``record_time == sim_time`` records every step => history shape ``(sim_steps+1, N)``.
    """
    from visionsim.simulate.heatsim.solver import HeatSimFEM

    sim_time_s = float(solver_cfg.get("sim_time_s", 1.0))
    timestep_s = float(solver_cfg.get("timestep_s", 0.05))
    domain = str(solver_cfg.get("domain", "POINTS")).upper()
    backend = str(solver_cfg.get("laplacian_backend", "ROBUST")).upper()
    device = str(solver_cfg.get("device", "cpu"))

    gen_params = SimpleNamespace(
        device=device,
        RHO=float(defaults["density_kg_m3"]) / _KGM3_TO_KGMM3,  # kg/mm^3 fallback scalar
        C=float(defaults["specific_heat_J_kgK"]),
        K=float(defaults["thermal_diffusivity_mm2_s"]),
        NUM_FRAME_DELTA=timestep_s * 60.0,
    )
    sim_params = SimpleNamespace(
        sim_radiation=True,
        sim_convection=False,  # CONVECTION_COEFF is 0 anyway
        add_tikhonov_reg=False,
        sim_time=sim_time_s,
        record_time=sim_time_s,  # record_attimestep == 0 => record all steps
    )

    fem = HeatSimFEM(
        gen_params,
        sim_params,
        laplacian_domain=domain,
        laplacian_backend=backend,
        robust_mollify_factor=float(solver_cfg.get("robust_mollify_factor", 1e-5)),
        pointcloud_neighbors=int(solver_cfg.get("pointcloud_neighbors", 30)),
    )

    is_points = domain == "POINTS"
    history = fem.perform_gt_heat_simulation(
        verts_np=combined.verts.copy(),
        faces_np=None if is_points else combined.faces.copy(),
        boundary_faces_np=None if is_points else combined.faces.copy(),
        # NOTE: boundary_verts_mask_override is honoured by the solver in POINTS
        # mode only.  In MESH mode the boundary is derived from face topology and
        # the per-vertex override is silently ignored — Dirichlet sources are not
        # correctly excluded in MESH mode (known follow-up).
        boundary_verts_mask_override=combined.boundary_mask,
        u0=combined.t0,
        irradiance_map=combined.irradiance,
        thermal_diffusivity_map=combined.alpha,
        density_map=combined.density,
        specific_heat_map=combined.c,
        emissivity_map=combined.eps,
    )
    return np.asarray(history, dtype=np.float64)


def _split_history(history: np.ndarray, combined: SimpleNamespace) -> dict:
    """Trim interior points and split ``(T, N_total)`` into per-object ``(T, N)``."""
    u = history
    # Guard: in MESH mode the solver compresses the vertex array to only
    # face-referenced vertices, so the column count can be LESS than
    # surface_count.  Slicing per-object offsets into a shorter array would
    # silently return wrong temperatures.  Raise loudly instead.
    if u.ndim == 2 and u.shape[1] < combined.surface_count:
        raise RuntimeError(
            f"_split_history: solver returned {u.shape[1]} columns but combined "
            f"geometry has {combined.surface_count} surface vertices.  This happens "
            f"in MESH mode when the mesh contains orphan/unreferenced vertices that "
            f"the solver drops.  Use POINTS domain (the supported M1 path) or ensure "
            f"every vertex is referenced by at least one face."
        )
    if combined.surface_count > 0 and u.ndim == 2 and u.shape[1] > combined.surface_count:
        u = u[:, : combined.surface_count]
    out: dict = {}
    for name, off, n in combined.layout:
        out[name] = np.ascontiguousarray(u[:, off : off + n])
    return out


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def solve_scene(scene: Any, *, defaults: dict, solver_cfg: dict, cache_root: Path) -> dict:
    """Cache-aware FEM heat solve for ``scene``.

    On a cache hit the stored per-object ``(timesteps, vertices)`` history is
    returned untouched. On a miss: gather meshes -> Direct-Kernel irradiance
    (W/m^2 -> W/mm^2) -> combine -> ``HeatSimFEM.perform_gt_heat_simulation`` ->
    split the combined history back per object -> write the cache.

    Returns ``{obj_name: (timesteps, vertices) ndarray}``.
    """
    cache_root = Path(cache_root)
    sim_objects = gather_meshes(scene)

    blend_path = Path(str(getattr(getattr(bpy, "data", None), "filepath", "") or ""))
    key_cfg = {
        "solver": dict(solver_cfg),
        "defaults": dict(defaults),
        "objects": sorted(o.name for o in sim_objects),
    }
    key = cache.cache_key(blend_path, key_cfg)

    cached = cache.read_temperatures(cache_root, key)
    if cached is not None:
        _log.debug("[heatsim.adapter] cache hit: %s", key)
        return cached

    if not sim_objects:
        cache.write_temperatures(cache_root, key, {}, {"objects": []})
        return {}

    flux_by_obj = _compute_irradiance(scene, sim_objects, solver_cfg, defaults)
    combined = _combine(sim_objects, flux_by_obj, defaults, solver_cfg)
    if combined is None:
        cache.write_temperatures(cache_root, key, {}, {"objects": []})
        return {}

    history = _run_solver(combined, solver_cfg, defaults)
    per_object = _split_history(history, combined)

    meta = {
        "solver_cfg": dict(solver_cfg),
        "objects": [name for name, _, _ in combined.layout],
        "timesteps": int(history.shape[0]) if history.ndim == 2 else 0,
        "surface_count": int(combined.surface_count),
    }
    cache.write_temperatures(cache_root, key, per_object, meta)
    _log.debug("[heatsim.adapter] solved %d object(s); history %s", len(per_object), history.shape)
    return per_object


def _write_point_float_attr(mesh: Any, name: str, values: np.ndarray) -> None:
    """(Re)create a POINT/FLOAT mesh attribute from ``values``."""
    if name in mesh.attributes:
        try:
            mesh.attributes.remove(mesh.attributes[name])
        except Exception:
            pass
    attr = mesh.attributes.new(name=name, type="FLOAT", domain="POINT")
    attr.data.foreach_set("value", np.asarray(values, dtype=np.float32))


def write_frame_attributes(scene: Any, history: dict, timestep: int, defaults: dict) -> None:
    """Write per-vertex temperatures for the chosen ``timestep`` (use ``-1`` for last).

    For every simulated mesh (present in ``history``) this writes a
    ``sim_temperature`` (FLOAT/POINT) attribute for that timestep plus a constant
    ``emissivity`` (FLOAT/POINT) attribute. Meshes that were NOT simulated get an
    OBJECT-level ``heatsim_default_temperature`` custom property (=
    ``defaults['initial_temperature_K']``) so downstream rendering still has a
    sane fallback.
    """
    default_T = float(defaults["initial_temperature_K"])
    for obj in scene.objects:
        if getattr(obj, "type", None) != "MESH":
            continue
        mesh = getattr(obj, "data", None)
        if mesh is None:
            continue

        hist = history.get(obj.name)
        if hist is None:
            obj["heatsim_default_temperature"] = default_T
            continue

        arr = np.asarray(hist)
        n = len(mesh.vertices)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != n:
            # Topology mismatch (modifiers) or empty history: fall back to a
            # constant object-level temperature rather than writing garbage.
            obj["heatsim_default_temperature"] = default_T
            continue

        row = np.asarray(arr[timestep], dtype=np.float32).reshape(-1)
        _write_point_float_attr(mesh, "sim_temperature", row)
        eps = float(resolve_material(obj, defaults)["emissivity"])
        _write_point_float_attr(mesh, "emissivity", np.full(n, eps, dtype=np.float32))
        mesh.update()
