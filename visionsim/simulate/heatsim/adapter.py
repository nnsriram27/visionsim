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

from visionsim.simulate.heatsim import cache, materials

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


def read_authored_irradiance_scale(scene: Any) -> Optional[float]:
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


def global_temperature_range(history: dict[str, Any], default_K: float) -> tuple[float, float]:
    """Global colormap range ``(tmin, tmax)`` in Kelvin over the solved scene.

    Spans the **final-timestep** temperatures of every solved object (M1 renders
    the final state on every frame), so the thermal preview colormap covers the
    actual data instead of a fixed 295-400 K band. The lower bound is floored at
    ``default_K`` — unsolved meshes are stamped at that temperature — and the span
    is widened to at least 1 K so a near-uniform field does not collapse to a
    single colour.

    Args:
        history: ``{obj_name: (timesteps, vertices) array}`` from :func:`solve_scene`.
        default_K: Default initial temperature stamped on unsolved meshes.

    Returns:
        ``(tmin, tmax)`` with ``tmax - tmin >= 1.0``.
    """
    finals = []
    for arr in history.values():
        a = np.asarray(arr, dtype=float)
        if a.size:
            finals.append(a[-1] if a.ndim >= 2 else a)
    if not finals:
        return float(default_K), float(default_K) + 1.0
    tmin = min(float(np.min(a)) for a in finals)
    tmax = max(float(np.max(a)) for a in finals)
    tmin = min(tmin, float(default_K))
    if tmax - tmin < 1.0:
        tmax = tmin + 1.0
    return tmin, tmax


def global_temperature_range_animated(history: dict[str, Any], default_K: float) -> tuple[float, float]:
    """Global colormap range ``(tmin, tmax)`` in Kelvin spanning EVERY frame of an animated solve.

    Like :func:`global_temperature_range`, but folds in the full per-frame history instead
    of only the final timestep. The animated (M2) field keeps evolving frame to frame, so a
    final-frame-only range would clip the preview colormap against later (typically hotter)
    frames -- this instead gives a single, stable scale for the whole sequence.

    Args:
        history: ``{obj_name: (n_frames, n_verts) array}`` from :func:`solve_scene_animated`.
        default_K: Default initial temperature stamped on unsolved meshes.

    Returns:
        ``(tmin, tmax)`` with ``tmax - tmin >= 1.0``.
    """
    arrays = [np.asarray(arr, dtype=float) for arr in history.values() if np.asarray(arr).size]
    if not arrays:
        return float(default_K), float(default_K) + 1.0
    tmin = min(float(np.min(a)) for a in arrays)
    tmax = max(float(np.max(a)) for a in arrays)
    tmin = min(tmin, float(default_K))
    if tmax - tmin < 1.0:
        tmax = tmin + 1.0
    return tmin, tmax


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


def _compute_irradiance(scene: Any, sim_objects: list, solver_cfg: dict, defaults: dict) -> dict:
    """Run the Direct-Kernel and return ``{obj: (N,) float64 W/m^2 absorbed}``."""
    from visionsim.simulate.heatsim import irradiance_kernel

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


def _combine(
    sim_objects: list,
    flux_by_obj: dict,
    defaults: dict,
    solver_cfg: dict,
    assignment: Optional[Any] = None,
) -> Optional[SimpleNamespace]:
    """Stack per-object geometry + material vectors into one solver-ready system.

    Surface vertices come first (layout records each object's slice); optional
    interior POINTS-mode samples are appended afterwards by
    :func:`_augment_interior_points` so the surface slices stay valid.

    When *assignment* (a parsed thermal sidecar) is supplied and an object has
    usable material slots, alpha/rho/c/eps/T0 and the Dirichlet mask are resolved
    **per vertex** from the slot assignment
    (:func:`materials.resolve_vertex_materials`) instead of being filled with one
    object-level constant. ``resolve_material`` still supplies the per-object
    fallback for unassigned slots, so addon-authored blends are unaffected. With
    ``assignment=None`` this function behaves exactly as before.
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

        per_vertex = None
        if assignment is not None:
            per_vertex = materials.resolve_vertex_materials(obj, assignment, mat)
            # resolve_vertex_materials walks the object's *base* mesh, so its arrays
            # are sized to len(obj.data.vertices). _combine operates on the *evaluated*
            # geometry from _extract_geometry, whose vertex count differs when the
            # object carries topology-changing modifiers (Subdivision, Array, ...).
            # When they disagree we cannot map slots to evaluated vertices, so fall
            # back to the object-level path for this object rather than crash - the
            # same shape-mismatch degradation write_frame_attributes already applies.
            if per_vertex is not None and int(per_vertex["alpha"].shape[0]) != n:
                _log.warning(
                    "[heatsim.adapter] '%s': base mesh has %d verts but evaluated geometry has %d "
                    "(topology-changing modifier); per-slot thermal materials skipped, using object-level values.",
                    obj.name, int(per_vertex["alpha"].shape[0]), n,
                )
                per_vertex = None

        flux = flux_by_obj.get(obj)
        if flux is not None and int(np.asarray(flux).reshape(-1).shape[0]) == n:
            irr = (np.asarray(flux, dtype=np.float64).reshape(-1) / _WM2_TO_WMM2) * irradiance_scale
        else:
            irr = np.zeros(n, dtype=np.float64)

        if per_vertex is not None:
            # Per-slot resolution: every field is already (N,). Dirichlet vertices
            # are pinned individually - alpha=0, no incident flux, excluded from the
            # radiation/convection boundary - exactly what the object-level branch
            # below does, just per vertex. T0 is the simulation's *initial condition*,
            # not a constitutive property: a FEM-participant vertex starts at ambient
            # even where it seams against a Dirichlet slot (materials.resolve_vertex_materials
            # area-weights T0 like any other continuous field, which would otherwise
            # pre-seed seam vertices with a slice of the reservoir's temperature before
            # the solver ever runs a step -- mirrors the object-level branch below,
            # which always starts a non-Dirichlet object at initial_temperature_K).
            rho = per_vertex["rho"]
            c_vec = per_vertex["c"]
            eps = per_vertex["eps"]
            dmask = per_vertex["dirichlet_mask"]
            t0 = np.where(dmask, per_vertex["t0"], mat["initial_temperature_K"])
            alpha = np.where(dmask, 0.0, per_vertex["alpha"])
            irr = np.where(dmask, 0.0, irr)
            bmask = ~dmask
        else:
            rho = np.full(n, mat["density_kg_m3"], dtype=np.float64)
            c_vec = np.full(n, mat["specific_heat_J_kgK"], dtype=np.float64)
            eps = np.full(n, mat["emissivity"], dtype=np.float64)
            if mat["thermal_role"] == "DIRICHLET_SOURCE":
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
        rho_l.append(rho)
        c_l.append(c_vec)
        eps_l.append(eps)
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

    Known per-vertex-materials gap: materials here are resolved object-level via
    :func:`resolve_material`, not per vertex via :func:`materials.resolve_vertex_materials`,
    so interior POINTS-domain samples do not reflect per-slot material variation.
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


def solve_scene(
    scene: Any, *, defaults: dict, solver_cfg: dict, cache_root: Path, assignment: Optional[Any] = None
) -> dict:
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
        "assignments": None if assignment is None else assignment.digest,
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
    combined = _combine(sim_objects, flux_by_obj, defaults, solver_cfg, assignment=assignment)
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


def _scene_fps(scene: Any) -> float:
    """``scene.render.fps / scene.render.fps_base``, guarded against a zero base."""
    r = scene.render
    fps = float(getattr(r, "fps", 24) or 24)
    fps_base = float(getattr(r, "fps_base", 1.0) or 1.0)
    if fps_base <= 0.0:
        fps_base = 1.0
    return fps / fps_base


def _order_fem_first(sim_objects: list, defaults: dict) -> tuple[list, set]:
    """Split ``sim_objects`` into FEM-participant-first, Dirichlet-source-last order.

    A Dirichlet source's evaluated vertex count may change frame to frame (e.g. a
    regenerated fluid mesh). Keeping it last in the combine order means its resize
    never shifts the ``_combine`` offsets of any FEM-participant object, so the
    FEM-participant surface prefix has a stable width across frames -- exactly the
    property :func:`solve_scene_animated` relies on to carry ``u_prev`` forward by
    index.
    """
    fem: list = []
    dirichlet: list = []
    dirichlet_names: set = set()
    for obj in sim_objects:
        role = resolve_material(obj, defaults)["thermal_role"]
        if role == "DIRICHLET_SOURCE":
            dirichlet.append(obj)
            dirichlet_names.add(obj.name)
        else:
            fem.append(obj)
    return fem + dirichlet, dirichlet_names


def _slot_level_dirichlet_mismatches(sim_objects: list, assignment: Any, defaults: dict) -> list:
    """Objects with a **slot-level** ``DIRICHLET_SOURCE`` assignment their own
    ``heat_sim_material.thermal_role`` does not already declare.

    :func:`_order_fem_first` (and the per-substep re-pin loop in
    :func:`solve_scene_animated`) only ever look at the object-level role, so a
    material slot the sidecar assigns ``DIRICHLET_SOURCE`` is never re-pinned
    between substeps in animated mode even though :func:`_combine` correctly
    zeroes its ``alpha``/``irradiance`` and clears its ``boundary_mask`` for the
    static solve. This is a detector for that gap, used to warn the caller
    rather than silently drifting -- see :func:`solve_scene_animated`.
    """
    names: list = []
    for obj in sim_objects:
        if resolve_material(obj, defaults)["thermal_role"] == "DIRICHLET_SOURCE":
            continue  # already re-pinned wholesale as an object-level Dirichlet source
        for slot in getattr(obj, "material_slots", []):
            material = getattr(slot, "material", None)
            if material is None:
                continue
            entry = assignment.entry_for(str(getattr(material, "name", "")))
            if entry is not None and entry.role == "DIRICHLET_SOURCE":
                names.append(obj.name)
                break
    return names


def solve_scene_animated(
    scene: Any,
    *,
    defaults: dict,
    solver_cfg: dict,
    cache_root: Path,
    frame_start: int,
    frame_end: int,
    every_n: int,
    substeps_per_frame: int,
    assignment: Optional[Any] = None,
) -> tuple[dict, list]:
    """Transient per-frame FEM solve over an animated scene (Phase 1 / M2).

    ``FEM_PARTICIPANT`` objects (stable topology) evolve: ``u_prev`` carries the
    previous frame's final state forward by vertex index. ``DIRICHLET_SOURCE``
    objects (topology may change frame to frame, e.g. a regenerated fluid mesh)
    are a constant-temperature reservoir -- re-extracted every frame purely for
    *position* (so they couple heat into nearby FEM-participant vertices through
    the POINTS kNN Laplacian; see ``solver._build_matrices``, which builds that
    Laplacian over ALL combined points regardless of source object) and reset to
    the reservoir temperature every frame. Their own field never evolves and is
    not recorded.

    On a vertex-count change of any Dirichlet source, the combined system is
    rebuilt from scratch for that frame (via :func:`_combine`); the
    FEM-participant prefix of ``u_prev`` is preserved unchanged and the Dirichlet
    slice is refilled with its (possibly keyframed) reservoir temperature. A
    vertex-count change on a FEM-participant object is NOT supported (matches
    heat-sim-blender's ANIMATE Phase 1) and raises ``RuntimeError`` with a clear
    message pointing at ``thermal_role``.

    POINTS domain only. Per the M2 design (irradiance re-bake is a deferred
    non-goal -- see the design spec), irradiance is not computed here: coupling
    into the FEM participants comes from the Dirichlet reservoir's *position* via
    the shared POINTS Laplacian, which is sufficient for a "hot pour" testbed.
    Known v1 limitation: interior-point-volume samples (POINTS domain,
    ``interior_point_ratio`` > 0) are reseeded at their configured initial
    temperature every frame rather than carried forward (deformation-aware
    interior continuity is an explicit M2 non-goal).

    Known limitation -- **slot-level Dirichlet sources are not supported here**:
    when *assignment* is given, a material slot the sidecar assigns
    ``DIRICHLET_SOURCE`` is pinned correctly in the static :func:`solve_scene`
    but is only re-pinned *once per frame* here, not after every substep (the
    per-substep re-pin loop below only knows about object-level
    ``thermal_role``). Across substeps the FEM/Dirichlet coupling then lets
    those vertices drift away from their reservoir temperature. A scene
    exhibiting this logs a warning naming the affected object(s); use the
    static solve for such scenes until per-vertex Dirichlet is supported in
    the animated substep loop.

    Returns ``({obj_name: (n_frames, n_surface_verts) ndarray}, frames)`` for
    FEM-participant objects only -- Dirichlet sources are not recorded, since
    their temperature is always the constant we configured. Cache-aware: a hit
    on ``cache.read_animated`` short-circuits the solve entirely.
    """
    cache_root = Path(cache_root)
    sim_objects = gather_meshes(scene)

    frame_start = int(frame_start)
    frame_end = int(frame_end)
    every_n = max(1, int(every_n))
    substeps_per_frame = max(1, int(substeps_per_frame))
    frames = list(range(frame_start, frame_end + 1, every_n))

    blend_path = Path(str(getattr(getattr(bpy, "data", None), "filepath", "") or ""))
    key_cfg = {
        "solver": dict(solver_cfg),
        "defaults": dict(defaults),
        "objects": sorted(o.name for o in sim_objects),
        "animated": True,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "every_n": every_n,
        "substeps": substeps_per_frame,
        "assignments": None if assignment is None else assignment.digest,
    }
    key = cache.cache_key(blend_path, key_cfg)
    cache_dir = cache_root / key

    cached = cache.read_animated(cache_dir)
    if cached is not None:
        _log.debug("[heatsim.adapter] animated cache hit: %s", key)
        return cached

    if not frames or not sim_objects:
        cache.write_animated(cache_dir, {}, frames, {"objects": []})
        return {}, frames

    ordered_objects, dirichlet_names = _order_fem_first(sim_objects, defaults)
    objects_by_name = {o.name: o for o in ordered_objects}

    # M2 design: irradiance re-bake is a deferred non-goal for animated mode --
    # ``_combine`` below is always called with an empty ``flux_by_obj``, so every
    # object's incident flux is always zero. The only possible heat source for an
    # animated solve is therefore a DIRICHLET_SOURCE reservoir; warn once so a
    # scene with neither doesn't silently stay at ambient for the whole run.
    if not dirichlet_names:
        _log.warning(
            "[heatsim.adapter] solve_scene_animated: no DIRICHLET_SOURCE object and "
            "animated mode never re-bakes irradiance -- this scene has no heat "
            "source, so the solved field will stay at ambient temperature for the "
            "entire run."
        )

    if assignment is not None:
        mismatched = _slot_level_dirichlet_mismatches(ordered_objects, assignment, defaults)
        if mismatched:
            _log.warning(
                "[heatsim.adapter] solve_scene_animated: %s: material slot(s) assigned "
                "DIRICHLET_SOURCE in the thermal sidecar, but the object-level "
                "thermal_role does not declare it. Slot-level Dirichlet sources are NOT "
                "re-pinned per substep in animated mode, so these vertices will drift "
                "away from their reservoir temperature within a frame -- use the static "
                "solve (solve_scene) for this scene until per-vertex Dirichlet is "
                "supported in the animated substep loop.",
                ", ".join(sorted(mismatched)),
            )

    from visionsim.simulate.heatsim.solver import HeatSimFEM

    device = str(solver_cfg.get("device", "cpu"))
    domain = str(solver_cfg.get("domain", "POINTS")).upper()
    backend = str(solver_cfg.get("laplacian_backend", "ROBUST")).upper()

    fps = _scene_fps(scene)
    dt = (1.0 / fps) / float(substeps_per_frame)

    gen_params = SimpleNamespace(
        device=device,
        RHO=float(defaults["density_kg_m3"]) / _KGM3_TO_KGMM3,
        C=float(defaults["specific_heat_J_kgK"]),
        K=float(defaults["thermal_diffusivity_mm2_s"]),
        NUM_FRAME_DELTA=dt * 60.0,
    )
    sim_params = SimpleNamespace(
        sim_radiation=True,
        sim_convection=False,
        add_tikhonov_reg=False,
        sim_time=0.0,
        record_time=0.0,
    )
    fem = HeatSimFEM(
        gen_params,
        sim_params,
        laplacian_domain=domain,
        laplacian_backend=backend,
        robust_mollify_factor=float(solver_cfg.get("robust_mollify_factor", 1e-5)),
        pointcloud_neighbors=int(solver_cfg.get("pointcloud_neighbors", 30)),
    )

    orig_frame = int(scene.frame_current)
    history_rows: dict = {}
    u_prev: Optional[np.ndarray] = None
    n_fem_surface: Optional[int] = None

    try:
        for f in frames:
            scene.frame_set(int(f))
            combined = _combine(ordered_objects, {}, defaults, solver_cfg, assignment=assignment)
            if combined is None:
                raise RuntimeError(
                    f"[heatsim.adapter] solve_scene_animated: no geometry at frame {f} "
                    "(all sim objects vanished mid-run)."
                )

            fem_entries = [(name, off, n) for name, off, n in combined.layout if name not in dirichlet_names]
            cur_n_fem_surface = sum(n for _, _, n in fem_entries)

            if u_prev is None:
                u_prev = combined.t0.copy()
                n_fem_surface = cur_n_fem_surface
            else:
                if cur_n_fem_surface != n_fem_surface:
                    raise RuntimeError(
                        "[heatsim.adapter] solve_scene_animated: a FEM_PARTICIPANT "
                        f"object's vertex count changed at frame {f} (expected "
                        f"{n_fem_surface} surface vertices, got {cur_n_fem_surface}). "
                        "Animated mode requires stable topology for FEM participants; "
                        "set thermal_role=DIRICHLET_SOURCE for meshes with changing "
                        "topology (e.g. fluids)."
                    )
                new_u_prev = combined.t0.copy()
                new_u_prev[:n_fem_surface] = u_prev[:n_fem_surface]
                u_prev = new_u_prev

            # Dirichlet pin: resolve each reservoir vertex's target temperature
            # fresh every frame (also covers a future keyframed
            # dirichlet_temperature_K) and hand the indices/values to
            # simulate_for_pose so it re-pins them internally AFTER EVERY
            # substep's CG solve, not just once on the returned array -- the
            # FEM/Dirichlet coupling weight in the solve would otherwise let
            # pinned nodes drift within a frame across substeps (and drift
            # further as substeps_per_frame grows).
            dir_idx: list[int] = []
            dir_val: list[float] = []
            for name, off, n in combined.layout:
                if name in dirichlet_names:
                    mat = resolve_material(objects_by_name[name], defaults)
                    t_target = mat["dirichlet_temperature_K"] or mat["initial_temperature_K"]
                    dir_idx.extend(range(off, off + n))
                    dir_val.extend([t_target] * n)

            states = fem.simulate_for_pose(
                combined.verts,
                combined.faces,
                combined.boundary_mask,
                u_prev,
                combined.irradiance,
                combined.alpha,
                combined.density,
                combined.c,
                combined.eps,
                num_substeps=substeps_per_frame,
                dt=dt,
                dirichlet_indices=dir_idx or None,
                dirichlet_values=dir_val or None,
            )  # (substeps, N); Dirichlet rows already pinned every substep.

            u_prev = states[-1].copy()
            for name, off, n in fem_entries:
                history_rows.setdefault(name, []).append(
                    np.asarray(states[-1, off : off + n], dtype=np.float64)
                )
    finally:
        scene.frame_set(orig_frame)

    history = {name: np.stack(rows, axis=0) for name, rows in history_rows.items()}

    meta = {
        "objects": sorted(history),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "every_n": every_n,
        "substeps": substeps_per_frame,
    }
    cache.write_animated(cache_dir, history, frames, meta)
    _log.debug("[heatsim.adapter] animated solve: %d object(s), %d frame(s)", len(history), len(frames))
    return history, frames


def _write_point_float_attr(mesh: Any, name: str, values: np.ndarray) -> None:
    """(Re)create a POINT/FLOAT mesh attribute from ``values``."""
    if name in mesh.attributes:
        try:
            mesh.attributes.remove(mesh.attributes[name])
        except Exception:
            pass
    attr = mesh.attributes.new(name=name, type="FLOAT", domain="POINT")
    attr.data.foreach_set("value", np.asarray(values, dtype=np.float32))


def _fallback_temperature_K(obj: Any, defaults: dict, default_T: float) -> float:
    """Object-level fallback temperature for ``write_frame_attributes``.

    A ``DIRICHLET_SOURCE`` (e.g. a topology-changing hot liquid whose
    evaluated vertex count doesn't line up with its base mesh, so it can't be
    given a per-vertex history) must still render at its reservoir
    temperature rather than ambient. Everything else (FEM participants)
    keeps the ambient ``initial_temperature_K`` default.
    """
    material = resolve_material(obj, defaults)
    if material["thermal_role"] == "DIRICHLET_SOURCE":
        return float(material["dirichlet_temperature_K"]) or float(material["initial_temperature_K"])
    return default_T


def write_frame_attributes(
    scene: Any, history: dict, timestep: int, defaults: dict, assignment: Optional[Any] = None
) -> None:
    """Write per-vertex temperatures for the chosen ``timestep`` (use ``-1`` for last).

    For every simulated mesh (present in ``history``) this writes a
    ``sim_temperature`` (FLOAT/POINT) attribute for that timestep plus a constant
    ``emissivity`` (FLOAT/POINT) attribute. Meshes that were NOT simulated get an
    OBJECT-level ``heatsim_default_temperature`` custom property so downstream
    rendering still has a sane fallback: ``defaults['initial_temperature_K']``
    (ambient) for FEM participants, or the object's own
    ``dirichlet_temperature_K`` reservoir temperature for a ``DIRICHLET_SOURCE``
    (e.g. a topology-changing hot liquid whose vertex count can't be tracked
    per-frame) so it still renders hot instead of at ambient.

    When *assignment* is supplied the ``emissivity`` attribute is resolved **per
    vertex** from the object's material slots rather than stamped as one
    object-level constant, so per-slot emissivity reaches the gray-body radiance
    shader. In LWIR that difference (polished metal ~0.05 vs painted ~0.9)
    dominates how the rendered frame looks.
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
            obj["heatsim_default_temperature"] = _fallback_temperature_K(obj, defaults, default_T)
            continue

        arr = np.asarray(hist)
        n = len(mesh.vertices)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != n:
            # Topology mismatch (modifiers) or empty history: fall back to a
            # constant object-level temperature rather than writing garbage.
            obj["heatsim_default_temperature"] = _fallback_temperature_K(obj, defaults, default_T)
            continue

        row = np.asarray(arr[timestep], dtype=np.float32).reshape(-1)
        _write_point_float_attr(mesh, "sim_temperature", row)

        material = resolve_material(obj, defaults)
        eps_vec = None
        if assignment is not None:
            per_vertex = materials.resolve_vertex_materials(obj, assignment, material)
            if per_vertex is not None:
                eps_vec = np.asarray(per_vertex["eps"], dtype=np.float32).reshape(-1)
        if eps_vec is None or eps_vec.shape[0] != n:
            eps_vec = np.full(n, float(material["emissivity"]), dtype=np.float32)
        _write_point_float_attr(mesh, "emissivity", eps_vec)
        mesh.update()
