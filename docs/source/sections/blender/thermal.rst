Thermal Modality
================

The thermal modality adds heat-transfer simulation outputs to any render produced by VisionSim.
When ``--config.include-thermal`` is set, VisionSim runs a finite-element-method (FEM)
heat-transfer solve on the scene geometry before the main render loop and produces three outputs
per frame alongside the standard RGB/depth passes.

Output directories
------------------

``temperature/``
    Per-pixel surface temperature in **Kelvin**, saved as a single-channel
    ``OPEN_EXR`` file.  This is a Cycles value AOV (``HeatSim_To``) that is
    co-rendered with the RGB pass; it adds no extra render samples.  Meshes that
    did not participate in the FEM solve (no per-vertex ``sim_temperature``)
    report their per-object ``heatsim_default_temperature`` fallback - the same
    value the gray-body radiance shader uses - so the two passes stay consistent.

``thermal_radiance/``
    Gray-body emission image produced by a **second Blender render** that
    replaces all scene materials with emission shaders driven by the solved
    ``sim_temperature`` vertex attribute.  The output is a 3-channel
    ``OPEN_EXR`` proportional to the Stefan-Boltzmann radiated power
    (``emissivity × σ × T⁴``).  Disable with ``--config.thermal.radiance False``
    to skip this extra render and save time.

``previews/temperature/``
    Turbo-colormap PNG derived from the temperature AOV, useful for quick
    visual inspection.  Controlled by ``--config.thermal.preview`` (default
    ``True``).

.. admonition:: Note

    ``temperature/`` and ``thermal_radiance/`` carry physically meaningful
    magnitudes (Kelvin and W/mm² respectively).  Load them with a library that
    preserves EXR float precision (e.g. ``OpenEXR``, ``imageio`` with the EXR
    plugin, or ``cv2``).

Basic CLI usage
---------------

.. code-block:: bash

    vsim blender.render-animation scene.blend out/ --config.include-thermal

Selected tuning flags:

.. code-block:: bash

    # skip the gray-body radiance render (faster)
    vsim blender.render-animation scene.blend out/ \
        --config.include-thermal \
        --config.thermal.radiance False

    # force CPU solve (no CUDA required)
    vsim blender.render-animation scene.blend out/ \
        --config.include-thermal \
        --config.thermal.device cpu

    # use the MESH FEM domain instead of the default POINTS point-cloud domain
    vsim blender.render-animation scene.blend out/ \
        --config.include-thermal \
        --config.thermal.domain MESH

    # set a global surface emissivity
    vsim blender.render-animation scene.blend out/ \
        --config.include-thermal \
        --config.thermal.emissivity 0.95

See :meth:`~visionsim.simulate.blender.BlenderService.exposed_include_thermal` for the
full parameter reference.

Per-object material model
--------------------------

Every mesh object in the scene can carry a ``heat_sim_material`` property group
(a ``HeatSimObjectMaterialProperties`` Blender PropertyGroup registered at
``bpy.types.Object.heat_sim_material``).  When present, its values override the
global ``ThermalConfig`` defaults for that object only; meshes without per-object
values fall back to the globals.

The per-object fields are:

``initial_temperature_K``
    Starting temperature (K) for the object's vertices.

``thermal_diffusivity_mm2_s``
    Thermal diffusivity in mm²/s (consistent with the mm-unit FEM domain).

``density_kg_m3``
    Material density (kg/m³).

``specific_heat_J_kgK``
    Specific heat capacity (J/kg·K).

``emissivity``
    Surface emissivity in [0, 1], used for both radiation boundary conditions
    and the gray-body radiance render.

``thermal_role``
    Either ``"FEM_PARTICIPANT"`` (default — full transient solve) or
    ``"DIRICHLET_SOURCE"`` (vertex temperatures pinned to a constant value every
    step, i.e. a heat source/sink with fixed temperature).

``dirichlet_temperature_K``
    Constant temperature applied when ``thermal_role = "DIRICHLET_SOURCE"``.
    Falls back to ``initial_temperature_K`` when set to 0.

Cache behavior
--------------

The FEM solve is **lazy and cached**.  On the first render of a given blend file
with a given set of solver parameters, VisionSim runs the full solve and writes
the per-object temperature histories to::

    <blend_file>.heatsim/<cache_key>/temperatures.npz

where ``<cache_key>`` is a 16-character SHA-1 digest derived from the blend
file path, its modification timestamp, and the solver configuration.  Subsequent
renders that use the same blend file and parameters **skip the solve entirely**
and load the cached result.

To prime the cache before a render run (useful when the solve is expensive and
you want rendering to proceed immediately):

.. code-block:: bash

    vsim blender.heatsim-solve scene.blend \
        --config.include-thermal \
        --config.thermal.device cpu

Because the cache key is anchored to the source blend file, a primed solve is
reused by all later ``vsim blender.render-animation`` calls against the same
blend at the same solver settings — even in a different output directory.

Implementation notes
--------------------

This section documents what was added to VisionSim to support the thermal
modality.

**Vendored solver package — ``visionsim/simulate/heatsim/``**

The thermal numerics live in a self-contained package vendored from
`heat-sim-blender` (provenance tracked in
``visionsim/simulate/heatsim/VENDOR.md``).  The package contains:

* ``solver.py`` — implicit backward-Euler FEM time-stepper with PyTorch sparse
  tensors (handles radiation and convection boundary terms).
* ``laplacian.py`` — Laplacian and mass-matrix construction via
  ``robust_laplacian`` or ``igl``.
* ``irradiance_kernel.py`` + sky/light helpers — Direct-Kernel irradiance
  computation that feeds absorbed solar flux as a heat-flux boundary condition
  to the FEM solver.
* ``adapter.py`` — scene-level glue: extracts geometry from the live Blender
  scene, resolves per-object material properties, calls the solver, and writes
  ``sim_temperature`` vertex attributes back to the mesh objects.
* ``cache.py`` — ``cache_key`` / ``write_temperatures`` / ``read_temperatures``
  helpers that implement the lazy NPZ cache described above.
* ``properties.py`` — the ``HeatSimObjectMaterialProperties`` PropertyGroup
  registered on ``bpy.types.Object.heat_sim_material``.
* ``thermal_shader.py`` — sets up the gray-body emission material tree used
  during the thermal radiance second render.
* ``constants.py`` — Stefan-Boltzmann constant and reference material library.

**The pre-render hook — thermal as the first stateful modality**

Every other VisionSim modality (depth, normals, flow, segmentation, …) is a
stateless single-render AOV pass: ``include_*`` wires compositor nodes and the
render loop does the rest without any prior computation.

Thermal is different.  It is the **first modality requiring a precompute step**:

1. ``exposed_prepare_thermal`` solves the FEM heat equation on the scene
   geometry (or loads the result from cache) and writes ``sim_temperature``
   vertex attributes onto every mesh *before* the render loop starts.  It also
   registers a ``temperature`` Cycles value AOV on each material.

2. ``exposed_include_thermal`` then wires the ``temperature`` AOV output socket
   into the compositor (temperature EXR + optional turbo-colormap preview) and
   arms an in-render-loop second-render hook for the gray-body radiance pass.

3. During the render loop, each call to ``exposed_render_current_frame`` detects
   the armed second-render flag and performs a full second Blender render with
   all scene materials replaced by emission shaders (output-node muting prevents
   the main RGB/AOV pass and the gray-body pass from clobbering each other's
   file outputs).

**Service methods, config, and dispatch**

* :meth:`~visionsim.simulate.blender.BlenderService.exposed_prepare_thermal` —
  runs (or loads) the FEM solve and prepares the scene for thermal rendering.
* :meth:`~visionsim.simulate.blender.BlenderService.exposed_include_thermal` —
  wires the compositor and arms the radiance second-render hook.
* :meth:`~visionsim.simulate.blender.BlenderService.exposed_heatsim_solve` —
  standalone solve-and-cache command (used by ``vsim blender.heatsim-solve``).
* ``ThermalConfig`` in ``visionsim/simulate/config.py`` — dataclass holding all
  thermal tuning parameters; included in ``RenderConfig.thermal`` and gated by
  ``RenderConfig.include_thermal``.
* ``visionsim/simulate/job.py`` — dispatches ``prepare_thermal`` +
  ``include_thermal`` before the render loop when ``config.include_thermal`` is
  set, passing ``**asdict(config.thermal)`` to both calls.
* ``vsim blender.heatsim-solve`` CLI command in
  ``visionsim/cli/blender.py`` — thin wrapper around ``exposed_heatsim_solve``
  that lets users prime the cache without triggering a render.
