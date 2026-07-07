Thermal Modality
================

The thermal modality adds heat-transfer simulation outputs to any render produced by VisionSim.
When ``--config.include-thermal`` is set, VisionSim runs a finite-element-method (FEM)
heat-transfer solve on the scene geometry before the main render loop and produces temperature
and thermal-camera outputs per frame alongside the standard RGB/depth passes.

Enabling thermal
----------------

Thermal is off by default.  Turn it on with a single flag on any render command:

.. code-block:: bash

    vsim blender.render-animation scene.blend out/ --config.include-thermal

Everything else is optional tuning, exposed under the ``--config.thermal.*`` namespace and
documented in `Parameters`_ below.

Outputs
-------

With thermal enabled, three output directories are written per frame in addition to the usual
passes:

``temperature/``
    Per-pixel surface temperature in **Kelvin**, saved as a single-channel
    ``OPEN_EXR`` file.  This is a Cycles value AOV co-rendered with the RGB pass,
    so it adds no extra render samples.  Meshes that did not participate in the
    FEM solve report their fallback ``initial_temperature_K`` so the value is
    defined everywhere the geometry is visible.

``thermal_radiance/``
    Gray-body thermal-camera image produced by a **second Blender render** that
    replaces all scene materials with emission shaders driven by the solved
    surface temperature.  The 3-channel ``OPEN_EXR`` is proportional to the
    Stefan-Boltzmann radiated power (``emissivity × σ × T⁴``).  This is the most
    expensive part of thermal; disable it with ``--config.thermal.radiance False``
    when you only need the temperature map.

``previews/temperature/``
    An **inferno-colormap PNG** derived from the temperature map, for quick
    visual inspection.  The colormap spans the **global temperature range of the
    solved scene** (its minimum to maximum), so a scene with only a small
    temperature rise still uses the full colormap instead of being crushed to one
    end.  Controlled by ``--config.thermal.preview`` (default ``True``).

.. admonition:: Note

    ``temperature/`` and ``thermal_radiance/`` carry physically meaningful
    magnitudes (Kelvin and radiated power).  Load them with a library that
    preserves EXR float precision (e.g. ``OpenEXR``, ``imageio`` with the EXR
    plugin, or ``cv2``); the ``previews/`` PNGs are for display only and are not
    quantitative.

Rendering with thermal
----------------------

Thermal composes with the normal render commands — you can request it alongside any other
modality.  A few common invocations:

.. code-block:: bash

    # temperature + radiance + preview, GPU solve (defaults)
    vsim blender.render-animation scene.blend out/ --config.include-thermal

    # temperature map only — skip the (expensive) gray-body radiance render
    vsim blender.render-animation scene.blend out/ \
        --config.include-thermal \
        --config.thermal.radiance False

    # force a CPU solve (no CUDA required)
    vsim blender.render-animation scene.blend out/ \
        --config.include-thermal \
        --config.thermal.device cpu

    # make the scene heat up more (larger temperature rise)
    vsim blender.render-animation scene.blend out/ \
        --config.include-thermal \
        --config.thermal.irradiance-scale 1000

Animated geometry
-----------------

Everything above is a **static** solve: one temperature field is computed and held constant for
every frame of the render. Setting ``--config.thermal.animated`` switches to a **per-frame
transient solve** instead, so geometry that moves or deforms over the timeline produces a genuine
thermal *animation* rather than a single frozen field. The motivating example is a hot liquid
pouring into a cup: frame by frame, the liquid's heat diffuses into the cup and the cup's surface
visibly warms up over the sequence.

.. code-block:: bash

    vsim blender.render-animation cup_pour.blend out/ \
        --config.include-thermal \
        --config.thermal.animated \
        --config.thermal.substeps-per-frame 4

Animated mode distinguishes two kinds of objects, selected per object via the ``thermal_role``
material override (see `Per-object material overrides`_):

``FEM_PARTICIPANT`` objects (default) — stable topology
    Objects whose vertex count never changes across the animation (e.g. the cup itself). Their
    temperature **evolves**: each frame's solve carries the previous frame's result forward, so
    heat accumulates and diffuses realistically over time.

``DIRICHLET_SOURCE`` objects — topology-changing sources
    Objects whose mesh is regenerated every frame with a different vertex count — the standard
    situation for a fluid simulation's surface as it pours and splashes. A per-vertex temperature
    history can't be carried forward for a mesh like this, so instead it is treated as a
    **constant-temperature source**: every frame it drives heat into the nearby FEM-participant
    objects at its fixed ``dirichlet_temperature_K``, but its own temperature never evolves and is
    not part of the solve output. Set ``thermal_role = "DIRICHLET_SOURCE"`` on the hot liquid to
    get this behavior.

.. admonition:: Note

    The fluid (or other topology-changing) mesh must already be **baked** before running an
    animated thermal solve — e.g. a Mantaflow fluid domain baked to disk. The solver only reads
    geometry at each frame; it does not run or advance the fluid simulation itself, so the mesh
    must already exist at every frame the thermal solve visits.

.. admonition:: Note

    Animated mode currently requires ``--config.thermal.domain POINTS``. Requesting
    ``animated`` together with ``domain MESH`` logs a warning and falls back to the static
    (M1) solve described above.

The four animated-mode parameters are listed in the `Solver`_ table below; they only take effect
when ``animated`` is ``True``.

Parameters
----------

All thermal parameters live under ``--config.thermal.*`` and are collected in the
``ThermalConfig`` dataclass (:mod:`visionsim.simulate.config`).  The tables below list every
parameter, its default, and the effect of changing it.  Values set here are **global defaults**;
individual objects can override the material parameters (see `Per-object material overrides`_).

Output control
~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 26 12 62

    * - Parameter
      - Default
      - Effect
    * - ``radiance``
      - ``True``
      - Render the gray-body ``thermal_radiance/`` image (a second render pass).
        Set ``False`` to skip it and render roughly twice as fast.
    * - ``preview``
      - ``True``
      - Save the inferno-colormap ``previews/temperature/`` PNG.  Set ``False`` to
        skip preview generation.

Material defaults
~~~~~~~~~~~~~~~~~

These set the physical material used for every mesh that has no per-object override.

.. list-table::
    :header-rows: 1
    :widths: 30 12 58

    * - Parameter
      - Default
      - Effect
    * - ``initial-temperature-K``
      - ``295.0``
      - Starting temperature (K) of every vertex, and the fallback reported for
        meshes that do not participate in the solve.  Shifts the whole baseline.
    * - ``thermal-diffusivity-mm2-s``
      - ``0.17``
      - How fast heat spreads through the surface (mm²/s).  Higher values
        equalize temperature across the object faster (smoother field); lower
        values keep sharper local hot spots.
    * - ``density-kg-m3``
      - ``1330.0``
      - Material density.  With specific heat it sets the thermal mass: higher
        density means a slower temperature change for the same heat input.
    * - ``specific-heat-J-kgK``
      - ``880.0``
      - Heat capacity (J/kg·K).  Higher values require more energy to raise the
        temperature, so the rise is slower and smaller.
    * - ``emissivity``
      - ``0.9``
      - Surface emissivity in ``[0, 1]``.  Governs radiative cooling in the solve
        and the brightness of the ``thermal_radiance`` render — higher emissivity
        means more radiative loss and a brighter thermal image.

Solver
~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 28 14 58

    * - Parameter
      - Default
      - Effect
    * - ``irradiance-scale``
      - ``100.0``
      - Scales the absorbed-heat input that drives the temperature rise.  This is
        the main knob for **how hot the scene gets** — larger values produce a
        larger temperature rise.  (If the blend file carries authored thermal
        scene settings, that authored value takes precedence over this flag.)
    * - ``sim-time-s``
      - ``1.0``
      - Total simulated time of the static solve (seconds).  Longer times let the
        scene heat closer to steady state.
    * - ``timestep-s``
      - ``0.05``
      - Solver timestep (seconds).  Smaller steps are more accurate and stable at
        higher cost; the solve runs ``sim-time-s / timestep-s`` steps.
    * - ``domain``
      - ``POINTS``
      - FEM domain.  ``POINTS`` solves on a surface point cloud (recommended,
        robust to imperfect meshes); ``MESH`` solves on the mesh connectivity
        directly.
    * - ``laplacian-backend``
      - ``ROBUST``
      - Discrete-Laplacian construction.  ``ROBUST`` tolerates non-manifold or
        low-quality meshes; ``IGL`` uses the libigl cotangent Laplacian.
    * - ``device``
      - ``cuda``
      - Compute device for the solve.  Falls back to ``cpu`` automatically when
        CUDA is unavailable.
    * - ``animated``
      - ``False``
      - Enable the per-frame transient solve described in `Animated geometry`_,
        instead of the static single-shot solve above.  Requires
        ``domain = POINTS``; with ``domain = MESH`` it logs a warning and falls
        back to the static solve unchanged.
    * - ``substeps-per-frame``
      - ``4``
      - Solver substeps computed within each rendered Blender frame in animated
        mode.  More substeps make the per-frame integration more stable and
        accurate, at extra solve cost; the internal timestep is
        ``(1 / fps) / substeps-per-frame``.
    * - ``frame-start`` / ``frame-end``
      - scene frame range
      - First/last frame of the animated solve.  Defaults to the blend file's
        own ``frame_start``/``frame_end``.  Narrowing this range solves (and
        caches) only the portion of the timeline you actually render.
    * - ``every-n-frames``
      - ``1``
      - Solve every Nth frame instead of every frame, to cut solve cost on long
        sequences.  Frames that are skipped hold the most recently solved
        field rather than triggering a fresh solve.

Radiance render and file formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 24 14 62

    * - Parameter
      - Default
      - Effect
    * - ``radiance-scale``
      - ``1.0``
      - Brightness multiplier for the ``thermal_radiance`` image only.  A display
        scale — it does not change the temperature solve.
    * - ``exr-codec``
      - ``DWAA``
      - EXR compression codec for the ``temperature`` and ``thermal_radiance``
        files (e.g. ``ZIP``, ``PIZ``, ``DWAA``, ``NONE``).
    * - ``bit-depth``
      - ``32``
      - EXR channel bit depth, ``16`` or ``32``.  Use ``16`` for smaller files
        where full float precision is not required.

Per-object material overrides
-----------------------------

The `Material defaults`_ above apply to the whole scene.  To give a specific object a different
material, add a ``heat_sim_material`` property group to it in the blend file (registered at
``bpy.types.Object.heat_sim_material``).  Any field that is set overrides the corresponding
global default **for that object only**; objects without the property group fall back to the
globals.

The per-object fields are:

``initial_temperature_K``
    Starting temperature (K) for the object's vertices.

``thermal_diffusivity_mm2_s``
    Thermal diffusivity in mm²/s.

``density_kg_m3``
    Material density (kg/m³).

``specific_heat_J_kgK``
    Specific heat capacity (J/kg·K).

``emissivity``
    Surface emissivity in ``[0, 1]``, used for both radiation boundary conditions
    and the gray-body radiance render.

``thermal_role``
    Either ``"FEM_PARTICIPANT"`` (default — full transient solve) or
    ``"DIRICHLET_SOURCE"`` (vertex temperatures pinned to a constant value every
    step, i.e. a fixed-temperature heat source or sink).

``dirichlet_temperature_K``
    Constant temperature applied when ``thermal_role = "DIRICHLET_SOURCE"``.
    Falls back to ``initial_temperature_K`` when set to 0.

.. admonition:: Animated scenes

    These two fields are the mechanism behind `Animated geometry`_.  Give the
    topology-changing object — a pouring liquid, or any mesh whose vertex count
    changes frame to frame — ``thermal_role = "DIRICHLET_SOURCE"`` and a
    ``dirichlet_temperature_K``; leave stable-topology objects like the cup at
    the default ``"FEM_PARTICIPANT"`` so their temperature evolves across the
    animation instead of staying fixed.

Solve caching
-------------

The FEM solve is **lazy and cached**.  On the first render of a given blend file with a given set
of solver parameters, VisionSim runs the full solve and writes the per-object temperature
histories to::

    <blend_file>.heatsim/<cache_key>/temperatures.npz

where ``<cache_key>`` is a 16-character digest derived from the blend file path, its modification
timestamp, and the solver configuration.  Subsequent renders that use the same blend file and the
same parameters **skip the solve entirely** and load the cached result.  Changing any solver
parameter (or editing the blend) produces a new cache key and triggers a fresh solve.

To prime the cache ahead of a render run — useful when the solve is expensive and you want
rendering to start immediately — run the solve on its own:

.. code-block:: bash

    vsim blender.heatsim-solve scene.blend \
        --config.include-thermal \
        --config.thermal.device cpu

Because the cache key is anchored to the source blend file, a primed solve is reused by all later
``vsim blender.render-animation`` calls against the same blend at the same solver settings — even
in a different output directory.

Using thermal from the API
--------------------------

The same parameters are available when driving VisionSim programmatically.  ``ThermalConfig`` holds
every setting shown above and is attached to ``RenderConfig.thermal`` (gated by
``RenderConfig.include_thermal``); the render job forwards it to the two service calls that set up
thermal on the Blender side:

* :meth:`~visionsim.simulate.blender.BlenderService.exposed_prepare_thermal` — runs (or loads from
  cache) the FEM solve and prepares the scene for thermal rendering.
* :meth:`~visionsim.simulate.blender.BlenderService.exposed_include_thermal` — wires the
  temperature AOV, the inferno preview, and the gray-body radiance pass into the compositor.
* :meth:`~visionsim.simulate.blender.BlenderService.exposed_heatsim_solve` — the standalone
  solve-and-cache entry point behind ``vsim blender.heatsim-solve``.

See :mod:`visionsim.simulate.config` for the full ``ThermalConfig`` field reference.
