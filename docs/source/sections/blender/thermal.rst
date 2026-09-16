Thermal rendering
=================

VisionSim integrates a transient temperature field on a fixed scene and renders its
final state from one or more camera poses. Enable it with ``--config.include-thermal``.
The thermal solve is performed once for the scene; camera animation does not move
or reheat the simulated geometry. Moving/deforming thermal geometry and per-frame
thermal solves are outside the supported workflow.

The thermal pipeline samples scene surfaces at either evaluated mesh vertices or
thermal atlas texels, bakes incoming light and albedo with Cycles, and integrates
heat on a robust point-cloud Laplacian [1]_. The two representations use the same
physics. Cycles uses the scene's world and lights, including environment
textures supported by Blender; no HDRI is required.

Quick start
-----------

A scene with visible, renderable meshes and Cycles-compatible materials can be
rendered with::

    vsim blender.render-animation path/to/scene.blend path/to/output --config.include-thermal \
        --config.thermal.render-domain AUTO \
        --config.thermal.device cpu

``AUTO`` is the default. It assigns atlas texels to sparse surfaces, locally
coarse triangles and evaluated meshes that cannot safely map solved vertices
back to the base mesh. Dense meshes with safe correspondence retain vertices.
``TEXEL`` forces the atlas for participating nondegenerate surfaces, while
``VERTEX`` requests the diagnostic vertex path. Atlas preparation creates
thermal-specific UV layers; it does not replace the material UV coordinates used
by RGB rendering. A required atlas with unusable UVs or rasterization fails
explicitly. ``VERTEX`` cannot write a field to reordered or changed topology
safely and also fails in that case.

``atlas_texel_density`` is a target in samples per square metre. Tile bounds
and ``atlas_texel_soft_max`` limit memory; if the budget reduces effective
density, VisionSim warns. Bake image size and bake sample count are separate:
the former limits lighting detail, the latter primarily controls Monte Carlo
noise. Neither output image resolution nor increasing bake samples restores a
heating pattern that was never represented by thermal sample points.

Materials and fixed sources
---------------------------

The global defaults live in :class:`visionsim.simulate.config.ThermalConfig`.
Object ``heat_sim_material`` properties explicitly authored in Blender override
them. An optional JSON sidecar assigns presets and fixed-temperature roles per
Blender material slot. Save the following as ``scene.thermal.json``::

    {
      "schema_version": 1,
      "scene": "scene.blend",
      "defaults": {"preset": "plaster"},
      "materials": {
        "Wood Panel": {"preset": "wood"},
        "Steel Fixture": {"preset": "metal_painted"},
        "Warm Plate": {
          "preset": "metal_painted",
          "role": "DIRICHLET_SOURCE",
          "dirichlet_K": 320.0
        }
      }
    }

Pass it with ``--config.thermal.assignments path/to/scene.thermal.json``.
Names must match Blender material names. Preset values are defined in
:mod:`visionsim.simulate.heatsim.materials`; inspect and adjust assignments
for the actual asset. A ``DIRICHLET_SOURCE`` pins its samples at the specified
Kelvin temperature throughout integration. Unassigned slots use the sidecar
default preset, then the global defaults. The sidecar's bytes affect the solve
cache key. Invalid structure raises; unknown presets or roles warn and use
fallbacks. Material-slot emissivity is preserved at atlas texels as well as
vertices. Object-level material properties apply when the sidecar does not
override that material slot.

Model and units
---------------

Geometry enters the solver in millimetres, diffusivity in mm²/s, density in
kg/m³, specific heat in J/(kg·K), irradiance in W/m², and temperature in K.
Incident irradiance is approximated from Cycles' outgoing diffuse bake using
the Lambertian hemisphere factor :math:`\pi`. Absorbed flux is incident flux
times ``1 - albedo``. This is a broadband absorption approximation; visible
albedo is not a measured spectral absorptivity. Thermal emission uses emissivity
and the Stefan–Boltzmann law [2]_. It is a broadband gray-body approximation,
not a calibrated, band-limited LWIR camera. The radiance output currently uses
an exitance-equivalent :math:`\sigma T^4` scale; do not interpret its pixel
values as W/(m²·sr) without the Lambertian :math:`1/\pi` conversion.

The solver integrates from the initial temperature over ``sim_time_s`` in
``timestep_s`` steps. It uses a robust point-cloud Laplacian and connects
nearby samples; this is an approximate surface/contact conduction model and
can couple nearby, physically separate surfaces. Dirichlet samples are pinned.
Convection is disabled. The ambient linearization currently uses 295 K in the
solver and 295.372 K for background thermal radiance; these are fixed model
values, separate from the configurable initial temperature. Material presets
are starting points rather than measured values for a given asset.

Outputs and cache
-----------------

The temperature AOV is an EXR in Kelvin, with an optional colormap PNG preview.
The optional second pass renders broadband gray-body thermal signal. ZIP EXR
compression is the quantitative default. Texel temperatures and emissivity are
stored as float data in a Non-Color atlas; alpha marks covered/dilated pixels.
The atlas is packed into Blender memory for the render. Packing alone does not
save a modified blend file. Save an output blend explicitly if a frozen result
must be reopened independently of the external ``.heatsim`` directory; the
source blend is never overwritten for this purpose.

A reusable temperature cache is accepted only for a clean saved blend whose
bytes and external image/library files can be hashed. Unsaved edits, missing
assets and corrupt or shape-mismatched archives trigger a new bake and solve.
The key also contains solver settings, material defaults, assignments and the
sampling layout. The render atlas path includes the final field and pixel data,
so changing temperatures or emissivity cannot reuse an older atlas with the
same tile layout. Generated thermal attributes/UVs make a Blender session
dirty; repeated preparation in that session therefore recomputes conservatively.
Preview-only settings do not enter the heat-solve key. Set
``--config.thermal.recompute`` to bypass reusable results and regenerate the
Cycles bakes and heat solve.

Cycles persistent data is disabled during thermal setup because reusing device
state across changing AOV/material passes has produced intermittent zero-K
surfaces. The two-pass output muting keeps normal outputs out of the radiance
pass and the radiance file out of the normal pass.

References
----------

.. [1] Sharp and Crane (2020), `A Laplacian for Nonmanifold Triangle Meshes
   <https://www.cs.cmu.edu/~kmcrane/Projects/NonmanifoldLaplace/index.html>`_.
   The ``robust_laplacian`` point-cloud implementation is used here.
.. [2] NIST, `CODATA value of the Stefan–Boltzmann constant
   <https://physics.nist.gov/cgi-bin/cuu/Value?sigma>`_.
