# Volume conduction: upstream has it, visionsim does not

**Date:** 2026-09-01
**Status:** Known gap. Deliberately not ported — scope decision, 2026-09-01.

## The finding

Every `632_cup_pour_*` render produced through visionsim solved **surface-only**
conduction. Heat crossed the cup wall by diffusing across the surface, never through the
volume. The original addon renders of the same blend did not have this limitation.

## Evidence that the cup was solved upstream, not here

The blend's stored ID properties record where its cache came from:

```
heatsim_data_abspath    = /Users/sriram/research/heat_sim_blender/heat-sim-blender/
                          blender_files/632_cup_pour.heatsim/latest/temperatures.npz
heatsim_run_mode        = TRANSIENT
heatsim_evaluated_verts = 9292
heatsim_num_timesteps   = 600
heatsim_temp_min / max  = 294.96 / 342.34 K
```

A macOS path inside the addon repo. The object is named `geometry_0_watertight`, and
watertightness is precisely what interior sampling requires: the inside test is a BVH
ray-parity check, which is only meaningful on a closed mesh.

## What upstream implements

`heat-sim-blender/addon/lib/fem_adapter.py` reads three **per-object** properties off
`heat_sim_material`:

| property | meaning |
|---|---|
| `enable_point_volume` | add interior sample points so heat propagates through volume, not only across the surface |
| `point_volume_ratio` | interior point count as a ratio of that object's surface vertex count |
| `point_volume_min_spacing_mm` | minimum spacing between interior points |

`_sample_interior_points_for_object` implements **Bridson Poisson-disk** sampling
restricted to the interior by BVH ray-parity, seeded deterministically from the object
name, with a centroid-line fallback when the BVH cannot be built.

## What visionsim has

`adapter._augment_interior_points` exists and splices interior nodes into the solve
correctly -- they carry the object's material, zero incident flux, and are excluded from
the boundary. The splice is not the problem. Four things are:

1. **It is unreachable.** It reads `solver_cfg["interior_point_ratio"]`, and the complete
   set of keys visionsim ever writes into `solver_cfg` is `sim_time_s`, `timestep_s`,
   `domain`, `laplacian_backend`, `device`, `irradiance_source`, `bake_samples`. The ratio
   is never set, so it defaults to `0.0` and the feature is permanently off. There is no
   `ThermalConfig` field and no CLI flag.
2. **It is global.** One ratio for every object, scaled by each object's vertex count. Even
   if exposed it could not target one object.
3. **The per-object schema has nothing to hang it on.** visionsim's `heat_sim_material`
   PropertyGroup has zero volume fields, and the sidecar schema has no volume block.
4. **The sampler is weaker.** Plain rejection sampling, no Poisson-disk spacing, no
   fallback. Its own docstring calls it "a compact stand-in for upstream's Bridson
   sampler".

## What could not be recovered

The blend does **not** persist the settings that were used. Blender only writes non-default
PropertyGroup values, and neither `heat_sim_material` nor scene-level `heat_sim_settings`
appear as ID properties in `632_cup_pour_v4.blend` or `632_cup_pour_v22_10fps.blend`. So
the ratio and spacing the addon renders used are not knowable from the file -- only that
the solve ran TRANSIENT for 600 timesteps. `632_cup_pour.heatsim/latest/manifest.json` on
the original macOS checkout would have recorded them.

## If it is ported later

The work is: a volume block on the `heat_sim_material` PropertyGroup **and** in the sidecar
schema (so it is settable per object without a UI), the Bridson sampler ported from
`fem_adapter.py`, and the global `interior_point_ratio` replaced by a per-object lookup.
`_augment_interior_points` itself can stay.

Until then, treat any visionsim thermal result for a thick-walled or hollow object as
surface conduction only, and do not compare it against addon renders of the same asset.
