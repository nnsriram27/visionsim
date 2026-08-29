# Cycles Irradiance Bake: Closing the Emissive-Geometry Gap

**Date:** 2026-08-29
**Branch:** notes on `heatsim-devdocs`; the feature landed on `heatsim`
**Scene:** `visionsim50/classroom.blend`
**Status:** Implemented and verified. Opt-in via `--config.thermal.irradiance-source CYCLES_BAKE`.

## 1. The gap

`docs/devlog/README.md` recorded the classroom rendering thermally flat as a
known-open item, correctly diagnosed as physical rather than a bug. This note closes it.

The analytic Direct Kernel gathers flux from exactly one source set:

```python
light_objs = [o for o in scene.objects
              if o.type == "LIGHT" and not o.hide_render and o.visible_get()]
```

Only objects of **type `LIGHT`**, plus a 9-coefficient SH world-sky term. Meshes enter
the kernel only as BVH occluders, never as emitters. Its own docstring states the rest of
the limits: *"no indirect bounce, no specular contribution, sky is unshadowed by definition."*

The classroom violates all three at once:

| | |
|---|---|
| daylight source | `dayLight_portal`, an **emissive material** at strength 20 |
| its extent | **6.17 m²** across `windows` and `hallWindow` |
| world background strength | **0** — the SH sky term is identically zero |
| lamp objects present | sun at 1 W, five 100 W spots, a 60 W point |

So the scene's *actual* light source contributed **exactly zero** thermal flux, and what
remained could not reach most surfaces because the kernel carries no bounce. The room
stayed at its 295 K initial condition.

Measured, frame 301: interior p1–p99 spread **4.43 K**, median 295.32 K — i.e. flat.

## 2. Options considered

**A — teach the analytic kernel about emissive meshes.** Enumerate meshes with an
Emission node, sample their surface area-weighted, treat each sample as a Lambertian
emitter reusing the existing `evaluate_area` form-factor machinery and Embree occlusion.

Rejected as insufficient on its own: it fixes surfaces with *direct line of sight* to the
windows, but an interior is mostly lit by bounce. The predicted result is a bright pool
near the glass and a still-cold room.

**B — bake irradiance with Cycles.** A path tracer resolves emissive geometry, indirect
bounce, portals and HDRI transport in one step, because that is what it does.

Chosen. The decisive practical point: **this was a port, not new work.** heat-sim-blender
already has `bake_irradiance_map` and `bake_shared_irradiance`; visionsim had deliberately
vendored only the albedo bake (`irradiance.py`'s header says so explicitly).

## 3. Implementation

`bake_irradiance_map` is `bake_albedo_map` with the passes inverted:

| | albedo bake (existing) | irradiance bake (new) |
|---|---|---|
| `use_pass_color` | `True` | `False` |
| `use_pass_direct` / `_indirect` | `False` | `True` |
| `pass_filter` | `{"COLOR"}` | `{"DIRECT", "INDIRECT"}` |

Everything else is shared: UV snapshot/restore, `prepare_object_bake_uv`,
`_prepare_image_nodes_for_bake`, the bake-UV material overrides, and the
pixels→per-vertex sampler. Cycles bakes outgoing radiance, so the result is scaled by
`CYCLES_LOUT_TO_IRRADIANCE` (= π), a constant that already existed in `constants.py`.

**The one contract difference, handled at the call sites:** the Direct Kernel returns
**absorbed** flux (post-(1−albedo)); a `DIFFUSE` bake with `COLOR` off returns **incident**
irradiance. The Cycles paths therefore apply (1 − albedo), reusing the albedo bake that
already runs for the kernel — so the conversion adds no Cycles work.

Both render domains are covered:

- `_compute_irradiance_cycles` — per-vertex, from `BakedFluxMap.vertex_flux`
- `_texel_irradiance_cycles` — per-texel, bilinear-sampling the bake at the same UVs
  `_texel_albedo` already samples

Selected by `ThermalConfig.irradiance_source`: `DIRECT_KERNEL` (default, behaviour
unchanged) or `CYCLES_BAKE`.

## 4. Result

Classroom, frame 301, everything else identical:

| metric | DIRECT_KERNEL | CYCLES_BAKE |
|---|---|---|
| min | 170.7 K | 173.8 K |
| p1 | 295.27 K | 295.63 K |
| **median** | **295.32 K** | **312.61 K** |
| **p99** | **299.70 K** | **384.25 K** |
| max | 306.9 K | 452.2 K |
| **interior spread (p1–p99)** | **4.43 K** | **88.62 K** |

A 20× increase in spread, and a median 17 K above ambient rather than pinned to the
initial condition. The room now actually heats.

## 5. Cost

Solve time on classroom went **130 s → 379 s** — about +4 minutes, against a ~35 minute
600-frame render, and cached like any other solve.

A micro-benchmark beforehand predicted only +2%: 0.95 s/object for COLOR vs 0.97 s/object
for DIRECT+INDIRECT at 512px/128spp. That **under-predicted the real cost** because it
timed only the three largest objects, which turned out not to be representative of the
165-object sweep. The lesson is to benchmark a random sample rather than the extremes.
The conclusion still holds — the bake is affordable and one-time — but the honest figure
is ~3× the solve, not ~2% of it.

## 6. Do not gate this on "animated"

An earlier draft of this proposal suggested defaulting to the Direct Kernel for animated
scenes and the bake for static ones. **That gates on the wrong property**, as the user
pointed out: irradiance depends on geometry and lights, not on the camera. These datasets
animate the *camera* while geometry and lights stay put, so the solve is static and the
bake is a one-time cost regardless of how much the camera moves.

The Direct Kernel's speed advantage only applies when **geometry or lights actually change
per frame** — e.g. cup_pour's Mantaflow liquid. If a gate is ever added, that is the
condition to test.

## 7. What this does not address

- The **~2.3 K texel-domain mottling** (`texel-noise-investigation.md`) is unrelated and
  unchanged; it is a discretisation problem in the kNN point-cloud Laplacian.
- A **residual ~0 K region** still appears on isolated frames despite the non-mesh
  stamping fix, so some object type or material configuration still escapes
  `stamp_default_temperatures`. The classroom's worst frame minimum is 1.2 K.
- Bake **sample count is not exposed**. The bake inherits the scene's Cycles samples, and
  an interior lit through 6 m² of glass is a high-variance configuration — irradiance
  noise feeds straight into the temperature field. Worth measuring convergence before
  treating baked values as quantitative.

## 8. Applicability to the rest of the dataset

This is not a classroom quirk. Any scene lit by emissive geometry rather than lamp objects
hits it, and archviz interiors are routinely built that way — window planes, light
portals, emissive fixture panels. The signature is a scene whose thermal render is flat at
its initial temperature while its RGB render is well lit.

Worth sweeping the ~50-scene dataset for: emissive materials present, world background
strength 0, and an interior thermal spread of only a few Kelvin.
