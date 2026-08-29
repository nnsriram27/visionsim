# Texel-Domain Noise: Investigation and Findings

**Date:** 2026-08-29
**Branch:** notes on `heatsim-devdocs`; the code fixes landed on `heatsim`
**Scene:** `visionsim50/kitchen1.blend`, frame 301, 512×512, `Camera.004`
**Status:** Three bugs fixed and verified. The dominant artifact (spatial mottling)
is **unresolved** and traced to the texel-domain discretisation itself; every
tunable parameter was eliminated by measurement.

## 1. Symptoms

Rendering kitchen1 with `--config.thermal.render-domain TEXEL` produced three
visually distinct artifacts:

1. Regions rendering at ~0 K (black in the preview).
2. ~3400 pixels below the 295 K solve floor, down to 41 K, on geometric edges.
3. Persistent speckles and low-frequency mottling on flat surfaces, which survived
   every render-side remedy.

The user's observation that led to the decisive framing: *"temperature seems okay,
but only the thermal radiance seems noisy."*

That is explained and is not a separate defect. Radiance is `ε·σT⁴`, so
`dR/R = 4·dT/T`. Measured amplification is 3.51× (TEXEL) and 3.61× (VERTEX) against
a theoretical 4.0×. Display compounds it: the temperature ripple spans 3.09% of its
colormap window, the radiance ripple 10.39% of its. Same defect, ~10× more visible
in radiance.

## 2. Baseline measurements

All numbers are frame 301, flat cabinet patch `[30:175, 70:300]`, ripple measured as
the std of `x − median_filter(x, 41)`.

| configuration | T ripple | R ripple | sub-295 K px | min | max |
|---|---|---|---|---|---|
| TEXEL (before fixes) | 2.330 K | 2.597% | 3394 | 41.3 K | 452.4 K |
| TEXEL (after fixes) | 2.330 K | 2.608% | **28** | **225.8 K** | 452.4 K |
| **VERTEX** | **0.382 K** | **0.448%** | **0** | 295.6 K | 311.6 K |

VERTEX is effectively clean. TEXEL is ~6× noisier at every spatial scale tested
(median-filter windows 9, 21, 41, 81).

## 3. Bugs found and fixed (on `heatsim`)

### 3.1 Value AOV accumulates through transparent surfaces

`ShaderNodeOutputAOV` is evaluated at **every** shading event along the path,
weighted by throughput. A Transparent BSDF has throughput 1, so alpha-cutout foliage
(`hojas`/`hojas.001`, a MixShader over a TransparentBSDF) makes a camera ray shade
one leaf after another and the AOV **sums** their temperatures.

Evidence: the ivy read **2103 K** on a scene whose hottest source is a 350 K lamp,
with pixel values landing on integer multiples of the leaf temperature — 254 px at
2×, 57 at 3×, 16 at 4×, 6 at 5×, a clean geometric decay. Identical under both
VERTEX and TEXEL, so render-time, not solver.

**Fix:** gate the AOV value on `LightPath.Transparent Depth < 0.5` so only the first
surface contributes. Plant 2170 K → **315 K**; scene max 2170 K → 452 K; scene median
unchanged (307.2 → 306.5 K).

**Trade-off:** where a ray crosses the transparent part of a cutout texture, the AOV
reports that surface rather than the geometry behind it. That is the documented
meaning of the pass and is strictly better than an unbounded sum.

The same bug exists upstream in heat-sim-blender `addon/lib/visualization.py:_add_aov_value`
(identical unguarded wiring). Fixed there too; its test suite passes.

### 3.2 Non-mesh renderables never get a temperature

`stamp_default_temperatures` and `write_frame_attributes` both guard on
`obj.type != "MESH": continue`. Materials are shared datablocks, so a CURVE sharing a
material with meshes gets the AOV chain patched in by `setup_temperature_aov` but has
neither `sim_temperature` (meshes only) nor the stamped default. Both Attribute nodes
fall back to 0 → **`T_effective = 0 K` for the whole object**.

`IVY_Curve` in kitchen1 is exactly this. Being thin, almost none of its footprint is
interior, so Cycles' reconstruction filter smeared its zeros into neighbouring
fully-covered pixels.

Causal verification: hiding `IVY_Curve` removed **729 of 3394** sub-floor pixels and
raised the frame minimum from 31.3 K to 224.4 K.

**Fix:** stamp the default on any renderable geometry type
(`MESH, CURVE, SURFACE, META, FONT`). Non-meshes cannot carry the per-vertex
attribute, so the object-level default is all they get — they render at ambient
instead of absolute zero.

### 3.3 Atlas alpha is bilinearly interpolated across validity boundaries

The atlas alpha is a hard {0,1} validity mask, but the image is sampled with `Linear`
interpolation and `CLIP` extension, which returns `(0,0,0,0)` outside a tile. Colour
and alpha therefore blend toward zero **at the same rate**, yielding a half-open gate
over a half-darkened temperature:

```
T_valid = 296.2 K (α=1) beside an invalid texel (T=0, α=0)
  → sampled Red = 148.1 K, sampled alpha = 0.50
  → Mix(Factor=0.50, A=295.37, B=148.1) = 221.7 K     ← below the 295 K floor
```

Compounding it, `_ATLAS_DILATE_ITERATIONS = 1` gave one texel of protective margin
against **25 invalid components sized 1–462 texels** inside tile interiors.
(`atlas.dilate()`'s `iterations=4` default is not what the call site uses.)

**Fix:** threshold the alpha (`GREATER_THAN 0.5`) so the gate stays categorical —
a texel is either valid or it is not, and an invalid one falls back to the vertex path
rather than blending a partially-zeroed colour. Dilation raised 1 → 8, with
`_ATLAS_PACKING_PADDING` 3 → 17 to preserve the `2·iterations ≤ padding` invariant
that stops adjacent tiles bleeding into each other.

Contributions isolated by ablation: gate alone 3394 → 956; gate + dilation → **28**.

## 4. The unresolved artifact: spatial mottling

The fixes above removed the *extreme* values but left the mottling **untouched**
(2.597% → 2.641% → 2.608% across baseline, gate-only, gate+dilation). Measuring at
several scales showed the 8× dilation moved fine-scale structure into coarse-scale
structure without reducing total error — fine-window detections fell 38 → 2 while
window-41 detections held at 1276 → 1277.

### Hypotheses eliminated, each by controlled measurement

| hypothesis | test | result |
|---|---|---|
| Cycles sampling noise | 256 → 2048 → 4096 spp | no change; artifact is deterministic (98.1% identical pixels across runs) |
| Denoiser artifacts | denoise on/off at each spp | blobs are denoiser-shaped but the underlying error is unchanged |
| Adaptive sampling ceiling | `adaptive_threshold` 0.05 → 0.002 | 37 → 36 speckle px |
| Coverage-alpha premultiplication | explicit divide by Render Layers alpha | **0.0004 K** change — α is 1.0 everywhere (closed room, zero background pixels) |
| Atlas gate / dilation | see §3.3 | fixes sub-floor px, mottling unchanged |
| PCG non-convergence | `tol` 1e-5 → 1e-9, `max_iter` 200 → 5000 | byte-identical; already converged |
| Texel density | 1500 → 400 → 150 /m² | 2.330 → 2.255 → 1.924 K (18% for 10×; not proportional) |
| Shadow-ray MC noise | 8 → 256 rays at the TEXEL call site | 1.01× (5.7× predicted); max diff 72.4 K so it did reach the solve |
| kNN conditioning | `n_neighbors` 30 → 100, `mollify` 1e-5 → 1e-3 | 1.00×; max diff 5.5 K, 84,942 px changed, so genuinely applied |

### Conclusion

The noise is intrinsic to building a **kNN point-cloud Laplacian over UV-rasterized
texel positions**. Their 3D spacing is irregular (rasterization is uniform in UV, not
in world space), producing a spatially-varying discretisation error of ~2.3 K that no
hyperparameter removes.

At density 150 the atlas carries roughly VERTEX's total point count and is still 5×
noisier — so it is not point *count*, it is point *placement*.

Note both domains use the **same** solver path: `laplacian_domain` defaults to
`"POINTS"` unconditionally and is never overridden. VERTEX is clean partly because
mesh vertices are better distributed, and partly because linear interpolation across
large triangles smooths whatever error exists. **TEXEL does not create more error so
much as it resolves error that VERTEX hides** — which makes the flatness problem that
motivated the atlas and this noise two faces of one trade-off.

### Recommended direction

Replace the texel-domain kNN point cloud with a proper **cotangent FEM Laplacian on
the texel triangulation**. The rasterizer already carries texel→source-face
adjacency, so the connectivity is available. Global assembly is retained, so
**cross-object conduction is unaffected** — this was an explicit user constraint, and
it rules out the cheaper "scope the kNN graph per object" alternative.

## 5. Infrastructure defects found along the way

These are independent of the mottling and worth fixing on their own.

### 5.1 The solve cache is unsound

`cache.cache_key` is built from `{blend path, mtime, solver_cfg}` only. Solver
behaviour that is **not** a `solver_cfg` field — linear-solver tolerance, iteration
cap, shadow-ray count — changes results without invalidating the cache. Two
experiments in this investigation silently returned stale solves; the tell was a 13 s
"solve" where a real one takes ~160 s.

Either move those constants into `solver_cfg`, or hash the solver module into the key.

### 5.2 Solver defaults are shadowed up to four levels deep

`n_neighbors` has independent defaults in four places:

```
laplacian.py                n_neighbors: int = 30
solver.py:179               kwargs.get("pointcloud_neighbors", 30)
adapter.py:1331             solver_cfg.get("pointcloud_neighbors", 30)
adapter.py:1674             (a second copy of the same default)
```

Patching any but the outermost changes nothing. The same pattern applies to
`mollify_factor` and to `direct_kernel_soft_shadow_rays`, which has *two*
independent call sites — `irradiance_kernel.py:486` (per-vertex objects) and
`adapter.py:886` (**atlas objects**) — so a change to one silently misses the other.

This directly caused two false "ruled out" conclusions during this investigation.
The diagnostic that catches it is checking `max |diff|` against the baseline rather
than trusting a summary statistic.

### 5.3 Shadow rays are under-sampled for accuracy

Raising `direct_kernel_soft_shadow_rays` 8 → 256 moved peak scene temperature
**452.4 K → 391.0 K** — a 61 K difference at hot spots. It does not cause the
mottling, but the default appears materially inaccurate, and the knob is not
reachable from `ThermalConfig` at all.

### 5.4 Emissivity does not reach the radiance render

`_build_gray_body_material` hardcodes `_DEFAULT_EMISSIVITY = 0.9` for every object.
Per-material emissivities from a `.thermal.json` sidecar therefore affect the
**solve** (radiative boundary condition) but never the rendered radiance. Measured
`R/(σT⁴)` is 0.986–1.034 scene-wide, consistent with that constant. Any expectation
of material contrast in `thermal_radiance/` (e.g. stainless ε=0.16 vs plaster ε=0.91)
is currently unmet.

## 6. Reproduction

```bash
conda activate visionsim          # required: sets LD_LIBRARY_PATH for peewee/sqlite
export CUDA_VISIBLE_DEVICES=4     # single GPU
rm -rf <scene>.blend.heatsim      # the cache will otherwise hide solver changes

visionsim blender.render-animation \
    /data/sriram/blender_files/visionsim50/kitchen1.blend out/ \
    --config.include-thermal \
    --config.thermal.assignments assets/thermal/kitchen1_all_fem.thermal.json \
    --config.thermal.device cuda --config.thermal.render-domain TEXEL \
    --config.thermal.sim-time-s 500 --config.thermal.timestep-s 1.0 \
    --frame-start 301 --frame-end 301
```

Swap `TEXEL` for `VERTEX` to get the clean reference. `sim-time-s` matters: the
default of 1.0 s leaves the room isothermal to 0.04 K; 500 s gives a ~68 K spread.

Ripple metric used throughout:

```python
patch = img[30:175, 70:300]
ripple = np.std(patch - scipy.ndimage.median_filter(patch, size=41))
```

## 7. Practical guidance until the operator is replaced

- **`render-domain VERTEX`** is clean today (0.382 K noise, zero sub-floor pixels) at
  the cost of flatness on low-poly surfaces.
- **TEXEL** buys spatial detail at ~2.3 K of numerical noise; some of the apparent
  detail is that noise.
- In either domain, mask `temperature/` pixels below the solve floor before
  quantitative use — 28 remain in TEXEL after the fixes.
