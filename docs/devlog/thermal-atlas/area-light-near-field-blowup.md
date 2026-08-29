# Area-Light Near-Field Singularity: A Latent Bug the Atlas Exposed

**Date:** 2026-08-29
**Branch:** notes on `heatsim-devdocs`; the fix landed on `heatsim` in `light_models.py`
**Scene:** `visionsim50/kitchen1.blend`, frames 1–12, 512×512, `Camera.004`
**Status:** Root cause proven positionally. Fixed and verified.

## 1. Symptom

The first ~12 frames of a TEXEL render were far worse than the rest of the sequence:

| frame | R median | R max | max/median | T max | non-finite radiance px |
|---|---|---|---|---|---|
| 1 | 488.8 | **inf** | — | **1515.9 K** | **2** |
| 2 | 488.8 | 65248 | 133.5 | 1447.1 K | 0 |
| 3 | 488.8 | **inf** | — | 1329.0 K | 1 |
| 5 | 490.8 | 57856 | 117.9 | 1027.8 K | 0 |
| 8 | 488.8 | 15536 | 31.8 | 740.9 K | 0 |
| 12 | 488.8 | 3604 | 7.4 | 515.0 K | 0 |
| 301 | 497.2 | 2190 | 4.4 | 452.7 K | 0 |

Visually: a searing spot at the lower-left of frame 1, and a dense "starfield" of bright
speckle across the ceiling. The scene's hottest boundary condition is a 350 K lamp and it
starts at 295 K, so **1515.9 K is provably numerical error**, and the radiance EXR contained
literal `inf`.

The monotonic decay across frames 1→12 is not a transient in the solve (this is a *static*
solve — one field held for every frame). It is the camera moving away, so the blow-up
subtends a shrinking solid angle.

## 2. Root cause: unbounded 1/d² in the area-light evaluator

`light_models.py` evaluates irradiance from point, spot and area lights with the inverse-square
law, guarded only by `np.maximum(d2, 1e-20)`. That guard prevents division by zero; it does
**not** bound the physics. As a receiver approaches a light, irradiance diverges.

Area lights are the case that actually fires here. They are evaluated by **stratified sampling**
— K sample points across the emitter surface, each treated as a point source — so a sample can
land arbitrarily close to a receiver even when the light's *centre* is metres away.

### Positional proof

Ray-casting the blow-up pixels of frame 1 through `Camera.004`:

| pixel (row, col) | object hit | dist to light **centre** | dist to light **surface** | unclamped irradiance | vs clamped |
|---|---|---|---|---|---|
| (378, 2) | `Vert.004` | 2.034 m | **0.157 m** | 504.8 W/m² | **311×** |
| (379, 1) | `Vert.004` | 2.013 m | **0.135 m** | 686.1 W/m² | **422×** |
| (377, 3) | `Vert.004` | 2.054 m | **0.182 m** | 377.7 W/m² | 233× |
| (380, 4) | `Vert.004` | 1.979 m | **0.147 m** | 576.1 W/m² | 355× |

The culprit light is `Area.002`: an AREA light of **size 2.7741 m**, **157.1 W**, positioned at
`(-5.249, -1.991, 0.438)` — i.e. a large panel whose lower edge comes close to floor level.

Two details make this easy to miss:

- **Centre distance hides it.** 2.0 m from the light's origin looks harmless. 0.135 m from its
  *surface* does not. Any diagnostic that measures to `light.matrix_world.translation` — as a
  first pass of this investigation did — sees nothing wrong.
- **The energy is real, not a typo.** 686 W/m² into a floor texel, ~400× what a clamped
  evaluation allows, is exactly the scale needed to drive a 295 K surface to 1516 K.

### Why the atlas appeared to be at fault

Every blow-up pixel hits **`Vert.004`** — the 172 m², **16-vertex** floor slab, which the atlas
upsamples to **74,389 texels**.

- Under `render-domain VERTEX`, the nearest floor *vertex* to `Area.002` is metres away. The
  singular region is never sampled.
- Under `render-domain TEXEL`, 74,389 rasterized texels blanket the slab and some land 13 cm
  from the panel, squarely inside the singularity.

**The atlas did not create this bug; it sampled the scene densely enough to find one that was
always present in the light model.** An earlier round of this investigation concluded "the texel
path is intrinsically broken" — that framing was wrong for this defect, though it remains correct
for the separate mottling issue (see `texel-noise-investigation.md`).

Note the independent convergence: an earlier agent traced a 6155.8 K blow-up to a single face of
`Vert.004` and attributed it to near-coincident cross-object kNN edges. Same object, same
symptom, but the mechanism is the light model, not the Laplacian.

### The secondary cascade

Once a patch of floor reaches 1516 K it emits `σT⁴` ≈ **700×** a 295 K surface. Every surface with
line of sight then collects enormous energy on the rare paths that hit it — which is the ceiling
starfield. One localized numerical defect, scene-wide visual consequences.

## 3. Fix

In `light_models.py`, floor `d²` at the light's own physical radius rather than at a numerical
epsilon, in `evaluate_point`, `evaluate_spot` and `evaluate_area`:

```python
_radius = float(getattr(light_obj.data, "shadow_soft_size", 0.0) or 0.0)
d2 = np.maximum(d2, max(_radius * _radius, 1e-6))
```

The justification is physical, not cosmetic: Blender models point and spot lights as **spheres of
radius `shadow_soft_size`**, and an area light as a surface of finite extent. The inverse-square
point-source falloff is only meaningful *outside* that extent. Inside it the emitter is not a point
and 1/d² is the wrong model, so saturating is more correct than diverging. The `1e-6` absolute
floor covers a literal zero-radius lamp.

## 4. Verification

Cache cleared before each solve (see `texel-noise-investigation.md` §5.1 — the cache does not key
on solver internals and will otherwise serve a stale result).

**Frame 1 — the blow-up frame:**

| metric | before | after |
|---|---|---|
| T max | 1515.9 K | **441.4 K** |
| pixels > 800 K | 23 | **0** |
| non-finite radiance px | 2 | **0** |
| radiance max (finite) | 65,248 | **1,992** |

**Frame 301 — regression check:**

| metric | before | after |
|---|---|---|
| T max | 452.7 K | 452.4 K |
| T min | 222.7 K | 225.8 K |
| sub-295 K px | 26 | 28 |
| median | 306.72 K | 306.86 K |

Unchanged away from lights, as intended.

## 5. What this does and does not solve

**Solves:** the catastrophic early-frame artifacts — impossible temperatures, `inf` values in the
radiance EXR, and the reflected starfield they caused.

**Does not solve:** the pervasive ~2.3 K spatial mottling. That is a separate defect, traced to
the kNN point-cloud Laplacian over UV-rasterized texel positions, with nine hypotheses eliminated
by measurement. See `texel-noise-investigation.md` §4. The recommended direction there is
unchanged: a cotangent FEM operator on the texel triangulation, retaining global assembly so
cross-object conduction is preserved.

## 6. Generalisation

This is a **scene-independent** bug, not a kitchen1 quirk. Any scene where a solve point lands
inside a light's physical extent will produce it, and the atlas makes that far more likely by
raising sample density on large, coarsely-tessellated surfaces — exactly the surfaces the atlas
exists to serve. Interior scenes with large panel or window lights near floors and walls are the
high-risk case, and the ~50-scene dataset is full of them.

Worth auditing the other scenes for the same signature: temperatures above the hottest boundary
condition, and non-finite values in `thermal_radiance/`.
