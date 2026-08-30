# Dataset Audit: bathroom1, officebuilding, diningroom

**Date:** 2026-08-29
**Branch:** notes on `heatsim-devdocs`
**Scenes:** `visionsim50/{bathroom1,officebuilding,diningroom}.blend`, 600 frames each,
512x512, TEXEL domain, `irradiance_source=CYCLES_BAKE`, `sim_time_s=500`, `timestep_s=1.0`
**Status:** officebuilding complete and audited; bathroom1 and diningroom in progress.

## 1. Why these three

Two of the fixes on `heatsim` are claimed to be **scene-independent**, and both
`area-light-near-field-blowup.md` §6 and `cycles-irradiance-source.md` §8 end by
recommending an audit of the wider dataset. These scenes were picked to test those
claims rather than to look nice:

| scene | why | tests |
|---|---|---|
| `bathroom1` | **0 light objects.** Lit entirely by `lamp ext sdb`, an emissive plane of ~100 m². | The Cycles bake. Under the Direct Kernel this scene receives *zero* flux and cannot help but render flat. |
| `officebuilding` | **45 lights, 4562 W total** — the dataset's largest lighting rig. | The near-field 1/d² clamp, under the heaviest load available. |
| `diningroom` | The dataset's most severe Cycles clamp (`sample_clamp_*` = 0.5), and **no emissive materials**. | The clamp-clearing fix, plus a control: no emissive geometry, so the bake and the kernel should broadly agree. |

Sidecars were authored for all three (no prior configs existed): 172 materials total, all
`FEM_PARTICIPANT`, `dirichlet_K: null`, `defaults: {}`. Committed as `d06ea76`.

## 2. officebuilding — result

600/600 frames. Sampled and full-sweep statistics:

| metric | value |
|---|---|
| non-finite radiance px, **all 600 frames** | **0** |
| `r/(eps*sigma*T^4)` | **1.1114 every frame** |
| median T | 295.34 K |
| typical interior spread (p1-p99) | 9-14 K, rising to 66.9 K at f301 |
| T max over all frames | 889.2 K (frame 391) |
| T min over all frames | 113.4 K |
| sub-295 K px | 0.2728% |

**The near-field clamp holds under load.** Zero non-finite radiance pixels across 600
frames with 45 lights is the headline: this is the configuration most likely to reproduce
the kitchen1 blow-up, and it did not. All 45 lights carry a uniform
`shadow_soft_size` of 0.0305 m, so the clamp has a real radius to floor against.

**On the ratio 1.1114.** It is constant to 4 significant figures on every frame sampled.
1/0.9 = 1.1111, so `thermal_radiance` is `sigma*T^4` with **emissivity 1.0**, not
`eps*sigma*T^4` at the `_DEFAULT_EMISSIVITY = 0.9` the shader module declares. Worth
resolving separately — see §4. For the purpose of this audit the useful reading is that
the ratio is *constant*, which is what confirms radiance is a clean deterministic function
of the temperature field rather than an independently noisy quantity.

**On the median sitting at ambient.** 295.34 K against a 295 K initial condition invites
the same "it's flat" reading that the classroom got, and here it would be wrong. Roughly
73% of each frame is unlit interior; the median is measuring that. The lit fraction
behaves correctly — sunlight through the windows lands as a sharp diagonal band on the
wall and floor that is clearly resolved in both the temperature and the radiance pass, and
frame 301 reaches a 66.9 K spread. The lesson is that **median-vs-initial is not a
flatness test for a scene with a large unlit fraction**; the classroom diagnosis in
`cycles-irradiance-source.md` was sound because that scene was flat *everywhere*, p99
included. Use the p99 and the lit-fraction spread instead.

## 3. The 889 K spot: a confirmed bake firefly

Frame 391 reaches 889.2 K, above any boundary condition in a scene whose only heat input
is irradiance. This is exactly the signature `area-light-near-field-blowup.md` §6 said to
audit for, so it was chased to ground. **It is not the near-field bug.**

Ray-casting the four offending pixels (rows 22-25, col 31) through the camera:

| | |
|---|---|
| object hit | `Plane.010` — a **1-polygon, 4-vertex, 3.05 x 3.05 m** framed picture |
| nearest light | `Spot`, 50 W, **4.390 m** away (4.360 m to its surface) |
| irradiance from it | ~0 W/m2 |

4.4 m is nowhere near the singularity. Tracing into the solve cache instead:

- The atlas holds **exactly one texel** above 600 K out of **1,234,640** (2 above 400 K).
- The value is present in the FEM solve itself, so it is not a rasterization artifact:
  `Plane.010` node **505 of 4096**, at 976.5 K, while the object's p99 is 335.1 K.

Steady state requires absorbed flux = `sigma*T^4`: 976.5 K needs ~51 kW/m2, against
~714 W/m2 for its neighbours at 335 K. **That single node is absorbing ~70x what the
surface around it absorbs**, which no smooth irradiance field produces. It is a Cycles
firefly in the baked irradiance map, sustained across all 501 solve steps because the bake
is computed once and held.

This confirms, with a concrete instance, the third known-open item in
`cycles-irradiance-source.md` §7: *"Bake sample count is not exposed... irradiance noise
feeds straight into the temperature field."* The bake inherits the scene's Cycles sample
count, and nothing filters its output before it becomes a boundary condition.

Note the profile of the affected object. `Plane.010` is a 4-vertex plane upsampled to 4096
texels — the same shape of object as kitchen1's `Vert.004` (16 vertices, 74,389 texels).
Sparse geometry with a large atlas allocation is where single-texel bake noise has the most
room to appear, for the same reason the near-field bug surfaced there: the atlas samples
densely enough to find defects that vertex-domain solving never touches.

**Recommended fix, in preference order:**

1. **Denoise the irradiance bake.** Cycles can denoise a bake pass directly. This is the
   smallest change and targets fireflies specifically.
2. **Outlier-reject per object before the solve.** Clamp baked texels to some multiple of
   a local median. Cheap and robust, but a tunable that has to be justified.
3. **Expose and raise the bake sample count.** Addresses the cause, costs the most; the
   bake is already ~3x the solve (`cycles-irradiance-source.md` §5).

Not implemented here deliberately: changing solver behaviour partway through a three-scene
sequence would have made the scenes non-comparable, which is the whole point of running
them together.

## 4. Per-material emissivity never reaches the radiance render

**First reading of the 1.1114 ratio was wrong, and the error is instructive.** It looked
like `thermal_radiance` was emitting `sigma*T^4` at eps = 1.0, ignoring the declared
`_DEFAULT_EMISSIVITY = 0.9`. Reading the shader shows it does apply eps:

```
out = Mix(Fac = 1 - eps, A = Emission(eps-weighted sigma*T^4), B = Diffuse)
    = eps*sigma*T^4*scale + (1 - eps)*L_in
```

The measured ratio is the **cavity effect**, not a missing factor. In a closed interior
near radiative equilibrium the reflected term `(1-eps)*L_in` is filled by surroundings at
a similar temperature, so `L_in ~ sigma*T^4` and the two terms sum back to `sigma*T^4`.
An enclosure behaves as a blackbody regardless of its wall emissivity — textbook, and
exactly what a *correct* gray-body shader should produce here. The near-perfect constancy
of the ratio is evidence the shader is right, not evidence it is broken.

**The real gap is a different one.** `_DEFAULT_EMISSIVITY` is baked into the Mix Fac as a
**scene-wide constant**. The radiance shader reads exactly four attributes —
`sim_temperature`, `heatsim_default_temperature`, the atlas UV layer and the atlas
coverage gate — and there is **no emissivity attribute node in the graph at all**.
Meanwhile `adapter.py` (`_write_emissivity_attr`) does write a per-vertex `emissivity`
POINT attribute from the sidecar presets, and its own docstring says this is
*"so per-slot emissivity reaches the gray-body radiance"*. It does not. The attribute is
written and then never read by the shader.

The magnitude of what is lost: the preset table spans **eps 0.05 (`aluminium_polished`) to
0.98 (`skin`)**, a 20x range across 29 presets, and the sidecars authored for these scenes
lean on it — officebuilding alone assigns `stainless_steel` (eps = 0.16) to 8 materials.
Every one of them renders at 0.9.

This matters specifically because the modality is **LWIR**. Emissivity contrast is a large
part of what a thermal camera actually distinguishes — a polished metal surface reads dark
and mirror-like against a matte wall at the same physical temperature, and that cue is
absent here by construction. Note the asymmetry: emissivity *does* reach the FEM solve
(`adapter.py` passes `emissivity_map=combined.eps`), so it correctly governs radiative
cooling in the physics. It is only the render that flattens it.

Fixing it looks small — add a `ShaderNodeAttribute` for `emissivity` and drive the Mix Fac
from `1 - eps` per-vertex instead of a constant, with the existing constant as the fallback
when the attribute is absent. The atlas path would need the same treatment the temperature
chain already has. Not attempted here; it is orthogonal to this audit.

## 5. diningroom — result

600/600 frames, and the cleanest of the three.

| metric | value |
|---|---|
| non-finite radiance px, all 600 frames | **0** |
| `r/(eps*sigma*T^4)` | 1.091 - 1.111 |
| T max over all frames | 415.3 K |
| interior spread (p1-p99) | 16.7 - 38.9 K |
| sub-295 K px | 0.3891% |

**The clamp-clearing fix is confirmed on the scene that most needed it.** diningroom carries
the dataset's harshest Cycles clamp (`sample_clamp_*` = 0.5). With the clamps inherited from
the RGB settings the radiance pass would be pinned near the clamp value and the ratio would
collapse the way the classroom's did to 0.0287; instead it sits at ~1.1, i.e. `r ~ sigma*T^4`.

Visually the field is structured and physical: sunlit walls heat, the dark chairs and
tabletop stay near ambient, and radiance tracks temperature closely. T max of 415.3 K is
consistent with sunlit surfaces and needs no special explanation. This scene was also the
control — no emissive materials — and it behaves the same under the bake as the kernel
would, which is the expected result.

## 6. bathroom1 — the render failed, and the cause is not yet identified

**Do not use `bathroom1_v1`.** Its temperature pass is **89.4% NaN across all 600 frames**,
and the finite remainder sits at exactly 295.00 K, the untouched initial condition. Only a
handful of objects render at all; the rest are holes. The radiance pass contains no
non-finite values, which is why the earlier headline check (`non-finite radiance px = 0`)
passed and the failure was caught only when temperature percentiles came back `nan`.
**Lesson: check both passes for non-finite values, not just radiance.**

In the solve cache the pattern is unambiguous: **all 358 objects, 99.80% NaN each**. The
arrays are `(501, n_nodes)` and 1/501 = 0.2%, so **only the initial condition survived** —
the solve went non-finite on the very first timestep, globally. That is the signature of
the global assembly: every object is concatenated into one linear system, so one poisoned
value takes down all 358 at once.

### Ruled out by measurement

| hypothesis | evidence against |
|---|---|
| ~~the Cycles bake~~ | **RETRACTED - the bake IS the trigger. See the correction below.** |
| degenerate geometry | 0 non-finite world coords, 0 zero-scale axes, 0 empty meshes, no object over 500 m |
| an invalid preset in the sidecar | all 29 preset names in all 8 sidecars validate against `_PRESET_TABLE` |
| NaN entering through the solve inputs | instrumented: `u_prev`, `B_step`, `Minv`, `M`, `L`, `alpha`, `rho`, `c`, `eps`, `irradiance`, `boundary_mask` **all finite** |
| a NaN-producing Laplacian | `robust_laplacian.point_cloud_laplacian` on the 124,318-point cloud returns 0 NaN in `L` and 0 NaN/zero/negative in `M` |
| the 30 exact-duplicate points | deduplicating changes nothing; officebuilding has **57,352** exact duplicates and solves fine |
| poor mass conditioning | real (M spans 4.7e-15 to 1.13) but **officebuilding is worse** on every measure and succeeds |
| stale bytecode shadowing the fixes | all **62** installed `.pyc` files validate against their sources; 0 stale |

### Correction: the bake is the trigger after all

The row above was wrong and is left struck through rather than deleted, because the
mistake is the instructive part. It rested on a **single** `DIRECT_KERNEL` run that
returned 99.80% NaN. Repeated properly, that result does not hold:

| configuration | runs | result |
|---|---|---|
| `DIRECT_KERNEL` | **5** | 0.00% NaN every time, identical max 368.0 K |
| `CYCLES_BAKE` | 1 | **99.80% NaN**, range [295.0, 295.0] |
| `CYCLES_BAKE` + `bake_samples=1024` | 1 | **99.80% NaN** - unchanged |

So `CYCLES_BAKE` reproduces the failure and `DIRECT_KERNEL` does not, which matches the
fact that `bathroom1_v1` was rendered with the bake. **The lesson is the obvious one: one
run is not a measurement.** A single observation was promoted to a ruled-out row in a
table, and it then steered the investigation away from the actual culprit for hours. The
five-run `DIRECT_KERNEL` result is trustworthy; the one-run bake result should itself be
repeated before it is leaned on too hard.

This also retires the "it only fails under concurrent load" correlation recorded below -
the real split was `CYCLES_BAKE` vs `DIRECT_KERNEL`, and the two failing runs happened to
be the bake ones.

**Raising bake samples does not fix it.** The natural hypothesis, given that bathroom1 is
lit *only* by a ~100 m2 emissive plane and is therefore the worst case in the dataset for
a noisy bake, was that extreme bake outliers were driving the divergence. At 1024 fixed
samples - a measured 3x noise reduction, see section 7 - the NaN is bit-for-bit as bad.
Whatever the bake does to this scene is **structural, not statistical**.

### The remaining puzzle



After the investigation the same scene solves **cleanly and deterministically** — five
consecutive runs (`DIRECT_KERNEL`, 1 and 2 frames, instrumented and not) all returned
**0.00% NaN and an identical max of 368.0 K**, with a sensible field of 302.8-366.7 K,
median 335.9 K. So bathroom1 is not intrinsically broken; something made two consecutive
runs diverge and then stopped.

The one property that distinguishes the two failing runs from the five successes is that
**both failures occurred while another render was running concurrently on a different
GPU**, and all five successes ran on an otherwise idle machine. That is suggestive, not
conclusive, and the mechanism by which load would inject NaN into a solve whose inputs are
all verified finite is not obvious.

What the evidence does support: the solve for this scene is **marginally stable**. It is
assembled globally in **float32** (`scipy_to_torch_sparse` casts to `torch.float32`) over a
system whose mass entries span 14 orders of magnitude, with per-row `|L|/M` reaching
3.5e15 on 5 rows. A system that conditioned only needs a small perturbation to diverge, and
float32 gives roughly 7 digits to absorb it. **The natural hardening step is to solve in
float64**, and to add a non-finite guard that fails loudly at the first diverged step
instead of writing 183 MB of NaN and rendering 600 frames from it.

### Standing recommendation

Any batch run over the wider dataset should assert on the solve before rendering: if
`temperatures.npz` contains non-finite values, stop. The current pipeline exits 0 and
produces a complete, plausible-looking, entirely unusable render — the most expensive
failure mode there is, and the fourth instance in this project of the pipeline succeeding
loudly while producing nothing useful (see `README.md`'s trap list).

## 7. Bake sample starvation (the blotching)

Reported from the rendered videos: the temperature and radiance passes look blotchy across
every `CYCLES_BAKE` scene. The hypothesis put to me -- that the bake's sample count was too
low -- was correct, and the mechanism is worse than a plain low cap.

`bake_irradiance_map` never set a sample count; it inherited whatever the blend carried.
Every visionsim50 scene checked ships:

| setting | value | Blender default |
|---|---|---|
| `cycles.samples` | 256 | 4096 |
| `cycles.use_adaptive_sampling` | **True** | True |
| `cycles.adaptive_threshold` | **0.05** | **0.01** |

The threshold is the real problem: at 5x looser than default, adaptive terminates texels far
below even the 256 cap.

Measured on diningroom's two largest objects, relative noise against a 7x7 median:

| config | Cube.012 | Bamboo Planter | bake time |
|---|---|---|---|
| 256 + adaptive 0.05 (shipped) | **9.62%** | **19.24%** | 2.2 s |
| 512 fixed | 4.01% | 9.62% | 2.7 s |
| **1024 fixed (chosen)** | **3.06%** | **7.85%** | 3.4 s |
| 2048 fixed | 2.40% | 6.64% | 5.1 s |
| 1024 fixed, denoising OFF | 3.06% | 7.85% | 3.4 s |
| 1024 + adaptive 0.01 | 4.17% | 8.98% | 3.4 s |

The mean is unchanged across configs (0.5336 -> 0.5452), so this is pure variance, not bias.

**Why it shows up as several-Kelvin blotches.** A steady-state surface satisfies
absorbed = eps*sigma*T^4, so T ~ E^(1/4) and a relative error in E is a quarter of that in T:
9.6% in E is ~2.4% in T, which at 300 K is **~7 K**. That is the observed scale. Confirmed in
the solved atlas itself (mean |A - median7| = 3.26 K, p99 = 24 K), so it is a property of the
temperature field, not of the render.

Two results worth keeping because they contradict the obvious guesses:

- **Denoising does nothing for a bake.** Output is identical to four decimal places with
  `use_denoising` on and off. `bpy.ops.object.bake` does not run the denoiser the way a
  rendered pass does. This retires "denoise the bake", which was ranked as the *preferred*
  fix in `cycles-irradiance-source.md` section 7 -- that recommendation was wrong.
- **Adaptive sampling must be disabled, not tightened.** At a matched 1024 cap, adaptive at
  the stock 0.01 threshold measured *worse* than fixed sampling (4.17% vs 3.06%), because it
  still cuts texels short.

Fixed in `bake_irradiance_map`, which now takes an explicit sample count, disables adaptive
sampling for the bake and restores both afterwards. Exposed as
`--config.thermal.bake-samples`, default 1024. Cost is ~1.55x the bake, and 2048 was rejected
as buying only a further 1.28x for another 1.5x of time.

## 8. The bug underneath: irradiance baked against the wrong mesh

The same review raised a second observation -- that some diningroom objects (the black
chairs) looked far too cold. That turned out to be a much larger defect than the blotching,
and it had been silently corrupting **every** `CYCLES_BAKE` render produced in this project.

**The solver and the bake disagreed about which mesh they were describing.**

| | mesh used |
|---|---|
| solver nodes (`adapter._extract_geometry`) | `obj.evaluated_get(depsgraph).data` -- **modifiers applied** |
| Cycles bake per-vertex sampling (`irradiance.py`) | `obj.data` -- **original** |

When a modifier changes the vertex count the two disagree, `_combine` drops a flux array whose
length does not match the node count, and **the object receives no absorbed flux at all**. It
holds its initial temperature, warmed only by conduction from its neighbours.

Measured on diningroom, per object, flux logged where it is computed and joined against the
solved field:

| group | objects | median flux | median rise |
|---|---|---|---|
| flux length **==** node count | 60 | 0.146 | **1.335 K** (max 43.6) |
| flux length **!=** node count | **229 (79%)** | **1.792** | **0.332 K** (max 0.47) |

Note the direction: the broken group was receiving **12x more flux** than the working one and
still did not heat. Its rise is 0.332-0.334 K essentially regardless of the flux computed for
it, because none of that flux was ever applied.

Three independent confirmations:

1. **Two instances of one asset.** `decoration_twig_branch.009` (584 verts, 584 nodes) rose
   **33.8 K**; `.007` (584 verts, **9305** nodes) rose **0.333 K** -- same material, same
   preset, same flux.
2. **Identical trajectories under different flux.** `Circle.000` (flux 1.664) and `Circle.004`
   (flux 1.102) produced bit-identical temperature curves at every one of 501 steps
   (0.002, 0.003, 0.008, ... 0.333). Flux that reaches a solve cannot fail to separate them.
3. **The fix moves exactly the affected set.** After sampling the evaluated mesh:

| metric | before | after |
|---|---|---|
| median rise | 0.332 K | **31.478 K** |
| objects pinned at ~0.33 K | 229 | **4** |
| objects above 2 K | 60 | **254** |
| `Circle.000` | 0.333 K | **25.413 K** (74,226 nodes vs 166 verts) |
| `decoration_twig_branch.007` | 0.333 K | **39.831 K** |
| `Circle.005` (already aligned) | 43.631 K | **43.632 K** -- unchanged |

That last row is the regression check: objects that were already correct do not move.

**Scope.** `CYCLES_BAKE` only. The Direct Kernel evaluates irradiance at the evaluated mesh's
own vertex positions and was always aligned -- which is precisely what `_extract_geometry`'s
docstring asserts:

> The evaluated mesh's vertex order/count matches the Direct-Kernel irradiance extraction
> (it uses the same `foreach_get('co')` path), so the returned flux aligns index-for-index.

The statement was true when written, and scoped to the Direct Kernel. Porting the bake in
`cycles-irradiance-source.md` introduced a second irradiance producer that did **not** satisfy
it, and nothing enforced the contract. Every `CYCLES_BAKE` render in this project -- classroom
v2/v3, officebuilding v1, diningroom v1, bathroom1 v1 -- is affected and must be redone.

**The lesson.** The failure was invisible in every headline metric. Non-finite counts were
clean, `r/(sigma*T^4)` was a textbook 1.0, spreads and maxima looked plausible, and the
rendered images were merely *unconvincing* rather than obviously broken -- a room where most
of the furniture sits at ambient still looks like a thermal image. It took an outside
observation ("those chairs look too dark") to expose it. A cheap invariant would have caught
it at the source: **assert that a flux array's length equals the node count instead of
silently dropping it.** That guard is the natural follow-up to this fix.

**Related, not yet fixed:** `bake_albedo_map` and `get_or_bake_vertex_albedo` sample
`obj.data` the same way. The consequence there is milder -- on mismatch the albedo falls back
to zeros, i.e. full absorption, rather than the object being dropped -- but it is the same
latent inconsistency and should be brought onto the evaluated mesh too.

## 9. bathroom1 resolved: two degenerate nodes, not precision

Section 6 left this open after eight hypotheses. The answer came from dumping every solver
input immediately before the time loop:

```
diag(L)        nonfinite=2   min=-inf     -> idx [118296, 118313]
Minv           zeros=2                    -> idx [118296, 118313]
implied diagA  max=inf  nonfinite=2
rho / c / rc / eps / vec_rad_A / B_step / diag(M) / alpha : all finite
```

`robust_laplacian` emitted **-inf on the Laplacian diagonal for 2 of 120,183 nodes** --
degenerate local neighbourhoods, where coincident or near-coincident points leave the local
tangent plane undefined (this cloud has 30 exactly-coincident points).

**Why two bad nodes destroyed all 120,183.** The damage propagates through a *scalar*:

1. `inf` on `diag(L)` makes that node's `diagA` inf,
2. so the Jacobi entry `Minv = 1/clamp(diagA, min=1e-12)` becomes exactly 0,
3. `A @ p` goes non-finite, so `denom = p . Ap` does too,
4. PCG guards only `if denom.abs() < 1e-20: break` -- **false for both inf and NaN** --
5. so `alpha = rz_old / denom` becomes NaN, and `x = x + alpha * p` is NaN *everywhere*.

That is why the field went from perfect (295.0 K, all finite) to entirely NaN in one step,
with no intermediate state to catch.

Fixed in `laplacian.py`: non-finite Laplacian entries are detected, the affected nodes are
isolated (row and column zeroed) with a loud warning, and non-finite/non-positive mass
entries are replaced with the median. An isolated node still exchanges with ambient and
still absorbs flux -- it just does not conduct -- so it degrades to a thermally
disconnected speck instead of destroying the solve. **99.8004% NaN -> 0.0000%**, range
[294.5, 354.6] K. No-op on well-conditioned clouds.

**Two hypotheses recorded because they were wrong, and cost real time:**

- **float32 precision.** A full float64 solve (`np.float32` -> `np.float64` throughout
  `solver.py`) reproduced the NaN *identically*. Reverted -- it would have cost FP64
  throughput on hardware that runs it at a fraction of FP32, for nothing.
- **Irradiance magnitude.** The `CYCLES_BAKE`/`DIRECT_KERNEL` split looked like a
  large-source effect, but `A` is assembled from `M`, `L`, `Tamb^3` and `h/(rho*c)` and
  **never sees the irradiance** -- `vec_rad_A` linearises about ambient, not about a
  flux-derived temperature. The bake only decided *whether the run reached* the bad node's
  influence; it never changed the matrix.

## 10. Why objects rendered as flat single-temperature patches

Reported from the videos: dining chairs each showing a *different* uniform temperature, and
officebuilding's floor a uniform orange plateau. One root cause, and it is a one-line guard.

`irradiance.prepare_object_bake_uv` opened with:

```python
mesh = obj.data
if mesh is None or not getattr(mesh, "uv_layers", None):
    return
```

`mesh.uv_layers` on a mesh with **zero** UV layers is an *empty collection*, which is falsy.
The guard -- clearly intended as a None-check -- therefore returned early on exactly the
meshes that needed a UV layer, while the very next block exists to create one
(`uv_layers.new()` + Smart Project).

**The cascade, each step individually reasonable:**

| step | consequence |
|---|---|
| no `HeatSim_Bake_UV` created | `_write_atlas_uv_layer` has no source layer |
| no `HeatSim_Atlas_UV` written | `build_atlas_plan` cannot read it off the evaluated mesh |
| object demoted to the per-vertex path | with a topology-changing modifier, per-vertex write-back is impossible |
| `write_frame_attributes` constant-fills | the whole object renders as ONE value: the mean of its solved field |

Measured on diningroom (289 objects, **85% with no authored UVs**):

| | before | after |
|---|---|---|
| selected for the atlas | 259 | 259 |
| demoted for a missing UV layer | **231** | **0** |
| survived to rasterize | **28** | **259** |
| constant-filled on the vertex path | 229 | -- |

The visible symptoms follow exactly. Each chair was constant-filled at *its own* mean, so
each became a differently-coloured flat patch -- the "different temperatures" was literally
each object's mean. officebuilding's floor (`Cube.006`, base 68 verts vs 408 evaluated) was
filled at **330.74 K**, the mean of a field genuinely spanning **295.8-445.8 K** (std 33.16 K,
range 150 K). Its `p99` was pinned at 330.74 on five of seven sampled frames, with
`max == p99` -- a flat shelf, which is what prompted the question.

**What was already right.** `build_atlas_plan` deliberately forces atlas participation when
per-vertex write-back is impossible (`select_for_atlas(..., writeback_possible=False)`), and
that rule fired correctly for all 245 such objects. They were selected and *then* dropped
downstream for want of a UV layer. The architecture was sound; one falsy-vs-None check
disabled it for 80% of the dataset.

**A wrong turn worth recording.** The demotion warning mentions "the modifier stack dropped
the named layer", and the surrounding comment claims the code "force[s] a depsgraph update"
before reading the evaluated mesh -- no such update call exists. That made a stale-depsgraph
read the obvious suspect. It was tested directly and **falsified**: the layer is visible on
the evaluated mesh both before and after an explicit `view_layer.update()`. The objects that
failed had no layer to propagate in the first place. A plausible mechanism named in a comment
is not evidence.

**Diagnostic note.** These logs are Rich-formatted and wrap long messages across lines, so
`grep -c` under-counts badly -- "demoted from the atlas" returned 0 when the true count was
234. Flatten first (`tr '\n' ' '`) before counting.
