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
| the Cycles bake (new code) | `DIRECT_KERNEL` reproduced the identical 99.80% NaN |
| degenerate geometry | 0 non-finite world coords, 0 zero-scale axes, 0 empty meshes, no object over 500 m |
| an invalid preset in the sidecar | all 29 preset names in all 8 sidecars validate against `_PRESET_TABLE` |
| NaN entering through the solve inputs | instrumented: `u_prev`, `B_step`, `Minv`, `M`, `L`, `alpha`, `rho`, `c`, `eps`, `irradiance`, `boundary_mask` **all finite** |
| a NaN-producing Laplacian | `robust_laplacian.point_cloud_laplacian` on the 124,318-point cloud returns 0 NaN in `L` and 0 NaN/zero/negative in `M` |
| the 30 exact-duplicate points | deduplicating changes nothing; officebuilding has **57,352** exact duplicates and solves fine |
| poor mass conditioning | real (M spans 4.7e-15 to 1.13) but **officebuilding is worse** on every measure and succeeds |
| stale bytecode shadowing the fixes | all **62** installed `.pyc` files validate against their sources; 0 stale |

### The awkward part: it no longer reproduces

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
