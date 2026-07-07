# VisionSim ↔ heat-sim-blender Thermal Parity — Design Spec

**Date:** 2026-07-07
**Branch:** `heatsim` (VisionSim repo)
**Status:** Approved (author auto-approved on the user's explicit standing delegation — "Auto approve the plans since I will be away")

## 1. Problem

When the same `bunny_textured.blend` is thermally solved and rendered by both
tools (heat-sim-blender addon vs VisionSim's vendored M1 path), the output
temperature fields diverge in two visible, independent ways:

1. **No texture imprint.** heat-sim's temperature field shows the bunny's
   checkerboard albedo (dark squares absorb more → hotter). VisionSim's field
   is smooth — the checkerboard is absent.
2. **~10× magnitude deficit.** Peak temperature rise over a 1.0 s solve:
   heat-sim ≈ **+1.86 K**, VisionSim ≈ **+0.19 K**.

The user's goal: VisionSim should reproduce heat-sim's result from the same
`.blend` — both the checkerboard imprint and the absolute magnitude — without
manual per-render tuning. The Cycles *irradiance* bake is explicitly out of
scope; the albedo bake is explicitly in scope.

## 2. Root Causes (empirically verified this session)

All three were confirmed by controlled A/B runs on the same 46 565-vertex
bunny, reading each tool's on-disk caches and instrumenting VisionSim's kernel.

### 2a. Incident irradiance E is NOT the problem (eliminates a red herring)

Instrumenting VisionSim's `compute_per_vertex_irradiance` and comparing to
heat-sim's kernel log:

| | direct_mean | sky_mean | **E_mean** | E_max |
|---|---|---|---|---|
| VisionSim | 1.17 | 0.16 | **1.33** W/m² | 5.07 |
| heat-sim  | — | — | **1.27** W/m² | 5.07 |

The direct-kernel irradiance is **already correct** in VisionSim (identical
E_max, ~equal E_mean). The two-lamp direct lighting dominates; the world HDRI
(`brown_photostudio_02_4k.exr`) is unloaded headless in *both* tools, so the
sky term is small and equal. **No irradiance-kernel changes are needed.** An
earlier "lost sky term / 5–10× smaller E" hypothesis was disproven by this
measurement.

### 2b. Missing checkerboard = albedo bake module not vendored

VisionSim vendored the albedo *consumption* path in full and functional:
`irradiance_kernel.get_or_bake_vertex_albedo` (3-tier: attribute → disk cache →
Cycles bake), `_bake_vertex_albedo_via_cycles` (Rec.709 luma reduction
`0.2126/0.7152/0.0722`, identical to heat-sim), and
`absorbed = E·(1 − albedo)` at `irradiance_kernel.py:506`.

The **only** missing piece is the leaf it calls:
`irradiance.bake_albedo_map` — the Cycles `DIFFUSE`/`COLOR` bake. The module
`visionsim/simulate/heatsim/irradiance.py` **does not exist** in VisionSim.

To keep that missing import from ever firing, `adapter._ensure_albedo_attr`
**pre-stamps a constant `albedo = 0.0` POINT/FLOAT attribute** on every sim
object (`adapter.py:196–229`, called from `_compute_irradiance` at
`adapter.py:231–233` with `defaults.get("albedo", 0.0)`). That pre-stamp makes
`get_or_bake_vertex_albedo` short-circuit at tier 1 (read attribute → 0.0) and
never bake. Net effect: VisionSim treats every surface as fully absorbing and
uniform → no checkerboard. heat-sim bakes the per-vertex checkerboard albedo
(≈0.4 mean), confirmed by its `absorbed_mean 0.77 < E_mean 1.27` and the
checkerboard temperature render.

### 2c. ~10× magnitude = `irradiance_scale` mismatch

heat-sim applies a scene-level irradiance boost before the FEM source term.
Both tools apply `flux_solver = (absorbed_W_m² / 1e6) × irradiance_scale`
(`adapter._combine` `/_WM2_TO_WMM2` then `× irradiance_scale`, identical to
heat-sim `fem_adapter.py:907`). The FEM solver, unit scaling (mm geometry,
W/mm² flux, kg/mm³ density, σ/1e6), and `dt` are **byte-identical** (vendored
verbatim; confirmed by file diff).

The divergence is purely the scale value:

- heat-sim: `bunny_textured.blend` authored `scene.heat_sim_settings.irradiance_scale = 1000.0` — heat-sim reads it from the blend.
- VisionSim: hardcodes `ThermalConfig.irradiance_scale = 100.0` (`config.py:177`) and **never reads the blend's authored value**.

`1000 / 100 = 10×` — exactly the observed magnitude gap. Verified: heat-sim's
solver flux logged as `[0, 0.005] W/mm²` for E_max=5.07 W/m² ⇒ scale ≈ 1000.

### 2d. Material is already correct; only a small ambient/initial-temp offset remains

Earlier analysis suspected VisionSim fell back to PVC defaults for the bunny.
A first-principles check disproves that. With the mm-shell lumped model
`ΔT/t ≈ flux/(ρc)`:

- heat-sim: `flux = (0.766/1e6)×1000 = 7.66e-4 W/mm²`, aluminum
  `ρc = (2700/1e9)×978 = 2.64e-3` → **0.29 K/s** ≈ measured mean 0.286 K. ✓
- VisionSim (current): `flux = (1.33/1e6)×100 = 1.33e-4`; **assuming aluminum**
  `ρc = 2.64e-3` → **0.050 K** ≈ measured mean 0.058 K ✓; assuming PVC
  `ρc = 1.17e-3` → 0.114 K ✗ (2× too high).

VisionSim's measured ΔT matches **aluminum**, so it **already resolves the
authored material correctly**. The earlier "PVC / density 1330" reconstruction
was mistaken. The only residual material-side difference is the *initial /
ambient* temperature default: VisionSim initializes at 295.0 K
(`ThermalConfig`) vs heat-sim's 295.372 K (its `AMBIENT_TEMP` fallback) — a
**0.372 K constant offset** on absolute temperature that does not affect the
temperature *rise*. This is a minor, optional alignment, not a driver of the
10× gap.

## 3. Goal / Success Criteria

Re-render `bunny_textured.blend` in both tools and compare the `HeatSim_To`
(temperature, Kelvin) EXR / per-vertex temperature:

- **Magnitude:** VisionSim peak ΔT and mean ΔT within **±15 %** of heat-sim's
  (target: heat-sim ≈ +1.86 K peak). ±15 % accommodates residual differences
  in albedo bake sampling, diffusion, and mesh evaluation.
- **Texture imprint:** VisionSim's temperature field shows the checkerboard,
  spatially correlated with heat-sim's (the dark squares are hotter).
- **No manual tuning:** parity achieved by reading the blend's authored
  settings, not by passing `--config.thermal.irradiance-scale 1000` on the CLI.

## 4. Design

Two core changes plus one optional alignment, all in
`visionsim/simulate/heatsim/`, consuming the blend's authored heat-sim
configuration faithfully. No irradiance-kernel physics changes.

**Expected result after the two core fixes:** VisionSim's solver inputs become
identical to heat-sim's (scale 1000, per-vertex checkerboard albedo, same
already-correct aluminum material), and the FEM solver is byte-identical, so
VisionSim's ΔT should land within a few percent of heat-sim's ≈+1.86 K peak —
well inside the ±15 % tolerance. The remaining slack absorbs albedo-bake
sampling and evaluated-mesh differences.

### 4.1 Port `bake_albedo_map` and unblock the albedo path (fixes 2b)

**New file** `visionsim/simulate/heatsim/irradiance.py` containing a single
public function ported from heat-sim `addon/lib/irradiance.py:bake_albedo_map`:

```
def bake_albedo_map(scene, obj, texture_size) -> BakedImage | None
```

- Runs a Cycles `bake(type="DIFFUSE", pass_filter={"COLOR"}, ...)` of the
  object's material base color into a float image (lighting-independent
  reflectivity ρ∈[0,1]).
- Returns an object exposing `.pixels` as an `(H, W, 3)` float array — the
  exact shape `_bake_vertex_albedo_via_cycles` already samples
  (`irradiance_kernel.py:206–239`). Match that contract precisely; the
  consumer is unchanged.
- Requires/creates the bake UV layer named by
  `constants.BAKE_UV_LAYER_NAME` (already referenced at
  `irradiance_kernel.py:215`) and a temporary bake image target, cleaned up
  after. Port heat-sim's UV/material/image setup verbatim where present;
  do not invent new behavior.
- Returns `None` on any failure (no UVs, no bakeable material, bake error) so
  the existing tier-3 fallback ("treat as black / full absorption") holds.

**Modify** `adapter._ensure_albedo_attr` / `_compute_irradiance` so the
constant pre-stamp no longer shadows the bake:

- Remove the unconditional `_ensure_albedo_attr(obj, 0.0)` pre-stamp before the
  kernel runs. Instead, call `get_or_bake_vertex_albedo` (which now reaches the
  real bake) and let it populate the per-vertex `albedo` attribute.
- Preserve the "missing → fully absorbing" safety: if the bake returns nothing
  for an object, that object is absent from the albedo map and
  `absorbed = e_total` (unchanged behavior at `irradiance_kernel.py:507–509`).
- A pre-existing valid `albedo` attribute is still honored (unchanged).

**Note on the Cycles constraint:** the user excluded the Cycles *irradiance*
bake, not the albedo bake ("definitely port the albedo baking"). The albedo
bake is a one-shot, cached-on-disk operation and VisionSim already spawns
Cycles for rendering, so the dependency is acceptable and matches heat-sim
exactly. (A Cycles-free direct base-color-texture-at-vertex-UV sampler would
also reproduce the checkerboard for image-texture materials, but it diverges
from heat-sim for procedural materials; porting `bake_albedo_map` is the
faithful, minimal-surface choice and reuses the already-vendored consumer.)

### 4.2 Honor the blend's authored `irradiance_scale` (fixes 2c)

In the thermal entry path (`blender.py:_thermal_solve`, where the `defaults`
dict is built at `blender.py:1510–1517`):

- Read `scene.heat_sim_settings.irradiance_scale` from the opened blend when
  the property group is present, and use it to populate
  `defaults["irradiance_scale"]`, overriding the `ThermalConfig` default.
- Fall back to `ThermalConfig.irradiance_scale` (100.0) when the blend has no
  authored heat-sim settings (VisionSim-native scenes with no addon data).
- This mirrors how per-object material is meant to be read from the blend: the
  blend's authored value is the source of truth for a heat-sim scene; the
  `ThermalConfig` value is the fallback default.

Result: `bunny_textured.blend` → `irradiance_scale = 1000` → the 10× gap closes.

### 4.3 (Optional) Align the ambient / initial-temperature default (addresses 2d)

Only if the parity comparison is done on *absolute* Kelvin (not ΔT) and the
0.372 K offset matters:

- Confirm VisionSim resolves the bunny to aluminum (expected — §2d). A quick
  `resolve_material("bunny")` dump suffices; no material-read change is
  expected to be needed.
- If desired, align VisionSim's ambient/initial-temperature default with
  heat-sim's `AMBIENT_TEMP` (295.372 K) so both baselines match, or read an
  authored `initial_temperature_K` from the blend when present. This removes a
  constant offset; it does **not** affect the temperature rise or the 10× gap.

This section is a verification-plus-optional-tweak, not a required fix. If the
acceptance comparison is expressed as ΔT (rise), skip it.

## 5. Non-Goals (YAGNI)

- **No Cycles irradiance bake port.** The direct kernel already produces the
  correct incident E (§2a).
- **No irradiance-kernel / solver / unit-scaling changes.** Verified identical.
- **No animated / M2 thermal.** The bunny scene is fully static; M1 is correct
  for it. (Animated flow parity is a separate, previously-scoped effort.)
- **No new CLI flags** beyond what already exists. Parity comes from reading the
  blend, not from new user-facing knobs.
- **No sky/world/HDRI reload work.** The HDRI is unloaded headless in both
  tools equally; it is not a source of divergence for this scene.

## 6. Testing & Verification

**Unit / component (headless, VisionSim `.venv`):**

- `bake_albedo_map` returns an object with `.pixels` of shape `(H, W, 3)` for
  the bunny; returns `None` for an object with no UVs / no material.
- After the adapter change, solving the bunny produces a per-vertex `albedo`
  attribute with `min≈0`, `max≈1`, `std > 0` (i.e. spatially varying, not the
  old constant 0). Assert `mean` in a plausible band (≈0.3–0.6 for the
  checkerboard).
- The `defaults` dict carries `irradiance_scale == 1000.0` when solving
  `bunny_textured.blend`.
- `resolve_material("bunny")` returns aluminum (ρ=2700, cp=978, α=97).

**Integration parity (the acceptance test):**

1. Fresh heat-sim solve+render of `bunny_textured.blend` (delete
   `bunny_textured.heatsim` first; aluminum, 1 s / frame-synced).
2. Fresh VisionSim solve+render (delete `bunny_textured.blend.heatsim` first).
3. Compare `HeatSim_To` temperature per vertex / EXR:
   - peak ΔT and mean ΔT within ±15 %;
   - checkerboard present in VisionSim and spatially correlated with heat-sim
     (e.g. per-vertex Pearson correlation of ΔT above a threshold, or a
     side-by-side To/ PNG showing matching squares).
4. Record both numbers and the correlation in the verification report.

**Determinism:** follow the harness rules — CPU where the comparison needs it,
fixed seeds, `temperature_follow_timeline = False`.

## 7. Risks & Mitigations

- **Cycles albedo bake is finicky** (heat-sim's own `albedo_cache.npz` was
  observed all-zeros in one stale run). Mitigation: the tier-3 fallback already
  degrades safely to full absorption; the unit test asserts `std > 0` so a
  silently-black bake fails loudly instead of shipping a smooth field.
- **Albedo reduces absorbed flux, widening magnitude if applied alone.**
  Porting albedo makes VisionSim absorb ≈0.77 instead of 1.33 (≈0.58×), which
  would *lower* magnitude if shipped without the `irradiance_scale` fix. The two
  changes are complementary and must land together; the parity test gates on
  both. Do not tune one to compensate for the other missing.
- **Over-porting irradiance.py.** Port only `bake_albedo_map` and its minimal
  helpers; do not vendor unrelated Cycles-irradiance functions from heat-sim's
  `irradiance.py`. Keep the new module small and single-purpose.

## 8. File Change Summary

| File | Change |
|---|---|
| `visionsim/simulate/heatsim/irradiance.py` | **Create** — port `bake_albedo_map` (Cycles COLOR bake → `.pixels` (H,W,3)) + minimal UV/image helpers |
| `visionsim/simulate/heatsim/adapter.py` | Stop pre-stamping constant albedo=0; call `get_or_bake_vertex_albedo` so the real bake runs (material read is already correct — no change) |
| `visionsim/simulate/blender.py` | In `_thermal_solve`, read `scene.heat_sim_settings.irradiance_scale` into `defaults`, fallback to `ThermalConfig` |
| `tests/…` (VisionSim) | Add albedo-bake + irradiance_scale + material-resolution component tests; parity acceptance script |

## 9. Outcome & Known Limitations (post-implementation)

**Outcome — parity achieved and verified.** Fresh solves of `bunny_textured.blend`
in both tools compared per-vertex: peak ΔT ratio **1.012**, mean ΔT ratio
**1.102**, per-vertex ΔT correlation **0.996** (checkerboard imprinted and
aligned). VisionSim's baked albedo mean 0.389 ≈ heat-sim's ~0.4. Implemented as
five commits on `heatsim`: (1) port `bake_albedo_map` into `irradiance.py` +
vendored `uv_utils.py`; (2) remove the constant-albedo pre-stamp; (3) honor the
blend-authored `irradiance_scale` via a raw ID-property read (VisionSim does not
register the addon's scene PropertyGroup, so `scene.get("heat_sim_settings")` is
used, not `scene.heat_sim_settings`); (4) an all-zero guard so a stale/degenerate
cached albedo re-bakes; (5) a parity acceptance harness.

**Known limitations / follow-ups (from the final whole-branch review):**

- **Albedo cache is not content-keyed (pre-existing).** The kernel albedo cache
  lives at `<stem>.heatsim/latest/albedo_cache.npz` — a stem-based path that
  collides cross-tool with heat-sim's cache dir, keyed by object *name*, not
  content. The Task-5 guard rejects an all-*zero* stale cache (the incident that
  broke parity), but a stale *non-zero* albedo (e.g. after swapping a material
  while the object name and vertex count are unchanged) would still be served —
  no re-bake. Benign for parity on a fixed blend; **until the cache is
  content/mtime-keyed, clear `*.heatsim/latest/albedo_cache.npz` after changing
  an object's material.**
- **Behavioral change from removing the pre-stamp (intended).** Every materialed
  M1 sim object now absorbs `E·(1−albedo)` (matching heat-sim per-object) instead
  of full `E`; objects lacking UVs/materials get an auto `smart_project` unwrap
  and/or a default gray material created *during the solve* (both inherited
  verbatim from heat-sim). This changes non-bunny M1 scene temperatures and adds
  solve-time mesh side effects — the intended correctness alignment with heat-sim.
- **`irradiance_scale` precedence (by design).** A blend-authored
  `heat_sim_settings.irradiance_scale` unconditionally overrides the
  ThermalConfig/CLI value (blend = source of truth per §3). An INFO log now
  surfaces when this override happens so the effective scale is not a surprise.
