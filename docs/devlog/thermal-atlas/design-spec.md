# Thermal Atlas: Texel-Domain Simulation + Texture-Based Rendering — Design Spec

**Date:** 2026-08-04
**Branch:** `thermal-atlas` (worktree off `heatsim` @ 4a86dde)
**Status:** Approved (user, 2026-08-04) — with the density-driven allocator revision
(selection by native vertex density; sizing per surface area; budget demoted to a
soft warn-only ceiling), per user review.

## 1. Problem

The thermal modality simulates AND renders on mesh vertices. Both halves break on
artist-authored geometry:

- **Simulation:** kitchen1's floor spans 8.5 × 9.6 m with **16 vertices**. The FEM
  solves 16 samples for ~80 m² of the most visible surface in the frame, while a
  decorative orchid gets 21k. Sample density is dictated by the artist's
  tessellation, which is uncorrelated with thermal importance.
- **Rendering:** the render reads the per-vertex `sim_temperature` attribute, so a
  4-vertex plane renders as a single bilinear gradient across its whole area. The
  500 s kitchen1 render (contact sheet, 2026-07-25) shows exactly this: walls and
  floor are flat color ramps with no spatial thermal detail.

The user's requirement: **rendering must not be limited by the base mesh's vertex
count** — a 4-vertex plane should render rich spatial temperature detail, the way it
renders rich *visible* detail from an image texture.

## 2. Research grounding

- **Professional IR simulators already work this way.** MuSES (ThermoAnalytics) and
  OKTAL-SE — the industry-standard EO/IR signature tools; OKTAL-SE uses Blender as a
  renderer — generate "infrared textures of targets from the retrieved temperature
  distribution, mapped onto 3D geometry models." Temperature lives in a texture, not
  in vertex data.
- **The identical problem was solved by lightmaps.** Baked lighting computes a field
  in UV space at a chosen texel resolution and renders any-poly geometry by sampling
  it. Standard requirements carry over: non-overlapping 0–1 UVs and margin dilation
  against bilinear seam bleeding.
- **Our solver is already sampling-agnostic.** The FEM runs in POINTS domain on a
  robust-Laplacian point cloud (Sharp & Crane, SGP 2020), explicitly designed for
  point clouds "even with irregular sampling or non-uniform density." It never sees
  mesh topology — so simulation density can be decoupled from the mesh by feeding it
  *sampled surface points*; no subdivision, no modifiers, no mesh writes.

Sources: thermoanalytics.com/product/muses, oktal-se.fr, github.com/nmwsharp/robust-laplacians-py,
therealmjp.github.io (SG series, baking lab).

## 3. Core idea: the texel is the simulation element

One resolution decision serves both halves. Each object gets a tile in a scene-wide
**temperature atlas**; the texels of that tile are simultaneously:

1. the **simulation sample points** (position + normal from the texel's UV→triangle
   inverse mapping), and
2. the **render source** (the thermal shader samples the atlas by UV).

```
mesh (any vertex count)
   └─ atlas tile sized by SURFACE AREA (budgeted, clamped)
        └─ texel grid = sim points: position, normal, face → material slot
             ├─ albedo per texel  (existing Cycles albedo bake — same UV space)
             ├─ irradiance per texel (existing Direct Kernel at arbitrary points)
             ├─ α, ρ, c, ε per texel (face → slot → sidecar preset; EXACT, no
             │                        seam averaging — each texel has one face)
             └─ POINTS-domain FEM solve (solver completely unchanged)
                  └─ final temperatures → atlas EXR (+ margin dilation)
                       └─ thermal AOV shader: UV → atlas sample → temperature
```

Why this fits this codebase unusually well:

- `bake_albedo_map` already produces **per-texel albedo in bake-UV space**; today we
  *average it down* to vertices. Texel-sim consumes it at native resolution.
- The smart-project UV machinery (with all the 2026-07-24 headless robustness fixes)
  is exactly the tile-unwrap tool.
- Per-slot material resolution becomes exact: a texel belongs to one face, one slot.
  The area-weighted seam compromise applies only to the legacy vertex path.
- The vertex path remains untouched as default and as per-pixel fallback.

## 4. Components

### 4.1 Atlas allocator — density-driven, not budget-driven

**Selection (which objects get texels at all):** an object joins the atlas iff its
native vertex density `verts / surface_area` is **below** `atlas_texel_density`.
Objects already sampled at or above the target density (dense scans, the 21k-vert
orchid) are *excluded*: they keep the per-vertex path for both simulation and
rendering — their real vertices stay in the solve cloud exactly as today. The
post-treatment invariant is that every surface is sampled at ≥ the target density,
by whichever representation already achieves it.

**Sizing (how big a tile):** tile texel count = `surface_area × atlas_texel_density`,
side lengths rounded to multiples of 4 (no power-of-two quantization — a big floor
must not be overshot 4×), clamped to [`tile_min`=16², `tile_max`=512²]. The per-area
rule is the contract; big objects get big tiles.

**Safety valve (soft, never an allocator):** if total texels + retained vertices
exceeds `atlas_texel_soft_max` (default 500k), scale the density down uniformly and
**warn loudly** with the effective density. It exists only to prevent an accidental
million-point solve; raising it is a one-flag decision. It never silently squeezes
quality below the warning threshold.

Tiles are shelf-packed into one atlas image (sized to fit, ≥2px inter-tile padding).
The object's smart-project bake UV is scaled/offset into its tile and stored as a
new UV layer `HeatSim_Atlas_UV` (per-mesh data ⇒ one shared atlas image works with
shared materials).

### 4.2 Texel sampler
Rasterize each object's loop-triangles in atlas-UV space (extend the existing
UV-rasterization logic used to read baked albedo back to vertices — same math,
inverted output). Per valid texel: 3D position (barycentric), interpolated normal,
source face index, material slot. Output: per-object texel table `(K, …)` +
`(tile, x, y)` addressing.

### 4.3 Sim-input assembly (per texel)
- **albedo**: sample the object's albedo bake at the texel (shared UV space ⇒ direct
  texel correspondence; fall back to the kernel's full-absorption default).
- **irradiance**: Direct Kernel evaluated at texel positions/normals (it already
  operates on arbitrary point arrays).
- **material**: face → slot → sidecar entry → preset (α, ρ, c, ε, role, dirichlet_K);
  object-level `resolve_material` fallback where the sidecar is silent.

### 4.4 Solve integration
`_combine` gains a TEXEL mode: the combined point cloud = texel points of
atlas-participating objects + plain vertices of excluded objects (no-UV /
degenerate / opt-out), with the existing per-object layout slices. POINTS domain,
solver untouched. Per-texel Dirichlet mask uses the same pinning treatment
(α=0, irr=0, boundary_mask=False, t0=reservoir).

### 4.5 Atlas writer
Final-timestep texel temperatures → 32-bit single-channel(+alpha) EXR atlas; N-pixel
margin dilation (push-out) so bilinear filtering never reads invalid texels; alpha=1
marks valid coverage. Stored in the `.heatsim` cache and loaded/packed as a Blender
image at render time.

### 4.6 Shader
Extend the thermal AOV node group (and the gray-body radiance input): 
`Attribute("HeatSim_Atlas_UV") → Image Texture(atlas, Non-Color, linear)` produces
atlas temperature; `Mix(vertex-path temperature, atlas temperature, atlas alpha)`
selects per pixel. Objects without atlas coverage sample alpha=0 and get exactly
today's behavior (per-vertex attribute → object-prop fallback chain). One shared
node group; no per-object material duplication.

### 4.7 Config
`ThermalConfig`: `render_domain: Literal["VERTEX","TEXEL"] = "VERTEX"` (default =
byte-identical current behavior), `atlas_texel_density: float` (texels/m²; default
set from the §5 benchmark), `atlas_tile_min: int = 16`, `atlas_tile_max: int = 512`,
`atlas_texel_soft_max: int = 500_000` (soft ceiling, warn + uniform density
rescale). All enter the solve cache key.

## 5. Validation (kitchen1 vertical slice)

1. **Solver benchmark first**: Laplacian build + 500-step solve timing at 50k / 150k /
   300k synthetic points, before locking the density default.
2. kitchen1, 500 s / 1 s steps, 50 frames, TEXEL mode: before/after contact sheet.
   Acceptance: the floor/walls show *spatial* temperature structure (not a 4-corner
   gradient); no tile-seam artifacts; excluded objects render as today.
3. Regression: `render_domain="VERTEX"` output byte-identical; full suite green.

## 6. Out of scope

- Animated (M2) mode — static only for the dataset.
- Mesh subdivision — superseded: sim density now comes from texels, render density
  from the atlas.
- classroom / bulk scenes — after the slice is approved.
- Temperature-EXR ground-truth path changes — the AOV output format is unchanged.

## 7. Risks

- **Solve cost at ~200k points** (Laplacian build is C++/sparse; solver is torch
  sparse on GPU). Mitigated by the benchmark task + budget knob.
- **UV quality on hard meshes** — smart-project can produce poor unwraps; failure
  degrades per-object to the vertex path (never worse than today).
- **Seam bleeding / tile borders** — margin dilation + ≥2px tile padding; validated
  visually in the acceptance sheet.
- **Cross-object conduction via kNN** — the combined point cloud connects nearby
  objects (same as today with vertices, now denser); acceptable, it approximates
  contact conduction.
