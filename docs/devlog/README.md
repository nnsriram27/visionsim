# Development log

Engineering notes that would otherwise be lost: investigation trails, design
rationale, and the reasoning behind decisions that commit messages record only as
conclusions.

## How this branch works

`heatsim-devdocs` is `heatsim` plus this directory. Nothing else differs.

- **Code changes go to `heatsim`.** Never commit code here.
- **Notes go here.** Never commit notes to `heatsim`.
- To pick up new code: `git checkout heatsim-devdocs && git merge heatsim`.
  That is always a clean merge, because the two branches never touch the same files.
- This branch never merges back into `heatsim`.

The point is that `heatsim` stays a clean code history you can read, review, and
merge to `main`, while the investigation trail survives somewhere durable rather
than in a gitignored scratch directory on one machine.

Notes live under `docs/devlog/` because `.superpowers/` and `docs/superpowers/`
are both in `.gitignore` — putting them here means no ignore-rule conflicts when
merging `heatsim` forward.

## Contents

### `thermal-atlas/`

The texel-atlas thermal rendering work (LWIR modality for the ~50-scene interior
dataset).

| File | What it is |
|---|---|
| `progress-ledger.md` | Running record of every root cause, with the evidence and the measurements that confirmed each fix. Includes corrections where an earlier diagnosis turned out wrong. |
| `texel-noise-investigation.md` | Why TEXEL renders are ~6× noisier than VERTEX. Three bugs fixed (AOV accumulation through transparent surfaces, non-mesh objects rendering at 0 K, bilinear blending of the atlas validity mask); the remaining spatial mottling traced to the texel-domain discretisation, with nine hypotheses eliminated by measurement. Also records an unsound solve cache and solver defaults shadowed four levels deep. |
| `design-spec.md` | Why the atlas exists: decoupling simulation and render resolution from mesh vertex density, so a 4-vertex plane can still render a detailed thermal field. |
| `implementation-plan.md` | The staged plan the work followed. |
| `task-N-brief.md` / `task-N-report.md` | Per-task briefs and their outcome reports. |

The ledger is the one worth reading first. It documents traps that cost real time,
including several where the pipeline exits 0 while producing nothing useful:

- The `visionsim` console script imports the **main checkout, not your worktree**,
  so worktree-only CLI flags silently vanish. Export `PYTHONPATH=<worktree>`.
- Thermal output is gated behind `--config.include-thermal` (default `False`).
  Without it a run writes RGB frames only and still exits 0.
- `blender.render-animation` takes two positionals (blend, outdir), and the atlas
  flag is `--config.thermal.render-domain`, not `--config.thermal.domain` (which
  is the unrelated FEM domain).
- The test suite needs `--executable <blender>` and `blender` on `PATH`.

## Known-open items

Recorded so they are not mistaken for oversights:

- **The classroom scene renders thermally flat.** This is physically correct under
  the current model rather than a bug: its world background strength is `0`, its sun
  is 1 W/m², and its daylight arrives through emissive light-portal surfaces. The
  irradiance kernel counts only lamp objects and the world sky term, and models no
  indirect bounce, so 99.8% of surface points receive under 1 W/m² and stay at their
  295 K initial condition. Changing this is a modeling decision about whether
  emissive surfaces and bounced light should carry thermal flux.
- **Atlas tiles are not fully rasterized.** Around 1.2% of kitchen1 pixels sample an
  unwritten texel and fall back to the per-object default of 295 K instead of a
  solved value, showing flat where there should be a gradient.
  *Update 2026-08-29:* partly addressed — see `texel-noise-investigation.md` §3.3.
  The unwritten texels were also blending *below* the solve floor, because the
  {0,1} validity mask is sampled with `Linear` interpolation so colour and alpha
  fall toward zero together. Thresholding the gate and raising
  `_ATLAS_DILATE_ITERATIONS` 1 → 8 cut sub-floor pixels 3394 → 28. The dilation
  count is the reason holes persisted: 25 invalid components of 1–462 texels sat
  inside tile interiors against a one-texel margin.
- **TEXEL is ~6× spatially noisier than VERTEX** (2.330 K vs 0.382 K ripple) and no
  tunable parameter changes it — sampling, denoising, PCG tolerance, texel density,
  shadow-ray count and kNN conditioning were each eliminated by measurement. Traced
  to the kNN point-cloud Laplacian built over UV-rasterized texel positions, whose
  3D spacing is irregular. Fixing it properly means a cotangent FEM operator on the
  texel triangulation. See `texel-noise-investigation.md` §4.
- **Node-group external inputs** in `occluders.py` are followed whether or not the
  group routes them to its output internally. Documented in that module's docstring;
  the failure direction is a spurious shadow rather than a deleted one.
