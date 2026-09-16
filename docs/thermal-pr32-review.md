# PR 32 thermal revision: review handoff

This is a local draft for review. No PR text or review replies have been posted.

## Draft PR description

Sparse interior walls could render nearly flat temperatures, while the thermal integration exposed several lighting and numerical methods with different failure modes. This revision keeps two spatial representations within one pipeline: AUTO uses true atlas texel simulation on coarse or unsafe meshes and vertices on dense meshes; Cycles supplies the baked heating; a robust point-cloud Laplacian integrates a fixed transient field. The analytic direct-light/SH9 path, mesh solver, animated thermal solve, duplicate preset tables and offline assignment helper are removed. Atlas emissivity, image identity, source cache checks, failure handling and output state restoration are tightened. The new thermal tutorial documents the model and a hand-authored sidecar.

Validation: Blender 5.2.1 LTS on Linux, CPU solver, retained thermal tests and the cube rendering fixture. The four-vertex localized-light wall is simulated at 256 and 1,024 texels (not vertex interpolation). At 32 bake samples, 64-pixel bake maps and a 0.1 s integration, temperature contrasts were 0.911 K and 0.952 K. Nearest-sample coarse/reference RMS was 0.0224 K, max 0.0894 K. One in-process timing of plan+bake+solve was 0.077 s and 0.060 s, respectively; peak memory was not measured. These are one-scene measurements, not an interior-scene benchmark or a proof that reported interior mottling is resolved. A saved-blend cache test verifies reuse in a second Blender process and an explicit recompute bypass.

## Review item disposition

| Item | Disposition and evidence | Suggested reply |
| --- | --- | --- |
| R1 assignment helper | Removed the large script and its helper-only tests; added a JSON sidecar tutorial. Runtime sidecar parsing remains tested. | “The assignment workflow is now a short documented JSON example; runtime validation remains.” |
| R2 private provenance | Removed private-repository provenance text from retained modules. | “Private development references are removed from the code and documentation.” |
| R3 comments | Removed stale task and vendoring comments in retained modules; some long explanatory comments remain and can still be shortened. | “I trimmed stale progress/provenance commentary; I kept explanations for non-obvious thermal invariants.” |
| R4 constants/materials | Deleted the duplicate preset tables and legacy constants module. `materials.py` owns presets; `physics.py` and `names.py` own physical values and identifiers. | “There is now one preset source and explicit shared units/identifiers.” |
| R5 methods | Cycles bake plus robust point-cloud integration only; AUTO, TEXEL and VERTEX are sampling representations. | “Both representations remain because sparse surfaces need actual additional solve samples.” |
| R6 failures | Required missing bakes, unsafe writeback, malformed atlas inputs and PCG nonconvergence raise. Recoverable Blender API failures now log diagnostics. | “A missing physics input cannot silently become zero heat.” |
| R7 SH9 | Removed SH9 and its direct-light path. Cycles world lighting remains; HDRI is optional. | “The old SH9 path did not require HDRI, but its approximation has been retired.” |
| R8 dependencies | `robust-laplacian` is declared and installed; no `embreex` consumer remains. | “The retained point-cloud solver requires robust-laplacian; embreex is no longer used.” |
| R9 references | Added robust Laplacian and Stefan–Boltzmann references to the thermal tutorial. Preset-specific material references remain to be established. | “Method and physical-constant references are documented; preset provenance still needs a source audit.” |
| R10 assumptions/cache | Replaced obsolete method docs with fixed-geometry, units, absorption, contact, radiance, cache and output contracts. Cache entries are schema-versioned, checked on read and exercised across Blender processes. | “The supported model and conservative cache behavior are documented and tested.” |
| R11 config/API | CLI now sends one serialized `ThermalConfig` to a validated Blender RPC boundary. Existing direct methods remain for compatibility; their internal keyword plumbing remains. | “The normal CLI workflow has one typed configuration boundary; I kept direct entry points compatible.” |
| R12 atlas/persistence | Atlas path hashes result pixels and layout, material nodes are rebound on reload, user-owned same-name images are protected. Save/reopen without the external atlas EXR passes. | “Packing gives a frozen saved result when an output blend is explicitly written; it does not overwrite the input.” |
| R13 persistent data | Correctness workaround remains. Saved output restores the previous persistent-data setting. Comparative render performance remains unmeasured. | “I retained the observed correctness workaround pending controlled timing and frame-error tests.” |
| R14 checks | Thermal-wide Ruff/mypy exclusions removed. Targeted Ruff/mypy and Sphinx builds pass. | “Retained thermal code is back under normal checks.” |
| I1 first-frame write | Animated thermal hook removed; fixed-field repeated-frame render passes. | “The old first-frame write belonged to the deferred animated path; its removal does not establish flicker causality.” |
| I2 muting | Two-pass output isolation remains with `finally` restoration; cube two-frame output test passes. Exception injection for every render failure point is still pending. | “Muting keeps the radiance pass from writing normal outputs and the normal pass from writing radiance.” |
| I3 CUDA index | Removed unconditional cu124 wheel index; optional `--torch-index-url` accepts a platform-appropriate source. | “Existing compatible torch installs are respected; the wheel source is selectable.” |
| I4 embreex | No direct-kernel/BVH consumer or embreex dependency remains. | “No embreex installation is necessary for the retained method.” |
| I5 ignores | No new ignore pattern was committed. An unrelated `.gitignore` edit already exists in the working tree and is left untouched. | “I left the existing local ignore change for separate review rather than staging it into this PR revision.” |

## Remaining integration gates

The affected representative interior and its reported texel mottling were not available for a controlled bake/sample/solve/atlas/AOV comparison. The coarse-wall test establishes one localized-light contrast and a density reference; it does not isolate that interior artifact. CPU Blender was exercised; CUDA, other platforms, and peak memory were not benchmarked. Render failure-injection, cross-scene material/AOV reuse, and a broader cache invalidation matrix also remain for follow-up. These gaps mean the revision should not yet be called merge-ready on rendering evidence alone.
