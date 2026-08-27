# PLAN: thermal atlas (texel-domain sim + texture rendering)
# docs/superpowers/plans/2026-08-04-thermal-atlas.md — worktree thermal-atlas @ 4a86dde
# 4 tasks + Task 0 benchmark. Spec approved w/ density-driven allocator revision.
# ENV: /home/sriram/research/visionsim/.venv/bin/python with PYTHONPATH=THIS WORKTREE
#   (venv editable install points at MAIN checkout!), LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libsqlite3.so.0,
#   Blender /home/sriram/softwares/blender-5.1.0-linux-x64/blender, pre-existing fail: test_cli::test_completions
Task 0 benchmark: rerunning (bawb95umf) after shape bugfix.
BASE: 4a86dde

## USER DIRECTIVES (2026-08-04 23:50 PDT) — BINDING FOR AUTONOMOUS CONTINUATION
1. ALL reviews on LOCAL glm-5.2 (mcp__local-llm__local_review; paths mode — target-range
   diffs can report "no diff"). NEVER Claude-model reviewers. Implementers may be Claude.
2. User out of usage + away. PAUSE until >= 2026-08-05 02:50 PDT, then CONTINUE
   AUTONOMOUSLY to completion + testing. ScheduleWakeup clamps at 3600s -> chain hops,
   check `date` each wake, only proceed past 02:50.

## STATE AT PAUSE
Task 0 benchmark: CPU rerun RUNNING (bg bbxi9gcp0, may finish during pause; on its
  notification: record N= lines to this ledger, nothing else, stay paused).
Task 1: commit b6e30fb, 18/18 tests, ruff+mypy clean. Claude review (before directive):
  Approved, 1 Important OPEN = add regression test for mirrored-UV/CW area2<0 swap branch
  in rasterize_tile (code verified correct by reviewer's own mirrored-UV construction:
  quad faces natural order, loop_uv two entries reversed, expect pos [10,15,0]-style
  barycentric match, full coverage, no double-claim; mutation-check by inverting swap).
  2 Minors: report-wording (position vs normal ordering claim); _DEGENERATE_UV_AREA_EPS
  name reused for 3D normal check.
## CONTINUATION PLAN (execute after 02:50)
1. Dispatch fixer (sonnet): mirrored-UV regression test (spec above), mutation-check,
   commit test(heatsim): regression test for mirrored-UV (CW) triangle rasterization.
2. Local re-review Task 1 (local_review paths=[atlas.py, test_heatsim_atlas.py]) -> close.
3. Task 2 per plan (brief via task-brief script), implementer sonnet, LOCAL review.
4. Task 3 same. 5. Task 4 controller: benchmark-informed density default (write into
   config + plan), VERTEX regression (full suite --executable), kitchen1 TEXEL render
   500s/50f/512^2, contact sheet TEXEL vs VERTEX, ledger + user summary.
Env reminders: venv=/home/sriram/research/visionsim/.venv/bin/python, PYTHONPATH=worktree
  ALWAYS, LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libsqlite3.so.0, blender 5.1.0 path above.
Task 0 benchmark UPDATE (00:20): CPU rerun TIMED OUT at 90min without completing even
  N=50k/500-step (no output lines). DATA POINT: bare-venv CPU torch is far slower than the
  in-Blender production solves (kitchen1 80k/500 steps completed within a render). At wake:
  do NOT re-run standalone; instead time the solve INSIDE the real kitchen1 TEXEL render
  (production path) and derive the density default from a SHORT-steps standalone run
  (e.g. 50 steps, scale x10) if a standalone number is still wanted. Remain paused.

## AUTONOMOUS RUN (from 02:52 PDT)
Task 1 CLOSED: +6719e61 mirrored-UV regression test (mutation-verified). GLM-5.2 review:
  no real defects (core math hand-verified); 1 medium = dilate() crashes on multi-channel
  `valid` -> contract fix (2D-only guard+doc) DELEGATED to Task 3 brief (only consumer).
Task 2 DONE: f89be73. 12 new tests, regression 69 passed (+35 adjacent), existing suites
  unmodified (git diff empty), 2 mutation checks. Concern noted: _augment_interior_points
  no-ops for TEXEL objects (documented). GLM review: dispatching now (background).
Task 3 DONE: 0d843ed. 364 passed full suite; parity tests RED->GREEN unedited; stubs regen.
  Shader gates atlas mix on alpha x object-prop coverage flag (origin-texel leak guard).
  Concerns: EXR ~2mK drift (accepted), seam bleed mitigated-not-eliminated (accepted).
GLM reviews (T2 diff via local_delegate + materials.py via local_review):
  BLOCKING: "atlas":null enters VERTEX cache key (hash changes, docstring lies). FIX.
  PERF: per-vertex irradiance+albedo computed then overwritten for atlas objs; scene BVH
    rebuilt per atlas object. FIX (matters for kitchen1 render).
  FIDELITY: texel path hardcodes 8 shadow rays ignoring direct_kernel_soft_shadow_rays. FIX.
  MATERIALS: out-of-band dirichlet_K dropped but role kept -> ambient SINK; degrade role
    to FEM + warn, KEEP legit no-K-pin-at-ambient behavior (existing test). FIX.
  reason/scene JSON null -> "None" strings (scene reaches cache keys). FIX.
  Accepted/no-action: texel sky-occlusion gap (documented; sky_occ off in our renders);
  dilate-not-called-in-T2 (T3 calls it); _augment_interior_points TEXEL no-op (documented).
Consolidated fix pass dispatching now.
Fix pass DONE: 4964874, all 5 findings, 122 tests green (out-of-band-dirichlet tests
  intentionally updated per behavior change; no other existing test edited).
GLM review of T3+fixes (f89be73..4964874, pyi excluded): running bg k4zwhxg5t.
Task 4 starting: VERTEX 2-frame regression render now; TEXEL render next; review triage
  when GLM lands (blockers would force TEXEL re-render, accepted).
T3 GLM review: F1 stale sim_temperature on VERTEX->TEXEL switch; F2 static-branch stamp
  clobbers Dirichlet fallback (mirror of M2 animated fix); F3 dilate iters==padding can
  bridge tiles; F4 TEXEL->VERTEX stale coverage gate. ALL FIXED e8c1b66 (105 tests green).
  GLM verified-correct: cache-key parity, scatter orientation, alpha, shader gate+CLIP,
  P1-P99 texel pooling, all 5 prior fixes, config threading x3 methods.
Fix verification (GLM on e8c1b66 diff): bg k2o2kb2yp. Routing policy: Claude subagents =
  code-writing ONLY; local GLM otherwise.
Task 4 RUNNING: TEXEL render GPU0 (bibhe6a16, 50f 512^2), post-fix VERTEX 2f GPU1
  (bydsputyq). Old pre-fix vertex job stopped. (pkill self-match, exit 144, harmless.)
GLM fix-verification: F1-F4 all correct; 1 refinement = dilation invariant is 2*iters<=padding
  (tiles dilate toward each other), comment was wrong. FIXED 2231b6e (comment + module assert
  + middle-gap-column test). 6 passed; 4 errors = conftest live-Blender guard (my renders),
  re-verify in final gate. REMAINING: TEXEL render bibhe6a16, VERTEX check bydsputyq, then
  contact sheet + full suite + summary.
VERTEX regression gate PASSED: exit 0, median 299.0 / P99 300.8 == pre-atlas 500s run
  (even P1~125 cold-outlier matches). Feature-off behavior preserved on the real scene.
Awaiting TEXEL render (bibhe6a16) -> contact sheet, atlas stats, full suite, summary.
TEXEL RENDER BUG (caught by controller field-stats gate): frame median 11.4K. Atlas EXR
  itself has B median 11.29 = sRGB_encode(295) EXACTLY -> write-side linear->sRGB transform
  (bpy.data.images.new defaults colorspace sRGB; image.save() encodes). Round-trip test
  passed because encode/decode are symmetric -> tested symmetry not absolute values.
  FIX: colorspace_settings.name='Non-Color' on create AND load; strengthen test to assert
  ABSOLUTE EXR values via OpenEXR. Then re-render TEXEL. Dispatching fixer.
Colorspace fix 7c40dc9: Non-Color tag on write-side image (load sides already tagged);
  empirical 11.23 -> 295.0; round-trip test strengthened to assert ABSOLUTE EXR values
  via standalone OpenEXR (the assertion that catches this class). 26 tests green.
TEXEL re-render bc04b8984 (solve cache reused, atlas rewritten through fixed writer).
TEXEL re-render (fixed): field P1=295 med=299.8 P75=307.3 P99=366.6, 93.2% room-band.
CONTACT SHEET: acceptance MET — floor shows warm pool + wall glow (VERTEX: flat gradients);
  lamps render as compact hot elements; dense objects unchanged; no visible tile seams.
FULL SUITE: 1 failed (pre-existing test_cli::test_completions, identical on base)/375 passed.
Videos: kitchen1_thermal_texel.mp4 + kitchen1_vertex_vs_texel.mp4 + comparison sheet in
  out/t4_texel/. Awaiting final whole-branch GLM review (kb2vn20tk) -> summary.

## FINAL WHOLE-BRANCH GLM REVIEW: APPROVE, no blockers.
  F1 (low, acknowledged): byte-identical claim has ONE documented exception — stamp-order
  fix corrects unsolved-Dirichlet VERTEX fallback from buggy-ambient to reservoir
  (deliberate, mirrors animated branch, breaks no test). FOLLOW-UP: optional VERTEX-mode
  regression test pinning it. F2 nit FIXED (density fallback 50->1500, commit below).
## PLAN COMPLETE. 9 commits on thermal-atlas. Suite 375 passed/1 pre-existing fail.
  Deliverables in out/t4_texel/: kitchen1_thermal_texel.mp4, kitchen1_vertex_vs_texel.mp4,
  kitchen1_texel_vs_vertex.png. Atlas 652x1036, ~130k valid texels + retained dense verts.
  NOT merged into heatsim; worktree intact for user review.

## DIAGNOSIS: why island/cabinets render black (user question, 2026-08-05)
NOT an atlas bug: VERTEX and TEXEL both show 69.4% of pixels at EXACTLY 295.000 K.
ROOT CAUSE (pre-existing): 13/39 kitchen1 objects carry TOPOLOGY-CHANGING MODIFIERS
  (Bevel + Geometry Nodes, EdgeSplit). Solver runs on the EVALUATED mesh; render reads a
  per-vertex sim_temperature attribute writable only on the BASE mesh. write_frame_attributes'
  shape guard (arr.shape[1] != len(mesh.vertices)) bails to the ambient object-prop fallback
  -> the ENTIRE solved field is discarded. e.g. Vert.001 base 128 -> eval 356, solved mean
  306.05K (HOTTEST object in scene) yet renders at 295.000.
  Frame 289 coverage: Vert.001 52.4% + cocina gas 8.3% + Vert.002 8.2% = 68.9% ~= 69.4% dark.
  In TEXEL mode my own base-vs-eval demotion guard sends the same objects down that same path.
ALBEDO IS CORRECT: corr(albedo, meanT) = -0.376 (negative = physical). Dark cabinets
  (albedo 0.001) are the hottest solved objects. Absorbed = E*(1-albedo) verified.
FIX (high value): the atlas is UV-ADDRESSED, not vertex-addressed -> immune to base/eval
  vertex-count mismatch. Rasterize from the EVALUATED mesh (eval geometry + eval loop UVs),
  write the UV layer to the base mesh so modifiers propagate it, and DON'T demote on
  count mismatch. Those 13 objects then render their real solved field.
SECONDARY (explains modest overall spread): irradiance mean only 0.55 W/m2 pre-scale --
  Direct Kernel models only LAMP objects (kitchen1 has just 2 area lamps + flat 0.196 grey
  world); the emissive MATERIALS that light the kitchen in RGB (ilum 4.0, DIC ILUM) contribute
  ZERO thermal flux, and there is no indirect bounce (stated in the kernel docstring).
FIX (1) evaluated-mesh rasterization: commit 8a2ad54. Atlas UV written to BASE mesh (so
  modifiers propagate), triangles+UVs+material_index now read from EVALUATED mesh, vertex-count
  demotion DELETED. Coordinate convention handled: _write_atlas_uv_layer stores atlas-GLOBAL
  UVs, rasterize_tile wants tile-LOCAL -> build_atlas_plan inverts. 65 tests green; replaced
  the test that asserted the old demotion; new real-bpy test with a Subdiv modifier proves
  promotion + texel positions inside world bbox.
  Verification re-render: bi7jffgeg (expect ambient fraction to drop well below 69.4%).
CACHE-INVALIDATION BUG found while verifying 8a2ad54: the atlas digest is computed from
  layout.tiles, which is allocated BEFORE the per-object rasterization/demotion loop -- so
  demoted-vs-participating objects produce the SAME digest. Result: after the fix the cache
  key was unchanged, the old solve+atlas EXR were reused, and my first "verification" render
  was a no-op (identical 69.4%). Cleared the cache and re-rendered (bb62szuin) for a true test.
  TODO: include per-object TEXEL COUNTS (actual participation) in _atlas_digest so a change in
  who participates invalidates the cache. Not yet fixed.
GLM review of 8a2ad54: production logic ALGEBRAICALLY CORRECT (evaluated rasterization +
  fwd/inv UV remap, incl. interpolation-commutativity for modifiers). 1 medium finding, TEST
  fidelity only: test_uv_failure_demotes_to_vertex_path_with_warning feeds TILE-LOCAL fake UVs
  where build_atlas_plan applies the atlas-global inverse remap -> passes for the wrong reason,
  fragile to tile offset. Fix: fake_uv should apply the forward remap like
  test_build_atlas_plan_vertex_count_mismatch_no_longer_demotes does.
BATCHED FIX PENDING (after render verification): (a) _atlas_digest must include actual
  per-object texel participation; (b) the test-fidelity fix above.

## FIX (1) VERIFIED ON REAL SCENE (clean-cache re-render)
  Vert.001: 356 vertex-path nodes -> 19,834 TEXELS (mean 304.5K); Vert.002 -> 4,825;
  cocina gas -> 2,580. Demotions: 0.
  AMBIENT PIXELS 69.4% -> 4.0%; median 295.00 -> 311.33 K; p75 298.92 -> 317.03.
  Visual: cabinets/island/countertops now show real thermal structure (was flat black).
  Deliverables: out/t5_texel/kitchen1_fix_before_after.png, kitchen1_thermal_fixed.mp4.
  Cache-digest + test-fidelity fixes: a7ee81c (70 tests green).
  NOTE for user: scene now reads uniformly HOT (median 311K/38C) -- low-albedo black
  cabinets absorb most, and irradiance_scale=100 over 500s is aggressive. That is a
  CALIBRATION knob, not a coverage bug. Offer to tune.

## VERIFICATION: stools/chairs (user question)
SIMULATED, YES: Circle.005-.008 mean 301.3-302.0 K (+6.3..+7.0), Vert.011/.012 309.2 K
  (+14.2, MORE than the cabinets), Vert.006 302.9, Vert.007 309.8. All on the VERTEX path
  with 396-960 nodes -- correctly EXCLUDED from the atlas by the density rule (396 verts on
  a small stool already exceeds 1500 texels/m2). Not a coverage gap.
NEW BUG FOUND (PRE-EXISTING, NOT from the atlas work): those objects render with ~60% of
  their pixels at EXACTLY 0.00 K (impossible; distinct from the 295 K ambient fallback).
  Whole frame: 3.6% of pixels ==0 K, 4.55% <100 K.
  PROOF it predates this branch: pre-atlas VERTEX render (Jul-25) frame 577 seat patch
  ==0K 60.2% vs today's TEXEL 60.3%; whole-frame 3.58% vs 3.61%. Identical.
  RULED OUT: missing temperature AOV (all 40 in-use materials get it); atlas involvement
  (these objects have no tile/atlas UV); EXR channel misread (single 'V' channel).
  NOTED: every affected object uses material MADERA BANQUETAS, and the affected ones are
  LINKED DUPLICATES sharing mesh data (Circle.005 x4 users, CUVert.011 x2) -- shared mesh
  means one shared sim_temperature attribute for 4 objects (last write wins). Prime suspect,
  not yet proven; the confirming diagnostic (inspect sim_temperature post-write-back) timed
  out on a cache miss.
  LATENT RISK for the atlas: if a shared mesh ever IS an atlas participant, its siblings each
  get a distinct tile but only ONE UV layer can exist on the shared mesh -> last write wins ->
  siblings sample the wrong tile. Not currently triggered in kitchen1 (no shared-mesh object
  is a participant) but should be guarded.

## ROOT-CAUSE + DESIGN DATA (2026-08-06)
BUG A -- SHARED-MESH COLLISION: **ROOT-CAUSED, PROVEN** by minimal repro (/tmp/repro_shared.py):
  two objects sharing one mesh datablock -> write_frame_attributes writes sim_temperature to
  obj.data for each -> LAST WRITE WINS -> A renders B's field (A wanted 310, got 350).
  kitchen1: Circle.005 mesh shared by 4 stools; CUVert.011 by 2 chairs. So 3 of 4 stools
  display the 4th stool's temperatures. Same hazard for the atlas (one mesh = ONE UV layer,
  so N siblings cannot have N tiles) -> atlas-everything would make this WORSE, not better.
BUG B -- 0-K PIXELS: NOT root-caused. Characterized: 3.61% of frame exactly 0 K; 74% of those
  are INTERIOR to solid blobs (57.5% >=2px deep) => real missing data, NOT thin-edge artifacts.
  PRE-EXISTING (pre-atlas 3.58% vs today 3.61%). RULED OUT: missing AOV (all 40 materials have
  it), AOV mis-wiring (identical graph on affected + working materials), missing
  heatsim_default_temperature (all 39 objects have 295.0), atlas involvement, EXR channel.
  Still open.
ATLAS-EVERYTHING COST (measured, kitchen1, 37 objects, 91,960 native verts):
  current hybrid (9 participants) = 229,456 nodes; solve+50f render ~1h50m (CPU torch).
  atlas-everything @750/m2  = 403,776 texels (4.4x native, 1.8x current)
  atlas-everything @1500/m2 = 796,144 texels (8.7x native, 3.5x current)
  atlas-everything @3000/m2 = 1,073,200 texels (11.7x native)
  DOWNSAMPLING HAZARD: at 1500/m2, 25 of 37 objects LOSE resolution vs their own vertices --
  orquidea 21,198 verts -> 256 texels (83x loss), Circle 6,870 -> 784, JARRON BACAN 5,712 -> 256.
  'never downsample' variant only 8% more texels (859,293) -> downsampling saves little; cost is
  dominated by large-area objects.
KEY PERF FINDING: the venv torch is CPU-ONLY ("Torch not compiled with CUDA enabled") while two
  RTX 2080s sit idle. Installing a CUDA torch is likely the single biggest lever on solve time,
  and would change the atlas-everything cost calculus entirely.

## BUG B (0-K) ROOT-CAUSED 2026-08-06
Zero mask = EXACTLY the two visible bar stools (9460/9461 zeros in bottom half).
Those stools are Vert.011/Vert.012: base 720 verts, solve 808 nodes (BEVEL+NODES modifier).
write_frame_attributes' shape guard therefore leaves sim_temperature ABSENT on their mesh
(confirmed directly: "sim_temperature=ABSENT default=295.0").
When the attribute is ABSENT the temperature AOV emits 0; the RADIANCE pass on the same
pixels correctly yields the 295 K fallback (radiance 431 vs 503 on ~307 K pixels -- ratio
matches (307/295)^4 exactly). So the temperature source chain is fine; only the AOV path
degrades to 0 on a missing attribute.
=> Same root class as the cabinet bug: solved-on-evaluated-mesh results cannot be written
back to the base mesh. Cabinets showed 295 (atlas demotion + object-prop fallback); stools
show 0 (no atlas tile AND no attribute).
SCENE-AGNOSTIC FIX PLAN (3 parts, all general -- not kitchen1-specific):
 1. SELECTION: force atlas participation whenever base_verts != evaluated_verts, regardless
    of density -- the vertex write-back path structurally cannot represent those objects.
 2. SAFETY NET: if an object still lands on the vertex path with a shape mismatch, write a
    constant-fill sim_temperature = mean of its solved field (warn), never leave it absent.
    Preserves the object's real heating (309 K, not 295) and kills the 0-K class outright.
 3. SHARED MESH: linked duplicates share one mesh => one sim_temperature and one UV layer.
    Make sim objects single-user (or handle explicitly) so siblings stop overwriting.

## SCENE-AGNOSTICISM VALIDATION (build_atlas_plan on 5 diverse scenes, 2026-08-06)
scene       objs  modmismatch  participants  texels   retained   total    rescaled
kitchen1      39      13            ~9       ~140k     ~90k      229k      -
classroom    176      15            50       235,124   61,386    296,510   YES
bathroom1    358      19             6         2,974   59,369     62,343   no
barbershop  1122     479           480       108,410  372,005    480,415   YES
staircase     10       0             5         9,134   46,298     55,432   no
junkshop      37      12            23       343,574  286,015    629,589   YES
NO FAILURES on any scene. Selection adapts sensibly (bathroom1 is already dense -> only 6
  participants; barbershop is huge -> 480).
KEY: the modifier-mismatch bug I just fixed affects 479/1122 = 43% of barbershop's objects
  -> this was NOT a kitchen1 quirk, it is endemic to artist-authored assets. Fix validated as
  high-value across the dataset.
CAVEAT FOUND: soft_max=500k counts texels+retained verts, but retained verts alone can exceed
  it (junkshop 286k retained, total 629k after rescale). The cap is soft by design (warn+
  rescale) and cannot shrink retained vertices -- acceptable, but the warning should say so.
COST SPREAD: 62k..630k nodes across scenes -> CPU-only solving is the bottleneck; CUDA matters.
Fixes verified at source: Vert.011/.012 sim_temperature now 309.26 K (was ABSENT -> 0 K in AOV);
  shared meshes 6 -> 0, idempotent.

## CUDA LANDED (2026-08-06): torch 2.12.1+cu126, cuda_available=True, 2x RTX 2080.
  Sparse-COO CUDA path (what the solver uses) verified working.
  Renders relaunched with --config.thermal.device cuda:
  kitchen1 GPU0 (bqnanbmi7), classroom GPU1 (bw1edxdlv), both TEXEL 500s/1s, 50f, caches cleared.
KITCHEN1 GPU RENDER (t6, all fixes): EXIT 0, 50/50 frames.
  SPEED: 01:21:37 -> 01:24:06 = 2m29s wall-clock TOTAL (solve+50 frames) on GPU,
  vs 1h50m for the same job on CPU (t5b) => ~44x faster.
  0-K: 3.61% -> 0.18% (residual == true background/void). ==295: 1.48% -> 0.75%.
  Stool region: 0.0% zeros, mean 315.5 K (was solid black / 0 K). VERIFIED VISUALLY.
  Deliverables: out/t6_kitchen1/{kitchen1_thermal_final.mp4, kitchen1_final_sheet.png,
  stool_fix_proof.png}

## CLASSROOM CROSS-SCENE RENDER (t6) -> EXPOSED A NEW BUG (2026-08-06)
classroom t6 rendered 50/50, exit 0, 0-K only 0.91% -- but 89.7% of the scene sat
  within 1 K of its 295 K initial temperature: the room barely heats, while the RGB
  clearly shows sun patches on the floor.
ROOT CAUSE (scene-agnostic, NOT a classroom quirk):
  irradiance_kernel._collect_scene_meshes_world() built the occluder BVH from EVERY
  renderable mesh, including transmissive geometry. classroom.blend is lit through two
  planes with a 'dayLight_portal' material (Cycles light portals, 4 and 8 verts) plus a
  'frostedGlass' box; the 1963 W exterior_fillLight reaches the interior ONLY through
  them. Treated as solid -> interior gets nothing.
  Scene lighting confirmed: world background strength = 0 (no sky/SH9 term at all),
  sun lamp energy = 1.0 W/m2 (negligible). So the portals were the whole daylight path.
MEASURED: scene-wide sampled irradiance 0.010 -> 0.043 W/m2 when portals/glass are
  excluded from the occluder set (4.3x). Still tiny -- the scene stays cold because of
  the SEPARATE, already-documented Direct Kernel limitation (no indirect bounce;
  emissive materials contribute no thermal flux, deferred by user decision).
FIX (ebc1b49): _casts_shadow(obj) -- skip as occluder when Cycles shadow visibility is
  off, or when ALL material slots are effectively clear (Glass/Transparent/Refraction
  BSDF, or Principled constant transmission >= 0.95). Partly-transmissive still occludes;
  no-material objects still occlude; a LINKED transmission input reads opaque (unknowable
  statically). Physics: glass is opaque in LWIR but passes most solar shortwave, which is
  the band this kernel integrates.
  Test: tests/test_heatsim_irradiance.py::test_transparent_and_shadowless_meshes_do_not_occlude
  (7 cases incl. mixed slots + collector/predicate agreement). 6/6 irradiance tests pass.

## HARNESS GOTCHA THAT COST TWO FALSE RENDERS (record this)
The `visionsim` console script imports the EDITABLE install rooted at the MAIN checkout
  (/home/sriram/research/visionsim), NOT this worktree -- so it silently lacks
  --config.thermal.render-domain and all atlas flags. Correct invocation from the worktree:
    export PYTHONPATH=/home/sriram/research/visionsim/.claude/worktrees/thermal-atlas
  Also: the CLI takes TWO POSITIONALS (blend, outdir), not --input-path/--output-dir, and
  the atlas flag is --config.thermal.render-domain (NOT --config.thermal.domain, which is
  the FEM domain POINTS|MESH).
  Test suite needs: --executable /home/sriram/softwares/blender-5.1.0-linux-x64/blender
  and `blender` on PATH (test_cli.py::test_completions fails with exit 127 otherwise --
  environmental, not a code failure; passes with PATH set).
  AND: thermal output is gated on --config.include-thermal (default False). Without it the
  run exits 0 and writes ONLY RGB frames -- a silent no-op that looks like success.
FULL WORKING INVOCATION (from the worktree):
  export PATH=<venv>/bin:$PATH
  export PYTHONPATH=/home/sriram/research/visionsim/.claude/worktrees/thermal-atlas
  CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libsqlite3.so.0 \
  visionsim blender.render-animation <blend> <outdir> --config.include-thermal \
    --config.thermal.assignments assets/thermal/<scene>.thermal.json \
    --config.thermal.device cuda --config.thermal.render-domain TEXEL \
    --config.thermal.sim-time-s 500 --config.thermal.timestep-s 1.0 \
    --frame-start 1 --frame-end 600 --frame-step 12

## ROOT-CAUSED: intermittent 0-K dropouts = Cycles PERSISTENT DATA (fce0ab2)
SYMPTOM: in a 50-frame sequence, whole objects rendered 0 K in a handful of frames.
  Frame 361 of classroom: 31.5% of the frame zero in the sequence, 0.00% when rendered
  ALONE with identical settings. Which frames broke changed between two runs of the
  SAME command (t8: frames 241,289,301,361,469,517; t9: only 301,517 -- and 517 got
  worse). That non-reproducibility is what disguised it as a data bug.
RULED OUT along the way: atlas data (blackBoard's tile held valid 295-304 K),
  shader/AOV wiring (identical to objects that render fine), image packing (already
  packed), cold vs warm cache, frame-start, frame-step, short sequences.
CAUSE: blender.py:793 sets scene.render.use_persistent_data = True for speed. The
  thermal passes swap materials + AOV wiring per frame; persistent data keeps stale
  device-side state.
FIX: disable it in exposed_prepare_thermal (thermal runs only; RGB-only keeps it).
VERIFIED: classroom mean 0-K 1.879% -> 0.030%, frames>1% zeros: 6 -> 0. Deterministic.
FULL SUITE: 385 passed, 0 failed.

## TWO REMAINING 0-K CAUSES IN KITCHEN1 (deterministic, NOT fixed -- next up)
kitchen1 mean 0-K stayed 2.61% after the persistent-data fix (unchanged run to run),
  so its zeros are a different, deterministic class. Ray-cast through the zero pixels
  of frame 001 identified two objects:
1. Vert.005: verts=36, material_slots=[] (NO material), no UV layers, but DOES carry a
   sim_temperature attribute -- it was simulated. With no material there is no thermal
   shader/AOV on it, so Cycles emits 0. => Objects with zero material slots need a
   default thermal material assigned before the AOV is wired. Scene-agnostic.
2. Vert.001: verts=128, mat='MADERA ESTANTES', HAS HeatSim_Atlas_UV, but 47 of its 101
   polygons sample 0 in the atlas -- its tile is only half-written. Atlas rasterization
   coverage gap (suspect UV islands vs tile bounds / dilation), distinct from #1.
Vert.003 (the MUROS wall) is FINE: 0/40 polys zero -- ruled out.

## BOTH KITCHEN1 0-K CAUSES RESOLVED -- and my earlier attribution was WRONG (137593d)
It turned out to be ONE cause, not two. setup_temperature_aov() iterates
  obj.material_slots, so a mesh with NO material slot never got the Attribute->OutputAOV
  chain; Cycles drew it with its implicit default surface, which carries no temperature
  AOV, so the pass read 0 K regardless of the solve.
  kitchen1's culprits: 'Vert' (24 verts) and 'Vert.005' (36 verts) -- the big walls.
  classroom has 13 material-less objects + 16 with an empty slot => endemic, not a quirk.
FIX: _ensure_temperature_material_slots() assigns a shared HeatSim_Default_Surface
  (a fresh node material IS Blender's default look: Principled 0.8 grey / rough 0.5, so
  the RGB pass is unchanged) and also fills empty slots, since faces bound to an empty
  slot render with the default surface too.
RESULT: kitchen1 mean 0-K over 50 frames 2.612% -> 0.000%; frames>1% zeros 18 -> 0
  (worst was 10.9%). classroom 0.0302% -> 0.0293% (no regression; its residual is true background).
  Proof: out/fix_kitchen1/wall_fix_proof.png

CORRECTION to the earlier ledger entry: I had attributed the big kitchen1 black region to
  Vert.001's half-written atlas tile. That was WRONG. The ray_cast-based pixel->object
  attribution I used was unreliable; the decisive disproof was setting Vert.001's
  heatsim_default_temperature to 400 K and re-rendering -- only 0.05% of the frame changed
  while the 10.98% zero region stayed at exactly 0. The material-less walls were the whole
  cause. Vert.001's mix/gate chain is wired CORRECTLY (Mix A=vertex path, B=atlas, factor=
  alpha*coverage) and its fallback works.

## REMAINING (minor, NOT a 0-K bug): atlas tiles are not fully rasterized
Vert.001's tile has 24.5% unwritten texels (alpha==0 exactly where T==0). Those areas
  correctly fall back to the per-object default 295 K rather than a solved value, so they
  render a flat 295 instead of the simulated gradient -- a fidelity gap, not a black hole.
  Scene-wide share of pixels sitting at exactly 295.0 (i.e. on the fallback):
  kitchen1 1.19%, classroom 0.41%. Worth chasing in the rasterizer/dilation later.

TEST GATE: 386 pass, 0 fail (test_cli.py::test_completions needs the venv bin on PATH; it
  is the only "failure" and is purely environmental).
