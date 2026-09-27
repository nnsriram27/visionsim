# Baby-bottle transient thermal rendering plan

Backend/stepping update: the revised implementation plan now lives in [Heat Sim Blender](../../../heat-sim-blender/scripts/babybottle_render/PLAN.md). Use that plan for new work: Heat Sim Blender only, geometry/Laplacian updates at solver subframes, an audited contact interface, and noise-free parameter fitting. The observations and preliminary calibration below remain useful.

Inspected 2026-09-23. This is a preparation/implementation plan, not a completed thermal simulation. Original blend and recordings were not modified. No branches were switched.

## Recommended first result

Use Heat Sim Blender's animated stepping infrastructure, with explicit contact conductance between separate bodies. Start with `01_Light_Force`, camera `Cam1_274849`, 640×512 at the thermal timestamps (~60 Hz), from approach through the cooling tail (~15 s). Follow with `02_Heavy_Force` using the same bottle material parameters. Ignore wrist cameras. Keep task scripts in this directory; place large results in `/data/sriram/babybottle_thermal/`. Backend changes, if needed, belong in Heat Sim Blender with focused physics checks.

Cam1 is 0.0605 m from `visible_24017931` in the reconstruction; the other thermal cameras are 0.548–0.841 m away. The blend already has Cam1 scenes for both trials. Camera alignment still needs image-space verification: all four thermal cameras currently use the same stored K (fx=fy=1500, cx=320, cy=256). Their metadata says supplied calibration; that is not itself a measured reprojection-error check. Use the camera's 640×512 dimensions, not the saved wrist render dimensions. Effective vertical field of view depends on the output aspect ratio and sensor fit.

## What the inspection established

- The blend contains both full 1–901-frame trial scenes at 60 Hz; frame 1 is t=0. The scene selected on file load is a shorter wrist scene, which must not determine the thermal render range.
- Each trial has four uint16 thermal arrays, typically 900×514×640×1 (trial 1 Cam0 has 901 frames). The two leading rows are telemetry, per `force_proj/src/thermal_force/signals/image.py`; image data are `frame[2:]`. Use actual timestamps, not array indices alone.
- RGB is BayerRG16, 4000×3000, with 180/181 frames over ~15 s. Recorded RGB timestamps are irregular; the nominal 15 Hz setting is not a synchronization rule. Inspection sheets show raw downsampled Bayer intensities, not color-debayered RGB.
- Cam1 shows a localized residual imprint after light contact, and a larger/stronger imprint after heavy contact. Heavy contact also lasts longer. Force cannot be inferred from thermal amplitude without separating dwell time, contact area and conductance.
- Both hands have 778 vertices and absolute shape-key animation. Motion evaluates correctly after updating the view layer. The inferred contact geometry is explicitly labeled as neither measured force nor pressure in the blend.
- Hand visibility is animated. A hidden hand still exists geometrically: it must cease thermal coupling when absent, even if a solver collects hidden meshes. Preserve bottle temperature through release and the remaining cooling interval.
- The bottle body, collar, dome and teat are separate meshes with estimated Solidify wall thicknesses. The body's evaluated vertex count is 27,264 versus 13,632 in the base mesh. Use an explicit simulation-to-render mapping; do not blindly write evaluated temperatures onto base vertices.
- The optical table alone has 200,413 vertices. It should be a background/boundary approximation for the first experiment, not part of one giant coupled point cloud. The mat and board are zero-thickness planes and require an assigned thermal thickness if simulated.

## Solver findings

The current VisionSim `heatsim` checkout has the static thermal pipeline. The requested branch is actually named `Srirams-MBP-6`; reading it with `git show` found `solve_scene_animated` in `visionsim/simulate/heatsim/adapter.py`. It preserves participant temperatures and rebuilds pose-dependent matrices. Its documented inter-object coupling uses the shared POINTS neighborhood graph; animated irradiance and interior-point continuity have limitations.

Heat Sim Blender has an `ANIMATE` scene mode and a per-frame loop in `addon/lib/fem_adapter.py` around lines 3330–3550. It re-evaluates geometry, calls `HeatSimFEM.simulate_for_pose`, carries temperature forward, and supports irradiance updates. It is the recommended implementation base for this experiment, not because VisionSim has no animation support.

Important gap: a Laplacian over all combined points can bridge nearby disconnected surfaces. Conversely, disconnected mesh components do not automatically conduct across a contact. Neither path by itself establishes a physically calibrated skin–polymer interface. `contact_heat.py` deposits mechanical dissipative energy from contact-event files; it is not the temperature-difference-driven interface required here. Do not interpret mechanical dissipation as the main source of the observed imprint.

Use per-body conduction plus a separately assembled contact operator. A stationary bottle can reuse its internal operator. Rigid hand translation/rotation alone does not change its intrinsic operator; deformation can. Contact pairs/areas must update with pose. Carry temperature in material coordinates/stable sample identities, not world-space nearest neighbors.

## Temperature calibration and starting values

Blackbody recordings exist alongside the trials: `x_blackbody_20`, `_30`, `_40`, `_50`. A manually inspected interior disk ROI on Cam1 (x=280:320, y=80:110 after telemetry removal), sampled at six times, gave median counts 21131, 22511, 24231, 26041 respectively. This assumes the directory suffixes are Celsius setpoints; verify acquisition records before calling this an absolute calibration.

Piecewise interpolation gives precontact whole-image medians of 24.88°C (light) and 24.44°C (heavy), and contact-frame warm percentiles around 34.0–34.5°C. These are broad image statistics, not segmented bottle/skin thermometry. They ignore emissivity, reflected radiance and cross-recording camera drift. The blackbody background changes across setpoints, so drift/stray heating needs explicit checking.

For a first diagnostic simulation, use the following declared assumptions; fit/replace them before a calibrated result:

| Quantity | Initial diagnostic setting | How to determine final value |
|---|---|---|
| Hand temperature | 307.4 K (~34.25°C), reservoir approximation | Eroded visible skin masks; camera calibration and emissivity correction; allow trial variation |
| Bottle/collar/cap T0 | ~298 K light, ~297.6 K heavy | Per-part precontact masks; distinguish object temperature from background median |
| Background/environment | ~298 K provisional | Stable high-emissivity reference; air temperature is not directly measured by image median |
| Polymer conductivity k | 0.2 W/(m K), numerical starting assumption | Identify actual polymer; bracket material uncertainty |
| Polymer density rho | 900 kg/m³, numerical starting assumption | Material identification/manufacturer data |
| Polymer heat capacity cp | 1900 J/(kg K), numerical starting assumption | Material identification/manufacturer data |
| Derived alpha | k/(rho cp) = 1.17e-7 m²/s for the above assumptions | Derive, do not independently fit all four quantities |
| Wall thickness | Read estimated Solidify thickness per part | Verify physical thickness; sweep uncertainty |
| Polymer emissivity | 0.95 initial opaque-surface approximation | Validate in sensor band; visible transparency is not an LWIR material model |
| Hand emissivity | 0.98 initial assumption | Verify camera-band model and calibration |
| Ambient convection h | 5 W/(m² K) initial numerical setting | Fit cooling curve jointly with radiation/inner-wall boundary; report uncertainty |
| Contact conductance hc | Explore 50, 200, 1000 W/(m² K) | Broad numerical sweep, not measured values; fit imprint evolution after geometry/dwell are fixed |

The unverified polymer numbers above are experiment initialization choices, not an assertion about this bottle's composition. Contents/fill state and hidden wall construction are unknown. Begin with an explicitly labeled empty/air-filled approximation and test its cooling response. A filled bottle needs a different inner-wall boundary/thermal mass. The silicone teat needs its own properties if it affects the observable result; otherwise leave hidden parts out of the first solve. Do not automatically assign every part the same polymer.

Store each part's role, T0_K, k_W_mK, rho_kg_m3, cp_J_kgK, thickness_m, emissivity, and inner/outer boundary conditions in a JSON sidecar. Store contact parameters separately. Audit conversions at the backend boundary: Heat Sim Blender uses millimetres internally (alpha in mm²/s is 1e6 times m²/s; density in kg/mm³ is 1e-9 times kg/m³). Prefer one SI-facing configuration and a single conversion layer.

## Ordered implementation

1. **Calibrate observations and time.** Confirm blackbody setpoints, camera mode and telemetry format. Fit a monotone response per camera, inspect temporal drift/FFC events, validate held-out blackbody samples, and retain raw counts plus uncertainty. Segment bottle parts/hand; estimate precontact temperatures and post-release curves. Establish each trial's timestamp origin from the reconstruction and check approach/release against recorded motion. Do not add stored timestamp offsets again without verifying their semantics.
2. **Verify Cam1 projection.** Compare bottle silhouettes, cap/collar boundaries and board landmarks in recorded versus projected frames. Score errors before adjusting pose/intrinsics; account for lens distortion and pixel aspect. Calibrate geometric alignment without changing thermal parameters to compensate. Use the other views to check ambiguities.
3. **Build the simulation domain.** Use a body-fitted thin-shell mesh with a justified effective heat capacity, or a layered/volume discretization if through-wall gradients matter. Use stable surface samples and a persistent mapping to the render mesh. Resolve the smallest visible imprint with several samples and demonstrate spatial convergence. Check all-frame topology/order, inverted triangles, gaps, intersections, and hand visibility. The 778-vertex hand may be adequate as a temperature reservoir/contact surface but is coarse for resolving skin temperature gradients.
4. **Implement contact heat exchange.** At each pose, find valid hand–bottle surface pairs using distance, facing normals and an area estimate. Use a geometric tolerance tied to reconstruction uncertainty, then test sensitivity. Apply q = hc (T_hand - T_bottle) over contact area; remove coupling on release. Separate rigidly connected bottle parts from moving skin contact. Do not couple close internal/external shell surfaces through arbitrary nearest-neighbor edges. Start with a pinned hand reservoir; later evolve both bodies with equal/opposite contact energy if needed. For a reservoir, record supplied energy explicitly. Mechanical frictional heat remains a separate optional term.
5. **Integrate transient heat.** Use implicit stepping aligned to thermal capture times; start with four substeps per 60-Hz interval and compare against eight. Include ambient convection/radiation and appropriate inner-wall conditions. Retain physical heat capacities when the hand mesh deforms. Simulate approach, hold, release, and the full cooling tail without resetting the bottle each frame.
6. **Fit a small parameter set.** First geometry/time and radiometric response; then baseline/skin temperatures; then effective contact conductance and cooling parameters. Share material properties across trials, allow measured dwell/area and trial skin temperature to differ. Fit selected light-trial frames and validate held-out times/heavy trial/other cameras. If separate conductance is needed for heavy force, identify it as a fitted effective parameter, not a measured pressure law. Report degeneracies among hc, contact area, wall thickness, k and sensor gain.
7. **Render and validate.** Export temperature arrays/EXR plus a separate camera-radiance/count prediction. Include emissivity and reflected environment; do not reuse RGB glass transmission as thermal transmission. Add measured blur/noise only after the heat model matches. Render fixed-scale comparison videos at 640×512, with synchronized real/simulated views and residuals. Score imprint location/area, temperature/count rise, spatial profile and cooling decay. Keep colorized MP4 separate from quantitative data.

## Checks before full videos

- Uniform isolated temperature remains uniform; no contact means no inter-body conduction.
- A known two-body contact test conserves energy (or balances reservoir supply), approaches equilibrium, and stops exchanging contact heat on separation.
- A moving warm source leaves a persistent cooling trail on a fixed body without temperature following it through space.
- Results converge with smaller timestep/finer sampling/contact tolerance; watch shell mass/thickness and metre–millimetre conversions.
- No one-frame temperature reset, invisible-hand heating, accidental table coupling, or automatic per-frame color rescaling.

First deliverable: a short Cam1 light-trial diagnostic covering touch and release, with heat budget and error curves. Then the full 15-s light/heavy videos; reuse the same simulated surface fields for the other fixed cameras after their individual radiometric/geometric calibration.

## Inspection artifacts and reproduction

Scripts here: `inspect_recordings.py`, `inspect_scene.py`, `probe_calibration.py`. Outputs: `/data/sriram/babybottle_thermal/inspection/` contains contact sheets, `recordings.json`, `scene_light.json`, `scene_heavy.json`, and `calibration_probe.json`. The earlier `scene.json` describes the initially selected wrist scene only; use the light/heavy files for the proposed work.

Run from the VisionSim root:

```bash
.venv/bin/python scripts/babybottle_render/inspect_recordings.py --output /data/sriram/babybottle_thermal/inspection
.venv/bin/python scripts/babybottle_render/probe_calibration.py
blender --background --disable-autoexec /data/sriram/blender_files/babybottle_digital_twin.blend --python scripts/babybottle_render/inspect_scene.py -- /data/sriram/babybottle_thermal/inspection/scene_light.json 01_Light_Force
blender --background --disable-autoexec /data/sriram/blender_files/babybottle_digital_twin.blend --python scripts/babybottle_render/inspect_scene.py -- /data/sriram/babybottle_thermal/inspection/scene_heavy.json 02_Heavy_Force
```

External technical references: [FLIR on non-radiometric Boson output and camera-temperature dependence](https://flir.custhelp.com/app/answers/detail/a_id/4148/~/flir-oem---boson-radiometry-%28absolute-temperature-measurement%29); [COMSOL contact conductance formulation](https://doc.comsol.com/6.3/doc/com.comsol.help.heat/heat_ug_ht_features.09.093.html). These support the calibration caution and interface formulation, not identification of the actual camera variant or bottle material.
