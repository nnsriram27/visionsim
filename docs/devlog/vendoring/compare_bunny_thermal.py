"""Compare per-vertex bunny temperature between heat-sim and VisionSim caches.

Acceptance harness for the thermal-parity port (see
docs/superpowers/plans/2026-07-07-visionsim-thermal-parity.md, Task 4).

Both tools solve the same ``bunny_textured.blend`` for 1.0 s. This reads each
tool's per-vertex temperature history from its on-disk cache, computes the
per-vertex temperature rise (dT = last - first), and checks:

  * peak dT and mean dT ratios (VisionSim / heat-sim) within [0.85, 1.15]
  * per-vertex dT correlation >= 0.8 (the checkerboard imprint is present in
    VisionSim and spatially aligned with heat-sim's)

Run with the FEM Blender (it has numpy on the path):
  BL=/net/acadia2a/data/sriram/blender-fem-research/blender
  "$BL" --background --python tests/parity/compare_bunny_thermal.py
"""

import glob

import numpy as np

HS = "/home/sriram/research/heat-sim-blender/blender_files/bunny_textured.heatsim/latest/temperatures.npz"
VS_GLOB = "/home/sriram/research/heat-sim-blender/blender_files/bunny_textured.blend.heatsim/**/temperatures.npz"


def load(path):
    d = np.load(path)
    key = "bunny" if "bunny" in d.files else d.files[0]
    t = d[key]  # (steps, N)
    return t[-1] - t[0]  # per-vertex dT over the run


def main():
    hs = load(HS)
    vs_matches = sorted(glob.glob(VS_GLOB, recursive=True))
    if not vs_matches:
        print("PARITY_FAIL no VisionSim cache found — run the solve first")
        return
    vs_path = vs_matches[-1]
    vs = load(vs_path)
    n = min(hs.shape[0], vs.shape[0])
    hs, vs = hs[:n], vs[:n]

    hs_peak, hs_mean = float(hs.max()), float(hs.mean())
    vs_peak, vs_mean = float(vs.max()), float(vs.mean())
    peak_ratio = vs_peak / hs_peak if hs_peak else float("nan")
    mean_ratio = vs_mean / hs_mean if hs_mean else float("nan")
    corr = float(np.corrcoef(hs, vs)[0, 1]) if n > 2 else float("nan")

    print(f"heat-sim  cache: {HS}")
    print(f"visionsim cache: {vs_path}")
    print(f"heat-sim  dT: peak={hs_peak:.4f} mean={hs_mean:.4f}")
    print(f"visionsim dT: peak={vs_peak:.4f} mean={vs_mean:.4f}")
    print(f"ratio (vs/hs): peak={peak_ratio:.3f} mean={mean_ratio:.3f}")
    print(f"per-vertex dT correlation (checkerboard imprint): {corr:.3f}")

    peak_ok = 0.85 <= peak_ratio <= 1.15
    mean_ok = 0.85 <= mean_ratio <= 1.15
    corr_ok = corr >= 0.8
    verdict = "PASS" if (peak_ok and mean_ok and corr_ok) else "FAIL"
    print(f"PARITY_{verdict} peak_ok={peak_ok} mean_ok={mean_ok} corr_ok={corr_ok}")


if __name__ == "__main__":
    main()
