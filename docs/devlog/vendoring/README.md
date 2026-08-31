# Vendoring and port scaffolding

Moved here from the `heatsim` code branch so that branch carries code only.
Nothing here is imported or executed by visionsim.

| File | What it is |
|---|---|
| `VENDOR.md` | Provenance for the files vendored from `heat-sim-blender` @ `e5b4afe`, and the rationale for their ruff/mypy exemptions. |
| `compare_bunny_thermal.py` | One-off acceptance harness from the parity port: solves `bunny_textured.blend` in both tools and compares per-vertex temperature rise. Never wired into pytest, and it hardcodes a Blender path that no longer exists. |

## The lint exemptions are still live

`pyproject.toml` on `heatsim` still excludes the vendored modules from ruff and mypy.
`VENDOR.md` is the justification for that list, so the config now carries a comment
pointing here rather than to a file in the code tree.

The rule it records is worth keeping: **the exemption is provenance-based.** A file keeps
it only while it stays close to upstream; once substantially rewritten, it is ours to
maintain and gets linted like everything else. Two files have already crossed that line
(`irradiance_kernel.py`, `irradiance.py`) and are ruff-linted, mypy-excluded only because
they import `bpy` unconditionally.

Note that the drift figures in `VENDOR.md` are a snapshot from the port and have not been
updated since; `irradiance.py` in particular has changed substantially since (bake sample
control, evaluated-mesh sampling, the UV-guard fix). Treat them as historical.
