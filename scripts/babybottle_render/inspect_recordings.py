"""Read sampled frames from original NPZs without unpacking full recordings."""
import argparse
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def header(stream):
    version = np.lib.format.read_magic(stream)
    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
              else np.lib.format.read_array_header_2_0)
    return reader(stream)


def inspect(path, output):
    result = {"path": str(path), "arrays": {}}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                shape, order, dtype = header(stream)
            result["arrays"][name] = {"shape": shape, "dtype": str(dtype)}
            if "tstamps" in name:
                values = np.load(archive.open(name), allow_pickle=False)
                delta = np.diff(values)
                result["arrays"][name].update(start=float(values[0]), end=float(values[-1]),
                    duration_s=float(values[-1]-values[0]), median_dt_s=float(np.median(delta)),
                    min_dt_s=float(delta.min()), max_dt_s=float(delta.max()))
            elif not shape:
                result["arrays"][name]["value"] = np.load(archive.open(name), allow_pickle=False).item()
        name = next(n for n in archive.namelist() if "frames" in n)
        frames = []
        with archive.open(name) as stream:
            shape, order, dtype = header(stream)
            if order:
                raise ValueError("Fortran-ordered frame array is unsupported")
            offset = stream.tell()
            frame_bytes = int(np.prod(shape[1:])) * dtype.itemsize
            indices = np.linspace(0, shape[0]-1, 7).astype(int)
            for index in indices:
                stream.seek(offset + int(index)*frame_bytes)
                frame = np.frombuffer(stream.read(frame_bytes), dtype=dtype).reshape(shape[1:]).squeeze()
                frames.append(frame)
        thermal = "thr" in name
        # Acquisition decoder in force_proj strips two LEADING telemetry rows.
        images = [f[2:] if thermal else f[::5, ::5] for f in frames]
        lo, hi = np.percentile(np.concatenate([f.ravel() for f in images]), [1, 99])
        fig, axes = plt.subplots(1, len(images), figsize=(21, 3.8))
        for ax, index, frame in zip(axes, indices, images):
            ax.imshow(frame, cmap="inferno" if thermal else "gray", vmin=lo, vmax=hi)
            ax.set_title(f"frame {index}")
            ax.axis("off")
        fig.suptitle(f"{path.stem} | raw counts, fixed scale {lo:.0f}–{hi:.0f}")
        fig.tight_layout()
        fig.savefig(output / (path.stem + ".jpg"), dpi=110)
        plt.close(fig)
        result["sample_indices"] = indices.tolist()
        result["sample_percentiles_raw"] = [np.percentile(f, [1, 10, 50, 90, 99]).tolist() for f in images]
        if thermal:
            result["telemetry_rows_percentiles_raw"] = [np.percentile(f[:2], [0, 50, 100]).tolist() for f in frames]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/data/sriram/thermal_tactile_force/force_interaction_data/source"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", nargs="+", default=["a_babybottle_1", "a_babybottle_2"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for trial in args.trials:
        for path in sorted((args.source / trial).glob("*.npz")):
            print(path.name, flush=True)
            results.append(inspect(path, args.output))
    (args.output / "recordings.json").write_text(json.dumps(results, indent=2) + "\n")
