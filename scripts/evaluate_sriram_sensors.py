"""Post-process and evaluate the sriram_room non-thermal sensor pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

from visionsim.dataset import Dataset


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as stream:
        json.dump(payload, stream, indent=2)


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(xyz)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(stream, np.column_stack((xyz, rgb)), fmt="%.6f %.6f %.6f %d %d %d")


def evaluate_stereo(root: Path, baseline_m: float) -> dict:
    left_frames = Dataset.from_path(root / "stereo" / "left" / "frames")
    right_frames = Dataset.from_path(root / "stereo" / "right" / "frames")
    left_points = Dataset.from_path(root / "stereo" / "left" / "points")
    if not (len(left_frames) == len(right_frames) == len(left_points)):
        raise ValueError("Stereo frame and point-map counts do not agree")

    measured_baselines = []
    rotation_errors_deg = []
    valid_depth_fractions = []
    disparity_percentiles = []
    preview_dir = root / "stereo" / "evaluation" / "disparity_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    for index in range(len(left_frames)):
        left, left_meta = left_frames[index]
        right, right_meta = right_frames[index]
        points, points_meta = left_points[index]
        left_pose = np.asarray(left_meta["transform_matrix"], dtype=float)
        right_pose = np.asarray(right_meta["transform_matrix"], dtype=float)
        measured_baselines.append(float(np.linalg.norm(right_pose[:3, 3] - left_pose[:3, 3])))
        relative_rotation = left_pose[:3, :3].T @ right_pose[:3, :3]
        cos_angle = np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1)
        rotation_errors_deg.append(float(np.degrees(np.arccos(cos_angle))))

        world = points.reshape(-1, 3)
        valid = np.any(world != 0, axis=1) & np.all(np.isfinite(world), axis=1)
        world_h = np.column_stack((world[valid], np.ones(valid.sum())))
        camera = (np.linalg.inv(left_pose) @ world_h.T).T[:, :3]
        axial_depth = -camera[:, 2]
        axial_depth = axial_depth[axial_depth > 0]
        valid_depth_fractions.append(float(len(axial_depth) / points.shape[0] / points.shape[1]))
        disparities = float(points_meta["fl_x"]) * baseline_m / axial_depth
        disparity_percentiles.append(np.percentile(disparities, [1, 50, 99]).tolist())

        depth_image = np.full(points.shape[:2], np.nan, dtype=float)
        flat_valid = np.flatnonzero(valid)
        positive = camera[:, 2] < 0
        depth_image.flat[flat_valid[positive]] = -camera[positive, 2]
        disparity_image = float(points_meta["fl_x"]) * baseline_m / depth_image
        finite = np.isfinite(disparity_image)
        lo, hi = np.percentile(disparity_image[finite], [1, 99])
        normalized = np.nan_to_num(np.clip((disparity_image - lo) / max(hi - lo, 1e-9)))
        iio.imwrite(preview_dir / f"{index:04d}.png", plt.colormaps["turbo"](normalized, bytes=True)[..., :3])

        anaglyph = np.empty_like(left[..., :3])
        anaglyph[..., 0] = left[..., 0]
        anaglyph[..., 1:] = right[..., 1:3]
        iio.imwrite(root / "stereo" / "evaluation" / f"anaglyph_{index:04d}.png", anaglyph)

    report = {
        "frames": len(left_frames),
        "requested_baseline_m": baseline_m,
        "measured_baseline_m": {
            "mean": float(np.mean(measured_baselines)),
            "max_abs_error": float(np.max(np.abs(np.asarray(measured_baselines) - baseline_m))),
        },
        "relative_rotation_error_deg_max": float(np.max(rotation_errors_deg)),
        "valid_depth_fraction_mean": float(np.mean(valid_depth_fractions)),
        "disparity_px_percentiles_1_50_99_per_frame": disparity_percentiles,
        "checks": {
            "baseline_within_0.1_mm": bool(np.max(np.abs(np.asarray(measured_baselines) - baseline_m)) < 1e-4),
            "parallel_optical_axes": bool(np.max(rotation_errors_deg) < 1e-4),
        },
    }
    _json(root / "stereo" / "evaluation" / "metrics.json", report)
    return report


def emulate_lidar(root: Path, channels: int, columns: int, range_noise_m: float, dropout: float) -> dict:
    points_ds = Dataset.from_path(root / "stereo" / "left" / "points")
    frames_ds = Dataset.from_path(root / "stereo" / "left" / "frames")
    rng = np.random.default_rng(20260913)
    frame_reports = []

    for index, ((points, meta), (colors, _)) in enumerate(zip(points_ds, frames_ds)):
        pose = np.asarray(meta["transform_matrix"], dtype=float)
        world = points.reshape(-1, 3)
        rgb = colors[..., :3].reshape(-1, 3)
        valid = np.any(world != 0, axis=1) & np.all(np.isfinite(world), axis=1)
        world, rgb = world[valid], rgb[valid]
        camera = (np.linalg.inv(pose) @ np.column_stack((world, np.ones(len(world)))).T).T[:, :3]
        ranges = np.linalg.norm(camera, axis=1)
        azimuth = np.arctan2(camera[:, 0], -camera[:, 2])
        elevation = np.arctan2(camera[:, 1], np.hypot(camera[:, 0], camera[:, 2]))
        usable = (camera[:, 2] < 0) & (ranges >= 0.5) & (ranges <= 20.0)
        world, rgb, camera, ranges, azimuth, elevation = (
            array[usable] for array in (world, rgb, camera, ranges, azimuth, elevation)
        )

        az_min, az_max = np.percentile(azimuth, [0.1, 99.9])
        el_min, el_max = np.percentile(elevation, [0.1, 99.9])
        az_bin = np.clip(((azimuth - az_min) / (az_max - az_min) * columns).astype(int), 0, columns - 1)
        el_bin = np.clip(((elevation - el_min) / (el_max - el_min) * channels).astype(int), 0, channels - 1)
        beam = el_bin * columns + az_bin
        order = np.lexsort((ranges, beam))
        chosen = order[np.r_[True, beam[order][1:] != beam[order][:-1]]]
        kept = rng.random(len(chosen)) >= dropout
        chosen = chosen[kept]

        ideal_range = ranges[chosen]
        noisy_range = np.clip(ideal_range + rng.normal(0.0, range_noise_m, len(chosen)), 0.5, 20.0)
        camera_noisy = camera[chosen] * (noisy_range / ideal_range)[:, None]
        world_noisy = (pose @ np.column_stack((camera_noisy, np.ones(len(chosen)))).T).T[:, :3]
        out_dir = root / "lidar" / "scans"
        _write_ply(out_dir / f"{index:04d}.ply", world_noisy, rgb[chosen])
        np.savez_compressed(
            out_dir / f"{index:04d}.npz",
            xyz_world=world_noisy.astype(np.float32),
            range_m=noisy_range.astype(np.float32),
            ring=el_bin[chosen].astype(np.uint8),
            azimuth_rad=azimuth[chosen].astype(np.float32),
            rgb=rgb[chosen].astype(np.uint8),
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        scatter = axes[0].scatter(np.degrees(azimuth[chosen]), el_bin[chosen], c=noisy_range, s=2, cmap="turbo")
        axes[0].set(xlabel="Azimuth (deg)", ylabel="Ring", title="Range image")
        fig.colorbar(scatter, ax=axes[0], label="Range (m)")
        axes[1].scatter(camera_noisy[:, 0], -camera_noisy[:, 2], c=noisy_range, s=2, cmap="turbo")
        axes[1].set(xlabel="Camera X (m)", ylabel="Forward (m)", title="Top-down scan", aspect="equal")
        fig.savefig(root / "lidar" / f"preview_{index:04d}.png", dpi=140)
        plt.close(fig)

        frame_reports.append(
            {
                "frame_index": index,
                "points": len(chosen),
                "occupied_beam_fraction": float(len(chosen) / (channels * columns)),
                "range_m_min_median_max": [float(v) for v in [noisy_range.min(), np.median(noisy_range), noisy_range.max()]],
                "range_noise_rmse_m": float(np.sqrt(np.mean((noisy_range - ideal_range) ** 2))),
            }
        )

    report = {
        "model": "forward-facing structured LiDAR sampled from VisionSim world-space point maps",
        "channels": channels,
        "columns": columns,
        "range_limits_m": [0.5, 20.0],
        "range_noise_sigma_m": range_noise_m,
        "random_dropout_probability": dropout,
        "frames": frame_reports,
    }
    _json(root / "lidar" / "evaluation.json", report)
    return report


def evaluate_events(root: Path, width: int, height: int) -> dict:
    event_path = root / "event_camera" / "events" / "events.txt"
    with open(root / "event_camera" / "events" / "params.json") as stream:
        params = json.load(stream)
    events = np.loadtxt(event_path, delimiter=",", dtype=np.int64, ndmin=2)
    timestamps, x, y, polarity = events.T
    duration_s = max((timestamps.max() - timestamps.min()) / 1e6, 1e-12)
    per_pixel = np.bincount(y * width + x, minlength=width * height)
    nominal_noise_events = (
        (params["shot_noise_rate_hz"] + params["leak_rate_hz"]) * width * height * duration_s
    )
    report = {
        "event_count": len(events),
        "duration_s": float(duration_s),
        "mean_event_rate_hz": float(len(events) / duration_s),
        "on_fraction": float(np.mean(polarity == 1)),
        "off_fraction": float(np.mean(polarity == -1)),
        "active_pixel_fraction": float(np.mean(per_pixel > 0)),
        "hot_pixel_fraction_over_10x_mean": float(np.mean(per_pixel > 10 * per_pixel.mean())),
        "nominal_noise_event_budget": float(nominal_noise_events),
        "nominal_noise_fraction_upper_bound": float(min(nominal_noise_events / len(events), 1.0)),
        "events_per_active_pixel_percentiles_50_95_99": np.percentile(per_pixel[per_pixel > 0], [50, 95, 99]).tolist(),
        "timestamp_range_us": [int(timestamps.min()), int(timestamps.max())],
        "checks": {
            "timestamps_monotonic": bool(np.all(np.diff(timestamps) >= 0)),
            "coordinates_in_bounds": bool(np.all((x >= 0) & (x < width) & (y >= 0) & (y < height))),
            "polarities_valid": bool(np.all(np.isin(polarity, [-1, 1]))),
            "both_polarities_present": bool(np.any(polarity == -1) and np.any(polarity == 1)),
        },
    }
    _json(root / "event_camera" / "evaluation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline-m", type=float, default=0.065)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--lidar-channels", type=int, default=64)
    parser.add_argument("--lidar-columns", type=int, default=512)
    parser.add_argument("--lidar-range-noise-m", type=float, default=0.015)
    parser.add_argument("--lidar-dropout", type=float, default=0.02)
    args = parser.parse_args()
    summary = {
        "stereo": evaluate_stereo(args.root, args.baseline_m),
        "lidar": emulate_lidar(
            args.root, args.lidar_channels, args.lidar_columns, args.lidar_range_noise_m, args.lidar_dropout
        ),
        "event_camera": evaluate_events(args.root, args.width, args.height),
    }
    _json(args.root / "evaluation_summary.json", summary)


if __name__ == "__main__":
    main()
