"""Validate and preview the full Livox Avia-inspired PLY sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

PLY_DTYPE = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<u2"), ("time_offset_us", "<f4")]
)


def read_ply(path: Path) -> np.ndarray:
    with open(path, "rb") as stream:
        header_lines = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Incomplete PLY header: {path}")
            decoded = line.decode("ascii").rstrip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        expected = [line for line in header_lines if line.startswith("element vertex ")]
        if len(expected) != 1:
            raise ValueError(f"Missing or ambiguous vertex count: {path}")
        count = int(expected[0].split()[-1])
        points: np.ndarray = np.fromfile(stream, dtype=PLY_DTYPE, count=count)
        if len(points) != count or stream.read(1):
            raise ValueError(f"PLY payload size does not match header: {path}")
        return points


def equal_pose_runs(poses: np.ndarray) -> list[list[int]]:
    equal_transition = np.max(np.abs(np.diff(poses, axis=0)), axis=(1, 2)) < 1e-12
    runs = []
    start = None
    for transition_index, equal in enumerate(equal_transition):
        if equal and start is None:
            start = transition_index + 1
        if start is not None and (not equal or transition_index == len(equal_transition) - 1):
            end = transition_index + 1 if not equal else transition_index + 2
            if end - start + 1 >= 5:
                runs.append([start, end])
            start = None
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    with open(args.root / "manifest.json") as stream:
        manifest = json.load(stream)
    with open(args.root / "poses.json") as stream:
        pose_records = json.load(stream)

    frames = [record["frame"] for record in pose_records]
    poses = np.asarray([record["transform_matrix"] for record in pose_records], dtype=float)
    sensor_files = sorted((args.root / "ply_sensor").glob("*.ply"))
    world_files = sorted((args.root / "ply_world").glob("*.ply"))
    expected_frames = list(range(manifest["frame_range_inclusive"][0], manifest["frame_range_inclusive"][1] + 1))
    if frames != expected_frames or len(sensor_files) != len(frames) or len(world_files) != len(frames):
        raise ValueError("Frame, pose, sensor-PLY, or world-PLY sequence is incomplete")

    total_points = 0
    min_range = np.inf
    max_range = -np.inf
    median_ranges = []
    maximum_transform_error = 0.0
    timestamp_checks = True
    finite_checks = True
    count_checks = True
    preview_frames = {1, 91, 181, 271, 361, 451, 541, 631}
    preview_dir = args.root / "previews"
    preview_dir.mkdir(exist_ok=True)

    for frame, sensor_path, world_path, pose in zip(frames, sensor_files, world_files, poses):
        sensor = read_ply(sensor_path)
        world = read_ply(world_path)
        count_checks &= len(sensor) == len(world) == manifest["rays_per_frame"]
        sensor_xyz = np.column_stack((sensor["x"], sensor["y"], sensor["z"])).astype(float)
        world_xyz = np.column_stack((world["x"], world["y"], world["z"])).astype(float)
        ranges = np.linalg.norm(sensor_xyz, axis=1) * manifest["scene_unit_scale_m"]
        finite_checks &= bool(np.all(np.isfinite(sensor_xyz)) and np.all(np.isfinite(world_xyz)))
        timestamp_checks &= bool(
            np.all(np.diff(sensor["time_offset_us"]) >= 0)
            and sensor["time_offset_us"][0] >= 0
            and sensor["time_offset_us"][-1] < manifest["frame_period_us"]
        )
        selected = np.arange(0, len(sensor), max(1, len(sensor) // 100))
        sensor_h = np.column_stack((sensor_xyz[selected], np.ones(len(selected))))
        transformed = (pose @ sensor_h.T).T[:, :3]
        maximum_transform_error = max(maximum_transform_error, float(np.max(np.abs(transformed - world_xyz[selected]))))
        total_points += len(sensor)
        min_range = min(min_range, float(ranges.min()))
        max_range = max(max_range, float(ranges.max()))
        median_ranges.append(float(np.median(ranges)))

        if frame in preview_frames:
            fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
            scatter = axis.scatter(sensor_xyz[:, 0], -sensor_xyz[:, 2], c=ranges, s=1, cmap="turbo")
            axis.set(xlabel="Sensor X (m)", ylabel="Forward (m)", title=f"Livox Avia-inspired scan — frame {frame}")
            axis.set_aspect("equal")
            fig.colorbar(scatter, ax=axis, label="Range (m)")
            fig.savefig(preview_dir / f"{frame:06d}.png", dpi=140)
            plt.close(fig)

    rotations = Rotation.from_matrix(poses[:, :3, :3])
    step_angles_deg = np.degrees((rotations[:-1].inv() * rotations[1:]).magnitude())
    dwell_runs = equal_pose_runs(poses)
    report = {
        "complete": True,
        "frames": len(frames),
        "coordinate_sequences": 2,
        "points_per_frame": manifest["rays_per_frame"],
        "total_points_per_coordinate_sequence": total_points,
        "duration_s": len(frames) / manifest["fps"],
        "global_timestamp_range_us": [pose_records[0]["timestamp_us"], pose_records[-1]["timestamp_us"]],
        "per_point_time_offset_us_max": (manifest["rays_per_frame"] - 1) / manifest["model"]["point_rate_hz"] * 1e6,
        "range_m_min_median_of_medians_max": [min_range, float(np.median(median_ranges)), max_range],
        "pose_step_angle_deg_min_median_max": [
            float(step_angles_deg.min()),
            float(np.median(step_angles_deg)),
            float(step_angles_deg.max()),
        ],
        "stationary_frame_runs_inclusive": dwell_runs,
        "checks": {
            "all_frames_present": True,
            "all_returns_present": bool(count_checks),
            "all_coordinates_finite": bool(finite_checks),
            "per_point_timestamps_monotonic_and_in_frame": bool(timestamp_checks),
            "sensor_to_world_transform_max_error_below_1e-5": maximum_transform_error < 1e-5,
            "eight_one_second_dwells_detected": len(dwell_runs) == 8
            and all(end - start + 1 == 25 for start, end in dwell_runs),
        },
        "sensor_to_world_transform_max_abs_error": maximum_transform_error,
    }
    with open(args.root / "validation_summary.json", "w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
