"""Create false-color and RGB-overlay fisheye disparity videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imageio.v3 as iio
import imageio_ffmpeg
import matplotlib
import numpy as np
from PIL import Image

from visionsim.dataset import Dataset


def start_encoder(path: Path, width: int, height: int) -> subprocess.Popen:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        "24",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-frames:v",
        "720",
        "-movflags",
        "+faststart",
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/sriram/blender_renders/sriram_room/fisheye_stereo_final"),
    )
    parser.add_argument(
        "--rgb-root",
        type=Path,
        default=Path("/data/sriram/blender_renders/sriram_room/room_panorama_160deg/frames"),
    )
    parser.add_argument("--baseline-m", type=float, default=0.065)
    parser.add_argument("--overlay-alpha", type=float, default=0.40)
    parser.add_argument("--disparity-min-px", type=float, default=4.85)
    parser.add_argument("--disparity-max-px", type=float, default=25.53)
    args = parser.parse_args()

    depths = Dataset.from_path(args.root / "central_geometry" / "depths")
    expected_rgb = [f"{frame:04d}.png" for frame in range(1, 721)]
    if len(depths) != 720:
        raise ValueError(f"Expected 720 depth frames, found {len(depths)}")
    if [path.name for path in sorted(args.rgb_root.glob("*.png"))] != expected_rgb:
        raise ValueError("Central RGB sequence is incomplete")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("overlay-alpha must be between zero and one")

    first_depth, first_meta = depths[0]
    height, width = first_depth.shape[:2]
    focal_px = float(first_meta["fl_x"])
    columns, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    film_x = columns - width / 2.0
    film_y = -(rows - height / 2.0)
    film_radius = np.hypot(film_x, film_y)
    theta = 2.0 * np.arcsin(np.clip(film_radius / (2.0 * focal_px), 0.0, 1.0))
    phi = np.arctan2(film_y, film_x)
    ray_x = np.sin(theta) * np.cos(phi)
    perpendicular_to_baseline = np.sqrt(np.maximum(0.0, 1.0 - ray_x**2))

    color_lut = (matplotlib.colormaps["turbo"](np.linspace(0.0, 1.0, 2048))[:, :3] * 255).astype(np.uint8)
    disparity_path = args.root / "sriram_room_fisheye_disparity_turbo.mp4"
    overlay_path = args.root / "sriram_room_fisheye_rgb_disparity_overlay.mp4"
    encoders = [start_encoder(disparity_path, width, height), start_encoder(overlay_path, width, height)]
    assert encoders[0].stdin is not None and encoders[1].stdin is not None

    sampled_percentiles = []
    try:
        for index in range(720):
            depth = np.asarray(depths[index][0]).squeeze().astype(float)
            numerator = args.baseline_m * depth * perpendicular_to_baseline
            denominator = depth**2 - (args.baseline_m / 2.0) ** 2
            angular_disparity = np.arctan2(numerator, denominator)
            disparity_px = focal_px * angular_disparity
            normalized = np.clip(
                (disparity_px - args.disparity_min_px) / (args.disparity_max_px - args.disparity_min_px),
                0.0,
                1.0,
            )
            lut_indices = np.rint(normalized * (len(color_lut) - 1)).astype(np.int32)
            heatmap = color_lut[lut_indices]

            with Image.open(args.rgb_root / f"{index + 1:04d}.png") as source:
                rgb = np.asarray(source.convert("RGB").resize((width, height), Image.Resampling.LANCZOS))
            overlay = np.clip(
                rgb.astype(float) * (1.0 - args.overlay_alpha) + heatmap.astype(float) * args.overlay_alpha,
                0.0,
                255.0,
            ).astype(np.uint8)

            encoders[0].stdin.write(heatmap.tobytes())
            encoders[1].stdin.write(overlay.tobytes())
            if index == 180:
                iio.imwrite(disparity_path.with_suffix(".png"), heatmap)
                iio.imwrite(overlay_path.with_suffix(".png"), overlay)
            if index % 30 == 0:
                sampled_percentiles.append(np.percentile(disparity_px, [1, 50, 99]).tolist())
            if index == 0 or (index + 1) % 24 == 0:
                print(f"Packaged {index + 1}/720 frames", flush=True)
    finally:
        for encoder in encoders:
            assert encoder.stdin is not None
            encoder.stdin.close()

    for encoder in encoders:
        return_code = encoder.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")

    videos = {}
    for name, path in (("false_color", disparity_path), ("rgb_overlay", overlay_path)):
        frame_count, duration_s = imageio_ffmpeg.count_frames_and_secs(str(path))
        if frame_count != 720 or abs(duration_s - 30.0) > 1e-6:
            raise ValueError(f"Bad {name} video timing: frames={frame_count}, duration={duration_s}")
        videos[name] = {
            "path": str(path),
            "frames": frame_count,
            "duration_s": duration_s,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    sampled = np.asarray(sampled_percentiles)
    report = {
        "complete": True,
        "representation": "fisheye binocular angular disparity expressed in equivalent center pixels",
        "fisheye_model": "equisolid",
        "baseline_m": args.baseline_m,
        "radial_focal_scale_px": focal_px,
        "color_map": "turbo: purple/blue=small disparity, yellow/red=large disparity",
        "fixed_color_limits_equivalent_px": [args.disparity_min_px, args.disparity_max_px],
        "overlay_alpha": args.overlay_alpha,
        "sampled_frame_disparity_px_1_50_99_global_min_median_max": {
            "minimum": sampled.min(axis=0).tolist(),
            "median": np.median(sampled, axis=0).tolist(),
            "maximum": sampled.max(axis=0).tolist(),
        },
        "resolution": [width, height],
        "fps": 24.0,
        "videos": videos,
        "checks": {
            "depth_sequence_complete": len(depths) == 720,
            "rgb_sequence_complete": True,
            "fixed_color_scale_across_time": True,
            "both_video_timings_exact": True,
        },
    }
    with open(args.root / "disparity_video_validation_summary.json", "w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
