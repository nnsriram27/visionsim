"""Render and package the full sriram-room fisheye stereo sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imageio.v3 as iio
import imageio_ffmpeg
import numpy as np

from visionsim.dataset import Dataset
from visionsim.simulate.blender import BlenderClients
from visionsim.simulate.config import RenderConfig
from visionsim.simulate.job import render_job

CAMERA_NAME = "Room panorama - 160 degree fisheye"


def render_eye(args: argparse.Namespace) -> None:
    offset = -args.baseline_m / 2 if args.mode == "left" else args.baseline_m / 2
    config = RenderConfig(
        executable=args.blender,
        camera_name=CAMERA_NAME,
        width=args.width,
        height=args.height,
        include_frames=True,
        include_thermal=False,
        previews=False,
        max_samples=args.samples,
        adaptive_threshold=args.adaptive_threshold,
        use_denoising=True,
        use_motion_blur=False,
        device_type="optix",
        camera_offset=(offset, 0.0, 0.0),
        jobs=1,
        timeout=-1,
        log_dir=args.output / "logs" / args.mode,
    )
    with BlenderClients.spawn(
        jobs=1,
        log=config.log_dir,
        timeout=config.timeout,
        executable=config.executable,
        autoexec=config.autoexec,
    ) as clients:
        render_job(
            clients,
            args.blend.resolve(),
            (args.output / args.mode).resolve(),
            config,
            frame_start=1,
            frame_end=720,
            frame_step=1,
            update_fn=None,
        )


def sequence_pattern(dataset: Dataset) -> str:
    first_path = Path(dataset[0][1]["file_path"])
    if first_path.name != "001.png":
        raise ValueError(f"Unexpected first frame path: {first_path}")
    return str(first_path.with_name("%03d.png"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_and_package(args: argparse.Namespace) -> None:
    left = Dataset.from_path(args.output / "left" / "frames")
    right = Dataset.from_path(args.output / "right" / "frames")
    if len(left) != 720 or len(right) != 720:
        raise ValueError(f"Expected 720 frames per eye; found left={len(left)}, right={len(right)}")

    baselines = []
    rotation_errors = []
    rotation_matrix_differences = []
    mean_absolute_differences = []
    sample_indices = sorted(set(np.linspace(0, 719, 25, dtype=int)))
    for index in range(720):
        _, left_meta = left[index]
        _, right_meta = right[index]
        left_pose = np.asarray(left_meta["transform_matrix"], dtype=float)
        right_pose = np.asarray(right_meta["transform_matrix"], dtype=float)
        baselines.append(float(np.linalg.norm(right_pose[:3, 3] - left_pose[:3, 3])))
        left_u, _, left_vt = np.linalg.svd(left_pose[:3, :3])
        right_u, _, right_vt = np.linalg.svd(right_pose[:3, :3])
        left_rotation = left_u @ left_vt
        right_rotation = right_u @ right_vt
        relative_rotation = left_rotation.T @ right_rotation
        cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
        rotation_errors.append(float(np.degrees(np.arccos(cosine))))
        rotation_matrix_differences.append(
            float(np.max(np.abs(left_pose[:3, :3] - right_pose[:3, :3])))
        )
        if index in sample_indices:
            left_image, _ = left[index]
            right_image, _ = right[index]
            mean_absolute_differences.append(float(np.mean(np.abs(left_image.astype(float) - right_image.astype(float)))))

    output_video = args.output / "sriram_room_fisheye_stereo_sbs.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        "24",
        "-start_number",
        "1",
        "-i",
        sequence_pattern(left),
        "-framerate",
        "24",
        "-start_number",
        "1",
        "-i",
        sequence_pattern(right),
        "-filter_complex",
        "[0:v][1:v]hstack=inputs=2[v]",
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "24",
        "-frames:v",
        "720",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    subprocess.run(command, check=True)
    frame_count, duration_s = imageio_ffmpeg.count_frames_and_secs(str(output_video))
    if frame_count != 720 or abs(duration_s - 30.0) > 1e-6:
        raise ValueError(f"Bad packaged video timing: frames={frame_count}, duration={duration_s}")

    left_poster, _ = left[180]
    right_poster, _ = right[180]
    divider = np.full((args.height, 4, 3), 235, dtype=np.uint8)
    poster = np.concatenate((left_poster[..., :3], divider, right_poster[..., :3]), axis=1)
    iio.imwrite(args.output / "sriram_room_fisheye_stereo_sbs.png", poster)

    report = {
        "complete": True,
        "source_blend": str(args.blend.resolve()),
        "camera": CAMERA_NAME,
        "projection": "Blender fisheye equisolid (scene camera settings preserved)",
        "layout": "left eye | right eye, full-resolution side-by-side",
        "frames_per_eye": 720,
        "fps": 24.0,
        "duration_s": duration_s,
        "eye_resolution": [args.width, args.height],
        "video_resolution": [args.width * 2, args.height],
        "requested_baseline_m": args.baseline_m,
        "measured_baseline_m_mean": float(np.mean(baselines)),
        "measured_baseline_m_max_abs_error": float(np.max(np.abs(np.asarray(baselines) - args.baseline_m))),
        "relative_rotation_error_deg_max": float(np.max(rotation_errors)),
        "relative_rotation_matrix_max_abs_difference": float(np.max(rotation_matrix_differences)),
        "sampled_left_right_mean_absolute_pixel_difference": {
            "min": float(np.min(mean_absolute_differences)),
            "mean": float(np.mean(mean_absolute_differences)),
            "max": float(np.max(mean_absolute_differences)),
        },
        "video": {
            "path": str(output_video),
            "codec": "H.264",
            "frames": frame_count,
            "bytes": output_video.stat().st_size,
            "sha256": sha256(output_video),
        },
        "checks": {
            "both_eyes_complete": len(left) == len(right) == 720,
            "baseline_within_0.1_mm": bool(np.max(np.abs(np.asarray(baselines) - args.baseline_m)) < 1e-4),
            "parallel_optical_axes": bool(np.max(rotation_matrix_differences) < 1e-7),
            "left_right_images_not_identical": bool(np.min(mean_absolute_differences) > 0.0),
            "video_timing_exact": frame_count == 720 and abs(duration_s - 30.0) <= 1e-6,
        },
    }
    with open(args.output / "validation_summary.json", "w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("left", "right", "package"), required=True)
    parser.add_argument("--blend", type=Path, default=Path("/data/sriram/blender_files/sriram_room.blend"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/sriram/blender_renders/sriram_room/fisheye_stereo_final"),
    )
    parser.add_argument("--blender", type=Path, default=Path("/home/sriram/.local/bin/blender"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--adaptive-threshold", type=float, default=0.04)
    parser.add_argument("--baseline-m", type=float, default=0.065)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "package":
        evaluate_and_package(args)
    else:
        render_eye(args)


if __name__ == "__main__":
    main()
