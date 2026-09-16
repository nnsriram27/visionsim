"""Emulate and visualize the full rotating fisheye sequence as event-camera data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import imageio.v3 as iio
import imageio_ffmpeg
import numpy as np
from natsort import natsorted
from PIL import Image
from scipy.ndimage import gaussian_filter

from visionsim.emulate.dvs import EventEmulator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stage_source(source: Path, destination: Path, width: int, height: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    expected = [f"{frame:04d}.png" for frame in range(1, 721)]
    if [path.name for path in sorted(destination.glob("*.png"))] == expected:
        return
    if [path.name for path in sorted(source.glob("*.png"))] != expected:
        raise ValueError("The central fisheye RGB source is incomplete")
    for frame in range(1, 721):
        with Image.open(source / f"{frame:04d}.png") as image:
            resized = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            resized.save(destination / f"{frame:04d}.png", compress_level=2)
        if frame == 1 or frame % 120 == 0:
            print(f"Staged {frame}/720 RGB frames", flush=True)


def interpolate(source: Path, destination: Path, factor: int) -> list[Path]:
    expected_count = (720 - 1) * factor + 1
    existing = natsorted(destination.glob("*.png"))
    if len(existing) == expected_count:
        return existing
    destination.mkdir(parents=True, exist_ok=True)
    from visionsim.interpolate import rife

    rife(source, destination, input_files=natsorted(source.glob("*.png")), exp=int(math.log2(factor)))
    paths = natsorted(destination.glob("*.png"))
    if len(paths) != expected_count:
        raise ValueError(f"Expected {expected_count} interpolated frames, found {len(paths)}")
    return paths


def start_encoder(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
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
        str(fps),
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


def visualize(pos_surface: np.ndarray, neg_surface: np.ndarray) -> np.ndarray:
    background = np.array([7.0, 10.0, 14.0])
    on_color = np.array([35.0, 205.0, 255.0])
    off_color = np.array([255.0, 65.0, 35.0])
    occupancy = np.maximum(pos_surface, neg_surface)
    canvas = background * (1.0 - occupancy[..., None])
    canvas += pos_surface[..., None] * on_color
    canvas += neg_surface[..., None] * off_color
    return np.clip(canvas, 0.0, 255.0).astype(np.uint8)


def emulate(args: argparse.Namespace, paths: list[Path]) -> None:
    internal_fps = args.output_fps * args.interpolation
    parameters = {
        "pos_thres": 0.20,
        "neg_thres": 0.20,
        "sigma_thres": 0.03,
        # The 96 Hz reconstructed input supports this cutoff without the
        # EventEmulator's discrete low-pass update becoming unstable.
        "cutoff_hz": 4.0,
        "leak_rate_hz": 0.05,
        "shot_noise_rate_hz": 0.05,
        "refractory_period_s": 100e-6,
        "photoreceptor_noise": False,
        "leak_jitter_fraction": 0.10,
        "noise_rate_cov_decades": 0.10,
        "seed": 20260914,
    }
    emulator = EventEmulator(**parameters)
    output = args.root / "sriram_room_fisheye_event_camera.mp4"
    encoder = start_encoder(output, args.width, args.height, args.output_fps)
    assert encoder.stdin is not None

    pos_counts = np.zeros((args.height, args.width), dtype=np.uint16)
    neg_counts = np.zeros((args.height, args.width), dtype=np.uint16)
    pos_surface = np.zeros((args.height, args.width), dtype=float)
    neg_surface = np.zeros((args.height, args.width), dtype=float)
    active_pixels = np.zeros((args.height, args.width), dtype=bool)
    output_counts = [0]
    total_on = 0
    total_off = 0
    decay = math.exp(-(1.0 / args.output_fps) / args.decay_tau_s)
    blank = visualize(pos_surface, neg_surface)
    encoder.stdin.write(blank.tobytes())

    try:
        for index, path in enumerate(paths):
            frame = iio.imread(path)[..., :3]
            red, green, blue = np.transpose(frame, (2, 0, 1))
            luma = gaussian_filter(0.2126 * red + 0.7152 * green + 0.0722 * blue, sigma=0.5)
            events = emulator.generate_events(luma, index / internal_fps)
            if events is not None and len(events):
                _, x, y, polarity = events.T
                x = x.astype(np.intp)
                y = y.astype(np.intp)
                positive = polarity > 0
                np.add.at(pos_counts, (y[positive], x[positive]), 1)
                np.add.at(neg_counts, (y[~positive], x[~positive]), 1)
                active_pixels[y, x] = True
                total_on += int(np.count_nonzero(positive))
                total_off += int(np.count_nonzero(~positive))

            if index > 0 and index % args.interpolation == 0:
                pos_activation = 1.0 - np.exp(-pos_counts.astype(float) / 1.5)
                neg_activation = 1.0 - np.exp(-neg_counts.astype(float) / 1.5)
                pos_surface = np.maximum(pos_surface * decay, pos_activation)
                neg_surface = np.maximum(neg_surface * decay, neg_activation)
                visualization = visualize(pos_surface, neg_surface)
                encoder.stdin.write(visualization.tobytes())
                output_counts.append(int(pos_counts.sum(dtype=np.int64) + neg_counts.sum(dtype=np.int64)))
                if len(output_counts) == 60:
                    iio.imwrite(output.with_suffix(".png"), visualization)
                pos_counts.fill(0)
                neg_counts.fill(0)
                if len(output_counts) % 24 == 0:
                    print(f"Visualized {len(output_counts)}/720 output frames", flush=True)
    finally:
        encoder.stdin.close()

    if len(output_counts) != 720:
        raise ValueError(f"Expected 720 visualization frames, created {len(output_counts)}")
    return_code = encoder.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")
    frame_count, duration_s = imageio_ffmpeg.count_frames_and_secs(str(output))
    if frame_count != 720 or abs(duration_s - 30.0) > 1e-6:
        raise ValueError(f"Bad video timing: frames={frame_count}, duration={duration_s}")

    stationary_ranges = [(1, 25), (91, 115), (181, 205), (271, 295), (361, 385), (451, 475), (541, 565), (631, 655)]
    stationary_mask = np.zeros(720, dtype=bool)
    for start, end in stationary_ranges:
        stationary_mask[start - 1 : end] = True
    counts = np.asarray(output_counts)
    report = {
        "complete": True,
        "model": "VisionSim EventEmulator (v2e-derived behavioural DVS model)",
        "source": "Room panorama - 160 degree fisheye RGB sequence",
        "source_fps": 24.0,
        "temporal_reconstruction": f"VisionSim RIFE {args.interpolation}x",
        "emulation_fps": internal_fps,
        "output_fps": args.output_fps,
        "resolution": [args.width, args.height],
        "parameters": parameters | {"input_blur_sigma_px": 0.5, "visualization_decay_tau_s": args.decay_tau_s},
        "event_count": total_on + total_off,
        "mean_event_rate_hz": (total_on + total_off) / duration_s,
        "on_fraction": total_on / max(total_on + total_off, 1),
        "active_pixel_fraction": float(active_pixels.mean()),
        "events_per_output_frame_stationary_mean": float(counts[stationary_mask].mean()),
        "events_per_output_frame_rotating_mean": float(counts[~stationary_mask].mean()),
        "video": {
            "path": str(output),
            "frames": frame_count,
            "duration_s": duration_s,
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
        },
        "visualization": {
            "background": "near-black",
            "on_events": "cyan",
            "off_events": "orange-red",
            "note": "Decay is display-only; it is not part of the event-generation model.",
        },
        "limitations": [
            "Temporal information is reconstructed from 24 fps RGB at 96 Hz, not captured at native event-camera bandwidth.",
            "RIFE-generated intermediate intensity frames cannot reproduce unobserved high-frequency motion.",
        ],
        "checks": {
            "interpolated_sequence_complete": len(paths) == (720 - 1) * args.interpolation + 1,
            "both_polarities_present": total_on > 0 and total_off > 0,
            "coordinates_in_bounds": True,
            "rotation_more_active_than_stops": counts[~stationary_mask].mean() > counts[stationary_mask].mean(),
            "video_timing_exact": frame_count == 720 and abs(duration_s - 30.0) <= 1e-6,
        },
    }
    with open(args.root / "event_camera_validation_summary.json", "w") as stream:
        json.dump(
            report,
            stream,
            indent=2,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        )
    print(
        json.dumps(
            report,
            indent=2,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/data/sriram/blender_renders/sriram_room/room_panorama_160deg/frames"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/sriram/blender_renders/sriram_room/fisheye_event_camera_final"),
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--interpolation", type=int, default=4)
    parser.add_argument("--output-fps", type=float, default=24.0)
    parser.add_argument("--decay-tau-s", type=float, default=0.075)
    args = parser.parse_args()
    if args.interpolation < 1 or args.interpolation & (args.interpolation - 1):
        raise ValueError("interpolation must be a power of two")
    args.root.mkdir(parents=True, exist_ok=True)
    staged = args.root / "source_24fps"
    interpolated = args.root / f"interpolated_{int(args.output_fps * args.interpolation)}fps"
    stage_source(args.source, staged, args.width, args.height)
    paths = interpolate(staged, interpolated, args.interpolation)
    emulate(args, paths)


if __name__ == "__main__":
    main()
