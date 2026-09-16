"""Render a reproducible non-thermal sensor pilot for sriram_room.blend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_sriram_sensors import emulate_lidar, evaluate_events, evaluate_stereo
from PIL import Image

from visionsim.cli.blender import render_animation
from visionsim.cli.emulate import events
from visionsim.cli.interpolate import dataset as interpolate_dataset
from visionsim.simulate.config import PointsConfig, RenderConfig


def _render_stereo(args: argparse.Namespace) -> None:
    common = {
        "executable": args.blender,
        "camera_name": args.camera,
        "width": args.width,
        "height": args.height,
        "include_frames": True,
        "include_depths": True,
        "include_points": True,
        "include_thermal": False,
        "points": PointsConfig(preview=True, bit_depth=32),
        "max_samples": args.samples,
        "adaptive_threshold": 0.03,
        "use_denoising": True,
        "use_motion_blur": False,
        "device_type": args.device,
        "jobs": 1,
        "timeout": -1,
    }
    for eye, offset in (("left", -args.baseline_m / 2), ("right", args.baseline_m / 2)):
        config = RenderConfig(**common, camera_offset=(offset, 0.0, 0.0), log_dir=args.output / "logs" / eye)
        render_animation(
            args.blend,
            args.output / "stereo" / eye,
            config=config,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            frame_step=args.frame_step,
        )


def _stage_event_source(args: argparse.Namespace) -> Path:
    output = args.output / "event_camera" / "source_24fps"
    output.mkdir(parents=True, exist_ok=True)
    for index in range(args.event_start, args.event_end + 1):
        source = args.event_source / f"{index:04d}.png"
        if not source.exists():
            raise FileNotFoundError(source)
        destination = output / f"{index:04d}.png"
        if destination.exists():
            continue
        with Image.open(source) as image:
            image.convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS).save(destination)
    return output


def _render_events(args: argparse.Namespace) -> None:
    source = _stage_event_source(args)
    interpolated = args.output / "event_camera" / "interpolated_768fps"
    interpolate_dataset(source, interpolated, pattern="*.png", n=32)
    events(
        interpolated,
        args.output / "event_camera" / "events",
        fps=24 * 32,
        pattern="**/*.png",
        pos_thres=0.20,
        neg_thres=0.20,
        sigma_thres=0.03,
        cutoff_hz=25.0,
        leak_rate_hz=0.1,
        shot_noise_rate_hz=1.0,
        refractory_period_s=100e-6,
        leak_jitter_fraction=0.10,
        noise_rate_cov_decades=0.10,
        seed=20260913,
        blur_sigma=0.5,
        preview_step=8,
        force=args.force_event,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=Path, default=Path("/data/sriram/blender_files/sriram_room.blend"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/sriram/blender_renders/sriram_room/nonthermal_sensors_v1"),
    )
    parser.add_argument("--event-source", type=Path, default=Path("/data/sriram/blender_renders/sriram_room/highlights_20s/frames"))
    parser.add_argument("--blender", type=Path, default=Path("/home/sriram/.local/bin/blender"))
    parser.add_argument("--camera", default="Highlights - 20 second matched edit")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda", "optix"], default="optix")
    parser.add_argument("--baseline-m", type=float, default=0.065)
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=599)
    parser.add_argument("--frame-step", type=int, default=299)
    parser.add_argument("--event-start", type=int, default=1)
    parser.add_argument("--event-end", type=int, default=25)
    parser.add_argument("--skip-stereo", action="store_true")
    parser.add_argument("--skip-events", action="store_true")
    parser.add_argument("--force-event", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_blend": str(args.blend.resolve()),
        "camera": args.camera,
        "resolution": [args.width, args.height],
        "stereo": {
            "baseline_m": args.baseline_m,
            "frames": list(range(args.frame_start, args.frame_end + 1, args.frame_step)),
            "samples": args.samples,
        },
        "event_camera": {
            "source": str(args.event_source.resolve()),
            "source_frames": [args.event_start, args.event_end],
            "input_fps": 24,
            "interpolation": "VisionSim RIFE 32x",
            "emulation_fps": 768,
            "parameters": {
                "pos_thres": 0.20,
                "neg_thres": 0.20,
                "sigma_thres": 0.03,
                "cutoff_hz": 25.0,
                "leak_rate_hz": 0.1,
                "shot_noise_rate_hz": 1.0,
                "refractory_period_s": 100e-6,
                "blur_sigma": 0.5,
            },
        },
        "lidar": {
            "source": "VisionSim left-eye world-space point maps",
            "channels": 64,
            "columns": 512,
            "range_limits_m": [0.5, 20.0],
            "range_noise_sigma_m": 0.015,
            "dropout_probability": 0.02,
        },
        "thermal_enabled": False,
    }
    with open(args.output / "manifest.json", "w") as stream:
        json.dump(manifest, stream, indent=2)

    if not args.skip_stereo:
        _render_stereo(args)
        evaluate_stereo(args.output, args.baseline_m)
        emulate_lidar(args.output, channels=64, columns=512, range_noise_m=0.015, dropout=0.02)
    if not args.skip_events:
        _render_events(args)
        evaluate_events(args.output, args.width, args.height)


if __name__ == "__main__":
    main()
