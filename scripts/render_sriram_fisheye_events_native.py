"""Render the fisheye sequence with VisionSim's native DVS preview style."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import imageio.v3 as iio
import imageio_ffmpeg
import numpy as np
from natsort import natsorted

from visionsim.cli.emulate import events

ROOT = Path("/data/sriram/blender_renders/sriram_room/fisheye_event_camera_visionsim_white")
SOURCE = Path("/data/sriram/blender_renders/sriram_room/fisheye_event_camera_final/interpolated_96fps")
NATIVE = ROOT / "visionsim_native_preview"
VIDEO = ROOT / "sriram_room_fisheye_events_visionsim_white.mp4"
FPS = 24
WIDTH = 960
HEIGHT = 540


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def create_native_previews() -> list[Path]:
    """Run VisionSim's event CLI with its v2e clean-preset parameters."""
    events(
        input_dir=SOURCE,
        output_dir=NATIVE,
        fps=96.0,
        pattern="*.png",
        pos_thres=0.2,
        neg_thres=0.2,
        sigma_thres=0.02,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        refractory_period_s=0.0,
        photoreceptor_noise=False,
        leak_jitter_fraction=0.0,
        noise_rate_cov_decades=0.0,
        seed=20260914,
        blur_sigma=0.0,
        preview_step=4,
        only_preview=True,
        force=True,
    )
    previews = natsorted((NATIVE / "preview").rglob("*.png"))
    if len(previews) != 719:
        raise ValueError(f"Expected 719 complete native preview windows, found {len(previews)}")
    return previews


def start_encoder() -> subprocess.Popen:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
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
        str(VIDEO),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def encode_and_evaluate(previews: list[Path]) -> None:
    # The native CLI yields 719 complete 4-sample windows from 2,877 inputs.
    # Prepending the emulator's initial blank state preserves 720 source-aligned
    # output frames and exactly 30 seconds at 24 fps.
    blank = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    encoder = start_encoder()
    assert encoder.stdin is not None
    encoder.stdin.write(blank.tobytes())

    active_counts = [0]
    on_counts = [0]
    off_counts = [0]
    for output_index, path in enumerate(previews, start=1):
        frame = iio.imread(path)[..., :3]
        if frame.shape != (HEIGHT, WIDTH, 3):
            raise ValueError(f"Unexpected native preview shape {frame.shape} at {path}")
        on = (frame[..., 0] == 0) & (frame[..., 1] == 0) & (frame[..., 2] == 255)
        off = (frame[..., 0] == 255) & (frame[..., 1] == 0) & (frame[..., 2] == 0)
        on_counts.append(int(on.sum()))
        off_counts.append(int(off.sum()))
        active_counts.append(int((on | off).sum()))
        encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
        if output_index == 12:
            iio.imwrite(ROOT / "stationary_preview.png", frame)
        if output_index == 60:
            iio.imwrite(ROOT / "rotating_preview.png", frame)
        if output_index % 120 == 0:
            print(f"Encoded {output_index + 1}/720 frames", flush=True)

    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed")

    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(VIDEO))
    counts = np.asarray(active_counts)
    stationary = np.zeros(720, dtype=bool)
    for start, end in [(1, 25), (91, 115), (181, 205), (271, 295), (361, 385), (451, 475), (541, 565), (631, 655)]:
        stationary[start - 1 : end] = True

    report = {
        "complete": True,
        "pipeline": "VisionSim emulate.events native preview",
        "model": "VisionSim EventEmulator / v2e clean preset",
        "source_fps": 24.0,
        "temporal_reconstruction": "VisionSim RIFE 4x",
        "emulation_fps": 96.0,
        "output_fps": FPS,
        "resolution": [WIDTH, HEIGHT],
        "parameters": {
            "pos_thres": 0.2,
            "neg_thres": 0.2,
            "sigma_thres": 0.02,
            "cutoff_hz": 0.0,
            "leak_rate_hz": 0.0,
            "shot_noise_rate_hz": 0.0,
            "refractory_period_s": 0.0,
            "blur_sigma": 0.0,
            "preview_step": 4,
        },
        "visualization": {
            "background": "white",
            "on_events": "blue",
            "off_events": "red",
            "semantics": "Native VisionSim binary per-pixel preview accumulated over four 96 Hz samples.",
        },
        "visible_active_pixels_total": int(counts.sum()),
        "visible_on_pixels_total": int(np.asarray(on_counts).sum()),
        "visible_off_pixels_total": int(np.asarray(off_counts).sum()),
        "active_pixels_per_frame_stationary_mean": float(counts[stationary].mean()),
        "active_pixels_per_frame_rotating_mean": float(counts[~stationary].mean()),
        "rotation_to_stationary_activity_ratio": float(
            counts[~stationary].mean() / max(counts[stationary].mean(), 1e-12)
        ),
        "video": {
            "path": str(VIDEO),
            "frames": frames,
            "duration_s": seconds,
            "bytes": VIDEO.stat().st_size,
            "sha256": digest(VIDEO),
        },
        "checks": {
            "native_preview_count_correct": len(previews) == 719,
            "both_polarities_visible": sum(on_counts) > 0 and sum(off_counts) > 0,
            "rotation_more_active_than_stops": bool(counts[~stationary].mean() > counts[stationary].mean()),
            "video_timing_exact": frames == 720 and abs(seconds - 30.0) <= 1e-6,
        },
        "limitations": [
            "Events are inferred from RGB frames reconstructed to 96 Hz, not captured by event-camera hardware.",
            "The native preview collapses repeated events of one polarity at the same pixel within each 1/24 s window.",
            "The clean preset omits real sensor background activity so reconstructed signal events remain interpretable.",
        ],
    }
    with (ROOT / "event_camera_validation_summary.json").open("w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if len(list(SOURCE.glob("*.png"))) != 2877:
        raise ValueError("The cached 96 Hz reconstructed source is incomplete")
    encode_and_evaluate(create_native_previews())


if __name__ == "__main__":
    main()
