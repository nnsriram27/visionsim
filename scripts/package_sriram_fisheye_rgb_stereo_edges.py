"""Create an RGB-preserving red/cyan stereo-edge visualization video."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imageio.v3 as iio
import imageio_ffmpeg
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel


def edge_strength(image: np.ndarray, threshold: float = 20.0, span: float = 60.0) -> np.ndarray:
    gray = image @ np.array([0.2126, 0.7152, 0.0722])
    gray = gaussian_filter(gray, 0.8)
    magnitude = np.hypot(sobel(gray, axis=1), sobel(gray, axis=0)) / 4.0
    return np.clip((magnitude - threshold) / span, 0.0, 1.0)


def compose(base: np.ndarray, left: np.ndarray, right: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    left_edge = edge_strength(left)
    right_edge = edge_strength(right)
    coverage = np.clip((left_edge + right_edge) * alpha, 0.0, 0.85)
    output = base * (1.0 - coverage[..., None])
    output[..., 0] += 255.0 * left_edge * alpha
    output[..., 1] += 255.0 * right_edge * alpha
    output[..., 2] += 255.0 * right_edge * alpha
    return np.clip(output, 0.0, 255.0).astype(np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stereo-root",
        type=Path,
        default=Path("/data/sriram/blender_renders/sriram_room/fisheye_stereo_final"),
    )
    parser.add_argument(
        "--rgb-root",
        type=Path,
        default=Path("/data/sriram/blender_renders/sriram_room/room_panorama_160deg/frames"),
    )
    args = parser.parse_args()

    left_dir = args.stereo_root / "left" / "frames" / "0000"
    right_dir = args.stereo_root / "right" / "frames" / "0000"
    expected_stereo = [f"{frame:03d}.png" for frame in range(1, 721)]
    expected_rgb = [f"{frame:04d}.png" for frame in range(1, 721)]
    if [path.name for path in sorted(left_dir.glob("*.png"))] != expected_stereo:
        raise ValueError("Left stereo sequence is incomplete")
    if [path.name for path in sorted(right_dir.glob("*.png"))] != expected_stereo:
        raise ValueError("Right stereo sequence is incomplete")
    if [path.name for path in sorted(args.rgb_root.glob("*.png"))] != expected_rgb:
        raise ValueError("Central RGB sequence is incomplete")

    output = args.stereo_root / "sriram_room_fisheye_rgb_stereo_edges.mp4"
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
        "960x540",
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
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    poster_path = output.with_suffix(".png")
    try:
        for frame in range(1, 721):
            with Image.open(args.rgb_root / f"{frame:04d}.png") as source:
                base = np.asarray(source.convert("RGB").resize((960, 540), Image.Resampling.LANCZOS)).astype(float)
            left = iio.imread(left_dir / f"{frame:03d}.png")[..., :3].astype(float)
            right = iio.imread(right_dir / f"{frame:03d}.png")[..., :3].astype(float)
            composed = compose(base, left, right)
            if frame == 181:
                iio.imwrite(poster_path, composed)
            process.stdin.write(composed.tobytes())
            if frame == 1 or frame % 24 == 0:
                print(f"Composed {frame}/720 frames", flush=True)
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")

    frame_count, duration_s = imageio_ffmpeg.count_frames_and_secs(str(output))
    if frame_count != 720 or abs(duration_s - 30.0) > 1e-6:
        raise ValueError(f"Bad video timing: frames={frame_count}, duration={duration_s}")
    report = {
        "complete": True,
        "representation": "central RGB with red/cyan stereo edge overlay",
        "base": "full central Room panorama - 160 degree fisheye RGB sequence",
        "edge_mapping": {"red": "left-eye edges", "cyan": "right-eye edges"},
        "edge_parameters": {"gaussian_sigma": 0.8, "threshold": 20.0, "span": 60.0, "alpha": 0.65},
        "resolution": [960, 540],
        "frames": frame_count,
        "fps": 24.0,
        "duration_s": duration_s,
        "codec": "H.264",
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "checks": {
            "all_three_source_sequences_complete": True,
            "video_timing_exact": True,
            "poster_created": poster_path.exists(),
        },
    }
    with open(args.stereo_root / "rgb_stereo_edges_validation_summary.json", "w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
