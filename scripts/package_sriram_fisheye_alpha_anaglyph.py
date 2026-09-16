"""Blend a subtle red/cyan fisheye anaglyph over the central RGB video."""

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
    parser.add_argument("--alpha", type=float, default=0.25)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")

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

    output = args.stereo_root / "sriram_room_fisheye_rgb_alpha_anaglyph.mp4"
    base_weight = 1.0 - args.alpha
    filter_graph = (
        "[0:v]scale=960:540[base];"
        "[1:v]lutrgb=g=0:b=0[left_red];"
        "[2:v]lutrgb=r=0[right_cyan];"
        "[left_red][right_cyan]blend=all_mode=addition[anaglyph];"
        f"[base][anaglyph]blend=all_expr='A*{base_weight}+B*{args.alpha}'[out]"
    )
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-framerate",
        "24",
        "-start_number",
        "1",
        "-i",
        str(args.rgb_root / "%04d.png"),
        "-framerate",
        "24",
        "-start_number",
        "1",
        "-i",
        str(left_dir / "%03d.png"),
        "-framerate",
        "24",
        "-start_number",
        "1",
        "-i",
        str(right_dir / "%03d.png"),
        "-filter_complex",
        filter_graph,
        "-map",
        "[out]",
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
        str(output),
    ]
    subprocess.run(command, check=True)

    frame_count, duration_s = imageio_ffmpeg.count_frames_and_secs(str(output))
    if frame_count != 720 or abs(duration_s - 30.0) > 1e-6:
        raise ValueError(f"Bad video timing: frames={frame_count}, duration={duration_s}")

    with Image.open(args.rgb_root / "0181.png") as source:
        base = np.asarray(source.convert("RGB").resize((960, 540), Image.Resampling.LANCZOS)).astype(float)
    left = iio.imread(left_dir / "181.png")[..., :3]
    right = iio.imread(right_dir / "181.png")[..., :3]
    anaglyph = np.empty_like(left)
    anaglyph[..., 0] = left[..., 0]
    anaglyph[..., 1:] = right[..., 1:]
    poster = np.clip(base * base_weight + anaglyph.astype(float) * args.alpha, 0.0, 255.0).astype(np.uint8)
    poster_path = output.with_suffix(".png")
    iio.imwrite(poster_path, poster)

    report = {
        "complete": True,
        "representation": "central RGB alpha-blended with classical red/cyan anaglyph",
        "alpha": args.alpha,
        "base_weight": base_weight,
        "anaglyph_mapping": {"red": "left eye", "green_blue": "right eye"},
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
    with open(args.stereo_root / "alpha_anaglyph_validation_summary.json", "w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
