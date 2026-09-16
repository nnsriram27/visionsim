"""Package the rendered fisheye stereo pair as a red/cyan anaglyph video."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imageio.v3 as iio
import imageio_ffmpeg
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("/data/sriram/blender_renders/sriram_room/fisheye_stereo_final"),
    )
    args = parser.parse_args()

    left_dir = args.root / "left" / "frames" / "0000"
    right_dir = args.root / "right" / "frames" / "0000"
    left_files = sorted(left_dir.glob("*.png"))
    right_files = sorted(right_dir.glob("*.png"))
    expected_names = [f"{frame:03d}.png" for frame in range(1, 721)]
    if [path.name for path in left_files] != expected_names or [path.name for path in right_files] != expected_names:
        raise ValueError("Expected complete left and right frame sequences numbered 001.png through 720.png")

    output = args.root / "sriram_room_fisheye_stereo_anaglyph.mp4"
    filter_graph = (
        "[0:v]lutrgb=g=0:b=0[left_red];"
        "[1:v]lutrgb=r=0[right_cyan];"
        "[left_red][right_cyan]blend=all_mode=addition[anaglyph]"
    )
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
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
        "[anaglyph]",
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

    left = iio.imread(left_dir / "181.png")[..., :3]
    right = iio.imread(right_dir / "181.png")[..., :3]
    poster = np.empty_like(left)
    poster[..., 0] = left[..., 0]
    poster[..., 1:] = right[..., 1:]
    poster_path = output.with_suffix(".png")
    iio.imwrite(poster_path, poster)

    report = {
        "complete": True,
        "representation": "red/cyan anaglyph",
        "channel_mapping": {"red": "left eye", "green_blue": "right eye"},
        "resolution": [960, 540],
        "frames": frame_count,
        "fps": 24.0,
        "duration_s": duration_s,
        "codec": "H.264",
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "source_frames_per_eye": 720,
        "checks": {"source_sequences_complete": True, "video_timing_exact": True},
    }
    with open(args.root / "anaglyph_validation_summary.json", "w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
