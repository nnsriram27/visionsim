"""Render a synchronized oblique-view MP4 from the Livox-inspired PLY sequence."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import matplotlib
import numpy as np
from evaluate_livox_avia import read_ply
from PIL import Image, ImageDraw, ImageFont


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


class ViewerCamera:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        eye = np.array([5.8, -6.7, 5.0])
        target = np.array([0.35, -0.15, 1.05])
        forward = normalize(target - eye)
        self.right = normalize(np.cross(forward, np.array([0.0, 0.0, 1.0])))
        self.up = normalize(np.cross(self.right, forward))
        self.forward = forward
        self.eye = eye
        self.focal = 0.5 * width / np.tan(np.radians(46.0) / 2.0)

    def project(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        relative = xyz - self.eye
        depth = relative @ self.forward
        uv = np.empty((len(xyz), 2), dtype=float)
        uv[:, 0] = self.width / 2 + self.focal * (relative @ self.right) / depth
        uv[:, 1] = self.height / 2 - self.focal * (relative @ self.up) / depth
        return uv, depth


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_world_line(
    draw: ImageDraw.ImageDraw,
    camera: ViewerCamera,
    start: np.ndarray,
    end: np.ndarray,
    fill: tuple[int, int, int],
    width: int = 1,
) -> None:
    uv, depth = camera.project(np.vstack((start, end)))
    if np.all(depth > 0):
        draw.line([tuple(uv[0]), tuple(uv[1])], fill=fill, width=width)


def draw_grid(draw: ImageDraw.ImageDraw, camera: ViewerCamera) -> None:
    for coordinate in np.arange(-3.0, 3.01, 0.5):
        major = abs(coordinate - round(coordinate)) < 1e-6
        color = (48, 58, 66) if major else (34, 42, 49)
        draw_world_line(
            draw,
            camera,
            np.array([-3.0, coordinate, 0.0]),
            np.array([3.0, coordinate, 0.0]),
            color,
        )
        draw_world_line(
            draw,
            camera,
            np.array([coordinate, -3.0, 0.0]),
            np.array([coordinate, 3.0, 0.0]),
            color,
        )


def splat_points(
    pixels: np.ndarray,
    camera: ViewerCamera,
    xyz: np.ndarray,
    colors: np.ndarray,
    radius: int,
) -> None:
    uv, depth = camera.project(xyz)
    xy = np.rint(uv).astype(np.int32)
    valid = (
        (depth > 0)
        & (xy[:, 0] >= radius)
        & (xy[:, 0] < camera.width - radius)
        & (xy[:, 1] >= radius)
        & (xy[:, 1] < camera.height - radius)
    )
    xy = xy[valid]
    depth = depth[valid]
    colors = colors[valid]
    # Painting distant points first provides a simple point-cloud depth test.
    order = np.argsort(depth)[::-1]
    xy = xy[order]
    colors = colors[order]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            pixels[xy[:, 1] + dy, xy[:, 0] + dx] = colors


def range_colors(ranges: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    values = np.clip((ranges - minimum) / (maximum - minimum), 0.0, 1.0)
    return (matplotlib.colormaps["turbo"](values)[:, :3] * 255).astype(np.uint8)


def sensor_frustum(pose: np.ndarray, distance: float = 0.75) -> tuple[np.ndarray, list[np.ndarray]]:
    origin = pose[:3, 3]
    rotation = pose[:3, :3]
    corners = []
    for horizontal in (-35.2, 35.2):
        for vertical in (-38.6, 38.6):
            direction = np.array(
                [
                    np.tan(np.radians(horizontal)),
                    np.tan(np.radians(vertical)),
                    -1.0,
                ]
            )
            corners.append(origin + rotation @ normalize(direction) * distance)
    return origin, corners


def add_overlay(
    image: Image.Image,
    frame: int,
    frame_count: int,
    fps: float,
    heading_deg: float,
    paused: bool,
    range_min: float,
    range_max: float,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(25)
    text_font = load_font(18)
    small_font = load_font(15)
    draw.rounded_rectangle((22, 18, 488, 119), radius=9, fill=(4, 9, 14, 205), outline=(85, 109, 124, 180))
    draw.text((38, 30), "Livox Avia-inspired room scan", font=title_font, fill=(240, 246, 250, 255))
    state_color = (89, 220, 143, 255) if paused else (255, 180, 71, 255)
    state = "PAUSED" if paused else "ROTATING"
    draw.text((39, 70), state, font=text_font, fill=state_color)
    draw.text(
        (150, 72),
        f"t = {(frame - 1) / fps:05.2f} s    heading = {heading_deg:06.1f}°",
        font=small_font,
        fill=(195, 207, 216, 255),
    )
    draw.text(
        (image.width - 177, 28),
        f"FRAME {frame:03d}/{frame_count}",
        font=small_font,
        fill=(218, 226, 231, 255),
    )

    bar_left, bar_top, bar_width, bar_height = image.width - 57, 87, 15, 210
    values = np.linspace(1.0, 0.0, bar_height)
    bar = (matplotlib.colormaps["turbo"](values)[:, :3] * 255).astype(np.uint8)
    image_array = np.asarray(image).copy()
    image_array[bar_top : bar_top + bar_height, bar_left : bar_left + bar_width] = bar[:, None, :]
    image.paste(Image.fromarray(image_array))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((bar_left - 1, bar_top - 1, bar_left + bar_width, bar_top + bar_height), outline=(220, 225, 228, 220))
    draw.text((bar_left - 9, bar_top - 23), f"{range_max:.1f} m", font=small_font, fill=(220, 225, 228, 255), anchor="ra")
    draw.text(
        (bar_left - 9, bar_top + bar_height + 3),
        f"{range_min:.1f} m",
        font=small_font,
        fill=(220, 225, 228, 255),
        anchor="ra",
    )
    draw.text((bar_left + 7, bar_top + bar_height + 31), "RANGE", font=small_font, fill=(190, 201, 208, 255), anchor="ma")
    draw.text(
        (25, image.height - 34),
        "Fixed oblique world view  •  vivid: current scan  •  blue-gray: accumulated returns",
        font=small_font,
        fill=(175, 190, 200, 255),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--history-capacity", type=int, default=60_000)
    parser.add_argument("--history-stride", type=int, default=20)
    parser.add_argument("--preset", default="ultrafast")
    parser.add_argument("--crf", type=int, default=18)
    args = parser.parse_args()

    with open(args.root / "manifest.json") as stream:
        manifest = json.load(stream)
    with open(args.root / "poses.json") as stream:
        pose_records = json.load(stream)
    world_files = sorted((args.root / "ply_world").glob("*.ply"))
    if len(world_files) != len(pose_records):
        raise ValueError("The PLY and pose sequences do not have matching frame counts")

    fps = float(manifest["fps"])
    output = args.output or args.root / "livox_avia_oblique_range.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    poster = output.with_suffix(".png")
    camera = ViewerCamera(args.width, args.height)
    range_min, range_max = 0.8, 3.4

    poses = np.asarray([record["transform_matrix"] for record in pose_records], dtype=float)
    forwards = np.einsum("nij,j->ni", poses[:, :3, :3], np.array([0.0, 0.0, -1.0]))
    headings = np.unwrap(np.arctan2(forwards[:, 1], forwards[:, 0]))
    headings = np.degrees(headings - headings[0])
    equal_transitions = np.max(np.abs(np.diff(poses, axis=0)), axis=(1, 2)) < 1e-12
    stationary = np.zeros(len(poses), dtype=bool)
    stationary[:-1] |= equal_transitions
    stationary[1:] |= equal_transitions

    history_xyz = np.empty((args.history_capacity, 3), dtype=np.float32)
    history_age = np.empty(args.history_capacity, dtype=np.int32)
    history_count = 0
    history_cursor = 0

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{args.width}x{args.height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for index, (path, pose) in enumerate(zip(world_files, poses)):
            frame = index + 1
            cloud = read_ply(path)
            xyz = np.column_stack((cloud["x"], cloud["y"], cloud["z"])).astype(np.float32)
            sensor_origin = pose[:3, 3]
            ranges = np.linalg.norm(xyz - sensor_origin, axis=1)

            additions = xyz[:: args.history_stride]
            count = len(additions)
            indices = (np.arange(count) + history_cursor) % args.history_capacity
            history_xyz[indices] = additions
            history_age[indices] = frame
            history_cursor = (history_cursor + count) % args.history_capacity
            history_count = min(args.history_capacity, history_count + count)

            pixels = np.full((args.height, args.width, 3), (12, 18, 23), dtype=np.uint8)
            image = Image.fromarray(pixels)
            draw = ImageDraw.Draw(image)
            draw_grid(draw, camera)
            pixels = np.asarray(image).copy()

            history = history_xyz[:history_count]
            ages = frame - history_age[:history_count]
            brightness = np.clip(92.0 - ages * 0.12, 38.0, 92.0).astype(np.uint8)
            history_colors = np.column_stack(
                (
                    (brightness * 0.50).astype(np.uint8),
                    (brightness * 0.83).astype(np.uint8),
                    brightness,
                )
            )
            splat_points(pixels, camera, history, history_colors, radius=0)
            splat_points(pixels, camera, xyz, range_colors(ranges, range_min, range_max), radius=1)

            image = Image.fromarray(pixels)
            draw = ImageDraw.Draw(image, "RGBA")
            origin, corners = sensor_frustum(pose)
            forward_tip = origin + pose[:3, :3] @ np.array([0.0, 0.0, -0.95])
            for corner in corners:
                draw_world_line(draw, camera, origin, corner, (255, 177, 66), width=2)
            draw_world_line(draw, camera, origin, forward_tip, (255, 244, 205), width=4)
            origin_uv, depth = camera.project(origin[None])
            if depth[0] > 0:
                x, y = origin_uv[0]
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(255, 195, 77), outline=(255, 255, 255), width=2)

            add_overlay(
                image,
                frame,
                len(world_files),
                fps,
                headings[index],
                bool(stationary[index]),
                range_min,
                range_max,
            )
            if frame == 181:
                image.save(poster)
            process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
            if frame == 1 or frame % 24 == 0 or frame == len(world_files):
                print(f"Rendered {frame}/{len(world_files)} frames", flush=True)
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")
    print(f"Video: {output}")
    print(f"Poster: {poster}")


if __name__ == "__main__":
    main()
