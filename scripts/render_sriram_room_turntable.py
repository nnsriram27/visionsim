"""Render a loopable, open-roof dollhouse orbit of sriram_room.blend."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def script_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/sriram/blender_renders/sriram_room/room_mesh_turntable"),
    )
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])


def point_camera(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def configure_scene(output: Path) -> bpy.types.Object:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.fps = 24
    scene.render.fps_base = 1.0

    ceiling = bpy.data.objects.get("Sloped ceiling - inferred")
    if ceiling is None:
        raise RuntimeError("Could not find the room ceiling object")
    ceiling.hide_render = True

    camera_data = bpy.data.cameras.new("Dollhouse turntable camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 13.0
    camera_data.clip_start = 0.1
    camera_data.clip_end = 200.0
    camera = bpy.data.objects.new("Dollhouse turntable camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    # Soft overhead illumination keeps the room readable after removing its roof.
    key_data = bpy.data.lights.new("Dollhouse softbox", "AREA")
    key_data.energy = 1400.0
    key_data.shape = "DISK"
    key_data.size = 9.0
    key = bpy.data.objects.new("Dollhouse softbox", key_data)
    scene.collection.objects.link(key)
    key.location = (0.0, -1.0, 13.0)

    fill_data = bpy.data.lights.new("Dollhouse fill", "AREA")
    fill_data.energy = 700.0
    fill_data.shape = "DISK"
    fill_data.size = 7.0
    fill = bpy.data.objects.new("Dollhouse fill", fill_data)
    scene.collection.objects.link(fill)
    fill.location = (10.0, 8.0, 8.0)
    point_camera(fill, Vector((1.8, -0.3, -1.5)))

    output.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output / "frames" / "frame_")
    return camera


def render() -> None:
    args = script_args()
    camera = configure_scene(args.output)
    frames_dir = args.output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    target = Vector((1.8, -0.3, -0.2))
    radius = 9.0
    height = 24.0
    total_frames = 120
    frame_numbers = [1] if args.preview else range(1, total_frames + 1)

    for frame in frame_numbers:
        angle = math.tau * (frame - 1) / total_frames + math.radians(35.0)
        camera.location = (
            target.x + radius * math.cos(angle),
            target.y + radius * math.sin(angle),
            height,
        )
        point_camera(camera, target)
        bpy.context.scene.frame_set(frame)
        suffix = "preview" if args.preview else f"{frame:04d}"
        bpy.context.scene.render.filepath = str(frames_dir / suffix)
        bpy.ops.render.render(write_still=True)
        print(f"Rendered dollhouse frame {frame}/{total_frames}", flush=True)


if __name__ == "__main__":
    render()
