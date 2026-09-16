"""Render a realistic cutaway still of sriram_room using its original lights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/sriram/blender_renders/sriram_room/room_mesh_still"),
    )
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])


def hide_cutaway_geometry() -> list[str]:
    names = [
        "Sloped ceiling - inferred",
        "Entry side wall",
        "Far wall inferred",
        "Full-length floor mirror",
        "Closet brass knob",
        "Closet jamb",
        "Closet jamb.001",
        "Closet lintel",
        "Closet panel door",
        "Closet raised panel",
        "Closet raised panel.001",
        "Entry brass knob",
        "Entry jamb",
        "Entry jamb.001",
        "Entry lintel",
        "Entry panel door",
        "Entry raised panel",
        "Entry raised panel.001",
        "Ceiling fan stem",
        "Fan motor",
        "Frosted globe",
        "Walnut fan blade",
        "Walnut fan blade.001",
        "Walnut fan blade.002",
        "Walnut fan blade.003",
    ]
    missing = [name for name in names if bpy.data.objects.get(name) is None]
    if missing:
        raise RuntimeError(f"Missing cutaway objects: {missing}")
    for name in names:
        bpy.data.objects[name].hide_render = True
    return names


def aim(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def render() -> None:
    args = arguments()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    hidden = hide_cutaway_geometry()

    scene = bpy.context.scene
    scene.frame_set(1)
    scene.render.resolution_x = 960 if args.preview else 1920
    scene.render.resolution_y = 540 if args.preview else 1080
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.use_file_extension = True

    if args.preview:
        scene.render.engine = "BLENDER_EEVEE"
    else:
        # Preserve the scene's authored Cycles lighting and exposure. No new
        # lights, world changes, or material overrides are introduced.
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 64
        scene.cycles.use_denoising = True
        scene.cycles.device = "GPU"
        preferences = bpy.context.preferences.addons['cycles'].preferences
        preferences.compute_device_type = 'CUDA'
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type == 'CUDA'

    data = bpy.data.cameras.new("Cutaway still camera")
    data.type = "PERSP"
    data.lens = 45.0
    data.sensor_width = 36.0
    data.clip_start = 0.1
    data.clip_end = 200.0
    camera = bpy.data.objects.new("Cutaway still camera", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    floor = bpy.data.objects['Subfloor']
    center = floor.matrix_world.translation
    target = Vector((center.x, center.y, 1.1))
    camera.location = target + Vector((7.95, 7.95, 6.678))
    aim(camera, target)

    filename = "cutaway_preview" if args.preview else "sriram_room_realistic_cutaway"
    scene.render.filepath = str(output / filename)
    bpy.ops.render.render(write_still=True)
    print(f"Rendered {scene.render.filepath}.png", flush=True)
    print(f"Hidden only: {hidden}", flush=True)


if __name__ == "__main__":
    render()
