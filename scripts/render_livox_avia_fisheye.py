"""Render Livox Avia-inspired PLY scans rigidly attached to a Blender camera.

Run with Blender, for example::

    blender --background sriram_room.blend --python scripts/render_livox_avia_fisheye.py -- \
        --output /data/sriram/blender_renders/sriram_room/livox_avia_fisheye

The scan follows the camera's evaluated world transform at every animation frame.
It approximates the Avia non-repetitive scan with a temporally continuous
low-discrepancy angular sequence; it is not a vendor-exact optical scanner model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from embreex import mesh_construction, rtcore_scene
from mathutils import Matrix

MODEL = {
    "name": "Livox Avia-inspired",
    "horizontal_fov_deg": 70.4,
    "vertical_fov_deg": 77.2,
    "point_rate_hz": 240_000,
    "range_precision_sigma_m": 0.02,
    "minimum_range_m": 0.1,
    "maximum_range_m": 450.0,
    "returns": 1,
    "scan_pattern": "temporally continuous R2 low-discrepancy approximation",
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/sriram/blender_renders/sriram_room/livox_avia_fisheye"),
    )
    parser.add_argument("--camera", default="Room panorama - 160 degree fisheye")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def angular_pattern(global_start: int, count: int) -> np.ndarray:
    """Return deterministic non-repeating camera-space unit ray directions."""
    index = global_start + np.arange(count, dtype=np.float64)
    # The irrational R2 increments prevent frame-to-frame pattern repetition.
    u = np.mod(0.5 + index * 0.7548776662466927, 1.0)
    v = np.mod(0.5 + index * 0.5698402909980532, 1.0)
    yaw = np.deg2rad((u - 0.5) * MODEL["horizontal_fov_deg"])
    pitch = np.deg2rad((v - 0.5) * MODEL["vertical_fov_deg"])
    directions = np.column_stack((np.tan(yaw), np.tan(pitch), -np.ones(count)))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions


def write_binary_ply(path: Path, points: np.ndarray, intensity: np.ndarray, offsets_us: np.ndarray) -> None:
    """Write XYZ, proxy intensity, and per-point firing time to binary PLY."""
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Livox Avia-inspired single-return scan\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property ushort intensity\nproperty float time_offset_us\n"
        "end_header\n"
    ).encode("ascii")
    records = np.empty(
        len(points),
        dtype=np.dtype(
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<u2"), ("time_offset_us", "<f4")]
        ),
    )
    records["x"], records["y"], records["z"] = points.T
    records["intensity"] = intensity
    records["time_offset_us"] = offsets_us
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as stream:
        stream.write(header)
        stream.write(records.tobytes())


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2)
    temporary.replace(path)


def action_fcurves(action: bpy.types.Action):
    """Iterate action F-curves across both legacy and Blender 4.4+ actions."""
    if bpy.app.version >= (4, 4, 0):
        for layer in action.layers:
            for strip in layer.strips:
                for slot in action.slots:
                    channelbag = strip.channelbag(slot)
                    if channelbag is not None:
                        yield from channelbag.fcurves
    else:
        yield from action.fcurves


def direct_camera_pose(camera: bpy.types.Object, frame: int) -> Matrix:
    """Evaluate an unparented, unconstrained camera action without rebuilding the scene."""
    location = camera.location.copy()
    rotation = camera.rotation_euler.copy()
    scale = camera.scale.copy()
    if camera.animation_data and camera.animation_data.action:
        for curve in action_fcurves(camera.animation_data.action):
            target = {"location": location, "rotation_euler": rotation, "scale": scale}.get(curve.data_path)
            if target is not None:
                target[curve.array_index] = curve.evaluate(frame)
    return Matrix.LocRotScale(location, rotation.to_quaternion(), scale)


def build_embree_scene(scene: bpy.types.Scene):
    """Build one static whole-scene Embree BVH from visible evaluated meshes."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices = []
    triangles = []
    vertex_offset = 0
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render or not obj.visible_get():
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.data
        if not mesh.vertices or not mesh.polygons:
            continue
        coordinates = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", coordinates)
        coordinates = coordinates.reshape(-1, 3)
        matrix_world = np.asarray(evaluated.matrix_world, dtype=np.float64)
        coordinates = coordinates @ matrix_world[:3, :3].T + matrix_world[:3, 3]
        mesh.calc_loop_triangles()
        indices = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", indices)
        indices = indices.reshape(-1, 3).astype(np.uint32) + vertex_offset
        vertices.append(coordinates.astype(np.float32))
        triangles.append(indices)
        vertex_offset += len(coordinates)

    vertices_array = np.ascontiguousarray(np.vstack(vertices), dtype=np.float32)
    triangles_array = np.ascontiguousarray(np.vstack(triangles), dtype=np.uint32)
    embree_scene = rtcore_scene.EmbreeScene()
    embree_mesh = mesh_construction.TriangleMesh(embree_scene, vertices_array, triangles_array)
    return embree_scene, embree_mesh, len(vertices_array), len(triangles_array)


def main() -> None:
    args = parse_args()
    scene = bpy.context.scene
    camera = scene.objects.get(args.camera)
    if camera is None or camera.type != "CAMERA":
        raise ValueError(f"Camera object not found: {args.camera!r}")

    frame_start = args.frame_start if args.frame_start is not None else scene.frame_start
    frame_end = args.frame_end if args.frame_end is not None else scene.frame_end
    fps = scene.render.fps / scene.render.fps_base
    rays_per_frame = round(MODEL["point_rate_hz"] / fps)
    frame_period_us = 1e6 / fps
    offsets_us = np.arange(rays_per_frame, dtype=np.float64) / MODEL["point_rate_hz"] * 1e6
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_blend": bpy.data.filepath,
        "camera": args.camera,
        "mount": "co-located; LiDAR -Z axis aligned with camera optical axis",
        "coordinate_outputs": {
            "ply_sensor": "per-frame LiDAR coordinates: +X right, +Y up, -Z forward",
            "ply_world": "Blender world coordinates in scene units",
        },
        "scene_unit_scale_m": scene.unit_settings.scale_length,
        "fps": fps,
        "frame_period_us": frame_period_us,
        "frame_range_inclusive": [frame_start, frame_end],
        "rays_per_frame": rays_per_frame,
        "ray_backend": "embreex 4.x DISTANCE queries; one static whole-scene BVH",
        "model": MODEL,
        "limitations": [
            "scan pattern approximates, but does not reproduce, Livox proprietary optics",
            "intensity is a range-only proxy, not calibrated 905 nm material reflectivity",
            "single return only; atmosphere, beam divergence, and multipath are not modeled",
            "scene geometry is sampled at the Blender frame time; within-frame object motion distortion is not modeled",
        ],
    }
    write_json(args.output / "manifest.json", manifest)

    animated_non_cameras = [
        obj
        for obj in scene.objects
        if obj.type != "CAMERA"
        and obj.animation_data
        and (obj.animation_data.action is not None or len(obj.animation_data.drivers) > 0)
    ]
    direct_pose_mode = not animated_non_cameras and camera.parent is None and not camera.constraints
    if not direct_pose_mode:
        raise RuntimeError("This batched renderer requires static geometry and an unparented, unconstrained camera")
    scene.frame_set(frame_start)
    print("Building static whole-scene Embree BVH...", flush=True)
    embree_scene, embree_mesh, vertex_count, triangle_count = build_embree_scene(scene)
    # The Python mesh wrapper owns Embree's geometry buffers and must remain alive.
    assert embree_mesh is not None
    manifest["scene_bvh"] = {"vertices": vertex_count, "triangles": triangle_count}
    write_json(args.output / "manifest.json", manifest)

    poses = []
    reports = []
    started = time.perf_counter()
    for frame in range(frame_start, frame_end + 1):
        sensor_path = args.output / "ply_sensor" / f"{frame:06d}.ply"
        world_path = args.output / "ply_world" / f"{frame:06d}.ply"
        pose = direct_camera_pose(camera, frame)
        rotation = np.asarray(pose.to_3x3(), dtype=np.float32)
        origin = np.asarray(pose.translation, dtype=np.float32)
        poses.append(
            {
                "frame": frame,
                "timestamp_us": (frame - frame_start) * frame_period_us,
                "transform_matrix": [[float(value) for value in row] for row in pose],
            }
        )

        if sensor_path.exists() and world_path.exists() and not args.force:
            reports.append({"frame": frame, "status": "skipped_existing"})
            continue

        directions_camera = angular_pattern((frame - frame_start) * rays_per_frame, rays_per_frame)
        frame_rng = np.random.default_rng(np.random.SeedSequence([args.seed, frame]))
        directions_world = np.ascontiguousarray(directions_camera @ rotation.T, dtype=np.float32)
        origins = np.ascontiguousarray(np.broadcast_to(origin, directions_world.shape), dtype=np.float32)
        ideal_distance_scene = np.asarray(
            embree_scene.run(origins, directions_world, query="DISTANCE"), dtype=np.float32
        ).reshape(-1)
        ideal_range_m = ideal_distance_scene * scene.unit_settings.scale_length
        valid = (
            np.isfinite(ideal_range_m)
            & (ideal_range_m >= MODEL["minimum_range_m"])
            & (ideal_range_m <= MODEL["maximum_range_m"])
        )
        noisy_range_m = np.maximum(
            MODEL["minimum_range_m"],
            ideal_range_m[valid] + frame_rng.normal(0.0, MODEL["range_precision_sigma_m"], valid.sum()),
        )
        noisy_distance_scene = noisy_range_m / max(scene.unit_settings.scale_length, 1e-12)
        sensor_array = np.asarray(directions_camera[valid] * noisy_distance_scene[:, None], dtype=np.float32)
        world_array = np.asarray(origin + directions_world[valid] * noisy_distance_scene[:, None], dtype=np.float32)
        intensity_array = np.asarray(65535 / (1.0 + 0.05 * noisy_range_m * noisy_range_m), dtype=np.uint16)
        offset_array = np.asarray(offsets_us[valid], dtype=np.float32)
        write_binary_ply(world_path, world_array, intensity_array, offset_array)
        write_binary_ply(sensor_path, sensor_array, intensity_array, offset_array)
        ranges = noisy_range_m
        reports.append(
            {
                "frame": frame,
                "status": "rendered",
                "rays": rays_per_frame,
                "returns": len(sensor_array),
                "return_fraction": len(sensor_array) / rays_per_frame,
                "range_m_min_median_max": [float(ranges.min()), float(np.median(ranges)), float(ranges.max())],
            }
        )
        if frame == frame_start or frame % 10 == 0 or frame == frame_end:
            elapsed = time.perf_counter() - started
            print(
                f"LIDAR_PROGRESS frame={frame}/{frame_end} returns={len(sensor_array)} elapsed_s={elapsed:.1f}",
                flush=True,
            )
            write_json(
                args.output / "progress.json",
                {"last_completed_frame": frame, "frames_completed": frame - frame_start + 1, "elapsed_s": elapsed},
            )

    write_json(args.output / "poses.json", poses)
    write_json(args.output / "evaluation.json", {"frames": reports, "elapsed_s": time.perf_counter() - started})
    write_json(
        args.output / "progress.json",
        {
            "complete": True,
            "last_completed_frame": frame_end,
            "frames_completed": frame_end - frame_start + 1,
            "elapsed_s": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
