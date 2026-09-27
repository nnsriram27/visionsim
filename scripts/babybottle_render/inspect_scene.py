"""Run with blender --background --disable-autoexec FILE --python SCRIPT -- OUTPUT."""
import json
import sys
from pathlib import Path
import bpy

scene_name = sys.argv[sys.argv.index("--") + 2] if len(sys.argv) > sys.argv.index("--") + 2 else "01_Light_Force"
scene = bpy.data.scenes[scene_name]
bpy.context.window.scene = scene
result = {"blender": bpy.app.version_string, "file": bpy.data.filepath,
          "scene": scene.name,
          "scenes": [{"name": s.name, "start": s.frame_start, "end": s.frame_end,
                      "camera": s.camera.name if s.camera else None, "properties": dict(s.items())}
                     for s in bpy.data.scenes],
          "frame_start": scene.frame_start, "frame_end": scene.frame_end,
          "fps": scene.render.fps / scene.render.fps_base,
          "unit_scale": scene.unit_settings.scale_length,
          "resolution": [scene.render.resolution_x, scene.render.resolution_y],
          "active_camera": scene.camera.name if scene.camera else None,
          "objects": [], "samples": {}}
for obj in scene.objects:
    entry = {"name": obj.name, "type": obj.type, "parent": obj.parent.name if obj.parent else None,
             "dimensions": list(obj.dimensions), "location": list(obj.matrix_world.translation),
             "hide_render": obj.hide_render, "custom": {k: str(obj[k])[:500] for k in obj.keys()}}
    if obj.type == "MESH":
        entry.update(vertices=len(obj.data.vertices), polygons=len(obj.data.polygons),
                     materials=[m.name if m else None for m in obj.data.materials],
                     modifiers=[{"name": m.name, "type": m.type} for m in obj.modifiers])
    if obj.type == "CAMERA":
        entry.update(lens_mm=obj.data.lens, sensor_width_mm=obj.data.sensor_width,
                     sensor_height_mm=obj.data.sensor_height, sensor_fit=obj.data.sensor_fit,
                     angle_x=obj.data.angle_x, angle_y=obj.data.angle_y)
    result["objects"].append(entry)
for frame in sorted(set([scene.frame_start, 130, 181, 240, 360, (scene.frame_start+scene.frame_end)//2, scene.frame_end])):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    sample = {}
    for obj in scene.objects:
        if obj.type not in {"MESH", "CAMERA"}:
            continue
        evaluated = obj.evaluated_get(graph)
        item = {"matrix_world": [list(row) for row in evaluated.matrix_world], "dimensions": list(evaluated.dimensions), "hide_render": obj.hide_render}
        if obj.type == "MESH":
            mesh = evaluated.to_mesh()
            item["vertices"] = len(mesh.vertices)
            item["polygons"] = len(mesh.polygons)
            if mesh.vertices:
                item["first_vertex"] = list(evaluated.matrix_world @ mesh.vertices[0].co)
                item["vertex_100"] = list(evaluated.matrix_world @ mesh.vertices[min(100, len(mesh.vertices)-1)].co)
            evaluated.to_mesh_clear()
        sample[obj.name] = item
    result["samples"][str(frame)] = sample
out = Path(sys.argv[sys.argv.index("--") + 1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n")
print("Inspection written to", out)
