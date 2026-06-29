from __future__ import annotations

import subprocess


def test_temperature_aov_registered(executable):
    code = (
        "import bpy;"
        "from visionsim.simulate.heatsim import thermal_shader as ts;"
        "bpy.ops.mesh.primitive_cube_add();"
        "vl=bpy.context.view_layer;"
        "ts.setup_temperature_aov(bpy.context.scene, vl);"
        "assert any(a.name=='temperature' for a in vl.aovs);"
        "print('THERMAL_AOV_OK')"
    )
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True)
    assert "THERMAL_AOV_OK" in out.stdout, out.stderr


def test_enter_restore_round_trip(executable):
    """enter_thermal_scene then restore_scene must leave materials, lights, and world unchanged."""
    code = (
        "import bpy;"
        "from visionsim.simulate.heatsim import thermal_shader as ts;"
        # Build a simple scene: cube with a named material + a point light.
        "bpy.ops.object.select_all(action='SELECT');"
        "bpy.ops.object.delete();"
        "bpy.ops.mesh.primitive_cube_add();"
        "cube = bpy.context.active_object;"
        "mat = bpy.data.materials.new('TestMat');"
        "cube.data.materials.append(mat);"
        "bpy.ops.object.light_add(type='POINT');"
        "light = bpy.context.active_object;"
        # Record original state.
        "orig_mat_name = cube.material_slots[0].material.name;"
        "orig_light_hide = light.hide_render;"
        "scene = bpy.context.scene;"
        "orig_world = scene.world;"
        # Round-trip.
        "state = ts.enter_thermal_scene(scene, radiance_scale=1.0);"
        "ts.restore_scene(scene, state);"
        # Verify materials restored.
        "restored_name = cube.material_slots[0].material.name if cube.material_slots[0].material else None;"
        "assert restored_name == orig_mat_name, f'material: expected {orig_mat_name!r}, got {restored_name!r}';"
        # Verify light visibility restored.
        "assert light.hide_render == orig_light_hide, f'light hide_render changed';"
        # Verify world restored.
        "assert scene.world is orig_world, f'world not restored: {scene.world!r} vs {orig_world!r}';"
        "print('ROUND_TRIP_OK')"
    )
    out = subprocess.run([str(executable), "-b", "--python-expr", code], capture_output=True, text=True)
    assert "ROUND_TRIP_OK" in out.stdout, out.stderr
