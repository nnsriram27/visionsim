import subprocess


def test_bake_albedo_map_returns_varying_pixels(executable):
    code = r"""
import bpy, numpy as np
from visionsim.simulate.heatsim import register, irradiance

register()
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.mesh.primitive_grid_add(x_subdivisions=12, y_subdivisions=12, size=2.0)
obj = bpy.context.active_object

mat = bpy.data.materials.new('checker_mat')
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes.get('Principled BSDF')
checker = nt.nodes.new('ShaderNodeTexChecker')
checker.inputs['Scale'].default_value = 6.0
nt.links.new(checker.outputs['Color'], bsdf.inputs['Base Color'])
obj.data.materials.append(mat)

bpy.context.scene.render.engine = 'CYCLES'
try:
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.cycles.samples = 4
except Exception:
    pass

baked = irradiance.bake_albedo_map(bpy.context.scene, obj, 128)
assert baked is not None, 'bake_albedo_map returned None'
px = baked.pixels
assert px.ndim == 3 and px.shape[2] == 3, f'bad pixel shape {px.shape}'
assert float(px.std()) > 0.1, f'expected high-contrast checker variation (std>0.1), got std={px.std()}'
mean = float(px.mean())
assert 0.0 <= mean <= 1.0, f'albedo mean out of range: {mean}'
print('ALBEDO_BAKE_OK', px.shape, round(mean, 3), round(float(px.std()), 3))
"""
    out = subprocess.run(
        [str(executable), "-b", "--python-expr", code],
        capture_output=True, text=True,
    )
    assert "ALBEDO_BAKE_OK" in out.stdout, out.stdout + "\n" + out.stderr
