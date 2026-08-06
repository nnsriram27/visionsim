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


def test_solve_produces_varying_albedo_attribute(executable, tmp_path):
    code = r"""
import bpy, numpy as np
from pathlib import Path
from visionsim.simulate.heatsim import register, adapter

register()
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.mesh.primitive_grid_add(x_subdivisions=20, y_subdivisions=20, size=2.0)
obj = bpy.context.active_object
obj.name = 'ThermalPlane'
obj.heat_simulation_enabled = True

mat = bpy.data.materials.new('checker_mat')
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes.get('Principled BSDF')
checker = nt.nodes.new('ShaderNodeTexChecker')
checker.inputs['Scale'].default_value = 6.0
nt.links.new(checker.outputs['Color'], bsdf.inputs['Base Color'])
obj.data.materials.append(mat)

bpy.ops.object.light_add(type='SUN')
bpy.context.active_object.data.energy = 10.0
world = bpy.context.scene.world
world.use_nodes = True
bg = world.node_tree.nodes.get('Background')
bg.inputs['Strength'].default_value = 1.0

defaults = dict(initial_temperature_K=295.0, thermal_diffusivity_mm2_s=0.17,
                density_kg_m3=1330.0, specific_heat_J_kgK=880.0, emissivity=0.9,
                irradiance_scale=100.0)
solver_cfg = dict(sim_time_s=0.1, timestep_s=0.05, domain='POINTS',
                  laplacian_backend='ROBUST', device='cpu')
adapter.solve_scene(bpy.context.scene, defaults=defaults,
                    solver_cfg=solver_cfg, cache_root=Path(r'{tmp}'))

mesh = obj.data
attr = mesh.attributes.get('albedo')
assert attr is not None, 'no albedo attribute after solve'
vals = np.zeros(len(mesh.vertices), dtype=np.float32)
attr.data.foreach_get('value', vals)
assert float(vals.std()) > 0.05, f'albedo not varying (std={{vals.std()}}) - bake did not run'
assert 0.0 < float(vals.mean()) < 1.0, f'albedo mean out of range: {{vals.mean()}}'
print('VARYING_ALBEDO_OK', round(float(vals.mean()), 3), round(float(vals.std()), 3))
""".replace("{tmp}", str(tmp_path))
    out = subprocess.run(
        [str(executable), "-b", "--python-expr", code],
        capture_output=True, text=True,
    )
    assert "VARYING_ALBEDO_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_authored_irradiance_scale_read_under_bpy(executable):
    code = r"""
import bpy
from visionsim.simulate.heatsim import register, adapter
register()
sc = bpy.context.scene
sc['heat_sim_settings'] = {'irradiance_scale': 1000.0}
val = adapter.read_authored_irradiance_scale(sc)
assert val == 1000.0, f'expected 1000.0, got {val!r}'
del sc['heat_sim_settings']
assert adapter.read_authored_irradiance_scale(sc) is None
print('AUTHORED_SCALE_OK')
"""
    out = subprocess.run(
        [str(executable), "-b", "--python-expr", code],
        capture_output=True, text=True,
    )
    assert "AUTHORED_SCALE_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_all_zero_cached_albedo_is_ignored_and_rebaked(executable):
    code = r"""
import bpy, numpy as np
from visionsim.simulate.heatsim import register, irradiance_kernel
register()
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()
bpy.ops.mesh.primitive_grid_add(x_subdivisions=12, y_subdivisions=12, size=2.0)
obj = bpy.context.active_object
mat = bpy.data.materials.new('checker'); mat.use_nodes = True
nt = mat.node_tree; bsdf = nt.nodes.get('Principled BSDF')
ck = nt.nodes.new('ShaderNodeTexChecker'); ck.inputs['Scale'].default_value = 6.0
nt.links.new(ck.outputs['Color'], bsdf.inputs['Base Color'])
obj.data.materials.append(mat)
bpy.context.scene.render.engine = 'CYCLES'
try: bpy.context.scene.cycles.device = 'CPU'; bpy.context.scene.cycles.samples = 4
except Exception: pass
nv = len(obj.data.vertices)
# Stale all-zeros disk cache for this object must NOT be served; must re-bake.
amap = irradiance_kernel.get_or_bake_vertex_albedo(
    bpy.context.scene, [obj], texture_size=128,
    disk_cache={obj.name: np.zeros(nv, dtype=np.float64)})
alb = amap.get(obj.name)
assert alb is not None, 'albedo absent'
assert float(alb.std()) > 0.05, f'zeros cache was served instead of re-baking (std={alb.std()})'
print('ZERO_CACHE_IGNORED_OK', round(float(alb.mean()),3), round(float(alb.std()),3))
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code],
                         capture_output=True, text=True)
    assert "ZERO_CACHE_IGNORED_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_all_zero_attribute_albedo_is_ignored_and_rebaked(executable):
    code = r"""
import bpy, numpy as np
from visionsim.simulate.heatsim import register, irradiance_kernel
register()
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()
bpy.ops.mesh.primitive_grid_add(x_subdivisions=12, y_subdivisions=12, size=2.0)
obj = bpy.context.active_object
mat = bpy.data.materials.new('checker'); mat.use_nodes = True
nt = mat.node_tree; bsdf = nt.nodes.get('Principled BSDF')
ck = nt.nodes.new('ShaderNodeTexChecker'); ck.inputs['Scale'].default_value = 6.0
nt.links.new(ck.outputs['Color'], bsdf.inputs['Base Color'])
obj.data.materials.append(mat)
bpy.context.scene.render.engine = 'CYCLES'
try: bpy.context.scene.cycles.device = 'CPU'; bpy.context.scene.cycles.samples = 4
except Exception: pass
nv = len(obj.data.vertices)
# Stale all-zeros mesh attribute for this object must NOT be served; must re-bake.
mesh = obj.data
attr = mesh.attributes.new(name='albedo', type='FLOAT', domain='POINT')
attr.data.foreach_set('value', np.zeros(nv, dtype=np.float64))
amap = irradiance_kernel.get_or_bake_vertex_albedo(
    bpy.context.scene, [obj], texture_size=128)
alb = amap.get(obj.name)
assert alb is not None, 'albedo absent'
assert float(alb.std()) > 0.05, f'zeros attribute was served instead of re-baking (std={alb.std()})'
print('ZERO_ATTR_IGNORED_OK', round(float(alb.mean()),3), round(float(alb.std()),3))
"""
    out = subprocess.run([str(executable), "-b", "--python-expr", code],
                         capture_output=True, text=True)
    assert "ZERO_ATTR_IGNORED_OK" in out.stdout, out.stdout + "\n" + out.stderr


def test_transparent_and_shadowless_meshes_do_not_occlude(executable):
    """Glass panes, Cycles light portals and shadow-disabled props must not
    block shortwave shadow rays; opaque, alpha-cutout and partly-transmissive
    geometry must keep casting shadows."""
    code = r"""
import bpy
from visionsim.simulate.heatsim import irradiance_kernel as ik

def mesh(name):
    bpy.ops.mesh.primitive_plane_add()
    o = bpy.context.active_object
    o.name = name
    return o

def make_mat(name, build):
    # Build a material whose Surface output is driven by build's shader.
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    tree = m.node_tree
    out = next(n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL')
    # Drop the default Principled so only what `build` returns feeds the output.
    for n in [n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED']:
        tree.nodes.remove(n)
    tree.links.new(build(tree).outputs[0], out.inputs['Surface'])
    return m

def add(tree, bl_idname):
    return tree.nodes.new(bl_idname)

def glass(tree):
    return add(tree, 'ShaderNodeBsdfGlass')

def transparent(tree):
    return add(tree, 'ShaderNodeBsdfTransparent')

def principled(weight):
    def build(tree):
        n = add(tree, 'ShaderNodeBsdfPrincipled')
        for key in ('Transmission Weight', 'Transmission'):
            if key in n.inputs:
                n.inputs[key].default_value = weight
        return n
    return build

def cutout_leaf(tree):
    # Opaque leaf mixed with a Transparent BSDF - standard alpha cutout.
    mix = add(tree, 'ShaderNodeMixShader')
    tree.links.new(transparent(tree).outputs[0], mix.inputs[1])
    tree.links.new(principled(0.0)(tree).outputs[0], mix.inputs[2])
    return mix

def transparent_plus_diffuse(tree):
    mix = add(tree, 'ShaderNodeMixShader')
    tree.links.new(transparent(tree).outputs[0], mix.inputs[1])
    tree.links.new(add(tree, 'ShaderNodeBsdfDiffuse').outputs[0], mix.inputs[2])
    return mix

def driven_transmission(tree):
    # Transmission fed by a texture - unknowable statically, must be opaque.
    n = add(tree, 'ShaderNodeBsdfPrincipled')
    key = next(k for k in ('Transmission Weight', 'Transmission') if k in n.inputs)
    tree.links.new(add(tree, 'ShaderNodeTexNoise').outputs[0], n.inputs[key])
    return n

def check(name, build, should_occlude):
    o = mesh(name)
    if build is not None:
        o.data.materials.append(make_mat(name + '_m', build))
    got = ik._casts_shadow(o)
    assert got == should_occlude, f'{name}: expected occlude={should_occlude}, got {got}'
    return o

check('no_slots', None, True)
check('glass', glass, False)
check('portal', transparent, False)
check('clear_principled', principled(1.0), False)
check('frosted', principled(0.5), True)
check('opaque', principled(0.0), True)
check('cutout_leaf', cutout_leaf, True)
check('decal', transparent_plus_diffuse, True)
check('driven', driven_transmission, True)

# Glass wrapped in a node group must still read as clear (review finding).
group = bpy.data.node_groups.new('GlassGroup', 'ShaderNodeTree')
g_out = group.nodes.new('NodeGroupOutput')
group.interface.new_socket('Shader', in_out='OUTPUT', socket_type='NodeSocketShader')
group.links.new(group.nodes.new('ShaderNodeBsdfGlass').outputs[0], g_out.inputs[0])

def grouped_glass(tree):
    n = tree.nodes.new('ShaderNodeGroup')
    n.node_tree = group
    return n

check('grouped_glass', grouped_glass, False)

# An opaque BSDF hidden inside a group must still occlude.
ogroup = bpy.data.node_groups.new('OpaqueGroup', 'ShaderNodeTree')
og_out = ogroup.nodes.new('NodeGroupOutput')
ogroup.interface.new_socket('Shader', in_out='OUTPUT', socket_type='NodeSocketShader')
ogroup.links.new(ogroup.nodes.new('ShaderNodeBsdfDiffuse').outputs[0], og_out.inputs[0])

def grouped_opaque(tree):
    n = tree.nodes.new('ShaderNodeGroup')
    n.node_tree = ogroup
    return n

check('grouped_opaque', grouped_opaque, True)

# An empty material slot renders opaque, so [None, glass] must keep its shadow.
empty_and_glass = mesh('empty_and_glass')
empty_and_glass.data.materials.append(None)
empty_and_glass.data.materials.append(make_mat('eg_glass', glass))
assert ik._casts_shadow(empty_and_glass), 'None slot + glass must occlude'

# Volume-only material (Surface unlinked) has no surface to block a ray.
volbox = mesh('volbox')
vm = bpy.data.materials.new('vol_m')
vm.use_nodes = True
vt = vm.node_tree
vout = next(n for n in vt.nodes if n.type == 'OUTPUT_MATERIAL')
for n in [n for n in vt.nodes if n.type == 'BSDF_PRINCIPLED']:
    vt.nodes.remove(n)
vt.links.new(vt.nodes.new('ShaderNodeVolumePrincipled').outputs[0], vout.inputs['Volume'])
volbox.data.materials.append(vm)
assert not ik._casts_shadow(volbox), 'volume-only material must not occlude'

# An unconnected leftover Principled must not make a glass pane opaque.
stray = mesh('stray')
m = bpy.data.materials.new('stray_m')
m.use_nodes = True
tree = m.node_tree
out = next(n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL')
tree.links.new(tree.nodes.new('ShaderNodeBsdfGlass').outputs[0], out.inputs['Surface'])
stray.data.materials.append(m)
assert not ik._casts_shadow(stray), 'orphaned Principled must not force opacity'

# One opaque slot is enough to keep the whole object casting shadows.
mixed = mesh('mixed')
mixed.data.materials.append(make_mat('mixed_glass', glass))
mixed.data.materials.append(make_mat('mixed_opaque', principled(0.0)))
assert ik._casts_shadow(mixed), 'mixed slots must occlude'

noshadow = check('noshadow_base', principled(0.0), True)
noshadow.visible_shadow = False
assert not ik._casts_shadow(noshadow), 'visible_shadow=False must not occlude'

# The BVH collector must agree with _casts_shadow.
collected = ik._collect_scene_meshes_world(bpy.context.scene)
expected = sum(
    1 for o in bpy.context.scene.objects
    if o.type == 'MESH' and not o.hide_render and o.visible_get() and ik._casts_shadow(o)
)
assert len(collected) == expected, (len(collected), expected)
print('OCCLUDER_FILTER_OK')
"""
    out = subprocess.run(
        [str(executable), "-b", "--python-expr", code],
        capture_output=True, text=True,
    )
    assert "OCCLUDER_FILTER_OK" in out.stdout, out.stdout + "\n" + out.stderr
