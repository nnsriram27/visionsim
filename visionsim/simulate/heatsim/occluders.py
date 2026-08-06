"""Which meshes block a shortwave (solar / lamp) shadow ray.

The thermal irradiance kernel integrates the shortwave band, and it builds a BVH
of occluders to decide what light reaches each surface. Deciding membership of
that set is a Blender-material question rather than a physics one, so it lives
here instead of in the vendored kernel:

* Geometry the artist marked as non-shadow-casting in Cycles is honoured.
* Clear geometry — glass panes, and the render-only "light portal" planes that
  let an exterior lamp into an interior — passes the beam. Real glass is opaque
  in the LWIR band but transmits most incident solar shortwave, which is the
  band being integrated.

Getting this wrong is asymmetric: wrongly skipping a mesh deletes a real shadow,
while wrongly keeping one starves every surface behind it. Anything that cannot
be resolved statically therefore reads as opaque.
"""

from __future__ import annotations

try:
    import bpy  # type: ignore
except ImportError:  # pragma: no cover - host interpreter without Blender
    bpy = None  # type: ignore

from typing import List, Optional

__all__ = ["casts_shadow", "material_is_clear", "surface_shader_nodes", "principled_is_clear"]


# A mesh only stops casting shadows once it is essentially clear glass; a
# partly-transmissive surface still blocks most of the beam.
_TRANSMISSION_OCCLUDER_CUTOFF = 0.95

_CLEAR_BSDF_NODES = frozenset({"BSDF_GLASS", "BSDF_TRANSPARENT", "BSDF_REFRACTION"})

# Shaders that put opaque energy on the surface. Their presence anywhere in the
# tree means the material is not clear glass, however it is mixed — this is what
# keeps alpha-cutout foliage (an opaque leaf mixed with a Transparent BSDF)
# casting shadows.
_OPAQUE_BSDF_NODES = frozenset({
    "BSDF_DIFFUSE", "BSDF_GLOSSY", "BSDF_ANISOTROPIC", "BSDF_VELVET", "BSDF_SHEEN",
    "BSDF_TOON", "BSDF_TRANSLUCENT", "BSDF_HAIR", "BSDF_HAIR_PRINCIPLED",
    "SUBSURFACE_SCATTERING", "EMISSION", "PRINCIPLED_VOLUME", "VOLUME_ABSORPTION",
    "VOLUME_SCATTER", "BACKGROUND",
})


def principled_is_clear(node: bpy.types.Node) -> bool:
    """Whether a Principled BSDF is driven fully transmissive by a constant.

    A linked transmission input cannot be evaluated statically, so it reads as
    opaque — the safe assumption, since guessing clear would delete a real
    shadow.
    """
    for key in ("Transmission Weight", "Transmission"):
        socket = node.inputs.get(key)
        if socket is None or socket.is_linked:
            continue
        try:
            if float(socket.default_value) >= _TRANSMISSION_OCCLUDER_CUTOFF:
                return True
        except (TypeError, ValueError):
            continue
    return False


def surface_shader_nodes(tree: bpy.types.NodeTree) -> Optional[List[bpy.types.Node]]:
    """Shader nodes actually reachable from the active Material Output's Surface.

    Walking the graph rather than scanning ``tree.nodes`` matters in both
    directions: Blender leaves an unconnected Principled node in every new
    material (which would otherwise read as opaque), and an alpha-cutout leaf
    genuinely routes an opaque shader into the output through a Mix Shader
    (which must read as opaque). The walk descends into node groups, since
    reusable glass/window shaders are commonly wrapped in one.

    Returns ``None`` when the material has no surface at all — no Material
    Output, or its Surface input left unlinked (a volume-only fog box, say).
    Cycles draws no surface there, so nothing can block a ray. That is distinct
    from an empty list, which means a surface exists but reached no shader.
    """
    outputs = [n for n in tree.nodes if n.type == "OUTPUT_MATERIAL"]
    if not outputs:
        return None
    active = next((n for n in outputs if getattr(n, "is_active_output", False)), outputs[0])
    surface = active.inputs.get("Surface")
    if surface is None or not surface.is_linked:
        return None

    seen: set = set()
    stack = [link.from_node for link in surface.links]
    reachable: List[bpy.types.Node] = []
    while stack:
        node = stack.pop()
        # Node names are only unique within a tree, so key on the tree too.
        key = (id(node.id_data), node.name)
        if key in seen:
            continue
        seen.add(key)
        reachable.append(node)

        group_tree = getattr(node, "node_tree", None)
        if node.type == "GROUP" and group_tree is not None:
            for inner in group_tree.nodes:
                if inner.type != "GROUP_OUTPUT":
                    continue
                for socket in inner.inputs:
                    for link in socket.links:
                        stack.append(link.from_node)

        for socket in node.inputs:
            for link in socket.links:
                stack.append(link.from_node)
    return reachable


def material_is_clear(mat: Optional[bpy.types.Material]) -> bool:
    """Whether ``mat`` passes essentially all shortwave light through.

    True only when the shaders feeding the surface output include a transmissive
    one and no opaque one.
    """
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return False
    nodes = surface_shader_nodes(mat.node_tree)
    if nodes is None:
        # No surface shader at all — Cycles renders nothing to block a ray.
        return True
    saw_clear = False
    for node in nodes:
        if node.type in _OPAQUE_BSDF_NODES:
            return False
        if node.type in _CLEAR_BSDF_NODES:
            saw_clear = True
        elif node.type == "BSDF_PRINCIPLED":
            if not principled_is_clear(node):
                return False
            saw_clear = True
    return saw_clear


def casts_shadow(obj: bpy.types.Object) -> bool:
    """Whether ``obj`` should block a shortwave (solar/lamp) shadow ray.

    Two classes of geometry are transparent to the beam even though they are
    renderable meshes, and treating them as solid starves every surface behind
    them:

    * Cycles ray-visibility has shadow casting switched off.
    * Every material slot is clear (glass panes, and the render-only "light
      portal" planes that let exterior lamps into an interior). Real glass is
      opaque in the LWIR band but transmits ~85% of incident solar shortwave,
      which is the band this kernel integrates.

    An object with no material slots is opaque by default.
    """
    if not getattr(obj, "visible_shadow", True):
        return False
    slots = [slot.material for slot in obj.material_slots]
    if not slots:
        return True
    # An empty slot renders with Blender's default opaque material, so faces
    # assigned to it cast a real shadow; material_is_clear(None) is False.
    return not all(material_is_clear(mat) for mat in slots)
