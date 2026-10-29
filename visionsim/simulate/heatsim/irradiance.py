"""Cycles baking and sampling for absorbed thermal flux."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import bpy  # type: ignore[import-not-found]
import numpy as np

from .material_slots import MaterialSlotState, restore_material_slots, set_material_override, snapshot_material_slots
from .names import ALBEDO_LAYER_NAME, BAKE_UV_LAYER_NAME, IRRADIANCE_LAYER_NAME
from .physics import CYCLES_LOUT_TO_IRRADIANCE
from .uv_utils import restore_uv_states, snapshot_uv_states


@dataclass
class BakedFluxMap:
    """Container for baked flux data coming from an image texture."""

    pixels: np.ndarray  # shape (H, W, 3) in linear space
    vertex_flux: np.ndarray  # per-vertex luminance * scale
    width: int
    height: int


def _ensure_uv_layer(obj):
    """Return an active UV layer or warn if missing."""
    mesh = obj.data
    if not mesh.uv_layers:
        scene = bpy.context.scene
        view_layer = bpy.context.view_layer
        prev_active = view_layer.objects.active
        prev_selection = [o for o in scene.objects if o.select_get()]
        try:
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            view_layer.objects.active = obj
            if obj.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            # Using 1.6% island margin to prevent texture bleeding artifacts
            bpy.ops.uv.smart_project(island_margin=0.016)
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"HeatSim: Failed to auto-unwrap UVs for {obj.name}: {exc}")
        finally:
            # Restore selection and active object/mode. Force OBJECT mode FIRST and
            # guard the operators: if smart_project raised above it left the context
            # in EDIT mode, and an unguarded object.select_all here would poll()-fail
            # and abort the whole scene's bake instead of just skipping this object.
            if getattr(bpy.context, "mode", "OBJECT") != "OBJECT":
                try:
                    bpy.ops.object.mode_set(mode="OBJECT")
                except Exception:
                    logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
            try:
                bpy.ops.object.select_all(action="DESELECT")
            except Exception:
                logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
            for o in prev_selection:
                try:
                    o.select_set(True)
                except Exception:
                    logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
            if prev_active:
                view_layer.objects.active = prev_active

    if not mesh.uv_layers:
        warnings.warn(f"HeatSim: {obj.name} has no UV map; cannot bake irradiance texture.")
        return None
    return mesh.uv_layers.active or mesh.uv_layers[0]


def prepare_object_bake_uv(obj: bpy.types.Object) -> None:
    """
    Ensure `BAKE_UV_LAYER_NAME` exists on the object and is unwrapped for baking.

    This mirrors the shared baking pipeline, but operates on a single object:
    - Create/activate `BAKE_UV_LAYER_NAME`
    - Smart Project unwrap into that layer to get a non-overlapping 0..1 bake UVs
    """
    if obj is None or obj.type != "MESH":
        return
    mesh = obj.data
    # An empty UV collection is falsy but still supports creating a bake layer.
    if mesh is None or getattr(mesh, "uv_layers", None) is None:
        return
    # Skip degenerate (zero-geometry) meshes: an empty object has no albedo to
    # bake, and bpy.ops.uv.smart_project.poll() fails on it in --background mode
    # (nothing to unwrap), which would otherwise abort the whole scene's bake.
    # Empty meshes cannot be baked.
    if len(mesh.polygons) == 0 or len(mesh.vertices) == 0:
        return

    # Ensure bake UV layer exists
    if BAKE_UV_LAYER_NAME not in mesh.uv_layers:
        try:
            mesh.uv_layers.new(name=BAKE_UV_LAYER_NAME)
        except Exception:  # noqa: BLE001
            return

    # Set bake UV active for unwrap + baking
    try:
        mesh.uv_layers[BAKE_UV_LAYER_NAME].active = True
        mesh.uv_layers[BAKE_UV_LAYER_NAME].active_render = True
    except Exception:
        logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
    ctx = bpy.context
    view_layer = ctx.view_layer
    prev_selection = list(getattr(ctx, "selected_objects", []) or [])
    prev_active = getattr(ctx, "active_object", None)

    try:
        # Establish a clean OBJECT-mode context before any selection operator.
        # object.select_all / mode_set poll() fail in --background when the context
        # is in EDIT mode with no valid active object (which happens on dense scenes
        # once any earlier object's unwrap misbehaves). Point 'active' at the object
        # we are about to bake first, so mode_set has a valid target to switch.
        if view_layer.objects.active is None:
            view_layer.objects.active = obj
        if getattr(bpy.context, "mode", "OBJECT") != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        view_layer.objects.active = obj

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        # Using 1.6% island margin to prevent texture bleeding artifacts.
        bpy.ops.uv.smart_project(island_margin=0.016)
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception as exc:  # noqa: BLE001
        # A per-object unwrap failure (smart_project.poll() and friends can fail in
        # --background on some meshes) must never abort the whole scene's bake. Warn
        # and keep whatever UVs the object already has; the albedo kernel treats a
        # missing/degenerate bake as full absorption, a documented fallback.
        warnings.warn(f"HeatSim: bake-UV prep failed for {obj.name}; using existing UVs. {exc}")
    finally:
        # Restore selection / active / mode best-effort, but always return to OBJECT mode:
        # this function enters EDIT mode above, and if smart_project raised, the next
        # object's object.select_all.poll() would fail in the leftover EDIT context.
        if getattr(bpy.context, "mode", "OBJECT") != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
        try:
            bpy.ops.object.select_all(action="DESELECT")
        except Exception:
            logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
        for o in prev_selection:
            try:
                o.select_set(True)
            except Exception:
                logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
        if prev_active and prev_active in getattr(ctx, "visible_objects", []):
            try:
                view_layer.objects.active = prev_active
            except Exception:
                logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)


def _ensure_bake_image(base_name: str, name_suffix: str, size: int):
    """Create or reuse a float image that receives baked data."""
    size = max(16, int(size))
    img_name = f"{base_name}_{name_suffix}"
    image = bpy.data.images.get(img_name)
    if image is None:
        image = bpy.data.images.new(img_name, width=size, height=size, alpha=False, float_buffer=True)
    else:
        if image.size[0] != size or image.size[1] != size:
            image.scale(size, size)
    # Prefer linear/non-color spaces; we treat all bake outputs numerically.
    preferred = ("Linear", "Linear Rec.709", "Non-Color")
    for name in preferred:
        if name in image.colorspace_settings.bl_rna.properties["name"].enum_items:
            image.colorspace_settings.name = name
            break
    # Clear previous contents
    if image.pixels:
        image.pixels.foreach_set([0.0] * (len(image.pixels)))
    return image


def _ensure_albedo_image(name_suffix: str, size: int):
    return _ensure_bake_image(ALBEDO_LAYER_NAME, name_suffix, size)


def _ensure_irradiance_image(name_suffix: str, size: int):
    return _ensure_bake_image(IRRADIANCE_LAYER_NAME, name_suffix, size)


def _prepare_image_nodes_for_bake(obj, image, node_name: str, node_label: str):
    """Add a fresh target to each private material, preserving copied source nodes."""
    seen = set()
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or mat.as_pointer() in seen:
            continue
        seen.add(mat.as_pointer())
        nodes = mat.node_tree.nodes
        # An authored texture can already use this name. Reusing it would change
        # the bake's source image, even though the material itself is private.
        img_node = nodes.new("ShaderNodeTexImage")
        img_node.name = node_name
        img_node.label = node_label
        img_node.image = image
        nodes.active = img_node
        for node in nodes:
            node.select = node == img_node


@dataclass
class _BakeMaterialUVOverride:
    """Private bake resources and the original assignments they replace."""

    objects: list[tuple[bpy.types.Object, MaterialSlotState]]
    materials: list[bpy.types.Material]
    node_groups: list[bpy.types.ShaderNodeTree]


def _pick_source_uv_for_object(
    obj: bpy.types.Object, uv_snapshot_map: dict[int, tuple[str | None, str | None]]
) -> str | None:
    """
    Choose the UV map name that represents the object's 'real' texturing UVs.
    Prefer the snapshot's active_render, then active, then any non-bake UV layer.
    """
    state = uv_snapshot_map.get(obj.as_pointer(), (None, None))
    active_name, render_name = state
    if render_name:
        return str(render_name)
    if active_name:
        return str(active_name)
    try:
        mesh = obj.data
        if mesh and getattr(mesh, "uv_layers", None):
            for uv in mesh.uv_layers:
                if uv.name != BAKE_UV_LAYER_NAME:
                    return uv.name
            if len(mesh.uv_layers) > 0:
                return mesh.uv_layers[0].name
    except Exception:
        logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
    return None


def _copy_bake_node_groups(mat: bpy.types.Material, state: _BakeMaterialUVOverride) -> None:
    """Copy nested groups while retaining sharing inside this private material."""
    copies = {}

    def copy_groups(tree):
        for node in tree.nodes:
            if node.type != "GROUP" or node.node_tree is None:
                continue
            source = node.node_tree
            key = source.as_pointer()
            if key not in copies:
                private = source.copy()
                private.use_fake_user = False
                # Register before recursion so partial-copy failures also unwind.
                state.node_groups.append(private)
                copies[key] = private
                copy_groups(private)
            node.node_tree = copies[key]

    copy_groups(mat.node_tree)


def _apply_uv_override_to_material(mat: bpy.types.Material, uv_name: str) -> None:
    """Pin top-level implicit image UVs in a private material.

    This retains the existing UV behavior. Linked coordinate paths, nested
    implicit UVs and tangent-space normal bases are handled in a separate change.
    """
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    uv_node = nodes.new("ShaderNodeUVMap")
    uv_node.name = f"HeatSim_OrigUV__{uv_name}"
    uv_node.label = "HeatSim Orig UV (Bake Override)"
    uv_node.uv_map = uv_name
    for node in nodes:
        if node.type != "TEX_IMAGE":
            continue
        vector = node.inputs.get("Vector")
        if vector is not None and not vector.is_linked:
            links.new(uv_node.outputs["UV"], vector)


def _install_bake_uv_material_overrides(objects: list[bpy.types.Object], uv_snapshot) -> _BakeMaterialUVOverride:
    """Isolate every changed mesh user from authored RGB materials and groups."""
    uv_states = {obj.as_pointer(): uv for obj, uv in (uv_snapshot or [])}
    mesh_states = {
        obj.data.as_pointer(): uv_states.get(obj.as_pointer(), (None, None))
        for obj in objects
        if obj is not None and obj.type == "MESH"
    }
    # Render-active UV is mesh state. Unselected users of that mesh also need
    # private source-UV materials while the target object's bake UV is active.
    affected = [obj for obj in bpy.data.objects if obj.type == "MESH" and obj.data.as_pointer() in mesh_states]
    for obj in affected:
        uv_states.setdefault(obj.as_pointer(), mesh_states[obj.data.as_pointer()])
    state = _BakeMaterialUVOverride(objects=[], materials=[], node_groups=[])
    copies = {}
    try:
        # A target slot on a bare shared mesh also appears on all its other users.
        # Capture every original assignment before installing the first override.
        state.objects = [(obj, snapshot_material_slots(obj)) for obj in affected]
        for obj, slots in state.objects:
            uv_name = _pick_source_uv_for_object(obj, uv_states)
            for index, source in enumerate(slots.materials or [None]):
                key = (source.as_pointer() if source is not None else None, uv_name)
                if key not in copies:
                    mat = source.copy() if source is not None else bpy.data.materials.new("HeatSim_Bake_Default")
                    state.materials.append(mat)
                    mat.use_fake_user = False
                    mat.use_nodes = True
                    _copy_bake_node_groups(mat, state)
                    if uv_name:
                        _apply_uv_override_to_material(mat, str(uv_name))
                    copies[key] = mat
                set_material_override(obj, index, copies[key])
    except Exception:
        _restore_bake_uv_material_overrides(state)
        raise
    return state


def _restore_bake_uv_material_overrides(state: _BakeMaterialUVOverride | None) -> None:
    if state is None:
        return
    errors = []
    # Clear every new object override before removing an originally bare shared
    # mesh's slots. Clearing the mesh first hides the other users' object arrays,
    # leaving references to temporary materials that can no longer be reached.
    for obj, slots in state.objects:
        if slots.data_materials:
            continue
        try:
            for slot in obj.material_slots:
                slot.link = "OBJECT"
                slot.material = None
                slot.link = "DATA"
        except Exception as exc:
            errors.append(exc)
            logging.getLogger(__name__).exception("Could not clear temporary bake slots on %s", obj.name)
    for obj, slots in state.objects:
        try:
            restore_material_slots(obj, slots)
        except Exception as exc:
            # Attempt every object even if one assignment could not be restored.
            errors.append(exc)
            logging.getLogger(__name__).exception("Could not restore bake material slots on %s", obj.name)
    for mat in state.materials:
        if mat.users:
            # Never unlink a still-assigned copy: doing so would clear that slot.
            errors.append(RuntimeError(f"Private bake material {mat.name!r} is still in use"))
        else:
            bpy.data.materials.remove(mat)
    for group in state.node_groups:
        # With all material copies released, references are between private groups
        # only. On a restore failure, keep groups that a retained copy still uses.
        if not errors or not group.users:
            bpy.data.node_groups.remove(group)
    if errors:
        raise RuntimeError("Could not completely restore private bake materials") from errors[0]


def _image_pixels_to_rgb(image) -> np.ndarray | None:
    """Convert a Blender image to (H, W, 3) numpy array."""
    w, h = image.size
    raw = np.array(image.pixels[:], dtype=np.float64)
    expected = w * h * 4
    if raw.size != expected:
        warnings.warn(f"HeatSim: Unexpected pixel count for baked image {image.name} ({raw.size} vs {expected}).")
        return None
    rgb = raw.reshape((h * w, 4))[:, :3]
    return rgb.reshape((h, w, 3))


def _bilinear_sample(rgb_pixels: np.ndarray, width: int, height: int, uv: np.ndarray) -> float:
    """Sample luminance from an RGB image at UV coordinates using bilinear filtering."""
    u = float(np.clip(uv[0], 0.0, 1.0))
    v = float(np.clip(uv[1], 0.0, 1.0))
    x = u * (width - 1)
    y = v * (height - 1)

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)

    tx = x - x0
    ty = y - y0

    c00 = rgb_pixels[y0, x0]
    c10 = rgb_pixels[y0, x1]
    c01 = rgb_pixels[y1, x0]
    c11 = rgb_pixels[y1, x1]

    c0 = c00 * (1 - tx) + c10 * tx
    c1 = c01 * (1 - tx) + c11 * tx
    rgb = c0 * (1 - ty) + c1 * ty
    return float(rgb @ np.array((0.2126, 0.7152, 0.0722), dtype=np.float64))


def _image_to_vertex_irradiance(
    loop_vertex_indices: np.ndarray,
    uv_data: np.ndarray,
    rgb_pixels: np.ndarray,
    width: int,
    height: int,
    scale: float,
    vert_count: int,
) -> np.ndarray:
    """Accumulate image luminance to vertices via UVs."""
    accum: np.ndarray = np.zeros(vert_count, dtype=np.float64)
    counts: np.ndarray = np.zeros(vert_count, dtype=np.int32)
    for loop_idx, vert_idx in enumerate(loop_vertex_indices):
        uv = uv_data[loop_idx]
        flux = _bilinear_sample(rgb_pixels, width, height, uv) * float(scale)
        accum[vert_idx] += flux
        counts[vert_idx] += 1

    counts[counts == 0] = 1
    return accum / counts


def _mesh_to_sample(obj):
    """Return the evaluated mesh used to index both Cycles vertex bakes.

    Sampling a pre-modifier mesh would misalign baked flux and albedo with the
    evaluated points passed to the solver. Fall back only when evaluated UVs or
    geometry are unavailable.
    """
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        candidate = obj.evaluated_get(depsgraph).data
        if candidate is not None and len(candidate.vertices) > 0 and len(candidate.uv_layers) > 0:
            return candidate
    except Exception:
        logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
    return obj.data


def bake_vertex_albedo(scene, obj, texture_size: int) -> np.ndarray:
    """Bake albedo and sample it on the evaluated mesh used by the heat solve."""
    from visionsim.simulate.heatsim.names import BAKE_UV_LAYER_NAME

    baked = bake_albedo_map(scene, obj, texture_size)
    if baked is None or baked.pixels is None:
        raise RuntimeError(f"Albedo bake failed for {obj.name!r}")
    mesh = _mesh_to_sample(obj)
    uv_layer = mesh.uv_layers.get(BAKE_UV_LAYER_NAME) or mesh.uv_layers.active
    if uv_layer is None:
        raise RuntimeError(f"No bake UV layer is available for {obj.name!r}")
    loop_count = len(mesh.loops)
    uv: np.ndarray = np.empty((loop_count, 2), dtype=np.float64)
    uv_layer.data.foreach_get("uv", uv.reshape(-1))
    vertex_indices: np.ndarray = np.empty(loop_count, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", vertex_indices)
    values = _image_to_vertex_irradiance(
        vertex_indices, uv, baked.pixels, baked.width, baked.height, 1.0, len(mesh.vertices)
    )
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"Albedo bake contains non-finite values for {obj.name!r}")
    return np.clip(values, 0.0, 1.0)


def bake_albedo_map(scene, obj, texture_size: int) -> BakedFluxMap | None:
    """
    Bake visible diffuse albedo (COLOR pass) for a single object.
    """
    if obj.type != "MESH":
        return None

    # Ensure object has source UVs for its real materials to sample from.
    if _ensure_uv_layer(obj) is None:
        return None

    # Snapshot UV state so we don't permanently change the user's UV selections.
    uv_snapshot = snapshot_uv_states([obj])

    # Ensure the bake UV exists and is unwrapped.
    prepare_object_bake_uv(obj)

    # Record which UV map this bake used (bake UV, matching shared pipeline).
    try:
        obj["heat_sim_albedo_uv"] = BAKE_UV_LAYER_NAME
    except Exception:
        logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
    image = _ensure_albedo_image(obj.name, texture_size)

    render = scene.render
    bake_settings = render.bake
    prev_engine = render.engine
    prev_settings = (
        getattr(bake_settings, "use_pass_direct", None),
        getattr(bake_settings, "use_pass_indirect", None),
        getattr(bake_settings, "use_pass_color", None),
        getattr(bake_settings, "target", None),
    )

    view_layer = scene.view_layers.active if hasattr(scene.view_layers, "active") else scene.view_layers[0]
    prev_active = view_layer.objects.active
    prev_selection = [o for o in scene.objects if o.select_get()]
    uv_override_state: _BakeMaterialUVOverride | None = None

    try:
        render.engine = "CYCLES"
        if hasattr(bake_settings, "use_pass_direct"):
            bake_settings.use_pass_direct = False
        if hasattr(bake_settings, "use_pass_indirect"):
            bake_settings.use_pass_indirect = False
        if hasattr(bake_settings, "use_pass_color"):
            bake_settings.use_pass_color = True
        if hasattr(bake_settings, "target"):
            bake_settings.target = "IMAGE_TEXTURES"

        # Keep textures sampling from the original UVs while baking into BAKE_UV_LAYER_NAME.
        uv_override_state = _install_bake_uv_material_overrides([obj], uv_snapshot)
        _prepare_image_nodes_for_bake(obj, image, "HeatSim_Albedo", "HeatSim Albedo")

        # object.select_all.poll() fails in --background when the context is in
        # EDIT mode; make sure we are in OBJECT mode (with a valid active object)
        # before selecting the bake target.
        if view_layer.objects.active is None:
            view_layer.objects.active = obj
        if getattr(bpy.context, "mode", "OBJECT") != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        view_layer.objects.active = obj

        with bpy.context.temp_override(scene=scene, view_layer=view_layer, active_object=obj, selected_objects=[obj]):
            bpy.ops.object.bake(
                type="DIFFUSE", pass_filter={"COLOR"}, target="IMAGE_TEXTURES", margin=8, margin_type="EXTEND"
            )

    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"HeatSim albedo bake failed for {obj.name}: {exc}")
        return None
    finally:
        render.engine = prev_engine
        if hasattr(bake_settings, "use_pass_direct") and prev_settings[0] is not None:
            bake_settings.use_pass_direct = prev_settings[0]
        if hasattr(bake_settings, "use_pass_indirect") and prev_settings[1] is not None:
            bake_settings.use_pass_indirect = prev_settings[1]
        if hasattr(bake_settings, "use_pass_color") and prev_settings[2] is not None:
            bake_settings.use_pass_color = prev_settings[2]
        if hasattr(bake_settings, "target") and prev_settings[3] is not None:
            bake_settings.target = prev_settings[3]

        try:
            _restore_bake_uv_material_overrides(uv_override_state)
        finally:
            restore_uv_states(uv_snapshot)

        # Guarded: this runs in a finally, so an unhandled poll() failure here
        # (context left in EDIT mode) would mask the real error and abort the
        # whole scene's bake.
        try:
            bpy.ops.object.select_all(action="DESELECT")
        except Exception:
            logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
        for item in prev_selection:
            try:
                item.select_set(True)
            except Exception:
                logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
        view_layer.objects.active = prev_active

    rgb_pixels = _image_pixels_to_rgb(image)
    if rgb_pixels is None:
        return None

    mesh = _mesh_to_sample(obj)
    mesh.calc_loop_triangles()
    if len(mesh.loop_triangles) == 0:
        return None

    # Use the bake UV layer for mapping baked pixels to vertices.
    uv_layer = mesh.uv_layers.get(BAKE_UV_LAYER_NAME) or (mesh.uv_layers.active or mesh.uv_layers[0])

    loop_indices: np.ndarray = np.zeros((len(mesh.loop_triangles), 3), dtype=np.int32)
    mesh.loop_triangles.foreach_get("loops", loop_indices.ravel())

    uv_data: np.ndarray = np.zeros((len(mesh.loops), 2), dtype=np.float64)
    uv_layer.data.foreach_get("uv", uv_data.ravel())

    loop_vertex_indices: np.ndarray = np.zeros(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vertex_indices)

    width, height = image.size
    vertex_luma = _image_to_vertex_irradiance(
        loop_vertex_indices,
        uv_data,
        rgb_pixels,
        width,
        height,
        1.0,
        len(mesh.vertices),
    )

    obj["heat_sim_albedo_image"] = image.name
    return BakedFluxMap(
        pixels=rgb_pixels,
        vertex_flux=vertex_luma,
        width=width,
        height=height,
    )


def bake_irradiance_map(scene, obj, texture_size: int, samples: int | None = None) -> BakedFluxMap | None:
    """Bake incident irradiance (DIFFUSE DIRECT+INDIRECT) for a single object.

    The mirror image of :func:`bake_albedo_map`: that bakes COLOR with the light
    passes off (a texture lookup, no light transport); this bakes the light passes
    with COLOR off, so the result is incoming light *independent of surface colour* -
    irradiance, not radiosity.

    Cycles includes emissive geometry, indirect lighting and world illumination.
    ``vertex_flux`` is incident W/m²; the caller multiplies by ``1 - albedo``
    to obtain absorbed flux.
    """
    if obj.type != "MESH":
        return None

    # Ensure object has source UVs for its real materials to sample from.
    if _ensure_uv_layer(obj) is None:
        return None

    # Snapshot UV state so we don't permanently change the user's UV selections.
    uv_snapshot = snapshot_uv_states([obj])

    # Ensure the bake UV exists and is unwrapped.
    prepare_object_bake_uv(obj)

    # Record which UV map this bake used (bake UV, matching shared pipeline).
    try:
        obj["heat_sim_flux_uv"] = BAKE_UV_LAYER_NAME
    except Exception:
        logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
    image = _ensure_irradiance_image(obj.name, texture_size)

    render = scene.render
    bake_settings = render.bake
    prev_engine = render.engine
    prev_settings = (
        getattr(bake_settings, "use_pass_direct", None),
        getattr(bake_settings, "use_pass_indirect", None),
        getattr(bake_settings, "use_pass_color", None),
        getattr(bake_settings, "target", None),
    )

    view_layer = scene.view_layers.active if hasattr(scene.view_layers, "active") else scene.view_layers[0]
    prev_active = view_layer.objects.active
    prev_selection = [o for o in scene.objects if o.select_get()]
    uv_override_state: _BakeMaterialUVOverride | None = None

    # Use the configured sample count for every bake texel, regardless of scene settings.
    cycles = getattr(scene, "cycles", None)
    prev_sampling = None
    if cycles is not None and samples is not None:
        prev_sampling = (
            getattr(cycles, "samples", None),
            getattr(cycles, "use_adaptive_sampling", None),
        )

    try:
        render.engine = "CYCLES"
        if prev_sampling is not None:
            if cycles is not None and hasattr(cycles, "use_adaptive_sampling"):
                cycles.use_adaptive_sampling = False
            if cycles is not None and hasattr(cycles, "samples"):
                cycles.samples = int(samples or 1)
        if hasattr(bake_settings, "use_pass_direct"):
            bake_settings.use_pass_direct = True
        if hasattr(bake_settings, "use_pass_indirect"):
            bake_settings.use_pass_indirect = True
        if hasattr(bake_settings, "use_pass_color"):
            bake_settings.use_pass_color = False
        if hasattr(bake_settings, "target"):
            bake_settings.target = "IMAGE_TEXTURES"

        # Keep textures sampling from the original UVs while baking into BAKE_UV_LAYER_NAME.
        uv_override_state = _install_bake_uv_material_overrides([obj], uv_snapshot)
        _prepare_image_nodes_for_bake(obj, image, "HeatSim_Irradiance", "HeatSim Irradiance")

        # object.select_all.poll() fails in --background when the context is in
        # EDIT mode; make sure we are in OBJECT mode (with a valid active object)
        # before selecting the bake target.
        if view_layer.objects.active is None:
            view_layer.objects.active = obj
        if getattr(bpy.context, "mode", "OBJECT") != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        view_layer.objects.active = obj

        with bpy.context.temp_override(scene=scene, view_layer=view_layer, active_object=obj, selected_objects=[obj]):
            bpy.ops.object.bake(
                type="DIFFUSE",
                pass_filter={"DIRECT", "INDIRECT"},
                target="IMAGE_TEXTURES",
                margin=8,
                margin_type="EXTEND",
            )

    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"HeatSim irradiance bake failed for {obj.name}: {exc}")
        return None
    finally:
        render.engine = prev_engine
        if prev_sampling is not None:
            if cycles is not None and prev_sampling[0] is not None and hasattr(cycles, "samples"):
                cycles.samples = prev_sampling[0]
            if cycles is not None and prev_sampling[1] is not None and hasattr(cycles, "use_adaptive_sampling"):
                cycles.use_adaptive_sampling = prev_sampling[1]
        if hasattr(bake_settings, "use_pass_direct") and prev_settings[0] is not None:
            bake_settings.use_pass_direct = prev_settings[0]
        if hasattr(bake_settings, "use_pass_indirect") and prev_settings[1] is not None:
            bake_settings.use_pass_indirect = prev_settings[1]
        if hasattr(bake_settings, "use_pass_color") and prev_settings[2] is not None:
            bake_settings.use_pass_color = prev_settings[2]
        if hasattr(bake_settings, "target") and prev_settings[3] is not None:
            bake_settings.target = prev_settings[3]

        try:
            _restore_bake_uv_material_overrides(uv_override_state)
        finally:
            restore_uv_states(uv_snapshot)

        # Guarded: this runs in a finally, so an unhandled poll() failure here
        # (context left in EDIT mode) would mask the real error and abort the
        # whole scene's bake.
        try:
            bpy.ops.object.select_all(action="DESELECT")
        except Exception:
            logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
        for item in prev_selection:
            try:
                item.select_set(True)
            except Exception:
                logging.getLogger(__name__).debug("Blender thermal operation failed", exc_info=True)
        view_layer.objects.active = prev_active

    rgb_pixels = _image_pixels_to_rgb(image)
    if rgb_pixels is None:
        return None

    mesh = _mesh_to_sample(obj)
    mesh.calc_loop_triangles()
    if len(mesh.loop_triangles) == 0:
        return None

    # Use the bake UV layer for mapping baked pixels to vertices.
    uv_layer = mesh.uv_layers.get(BAKE_UV_LAYER_NAME) or (mesh.uv_layers.active or mesh.uv_layers[0])

    loop_indices: np.ndarray = np.zeros((len(mesh.loop_triangles), 3), dtype=np.int32)
    mesh.loop_triangles.foreach_get("loops", loop_indices.ravel())

    uv_data: np.ndarray = np.zeros((len(mesh.loops), 2), dtype=np.float64)
    uv_layer.data.foreach_get("uv", uv_data.ravel())

    loop_vertex_indices: np.ndarray = np.zeros(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vertex_indices)

    width, height = image.size
    vertex_luma = _image_to_vertex_irradiance(
        loop_vertex_indices,
        uv_data,
        rgb_pixels,
        width,
        height,
        CYCLES_LOUT_TO_IRRADIANCE,
        len(mesh.vertices),
    )

    obj["heat_sim_irradiance_image"] = image.name
    return BakedFluxMap(
        pixels=rgb_pixels,
        vertex_flux=vertex_luma,
        width=width,
        height=height,
    )
