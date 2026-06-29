"""Thermal AOV + gray-body radiance shader support for visionsim.

Provides four entry points consumed by the thermal render pipeline:

* :func:`setup_temperature_aov`      — register a ``temperature`` value AOV on a view-layer
                                       and append ``Attribute → ShaderNodeOutputAOV`` to
                                       every material; returns the compositor socket name.
* :func:`enter_thermal_scene`        — swap scene to gray-body emission materials, disable
                                       lights, set a gray world; returns a restore state dict.
* :func:`restore_scene`              — undo :func:`enter_thermal_scene` from the state dict.
* :func:`stamp_default_temperatures` — stamp per-object ``heatsim_default_temperature``
                                       OBJECT-domain custom properties as a shader fallback.

Port of heat-sim-blender ``addon/lib/visualization.py`` (@ 543ee81) into visionsim
conventions: guarded ``bpy`` import, rich logger, Google-style docstrings, ruff 121 chars.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import bpy  # type: ignore
except ImportError:
    bpy = None  # type: ignore

_log = logging.getLogger("rich")

# Stefan-Boltzmann constant (SI, W/m²·K⁴).  Used as a magnitude knob × radiance_scale
# in the shader.  Do NOT unify with the solver σ (constants.py uses mm-scaled W/mm²·K⁴).
_SIGMA_SI: float = 5.670374419e-8

# Defaults used when the scene provides no overrides.
_DEFAULT_EMISSIVITY: float = 0.9
_AMBIENT_K: float = 295.372
_THERMAL_WORLD_NAME: str = "HeatSim_Thermal_World"

# Keys inside the opaque state dict returned by enter_thermal_scene.
_KEY_WORLD: str = "orig_world"
_KEY_MATERIALS: str = "orig_materials"
_KEY_LIGHT_HIDE_RENDER: str = "light_hide_render"
_KEY_LIGHT_HIDE_VIEWPORT: str = "light_hide_viewport"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_gray_body_material(radiance_scale: float) -> Any:
    """Create (or return a cached) gray-body emission material for thermal rendering.

    Shader graph — gray-body Kirchhoff (ε + ρ = 1) with Stefan-Boltzmann T⁴ emission:

        T_eff ─→ POWER(4) ─→ MUL(σ) ─→ MUL(radiance_scale) ─→ Emission(Color=1, Str)─┐
                                                                                        ├─ Mix(Fac=1-ε) ─→ Out
                                                               Diffuse(Color=1, R=0) ──┘

    ``T_eff`` is read from per-vertex ``sim_temperature`` (falling back to the per-object
    ``heatsim_default_temperature`` custom property when the attribute is absent).
    Emissivity is a fixed constant ``_DEFAULT_EMISSIVITY`` baked into the Mix Fac.

    Args:
        radiance_scale: Scalar multiplier after σ·T⁴ — tune this to adjust rendered
            radiance brightness relative to physical W/m² magnitudes.

    Returns:
        A ``bpy.types.Material`` configured for gray-body thermal rendering.
    """
    mat_name = f"HeatSim_ThermalShader_vs_{radiance_scale:.6g}"
    existing = bpy.data.materials.get(mat_name)
    if existing is not None:
        return existing

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # -- Per-vertex temperature (zero when attribute absent) --------------------
    temp_attr = nodes.new("ShaderNodeAttribute")
    temp_attr.attribute_name = "sim_temperature"
    temp_attr.attribute_type = "GEOMETRY"
    temp_attr.location = (-800.0, 300.0)

    # -- Per-object fallback temperature (stamped by stamp_default_temperatures) -
    default_temp_attr = nodes.new("ShaderNodeAttribute")
    default_temp_attr.attribute_name = "heatsim_default_temperature"
    default_temp_attr.attribute_type = "OBJECT"
    default_temp_attr.location = (-800.0, 500.0)

    # is_valid = sim_temperature > 1.0  (a real physical Kelvin value)
    temp_is_valid = nodes.new("ShaderNodeMath")
    temp_is_valid.operation = "GREATER_THAN"
    temp_is_valid.location = (-600.0, 400.0)
    temp_is_valid.inputs[1].default_value = 1.0
    links.new(temp_attr.outputs["Fac"], temp_is_valid.inputs[0])

    # delta = sim_temperature - default_T
    temp_delta = nodes.new("ShaderNodeMath")
    temp_delta.operation = "SUBTRACT"
    temp_delta.location = (-600.0, 260.0)
    links.new(temp_attr.outputs["Fac"], temp_delta.inputs[0])
    links.new(default_temp_attr.outputs["Fac"], temp_delta.inputs[1])

    # scaled = is_valid × delta  (zero when the per-vertex attr is absent)
    temp_scaled = nodes.new("ShaderNodeMath")
    temp_scaled.operation = "MULTIPLY"
    temp_scaled.location = (-440.0, 330.0)
    links.new(temp_is_valid.outputs["Value"], temp_scaled.inputs[0])
    links.new(temp_delta.outputs["Value"], temp_scaled.inputs[1])

    # T_effective = default_T + scaled  (= sim_temperature when valid, else default_T)
    temp_effective = nodes.new("ShaderNodeMath")
    temp_effective.operation = "ADD"
    temp_effective.location = (-280.0, 430.0)
    links.new(default_temp_attr.outputs["Fac"], temp_effective.inputs[0])
    links.new(temp_scaled.outputs["Value"], temp_effective.inputs[1])

    # -- Stefan-Boltzmann T⁴ chain ---------------------------------------------
    temp_pow4 = nodes.new("ShaderNodeMath")
    temp_pow4.operation = "POWER"
    temp_pow4.location = (-100.0, 430.0)
    temp_pow4.inputs[1].default_value = 4.0
    links.new(temp_effective.outputs["Value"], temp_pow4.inputs[0])

    sigma_mul = nodes.new("ShaderNodeMath")
    sigma_mul.operation = "MULTIPLY"
    sigma_mul.location = (80.0, 430.0)
    sigma_mul.inputs[1].default_value = _SIGMA_SI
    links.new(temp_pow4.outputs["Value"], sigma_mul.inputs[0])

    scale_mul = nodes.new("ShaderNodeMath")
    scale_mul.operation = "MULTIPLY"
    scale_mul.location = (260.0, 430.0)
    scale_mul.inputs[1].default_value = float(radiance_scale)
    links.new(sigma_mul.outputs["Value"], scale_mul.inputs[0])

    # -- Gray-body shader: Mix(Fac=1-ε, Emission, Diffuse) ----------------------
    # Cycles Mix Shader: out = (1-Fac)·A + Fac·B
    # With Fac=1-ε, A=Emission, B=Diffuse: out = ε·σT⁴·scale + (1-ε)·L_in  ✓
    emission = nodes.new("ShaderNodeEmission")
    emission.location = (440.0, 430.0)
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    links.new(scale_mul.outputs["Value"], emission.inputs["Strength"])

    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.location = (440.0, 200.0)
    diffuse.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    diffuse.inputs["Roughness"].default_value = 0.0  # Lambertian

    mix_shader = nodes.new("ShaderNodeMixShader")
    mix_shader.location = (640.0, 340.0)
    # Fac = 1 - ε so the Emission slot gets weight ε and Diffuse gets weight 1-ε.
    mix_shader.inputs["Fac"].default_value = 1.0 - _DEFAULT_EMISSIVITY
    links.new(emission.outputs["Emission"], mix_shader.inputs[1])
    links.new(diffuse.outputs["BSDF"], mix_shader.inputs[2])

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (840.0, 340.0)
    links.new(mix_shader.outputs["Shader"], out.inputs["Surface"])

    mat["heatsim_thermal_radiance_scale"] = float(radiance_scale)
    mat["heatsim_thermal_default_emissivity"] = _DEFAULT_EMISSIVITY
    return mat


def _append_temperature_aov_nodes(mat: Any, aov_name: str) -> None:
    """Append a temperature value chain → ``ShaderNodeOutputAOV(aov_name)`` to *mat*.

    The AOV mirrors the gray-body shader's is-valid/default blend rather than wiring
    ``sim_temperature`` straight through: where the per-vertex ``sim_temperature``
    attribute is missing/invalid (≤ 1 K) the AOV reports the per-object
    ``heatsim_default_temperature`` instead of 0 K, so ``temperature/`` and
    ``thermal_radiance/`` agree for un-simulated meshes.

        T_eff = default_T + (sim_temperature > 1) · (sim_temperature − default_T)

    Idempotent: skips materials that already have an OutputAOV node with the same name.
    """
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Skip if already wired.
    for node in nodes:
        if node.type == "OUTPUT_AOV":
            if getattr(node, "aov_name", None) == aov_name or node.name == aov_name:
                return

    # Track nodes we add so we can clean up on a mid-build failure.
    added: list[Any] = []

    def _new(node_type: str) -> Any:
        node = nodes.new(node_type)
        added.append(node)
        return node

    try:
        # -- per-vertex temperature (0 when the attribute is absent) ------------
        temp_attr = _new("ShaderNodeAttribute")
        temp_attr.attribute_name = "sim_temperature"
        temp_attr.attribute_type = "GEOMETRY"
        temp_attr.location = (-600.0, -400.0)

        # -- per-object fallback temperature (stamp_default_temperatures) -------
        default_temp_attr = _new("ShaderNodeAttribute")
        default_temp_attr.attribute_name = "heatsim_default_temperature"
        default_temp_attr.attribute_type = "OBJECT"
        default_temp_attr.location = (-600.0, -560.0)

        # is_valid = sim_temperature > 1.0
        temp_is_valid = _new("ShaderNodeMath")
        temp_is_valid.operation = "GREATER_THAN"
        temp_is_valid.location = (-420.0, -360.0)
        temp_is_valid.inputs[1].default_value = 1.0
        links.new(temp_attr.outputs["Fac"], temp_is_valid.inputs[0])

        # delta = sim_temperature - default_T
        temp_delta = _new("ShaderNodeMath")
        temp_delta.operation = "SUBTRACT"
        temp_delta.location = (-420.0, -520.0)
        links.new(temp_attr.outputs["Fac"], temp_delta.inputs[0])
        links.new(default_temp_attr.outputs["Fac"], temp_delta.inputs[1])

        # scaled = is_valid * delta  (zero when sim_temperature is absent)
        temp_scaled = _new("ShaderNodeMath")
        temp_scaled.operation = "MULTIPLY"
        temp_scaled.location = (-240.0, -440.0)
        links.new(temp_is_valid.outputs["Value"], temp_scaled.inputs[0])
        links.new(temp_delta.outputs["Value"], temp_scaled.inputs[1])

        # T_effective = default_T + scaled
        temp_effective = _new("ShaderNodeMath")
        temp_effective.operation = "ADD"
        temp_effective.location = (-60.0, -400.0)
        links.new(default_temp_attr.outputs["Fac"], temp_effective.inputs[0])
        links.new(temp_scaled.outputs["Value"], temp_effective.inputs[1])

        aov_node = _new("ShaderNodeOutputAOV")
    except Exception as exc:
        _log.debug("Could not build temperature AOV chain on %r: %s", mat.name, exc)
        for node in added:
            try:
                nodes.remove(node)
            except Exception:
                pass
        return

    aov_node.name = aov_name
    if hasattr(aov_node, "aov_name"):
        try:
            aov_node.aov_name = aov_name
        except Exception:
            pass
    aov_node.location = (140.0, -400.0)

    sock = aov_node.inputs.get("Value") or (aov_node.inputs[0] if aov_node.inputs else None)
    if sock is not None:
        try:
            links.new(temp_effective.outputs["Value"], sock)
        except Exception as exc:
            _log.debug("Could not link AOV socket on %r: %s", mat.name, exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_temperature_aov(scene: Any, view_layer: Any) -> str:
    """Register a ``temperature`` value AOV on *view_layer* and wire it into scene materials.

    For every mesh material in the scene, appends an
    ``Attribute("sim_temperature") → ShaderNodeOutputAOV("temperature")`` chain so that
    Cycles writes raw temperature values into the AOV render pass.

    Args:
        scene: The Blender scene (``bpy.types.Scene``).
        view_layer: The view layer on which to register the AOV (``bpy.types.ViewLayer``).

    Returns:
        The compositor socket name ``"temperature"`` — wire this to the
        ``CompositorNodeOutputFile`` input for the temperature render pass.
    """
    aov_name = "temperature"

    # 1. Register the AOV on the view layer (idempotent).
    existing_names: set[str] = set()
    try:
        existing_names = {a.name for a in view_layer.aovs}
    except Exception:
        pass

    if aov_name not in existing_names:
        try:
            aov = view_layer.aovs.add()
            aov.name = aov_name
            if hasattr(aov, "type"):
                try:
                    aov.type = "VALUE"
                except Exception:
                    try:
                        aov.type = "FLOAT"
                    except Exception:
                        pass
        except Exception as exc:
            _log.warning("Could not register temperature AOV on view layer: %s", exc)

    # 2. Append Attribute → OutputAOV chain to every material-using mesh in the scene.
    patched = 0
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or not mat.use_nodes:
                continue
            _append_temperature_aov_nodes(mat, aov_name)
            patched += 1

    _log.debug("setup_temperature_aov: AOV %r registered; %d material slot(s) patched", aov_name, patched)
    return aov_name


def enter_thermal_scene(scene: Any, *, radiance_scale: float) -> dict:
    """Swap *scene* into thermal rendering state for a gray-body radiance pass.

    Replaces every mesh object's materials with the gray-body emission material,
    hides all light objects from render, and sets a uniform-background thermal world.
    Returns an opaque state dict; pass it to :func:`restore_scene` to undo all changes.

    Args:
        scene: The Blender scene.
        radiance_scale: Multiplier applied after σ·T⁴; controls the brightness of the
            rendered radiance image relative to physical W/m² values.

    Returns:
        An opaque state dict for passing to :func:`restore_scene`.

    Note:
        Self-restoring: the ``state`` dict is populated *as* the scene is mutated,
        and any exception mid-swap (e.g. ``materials.clear()`` raising on
        library-linked/overridden mesh data) triggers a full :func:`restore_scene`
        rollback before the error is re-raised.  This guarantees the scene is never
        left half-swapped even though the caller has not yet received ``state``.
    """
    state: dict[str, Any] = {}

    # Bind the (mutable) record containers into ``state`` up-front so a rollback
    # during the loops below sees everything recorded so far.
    orig_materials: dict[str, list[Any]] = {}
    hide_render: dict[str, bool] = {}
    hide_viewport: dict[str, bool] = {}
    state[_KEY_MATERIALS] = orig_materials
    state[_KEY_LIGHT_HIDE_RENDER] = hide_render
    state[_KEY_LIGHT_HIDE_VIEWPORT] = hide_viewport

    try:
        thermal_mat = _build_gray_body_material(radiance_scale)

        # -- Save per-object material assignments and swap to thermal material --
        # Record AFTER clear() succeeds: if clear() raises on a non-editable mesh
        # the object is left untouched and is (correctly) not in the restore map,
        # so the rollback never double-clears it.
        for obj in scene.objects:
            if obj.type != "MESH":
                continue
            mat_names = [slot.material.name if slot.material else None for slot in obj.material_slots]
            obj.data.materials.clear()
            orig_materials[obj.name] = mat_names
            obj.data.materials.append(thermal_mat)

        # -- Save and disable all lights (hide from render + viewport) ----------
        for obj in scene.objects:
            if obj.type == "LIGHT":
                hide_render[obj.name] = bool(obj.hide_render)
                hide_viewport[obj.name] = bool(obj.hide_viewport)
                obj.hide_render = True
                obj.hide_viewport = True

        # -- Save world and replace with a uniform gray thermal world -----------
        orig_world = scene.world
        state[_KEY_WORLD] = orig_world.name if orig_world is not None else None

        thermal_world = bpy.data.worlds.get(_THERMAL_WORLD_NAME)
        if thermal_world is None:
            thermal_world = bpy.data.worlds.new(_THERMAL_WORLD_NAME)
            thermal_world.use_nodes = True
            wnodes = thermal_world.node_tree.nodes
            wlinks = thermal_world.node_tree.links
            wnodes.clear()
            bg = wnodes.new("ShaderNodeBackground")
            bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            bg.inputs["Strength"].default_value = _AMBIENT_K
            wout = wnodes.new("ShaderNodeOutputWorld")
            wlinks.new(bg.outputs["Background"], wout.inputs["Surface"])
        scene.world = thermal_world
    except Exception:
        # Roll back every change recorded so far, then re-raise so the caller sees
        # the original failure (with the scene already restored, not half-swapped).
        restore_scene(scene, state)
        raise

    return state


def restore_scene(scene: Any, state: dict) -> None:
    """Restore *scene* to the state captured by :func:`enter_thermal_scene`.

    Args:
        scene: The Blender scene (must be the same scene passed to
            :func:`enter_thermal_scene`).
        state: The opaque dict returned by :func:`enter_thermal_scene`.
    """
    # -- Restore per-object materials ------------------------------------------
    orig_materials: dict[str, list[Any]] = state.get(_KEY_MATERIALS, {})
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        saved = orig_materials.get(obj.name)
        if saved is None:
            continue
        obj.data.materials.clear()
        for mat_name in saved:
            mat = bpy.data.materials.get(mat_name) if mat_name else None
            obj.data.materials.append(mat)

    # -- Restore light visibility -----------------------------------------------
    hide_render: dict[str, bool] = state.get(_KEY_LIGHT_HIDE_RENDER, {})
    hide_viewport: dict[str, bool] = state.get(_KEY_LIGHT_HIDE_VIEWPORT, {})
    for obj in scene.objects:
        if obj.type == "LIGHT":
            if obj.name in hide_render:
                obj.hide_render = hide_render[obj.name]
            if obj.name in hide_viewport:
                obj.hide_viewport = hide_viewport[obj.name]

    # -- Restore world ----------------------------------------------------------
    orig_world_name = state.get(_KEY_WORLD)
    if orig_world_name is not None:
        w = bpy.data.worlds.get(orig_world_name)
        if w is not None:
            scene.world = w
    elif _KEY_WORLD in state:
        # World was None before enter_thermal_scene.
        scene.world = None  # type: ignore[assignment]


def stamp_default_temperatures(scene: Any, *, default_K: float) -> None:
    """Stamp the ``heatsim_default_temperature`` custom property on every mesh object.

    The gray-body emission shader reads this as a fallback temperature wherever the
    per-vertex ``sim_temperature`` attribute is absent (objects not participating in
    the FEM solve, or fluid meshes regenerated each frame by Mantaflow).  Cheap to
    call every frame.

    Args:
        scene: The Blender scene.
        default_K: Fallback temperature in Kelvin to stamp on all mesh objects.
    """
    t = float(default_K)
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        obj["heatsim_default_temperature"] = t
