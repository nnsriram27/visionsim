# NOTE: This needs to be imported by blender to work properly.

# Turbo colormap stop data sourced from heat-sim-blender temperature_viz.get_turbo_colormap() (@ 543ee81).

from __future__ import annotations

import bpy  # type: ignore

from .common import MAPRANGE_NODE, new_socket, set_clamp


def thermal_preview_node_group(tmin: float = 295.0, tmax: float = 400.0) -> bpy.types.NodeTree:
    """Compositor node group mapping a scalar temperature (K) to a turbo-coloured RGB image.

    Structure: ``Temperature`` float INPUT → MapRange([tmin, tmax] → [0, 1]) →
    ``ShaderNodeValToRGB`` with turbo colour stops → ``Image`` colour OUTPUT.

    Args:
        tmin: Minimum temperature in Kelvin — maps to the cool (dark blue) end of turbo.
        tmax: Maximum temperature in Kelvin — maps to the hot (dark red) end of turbo.

    Returns:
        A ``CompositorNodeTree`` named ``"ThermalPreview"``.
    """
    ng = bpy.data.node_groups.new(type="CompositorNodeTree", name="ThermalPreview")

    if bpy.app.version >= (4, 3, 0):
        ng.default_group_node_width = 140

    # -- Sockets -----------------------------------------------------------------
    # Output first so the interface order matches the socket panel.
    new_socket(ng, name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    new_socket(ng, name="Temperature", in_out="INPUT", socket_type="NodeSocketFloat")

    # -- Nodes -------------------------------------------------------------------
    group_input = ng.nodes.new("NodeGroupInput")
    group_input.name = "Group Input"

    group_output = ng.nodes.new("NodeGroupOutput")
    group_output.name = "Group Output"
    group_output.is_active_output = True

    # MapRange: Temperature (K) → [0, 1]
    map_range = ng.nodes.new(MAPRANGE_NODE)
    map_range.name = "TempNormalize"
    set_clamp(map_range, True)
    map_range.inputs[1].default_value = float(tmin)   # From Min
    map_range.inputs[2].default_value = float(tmax)   # From Max
    map_range.inputs[3].default_value = 0.0           # To Min
    map_range.inputs[4].default_value = 1.0           # To Max

    # Turbo colour ramp (ShaderNodeValToRGB is correct for the Blender ≥5.0 compositor).
    color_ramp = ng.nodes.new("ShaderNodeValToRGB")
    color_ramp.name = "TurboRamp"
    ramp = color_ramp.color_ramp
    ramp.interpolation = "LINEAR"

    # Turbo colormap stops — mirrors temperature_viz.get_turbo_colormap() exactly.
    _turbo_stops = [
        (0.00, (0.18995, 0.07176, 0.23217, 1.0)),  # dark blue
        (0.10, (0.25107, 0.25237, 0.63374, 1.0)),  # blue
        (0.20, (0.19943, 0.47510, 0.82373, 1.0)),  # light blue
        (0.30, (0.12394, 0.67124, 0.75369, 1.0)),  # cyan
        (0.40, (0.21960, 0.80968, 0.53235, 1.0)),  # green-cyan
        (0.50, (0.48137, 0.89499, 0.29948, 1.0)),  # green
        (0.60, (0.73563, 0.91465, 0.18511, 1.0)),  # yellow-green
        (0.70, (0.92989, 0.83247, 0.14869, 1.0)),  # yellow
        (0.80, (0.99324, 0.61749, 0.11615, 1.0)),  # orange
        (0.90, (0.96102, 0.37572, 0.11219, 1.0)),  # orange-red
        (1.00, (0.84620, 0.17942, 0.15089, 1.0)),  # dark red
    ]

    # Mirror colorize_indices_node_group: clear existing, resize, then fill.
    while len(ramp.elements) < len(_turbo_stops):
        ramp.elements.new(0.5)
    while len(ramp.elements) > len(_turbo_stops):
        ramp.elements.remove(ramp.elements[-1])
    for i, (pos, color) in enumerate(_turbo_stops):
        ramp.elements[i].position = pos
        ramp.elements[i].color = color

    # -- Locations ---------------------------------------------------------------
    group_input.location = (-300.0, 0.0)
    map_range.location = (-100.0, 0.0)
    color_ramp.location = (150.0, 0.0)
    group_output.location = (500.0, 0.0)

    # -- Links -------------------------------------------------------------------
    ng.links.new(group_input.outputs[0], map_range.inputs[0])
    ng.links.new(map_range.outputs[0], color_ramp.inputs[0])
    ng.links.new(color_ramp.outputs[0], group_output.inputs[0])

    return ng
