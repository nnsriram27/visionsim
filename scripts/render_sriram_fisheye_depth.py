"""Render the central fisheye metric-depth sequence for disparity visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

from visionsim.simulate.blender import BlenderClients
from visionsim.simulate.config import DepthsConfig, RenderConfig
from visionsim.simulate.job import render_job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=Path, default=Path("/data/sriram/blender_files/sriram_room.blend"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/sriram/blender_renders/sriram_room/fisheye_stereo_final/central_geometry"),
    )
    parser.add_argument("--blender", type=Path, default=Path("/home/sriram/.local/bin/blender"))
    args = parser.parse_args()
    config = RenderConfig(
        executable=args.blender,
        camera_name="Room panorama - 160 degree fisheye",
        width=960,
        height=540,
        include_frames=False,
        include_depths=True,
        depths=DepthsConfig(preview=False, bit_depth=32, exr_codec="ZIP"),
        include_thermal=False,
        previews=False,
        max_samples=1,
        adaptive_threshold=0.0,
        use_denoising=False,
        use_motion_blur=False,
        device_type="optix",
        jobs=1,
        timeout=-1,
        log_dir=args.output / "logs",
    )
    with BlenderClients.spawn(
        jobs=1,
        log=config.log_dir,
        timeout=config.timeout,
        executable=config.executable,
        autoexec=config.autoexec,
    ) as clients:
        render_job(
            clients,
            args.blend.resolve(),
            args.output.resolve(),
            config,
            frame_start=1,
            frame_end=720,
            frame_step=1,
            update_fn=None,
        )


if __name__ == "__main__":
    main()
