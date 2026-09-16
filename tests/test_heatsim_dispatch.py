"""Host-side (no Blender) unit tests for the thermal include-dispatch in ``render_job``.

The parity test (``test_docstrings.test_output_configs``) only checks that the
``ThermalConfig`` dataclass and the ``exposed_include_thermal`` signature agree; it
does NOT exercise the job-level dispatch.  These tests close that gap by driving
``render_job`` with a fake client and asserting that ``config.include_thermal``
triggers ``prepare_thermal(**asdict(config.thermal))`` immediately followed by
``include_thermal(**asdict(config.thermal))`` (and that nothing fires when disabled).
"""

from __future__ import annotations

from dataclasses import asdict

from visionsim.simulate.config import RenderConfig
from visionsim.simulate.job import render_job


class _RecordingClient:
    """Fake Blender client: records every method call as ``(name, args, kwargs)``.

    Every attribute access resolves to a no-op recorder, so ``render_job`` can run
    end-to-end without a real Blender process while we inspect the call sequence.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        def _record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "common_animation_range":
                return range(1, 11)

        return _record


def test_render_job_dispatches_thermal_prepare_then_include():
    client = _RecordingClient()
    config = RenderConfig(include_thermal=True)

    render_job(client, "scene.blend", "out", config=config, dry_run=True)

    names = [name for name, _, _ in client.calls]
    assert "prepare_thermal" in names, names
    assert "include_thermal" in names, names

    # prepare_thermal must come *immediately* before include_thermal.
    i = names.index("prepare_thermal")
    assert names[i + 1] == "include_thermal", names

    expected = asdict(config.thermal)
    _prep_name, prep_args, prep_kwargs = client.calls[i]
    _incl_name, incl_args, incl_kwargs = client.calls[i + 1]
    assert prep_args == () and incl_args == ()
    assert prep_kwargs == expected
    assert incl_kwargs == expected


def test_render_job_skips_thermal_when_disabled():
    client = _RecordingClient()
    config = RenderConfig(include_thermal=False)

    render_job(client, "scene.blend", "out", config=config, dry_run=True)

    names = [name for name, _, _ in client.calls]
    assert "prepare_thermal" not in names, names
    assert "include_thermal" not in names, names


def test_render_job_selects_named_camera_after_initialize():
    client = _RecordingClient()
    config = RenderConfig(camera_name="Stereo Rig")

    render_job(client, "scene.blend", "out", config=config, dry_run=True)

    names = [name for name, _, _ in client.calls]
    assert names[:2] == ["initialize", "select_camera"]
    assert client.calls[1][1] == ("Stereo Rig",)


def test_render_job_preserves_blend_active_camera_by_default():
    client = _RecordingClient()

    render_job(client, "scene.blend", "out", config=RenderConfig(), dry_run=True)

    assert "select_camera" not in [name for name, _, _ in client.calls]


def test_camera_offset_only_bakes_requested_render_frames():
    client = _RecordingClient()
    config = RenderConfig(camera_offset=(-0.0325, 0.0, 0.0))

    render_job(
        client,
        "scene.blend",
        "out",
        config=config,
        frame_start=2,
        frame_end=8,
        frame_step=3,
        dry_run=True,
    )

    current_frames = [args[0] for name, args, _ in client.calls if name == "set_current_frame"]
    assert current_frames == [2, 5, 8, 2, 5, 8]
