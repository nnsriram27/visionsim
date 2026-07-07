from __future__ import annotations

from dataclasses import asdict

from visionsim.simulate.config import ThermalConfig


def test_thermal_config_animated_defaults():
    cfg = ThermalConfig()
    assert cfg.animated is False
    assert cfg.substeps_per_frame == 4
    assert cfg.frame_start is None
    assert cfg.frame_end is None
    assert cfg.every_n_frames == 1

    fields = asdict(cfg)
    for key in ("animated", "substeps_per_frame", "frame_start", "frame_end", "every_n_frames"):
        assert key in fields
