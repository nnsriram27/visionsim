from __future__ import annotations

import numpy as np

from visionsim.simulate.heatsim import cache


def test_cache_roundtrip_and_miss(tmp_path):
    key = cache.cache_key(tmp_path / "scene.blend", {"dt": 0.05, "domain": "POINTS"})
    assert isinstance(key, str) and key

    assert cache.read_temperatures(tmp_path, key) is None  # miss before write

    per_object = {"cup": np.full((4, 10), 295.0), "plate": np.full((4, 7), 296.0)}
    out = cache.write_temperatures(tmp_path, key, per_object, {"num_timesteps": 4})
    assert out.exists()

    back = cache.read_temperatures(tmp_path, key)
    assert back is not None
    assert set(back) == {"cup", "plate"}
    assert np.allclose(back["cup"], 295.0) and back["plate"].shape == (4, 7)
