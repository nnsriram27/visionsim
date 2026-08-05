from __future__ import annotations

import math

import numpy as np
import pytest

from visionsim.simulate.heatsim import atlas


# ---------------------------------------------------------------------------
# surface_area_m2
# ---------------------------------------------------------------------------


def test_surface_area_of_unit_quad():
    # A 1000mm x 1000mm quad (two triangles) is exactly 1 m^2.
    verts_mm = np.array(
        [[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [1000.0, 1000.0, 0.0], [0.0, 1000.0, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    assert atlas.surface_area_m2(verts_mm, faces) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# select_for_atlas
# ---------------------------------------------------------------------------


def test_selection_excludes_dense_objects():
    density = 500.0
    # Orchid-like: 21k verts crammed into 2 m^2 -> ~10500 verts/m^2, already denser than target.
    assert atlas.select_for_atlas(n_verts=21_000, area_m2=2.0, density=density) is False
    # Floor-like: 16 verts spanning 80 m^2 -> 0.2 verts/m^2, far under target -> joins the atlas.
    assert atlas.select_for_atlas(n_verts=16, area_m2=80.0, density=density) is True


def test_selection_area_zero_uses_eps_guard():
    # area_m2=0 must not raise (division guarded by eps); any positive vertex count is "dense".
    assert atlas.select_for_atlas(n_verts=1, area_m2=0.0, density=500.0) is False
    # Zero verts and zero area: 0/eps = 0, which is < any positive density -> included.
    assert atlas.select_for_atlas(n_verts=0, area_m2=0.0, density=500.0) is True


# ---------------------------------------------------------------------------
# allocate: sizing
# ---------------------------------------------------------------------------


def test_tile_size_scales_with_area_and_respects_min_max_and_multiple_of_4():
    density = 1500.0
    areas = {
        "tiny": 0.01,  # K=15,   ceil(sqrt)=4,    mult4=4,    clamped up to tile_min=16
        "mid": 1.0,  # K=1500, ceil(sqrt)=39,   mult4=40,   no clamp
        "huge": 1000.0,  # K=1.5e6, ceil(sqrt)=1225, mult4=1228, clamped down to tile_max=512
    }
    layout = atlas.allocate(areas, density, tile_min=16, tile_max=512)

    assert layout.tiles["tiny"].size == (16, 16)
    assert layout.tiles["mid"].size == (40, 40)
    assert layout.tiles["huge"].size == (512, 512)
    for spec in layout.tiles.values():
        w, h = spec.size
        assert w == h  # tiles are square
        assert w % 4 == 0
        assert 16 <= w <= 512


def test_big_object_not_power_of_two_overshot():
    # 81 m^2 @ 1500/m^2: K=121500, ceil(sqrt)=349, rounded up to mult-of-4 = 352.
    # A naive power-of-two quantizer would jump straight to 512 (a ~2.1x-per-side, ~4.4x-area
    # overshoot); the spec explicitly forbids that.
    layout = atlas.allocate({"floor": 81.0}, 1500.0)
    side = layout.tiles["floor"].size[0]
    assert side == 352
    assert side != 512


# ---------------------------------------------------------------------------
# allocate: soft-max rescale
# ---------------------------------------------------------------------------


def test_soft_max_rescales_uniformly_and_warns():
    # area=100 m^2 @ density=100/m^2 -> K=10000 -> side=100 exactly (already mult of 4, no clamp),
    # so tiles_total == 10000 exactly, keeping the rescale arithmetic hand-checkable.
    areas = {"big": 100.0}
    density = 100.0
    soft_max = 5000

    with pytest.warns(UserWarning):
        layout = atlas.allocate(areas, density, soft_max=soft_max)

    assert layout.rescaled is True
    # ratio = soft_max / tiles_total = 5000/10000 = 0.5 -> effective_density = 50.0
    assert layout.effective_density == pytest.approx(50.0)
    assert layout.effective_density < density
    # New K = 100*50=5000 -> ceil(sqrt)=71 -> rounded up to mult-of-4 = 72.
    assert layout.tiles["big"].size == (72, 72)


def test_soft_max_counts_retained_vertices():
    areas = {"big": 100.0}  # K=10000 -> side=100 -> tiles_total=10000, same as above
    density = 100.0
    soft_max = 15_000

    # Without retained vertices, tiles_total (10000) is under soft_max (15000) -> no rescale.
    baseline = atlas.allocate(areas, density, soft_max=soft_max, retained_vertex_count=0)
    assert baseline.rescaled is False
    assert baseline.effective_density == pytest.approx(density)

    # With 6000 retained vertices, tiles_total+retained = 16000 > 15000 -> rescale must trigger,
    # even though tiles_total alone was fine. This is the "retained vertices count" contract.
    with pytest.warns(UserWarning):
        with_retained = atlas.allocate(areas, density, soft_max=soft_max, retained_vertex_count=6000)
    assert with_retained.rescaled is True
    # ratio = (soft_max - retained)/tiles_total = (15000-6000)/10000 = 0.9 -> density=90.0
    assert with_retained.effective_density == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# allocate: shelf packing
# ---------------------------------------------------------------------------


def _rects_have_padding_gap(r1, r2, padding):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    if x1 + w1 <= x2:
        xgap = x2 - (x1 + w1)
    elif x2 + w2 <= x1:
        xgap = x1 - (x2 + w2)
    else:
        xgap = None

    if y1 + h1 <= y2:
        ygap = y2 - (y1 + h1)
    elif y2 + h2 <= y1:
        ygap = y1 - (y2 + h2)
    else:
        ygap = None

    return (xgap is not None and xgap >= padding) or (ygap is not None and ygap >= padding)


def test_shelf_pack_no_overlap_and_padding():
    areas = {"a": 0.01, "b": 1.0, "c": 4.0, "d": 0.2, "e": 9.0}
    padding = 2
    layout = atlas.allocate(areas, density=1500.0, padding=padding)

    assert set(layout.tiles) == set(areas)
    rects = {name: (spec.offset[0], spec.offset[1], spec.size[0], spec.size[1]) for name, spec in layout.tiles.items()}

    names = list(rects)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert _rects_have_padding_gap(rects[names[i]], rects[names[j]], padding), (names[i], names[j])

    # Every tile must actually fit inside the reported atlas bounds.
    atlas_w, atlas_h = layout.atlas_size
    assert atlas_w % 4 == 0 and atlas_h % 4 == 0
    for x, y, w, h in rects.values():
        assert x >= 0 and y >= 0
        assert x + w <= atlas_w
        assert y + h <= atlas_h


def test_allocate_empty_areas():
    layout = atlas.allocate({}, density=500.0)
    assert layout.tiles == {}
    assert layout.atlas_size == (0, 0)
    assert layout.rescaled is False
    assert layout.effective_density == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# rasterize_tile
# ---------------------------------------------------------------------------


def _quad_mesh(size_mm=80.0):
    verts_mm = np.array(
        [[0.0, 0.0, 0.0], [size_mm, 0.0, 0.0], [size_mm, size_mm, 0.0], [0.0, size_mm, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    loop_uv = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ]
    )
    return verts_mm, faces, loop_uv


def test_rasterize_full_cover_quad():
    size_mm = 80.0
    tile = (8, 8)
    verts_mm, faces, loop_uv = _quad_mesh(size_mm)

    out = atlas.rasterize_tile(verts_mm, faces, loop_uv, tile)

    w, h = tile
    assert out["xy"].shape[0] == w * h
    # every texel covered exactly once
    seen = set(map(tuple, out["xy"].tolist()))
    assert seen == {(x, y) for x in range(w) for y in range(h)}

    # normals all point +Z (verts are CCW in the XY plane)
    assert np.allclose(out["normal"], np.array([0.0, 0.0, 1.0]), atol=1e-9)

    # positions interpolate linearly with texel-center UV
    for xy, pos in zip(out["xy"], out["position_mm"]):
        x, y = xy
        expected = np.array([(x + 0.5) / w * size_mm, (y + 0.5) / h * size_mm, 0.0])
        assert pos == pytest.approx(expected, abs=1e-9)

    assert np.isfinite(out["position_mm"]).all()
    assert np.isfinite(out["normal"]).all()


def test_rasterize_half_tile_triangle():
    tile = (8, 8)
    verts_mm = np.array([[0.0, 0.0, 0.0], [80.0, 0.0, 0.0], [0.0, 80.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    loop_uv = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])

    out = atlas.rasterize_tile(verts_mm, faces, loop_uv, tile)

    w, h = tile
    n = out["xy"].shape[0]
    # roughly half the tile, and strictly none of it
    assert 0 < n < w * h
    assert 20 <= n <= 44
    xs, ys = out["xy"][:, 0], out["xy"][:, 1]
    assert (xs >= 0).all() and (xs < w).all()
    assert (ys >= 0).all() and (ys < h).all()
    # no duplicate texels
    assert len(set(map(tuple, out["xy"].tolist()))) == n


def test_rasterize_no_double_claim():
    # Same abutting-triangle-pair setup as the full-cover test, but the assertion here is
    # specifically about the shared diagonal edge: no texel may be claimed by both triangles.
    tile = (16, 16)
    verts_mm, faces, loop_uv = _quad_mesh(160.0)

    out = atlas.rasterize_tile(verts_mm, faces, loop_uv, tile)

    w, h = tile
    xy_rows = list(map(tuple, out["xy"].tolist()))
    assert len(xy_rows) == w * h  # full coverage, no gaps
    assert len(set(xy_rows)) == len(xy_rows)  # no texel appears twice

    # Confirm both triangles actually contributed (i.e. this is a real two-triangle test, not one
    # triangle silently covering everything).
    assert set(out["face"].tolist()) == {0, 1}


def test_rasterize_degenerate_triangle_skipped():
    tile = (8, 8)
    verts_mm = np.array([[0.0, 0.0, 0.0], [80.0, 0.0, 0.0], [40.0, 0.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    # Colinear UVs -> zero UV-space area.
    loop_uv = np.array([[[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]]])

    out = atlas.rasterize_tile(verts_mm, faces, loop_uv, tile)

    assert out["xy"].shape[0] == 0
    assert out["position_mm"].shape == (0, 3)
    assert out["normal"].shape == (0, 3)
    assert out["face"].shape == (0,)
    for key in out:
        assert np.isfinite(out[key]).all() if out[key].size else True


def test_rasterize_degenerate_triangle_mixed_with_valid_no_nan():
    # A degenerate triangle followed by a valid one: the degenerate one must not poison the rest.
    tile = (8, 8)
    verts_mm = np.array(
        [[0.0, 0.0, 0.0], [80.0, 0.0, 0.0], [40.0, 0.0, 0.0], [80.0, 80.0, 0.0], [0.0, 80.0, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 3, 4]])
    loop_uv = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]],  # degenerate
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ]
    )

    out = atlas.rasterize_tile(verts_mm, faces, loop_uv, tile)

    assert out["xy"].shape[0] == 8 * 8
    assert np.isfinite(out["position_mm"]).all()
    assert np.isfinite(out["normal"]).all()
    assert 0 not in out["face"].tolist()  # the degenerate triangle (face 0) claims nothing


# ---------------------------------------------------------------------------
# dilate
# ---------------------------------------------------------------------------


def test_dilate_fills_margin_and_preserves_valid():
    image = np.zeros((5, 5), dtype=np.float64)
    valid = np.zeros((5, 5), dtype=bool)
    image[2, 2] = 10.0
    valid[2, 2] = True

    original = image.copy()
    out = atlas.dilate(image, valid, iterations=4)

    # The one originally-valid texel is untouched.
    assert out[2, 2] == pytest.approx(original[2, 2])

    # After one push-out pass, its 8 direct neighbors take the mean of valid 8-neighbors, i.e. 10.
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        assert out[2 + dy, 2 + dx] == pytest.approx(10.0)

    # With enough iterations (max Chebyshev distance from center on a 5x5 grid is 2), every texel
    # ends up filled - no leftover zeros from an unfilled/no-valid-neighbor texel.
    assert np.all(out > 0.0)

    # A corner (Chebyshev distance 2, reached only on the 2nd pass) should also end up close to
    # 10 given the small, uniform-source geometry here.
    assert out[0, 0] > 0.0


def test_dilate_preserves_valid_and_is_noop_when_all_valid():
    image = np.arange(9, dtype=np.float64).reshape(3, 3)
    valid = np.ones((3, 3), dtype=bool)
    out = atlas.dilate(image, valid, iterations=4)
    assert np.array_equal(out, image)


def test_dilate_zero_iterations_is_noop():
    image = np.array([[1.0, 0.0], [0.0, 0.0]])
    valid = np.array([[True, False], [False, False]])
    out = atlas.dilate(image, valid, iterations=0)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 1] == pytest.approx(0.0)
    assert out[1, 0] == pytest.approx(0.0)
    assert out[1, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# performance sanity (not a hard requirement of the brief's named tests, but the brief calls out
# a concrete target: a 352^2 tile with ~2k triangles well under a second)
# ---------------------------------------------------------------------------


def test_rasterize_performance_sanity():
    import time

    rng = np.random.default_rng(0)
    n_tris = 2000
    tile = (352, 352)

    # Build a grid of small quads (2 triangles each) tiling the UV unit square, so triangles are
    # spatially local (like real UV islands) rather than degenerate/overlapping garbage.
    grid_n = int(math.ceil(math.sqrt(n_tris / 2)))
    step = 1.0 / grid_n
    verts = []
    faces = []
    loop_uv = []
    for gy in range(grid_n):
        for gx in range(grid_n):
            if len(faces) >= n_tris:
                break
            u0, v0 = gx * step, gy * step
            u1, v1 = u0 + step, v0 + step
            base = len(verts)
            z = float(rng.uniform(-1.0, 1.0))
            verts.extend(
                [
                    [u0 * 1000, v0 * 1000, z],
                    [u1 * 1000, v0 * 1000, z],
                    [u1 * 1000, v1 * 1000, z],
                    [u0 * 1000, v1 * 1000, z],
                ]
            )
            faces.append([base, base + 1, base + 2])
            loop_uv.append([[u0, v0], [u1, v0], [u1, v1]])
            if len(faces) < n_tris:
                faces.append([base, base + 2, base + 3])
                loop_uv.append([[u0, v0], [u1, v1], [u0, v1]])

    verts_mm = np.array(verts)
    faces_arr = np.array(faces[:n_tris])
    loop_uv_arr = np.array(loop_uv[:n_tris])

    start = time.perf_counter()
    out = atlas.rasterize_tile(verts_mm, faces_arr, loop_uv_arr, tile)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"rasterize_tile took {elapsed:.3f}s for {n_tris} triangles on a {tile} tile"
    assert out["xy"].shape[0] > 0
