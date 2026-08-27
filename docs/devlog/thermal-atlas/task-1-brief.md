### Task 1: `atlas.py` — selection, allocation, packing, rasterization (pure numpy)

**Files:** Create `visionsim/simulate/heatsim/atlas.py`; Test `tests/test_heatsim_atlas.py`.

**Produces (exact signatures):**
```python
@dataclass(frozen=True)
class TileSpec:
    obj_name: str
    size: tuple[int, int]        # (w, h) texels, multiples of 4, tile_min<=side<=tile_max
    offset: tuple[int, int]      # (x, y) in atlas pixels
@dataclass(frozen=True)
class AtlasLayout:
    atlas_size: tuple[int, int]
    tiles: dict[str, TileSpec]
    effective_density: float     # after any soft-max rescale
    rescaled: bool

def surface_area_m2(verts_mm: np.ndarray, faces: np.ndarray) -> float
def select_for_atlas(n_verts: int, area_m2: float, density: float) -> bool
    # True iff n_verts / max(area_m2, eps) < density
def allocate(areas: dict[str, float], density: float, *, tile_min: int = 16,
             tile_max: int = 512, soft_max: int = 500_000,
             retained_vertex_count: int = 0, padding: int = 2) -> AtlasLayout
    # texel count = area*density; side = ceil(sqrt(K)) rounded UP to multiple of 4,
    # clamped [tile_min, tile_max]; if sum(sides^2)+retained > soft_max: rescale
    # density uniformly (sqrt factor on sides), set rescaled=True, warnings.warn.
    # Shelf-pack sorted by height desc; atlas_size grows to fit (multiple of 4).
def rasterize_tile(verts_mm, faces, loop_uv, tile_size) -> dict[str, np.ndarray]
    # loop_uv: (n_faces, 3, 2) triangle UVs in [0,1] tile-local space.
    # Rasterize each triangle over texel centers ((x+.5)/W, (y+.5)/H); half-open
    # edge rule; later triangles do NOT overwrite earlier ones (first hit wins).
    # Returns {"xy": (K,2) int, "position_mm": (K,3), "normal": (K,3) unit face
    # normals, "face": (K,) int} for covered texels only.
def dilate(image: np.ndarray, valid: np.ndarray, iterations: int = 4) -> np.ndarray
    # Push-out: each invalid texel takes the mean of valid 8-neighbors; repeat.
```

**Test cases to write first (names are the contract):**
- `test_surface_area_of_unit_quad` (two triangles, 1 m² from mm coords)
- `test_selection_excludes_dense_objects` (orchid-like: high verts/area → False; floor-like 16 verts/80 m² → True)
- `test_tile_size_scales_with_area_and_respects_min_max_and_multiple_of_4`
- `test_big_object_not_power_of_two_overshot` (81 m² @1500/m² → side ≈ 352, NOT 512)
- `test_soft_max_rescales_uniformly_and_warns` (pytest.warns; effective_density < requested; rescaled=True) and `test_soft_max_counts_retained_vertices`
- `test_shelf_pack_no_overlap_and_padding` (pairwise tile rect disjointness incl. padding)
- `test_rasterize_full_cover_quad` (quad covering whole tile → every texel covered, positions interpolate linearly, normals unit +Z)
- `test_rasterize_half_tile_triangle` (~half texels covered, none outside)
- `test_rasterize_no_double_claim` (two abutting triangles: each texel claimed exactly once)
- `test_rasterize_degenerate_triangle_skipped` (zero-area UV triangle → no texels, no NaN)
- `test_dilate_fills_margin_and_preserves_valid`

**Steps:** write tests → RED → implement → GREEN → `inv lint && inv type-check` clean for the new files → commit `feat(heatsim): atlas allocation, packing and texel rasterization (pure numpy)`.

---

