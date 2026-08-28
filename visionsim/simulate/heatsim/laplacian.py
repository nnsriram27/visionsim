# Vendored from heat-sim-blender:addon/lib/robust_laplacian_backend.py @ e5b4afe
"""
Optional integration with `robust_laplacian` (Sharp & Crane SGP 2020).

This module is safe to import even when the dependency is missing.
Callers should check `HAS_ROBUST_LAPLACIAN` before using.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import robust_laplacian  # type: ignore

    HAS_ROBUST_LAPLACIAN = True
    ROBUST_IMPORT_ERROR: Optional[str] = None
except Exception as e:  # pragma: no cover
    robust_laplacian = None
    HAS_ROBUST_LAPLACIAN = False
    ROBUST_IMPORT_ERROR = str(e)


def mesh_laplacian_and_mass(
    verts: np.ndarray,
    faces: np.ndarray,
    mollify_factor: float = 1e-5,
):
    """
    Build robust mesh Laplacian + lumped mass matrix.

    Returns:
        (L, M) as SciPy sparse matrices.
    """
    if not HAS_ROBUST_LAPLACIAN:  # pragma: no cover
        raise ImportError(
            "robust_laplacian is not available. "
            f"Import error: {ROBUST_IMPORT_ERROR or 'unknown'}"
        )

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    L, M = robust_laplacian.mesh_laplacian(verts, faces, mollify_factor=mollify_factor)
    return L, M


def point_cloud_laplacian_and_mass(
    points: np.ndarray,
    mollify_factor: float = 1e-5,
    n_neighbors: int = 30,
):
    """
    Build robust point-cloud Laplacian + diagonal lumped mass matrix.

    Returns:
        (L, M) as SciPy sparse matrices.
    """
    if not HAS_ROBUST_LAPLACIAN:  # pragma: no cover
        raise ImportError(
            "robust_laplacian is not available. "
            f"Import error: {ROBUST_IMPORT_ERROR or 'unknown'}"
        )

    points = np.asarray(points, dtype=np.float64)
    # Clamp n_neighbors to len(points)-1 (defensive, matches the scipy fallback
    # in solver.py: prevents robust_laplacian "k+1 is greater than number of
    # points" crash on small point clouds; no-op on real dense meshes).
    n_neighbors = max(1, min(int(n_neighbors), len(points) - 1))
    L, M = robust_laplacian.point_cloud_laplacian(
        points, mollify_factor=mollify_factor, n_neighbors=int(n_neighbors)
    )
    return L, M
