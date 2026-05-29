"""Rendering helpers shared by GUI and future processing tools."""

from __future__ import annotations


def default_detector(diffraction_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """Return a conservative circular detector for a diffraction image."""

    qy, qx = diffraction_shape
    center_y = (qy - 1) / 2
    center_x = (qx - 1) / 2
    radius_outer = max(1.0, min(qy, qx) / 6)
    return center_y, center_x, 0.0, radius_outer

