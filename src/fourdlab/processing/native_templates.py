"""Native CIF and diffraction-template helpers.

This module intentionally does not import py4DSTEM.  It provides a compact
crystal representation that is sufficient for calibration and sparse template
matching in the Diffraction Analysis window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NativeCrystal:
    """Minimal crystal model parsed from CIF cell parameters."""

    path: Path
    a: float
    b: float
    c: float
    alpha_deg: float
    beta_deg: float
    gamma_deg: float
    direct_basis: np.ndarray
    reciprocal_basis: np.ndarray


@dataclass
class NativeCifZonePreview:
    """Generated CIF diffraction spots for one zone axis."""

    qx: np.ndarray
    qy: np.ndarray
    q_radius: np.ndarray
    intensity: np.ndarray
    hkl: np.ndarray
    q_profile: np.ndarray
    intensity_profile: np.ndarray
    matrix: np.ndarray
    zone_axis: np.ndarray


def load_native_crystal(cif_path: str | Path) -> NativeCrystal:
    """Load CIF cell constants into a native crystal model."""

    path = Path(cif_path)
    if not path.exists():
        raise FileNotFoundError(f"CIF file not found: {path}")
    values = _parse_cif_scalars(path)
    required = {
        "_cell_length_a": "a",
        "_cell_length_b": "b",
        "_cell_length_c": "c",
        "_cell_angle_alpha": "alpha",
        "_cell_angle_beta": "beta",
        "_cell_angle_gamma": "gamma",
    }
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"CIF is missing required cell fields: {', '.join(missing)}")
    a = values["_cell_length_a"]
    b = values["_cell_length_b"]
    c = values["_cell_length_c"]
    alpha = values["_cell_angle_alpha"]
    beta = values["_cell_angle_beta"]
    gamma = values["_cell_angle_gamma"]
    direct = _direct_lattice(a, b, c, alpha, beta, gamma)
    reciprocal = np.linalg.inv(direct).T
    return NativeCrystal(
        path=path,
        a=float(a),
        b=float(b),
        c=float(c),
        alpha_deg=float(alpha),
        beta_deg=float(beta),
        gamma_deg=float(gamma),
        direct_basis=direct,
        reciprocal_basis=reciprocal,
    )


def generate_cif_zone_preview(
    crystal: NativeCrystal,
    k_max: float,
    *,
    zone_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    sigma_excitation: float = 0.03,
    intensity_weight_mode: str = "sqrt",
    points: int = 250,
) -> NativeCifZonePreview:
    """Generate native reciprocal spots for a selected CIF zone axis."""

    axis_lattice = _normalize(np.asarray(zone_axis, dtype=np.float64))
    axis_cart = _normalize(crystal.direct_basis @ axis_lattice)
    basis_x, basis_y = _projection_basis(axis_cart)
    hkl, g_cart = _candidate_reflections(crystal, float(k_max))
    if hkl.shape[0] == 0:
        return _empty_preview(axis_cart, basis_x, basis_y, points, float(k_max))

    zone_error = np.abs(hkl @ axis_lattice)
    excitation = np.exp(-(zone_error**2) / (2.0 * max(float(sigma_excitation), 1.0e-6) ** 2))
    exact_zone = zone_error < 0.5
    keep = exact_zone | (excitation > 0.05)
    hkl = hkl[keep]
    g_cart = g_cart[keep]
    excitation = excitation[keep]

    qx = g_cart @ basis_x
    qy = g_cart @ basis_y
    radius = np.sqrt(qx**2 + qy**2)
    keep = (radius > 1.0e-12) & (radius <= float(k_max))
    hkl = hkl[keep]
    qx = qx[keep]
    qy = qy[keep]
    radius = radius[keep]
    excitation = excitation[keep]

    intensity = _reflection_intensity(hkl, radius, excitation, intensity_weight_mode)
    order = np.argsort(radius)
    qx = qx[order]
    qy = qy[order]
    radius = radius[order]
    intensity = intensity[order]
    hkl = hkl[order]
    q_profile, intensity_profile = _radial_profile_from_q(
        radius,
        intensity,
        k_max=float(k_max),
        points=points,
    )
    matrix = np.column_stack((basis_x, basis_y, axis_cart))
    return NativeCifZonePreview(
        qx=qx,
        qy=qy,
        q_radius=radius,
        intensity=intensity,
        hkl=hkl,
        q_profile=q_profile,
        intensity_profile=intensity_profile,
        matrix=matrix,
        zone_axis=axis_cart,
    )


def rotated_zone_template(
    preview: NativeCifZonePreview,
    in_plane_angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a zone preview in-plane and return qxy plus its orientation matrix."""

    theta = np.deg2rad(float(in_plane_angle_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    rot2 = np.asarray([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    qxy = np.column_stack((preview.qx, preview.qy)) @ rot2.T
    rot3 = np.asarray(
        [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return qxy, preview.matrix @ rot3


def zone_axes_in_cone(
    center: tuple[float, float, float],
    max_angle_deg: float,
    step_deg: float,
) -> list[np.ndarray]:
    """Generate normalized zone axes in a cone around a center axis."""

    ref = _normalize(np.asarray(center, dtype=np.float64))
    basis_x, basis_y = _projection_basis(ref)
    max_angle = max(0.0, float(max_angle_deg))
    step = max(0.25, float(step_deg))
    axes = [ref]
    if max_angle <= 0:
        return axes
    for polar in np.arange(step, max_angle + step * 0.5, step):
        circumference = max(6, int(np.ceil(360.0 / step)))
        for azimuth in np.linspace(0.0, 360.0, circumference, endpoint=False):
            p = np.deg2rad(float(polar))
            a = np.deg2rad(float(azimuth))
            axis = (
                np.cos(p) * ref
                + np.sin(p) * (np.cos(a) * basis_x + np.sin(a) * basis_y)
            )
            axes.append(_normalize(axis))
    return axes


def _parse_cif_scalars(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith("_"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        token = parts[1].strip().strip("'\"")
        token = token.split("#", 1)[0].strip()
        if "(" in token:
            token = token.split("(", 1)[0]
        try:
            values[parts[0].lower()] = float(token)
        except ValueError:
            continue
    return values


def _direct_lattice(
    a: float,
    b: float,
    c: float,
    alpha_deg: float,
    beta_deg: float,
    gamma_deg: float,
) -> np.ndarray:
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)
    gamma = np.deg2rad(gamma_deg)
    cos_a = float(np.cos(alpha))
    cos_b = float(np.cos(beta))
    cos_g = float(np.cos(gamma))
    sin_g = max(float(np.sin(gamma)), 1.0e-12)
    ax = np.asarray([a, 0.0, 0.0], dtype=np.float64)
    bx = np.asarray([b * cos_g, b * sin_g, 0.0], dtype=np.float64)
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz2 = c**2 - cx**2 - cy**2
    cz = np.sqrt(max(float(cz2), 1.0e-12))
    return np.column_stack((ax, bx, np.asarray([cx, cy, cz], dtype=np.float64)))


def _candidate_reflections(crystal: NativeCrystal, k_max: float) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(crystal.reciprocal_basis, axis=0)
    min_norm = max(float(np.min(norms)), 1.0e-8)
    limit = int(np.ceil(float(k_max) / min_norm)) + 2
    values = np.arange(-limit, limit + 1, dtype=np.int64)
    hh, kk, ll = np.meshgrid(values, values, values, indexing="ij")
    hkl = np.column_stack((hh.ravel(), kk.ravel(), ll.ravel()))
    hkl = hkl[np.any(hkl != 0, axis=1)]
    g_cart = hkl @ crystal.reciprocal_basis.T
    radius = np.linalg.norm(g_cart, axis=1)
    keep = radius <= float(k_max) * 1.05
    return hkl[keep], g_cart[keep]


def _reflection_intensity(
    hkl: np.ndarray,
    radius: np.ndarray,
    excitation: np.ndarray,
    mode: str,
) -> np.ndarray:
    base = 1.0 / np.maximum(radius, 1.0e-6) ** 2
    parity = 1.0 + 0.12 * ((np.abs(hkl).sum(axis=1) % 2) == 0)
    raw = np.clip(base * parity * np.maximum(excitation, 0.05), 0.0, None)
    if mode == "uniform":
        out = np.ones_like(raw)
    elif mode == "raw":
        out = raw
    elif mode == "log":
        out = np.log1p(raw)
    else:
        out = np.sqrt(raw)
    peak = float(np.max(out)) if out.size else 1.0
    return out / max(peak, 1.0e-12)


def _radial_profile_from_q(
    q_radius: np.ndarray,
    intensity: np.ndarray,
    *,
    k_max: float,
    points: int,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.linspace(0.0, float(k_max), max(2, int(points)))
    profile = np.zeros_like(q)
    if q_radius.size == 0:
        return q, profile
    step = q[1] - q[0] if q.size > 1 else 1.0
    indices = np.clip(np.rint(q_radius / max(step, 1.0e-12)).astype(np.int64), 0, q.size - 1)
    np.add.at(profile, indices, np.asarray(intensity, dtype=np.float64))
    peak = float(np.max(profile))
    if peak > 0:
        profile = profile / peak
    return q, profile


def _empty_preview(
    axis: np.ndarray,
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    points: int,
    k_max: float,
) -> NativeCifZonePreview:
    q = np.linspace(0.0, k_max, max(2, int(points)))
    return NativeCifZonePreview(
        qx=np.zeros(0, dtype=np.float64),
        qy=np.zeros(0, dtype=np.float64),
        q_radius=np.zeros(0, dtype=np.float64),
        intensity=np.zeros(0, dtype=np.float64),
        hkl=np.zeros((0, 3), dtype=np.int64),
        q_profile=q,
        intensity_profile=np.zeros_like(q),
        matrix=np.column_stack((basis_x, basis_y, axis)),
        zone_axis=axis,
    )


def _projection_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(axis, helper))) > 0.9:
        helper = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    basis_x = _normalize(np.cross(helper, axis))
    basis_y = _normalize(np.cross(axis, basis_x))
    return basis_x, basis_y


def _normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        raise ValueError("Vector must be non-zero.")
    return arr / norm
