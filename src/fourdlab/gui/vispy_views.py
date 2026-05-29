"""VisPy-backed analysis views with Qt fallbacks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


@dataclass
class AnalysisOverlay:
    """Optional overlay data for an analysis result."""

    points: np.ndarray | None = None
    point_color: tuple[float, float, float, float] = (0.0, 0.75, 1.0, 1.0)
    lines: np.ndarray | None = None
    line_color: tuple[float, float, float, float] = (0.2, 0.95, 0.4, 1.0)


@dataclass
class AnalysisResult:
    """One image-like analysis output shown in the result strip and main view."""

    key: str
    title: str
    kind: str
    image: np.ndarray
    overlays: list[AnalysisOverlay] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class VispyImageView(QWidget):
    """2D VisPy image canvas embedded in Qt, falling back to QLabel if needed."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._canvas = None
        self._view = None
        self._image_visual = None
        self._overlay_visuals = []
        self._fallback = QLabel(alignment=Qt.AlignCenter)
        self._fallback.setMinimumSize(0, 0)
        self._fallback.setScaledContents(True)
        self._fallback.setStyleSheet("QLabel { background: #11151c; color: #d6dbe5; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        try:
            if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
                raise RuntimeError("VisPy OpenGL canvas is disabled for offscreen Qt.")
            from vispy import scene
            from vispy.app import use_app

            use_app("pyqt5")
            self._canvas = scene.SceneCanvas(keys="interactive", bgcolor="#11151c", show=False)
            self._view = self._canvas.central_widget.add_view()
            self._view.camera = "panzoom"
            self._image_visual = scene.visuals.Image(
                np.zeros((2, 2), dtype=np.float32),
                parent=self._view.scene,
                interpolation="nearest",
            )
            layout.addWidget(self._canvas.native)
        except Exception as exc:
            self._canvas = None
            self._fallback.setText(f"VisPy view unavailable\n{exc}")
            layout.addWidget(self._fallback)

    def set_result(self, result: AnalysisResult) -> None:
        image = np.asarray(result.image)
        if image.ndim == 2:
            display = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
            display = _normalize(display)
        else:
            display = _rgb_image(image)
        if self._canvas is None or self._image_visual is None:
            self._fallback.setPixmap(_pixmap_from_rgb(_rgb_image(display)))
            return

        self._image_visual.set_data(display)
        try:
            self._view.camera.set_range()
        except Exception:
            pass
        self._clear_overlays()
        self._add_overlays(result.overlays)
        self._canvas.update()

    def _clear_overlays(self) -> None:
        for visual in self._overlay_visuals:
            try:
                visual.parent = None
            except Exception:
                pass
        self._overlay_visuals = []

    def _add_overlays(self, overlays: list[AnalysisOverlay]) -> None:
        if self._view is None:
            return
        from vispy import scene

        for overlay in overlays:
            if overlay.points is not None and np.asarray(overlay.points).size:
                points = np.asarray(overlay.points, dtype=np.float32)
                markers = scene.visuals.Markers(parent=self._view.scene)
                markers.set_data(
                    points[:, :2],
                    face_color=overlay.point_color,
                    edge_color=overlay.point_color,
                    size=8,
                )
                self._overlay_visuals.append(markers)
            if overlay.lines is not None and np.asarray(overlay.lines).size:
                lines = np.asarray(overlay.lines, dtype=np.float32)
                line = scene.visuals.Line(
                    pos=lines.reshape(-1, 2),
                    color=overlay.line_color,
                    connect="segments",
                    parent=self._view.scene,
                )
                self._overlay_visuals.append(line)


class VispyOrientationMapView(VispyImageView):
    """Orientation-map view that appends a triangular legend to the RGB image."""

    def set_orientation_map(
        self,
        image_rgb: np.ndarray,
        *,
        vertex_labels: list[str] | None = None,
        key: str = "orientation_map",
        title: str = "Orientation Map",
    ) -> AnalysisResult:
        combined = orientation_map_with_triangle_legend(image_rgb, vertex_labels or ["v1", "v2", "v3"])
        result = AnalysisResult(key=key, title=title, kind="orientation_map", image=combined)
        self.set_result(result)
        return result


def orientation_map_with_triangle_legend(image_rgb: np.ndarray, labels: list[str]) -> np.ndarray:
    """Return an RGB image with a triangular color legend appended on the right."""

    image = _rgb_image(image_rgb)
    h = image.shape[0]
    legend_size = max(180, min(320, h))
    legend = _triangle_legend(legend_size)
    canvas = np.ones((max(h, legend_size), image.shape[1] + legend_size + 28, 3), dtype=np.float32)
    canvas[: image.shape[0], : image.shape[1]] = image
    y0 = (canvas.shape[0] - legend_size) // 2
    x0 = image.shape[1] + 20
    canvas[y0 : y0 + legend_size, x0 : x0 + legend_size] = legend
    return canvas


def _triangle_legend(size: int) -> np.ndarray:
    yy, xx = np.indices((size, size), dtype=np.float64)
    p0 = np.asarray([size / 2.0, size * 0.10])
    p1 = np.asarray([size * 0.10, size * 0.88])
    p2 = np.asarray([size * 0.90, size * 0.88])
    denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
    w0 = ((p1[1] - p2[1]) * (xx - p2[0]) + (p2[0] - p1[0]) * (yy - p2[1])) / denom
    w1 = ((p2[1] - p0[1]) * (xx - p2[0]) + (p0[0] - p2[0]) * (yy - p2[1])) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
    colors = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.3, 1.0]], dtype=np.float64)
    out = np.ones((size, size, 3), dtype=np.float32)
    out[inside] = (
        w0[inside][:, None] * colors[0]
        + w1[inside][:, None] * colors[1]
        + w2[inside][:, None] * colors[2]
    )
    return out


def thumbnail_pixmap(image: np.ndarray, width: int = 120, height: int = 76) -> QPixmap:
    pixmap = _pixmap_from_rgb(_rgb_image(image))
    return pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _normalize(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    low, high = np.percentile(finite, (1.0, 99.0))
    if high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        high = low + 1.0
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _rgb_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        norm = _normalize(arr)
        return np.dstack((norm, norm, norm)).astype(np.float32)
    rgb = np.asarray(arr[..., :3], dtype=np.float32)
    if rgb.max(initial=0.0) > 1.0:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _pixmap_from_rgb(image: np.ndarray) -> QPixmap:
    rgb8 = np.ascontiguousarray(np.clip(_rgb_image(image) * 255.0, 0, 255).astype(np.uint8))
    h, w = rgb8.shape[:2]
    qimage = QImage(rgb8.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)
