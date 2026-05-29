"""Dialog for center-disk correction."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from fourdlab.gui.workers import RunningTask, start_background_task
from fourdlab.processing.center_correction import (
    CenterCorrectionCancelled,
    CenterCorrectionConfig,
    CenterCorrectionResult,
    align_center_disks,
)


class CenterCorrectionDialog(QDialog):
    """Run COM-based center-disk correction and apply it back to the viewer."""

    def __init__(self, viewer) -> None:
        super().__init__(viewer)
        self.viewer = viewer
        self.result: CenterCorrectionResult | None = None
        self._task: RunningTask | None = None
        self.setWindowTitle("Center Correction")
        self.resize(760, 520)
        self._build_ui()
        self._load_viewer_selection()

    def closeEvent(self, event) -> None:
        if self._task is not None:
            self._task.request_stop()
            self.status_label.setText("A background task is stopping; close again after it stops.")
            event.ignore()
            return
        super().closeEvent(event)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.selection_label = QLabel("")
        self.selection_label.setWordWrap(True)
        layout.addWidget(self.selection_label)

        self.target_y = QDoubleSpinBox()
        self.target_x = QDoubleSpinBox()
        self.target_y.setDecimals(3)
        self.target_x.setDecimals(3)
        self.target_y.setSingleStep(0.25)
        self.target_x.setSingleStep(0.25)

        self.subpixel = QCheckBox("Subpixel shift")
        self.subpixel.setChecked(True)
        self.order = QSpinBox()
        self.order.setRange(0, 5)
        self.order.setValue(1)

        form = QFormLayout()
        form.addRow("Target Y", self.target_y)
        form.addRow("Target X", self.target_x)
        form.addRow(self.subpixel)
        form.addRow("Interpolation Order", self.order)
        layout.addLayout(form)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.status_label)

        self._build_preview(layout)

        run_button = QPushButton("Run")
        run_button.clicked.connect(self.run_correction)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_correction)
        self.apply_button = QPushButton("Apply to Viewer")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_to_viewer)

        buttons = QDialogButtonBox()
        buttons.addButton(run_button, QDialogButtonBox.ActionRole)
        buttons.addButton(self.stop_button, QDialogButtonBox.ActionRole)
        buttons.addButton(self.apply_button, QDialogButtonBox.AcceptRole)
        close_button = buttons.addButton(QDialogButtonBox.Close)
        close_button.clicked.connect(self.close)
        layout.addWidget(buttons)

    def _build_preview(self, layout: QVBoxLayout) -> None:
        import pyqtgraph as pg

        preview_row = QHBoxLayout()
        self.before_preview = pg.ImageView()
        self.after_preview = pg.ImageView()
        for view in (self.before_preview, self.after_preview):
            view.ui.roiBtn.hide()
            view.ui.menuBtn.hide()

        before_col = QVBoxLayout()
        before_col.addWidget(QLabel("Before mean DP"))
        before_col.addWidget(self.before_preview)
        after_col = QVBoxLayout()
        after_col.addWidget(QLabel("Corrected mean DP"))
        after_col.addWidget(self.after_preview)
        preview_row.addLayout(before_col)
        preview_row.addLayout(after_col)
        layout.addLayout(preview_row)

    def _load_viewer_selection(self) -> None:
        cube = self.viewer.cube
        if cube is None:
            self.selection_label.setText("No datacube is loaded.")
            return
        qy, qx = cube.diffraction_shape
        self.target_y.setRange(0, qy - 1)
        self.target_x.setRange(0, qx - 1)
        self.target_y.setValue((qy - 1) / 2.0)
        self.target_x.setValue((qx - 1) / 2.0)
        self.selection_label.setText(self._selection_text())

    def _selection_text(self) -> str:
        return (
            "Using current viewer diffraction ROI as center disk:\n"
            f"shape={self.viewer.detector_shape}, "
            f"center=({self.viewer.detector_y.value():.2f}, "
            f"{self.viewer.detector_x.value():.2f}), "
            f"radii=({self.viewer.detector_inner.value():.2f}, "
            f"{self.viewer.detector_outer.value():.2f})"
        )

    def run_correction(self) -> None:
        cube = self.viewer.cube
        if cube is None:
            QMessageBox.warning(self, "Center Correction", "Load a datacube first.")
            return
        if self._task is not None:
            self.status_label.setText("Center correction is already running.")
            return

        try:
            mask = self.viewer.current_diffraction_detector_mask()
            config = CenterCorrectionConfig(
                mask=mask,
                target_y=self.target_y.value(),
                target_x=self.target_x.value(),
                subpixel=self.subpixel.isChecked(),
                order=self.order.value(),
            )

            def progress(done: int, total: int) -> None:
                self.status_label.setText(f"Correcting {done}/{total}")

            self.status_label.setText("Running center correction...")

            def task(progress_emit, stop_requested):
                return align_center_disks(
                    cube,
                    config,
                    progress=progress_emit,
                    stop_requested=stop_requested,
                )

            def finished(result: CenterCorrectionResult) -> None:
                self.result = result
                self.apply_button.setEnabled(True)
                self._update_preview()
                self.status_label.setText(
                    "Done. "
                    f"dy {self.result.shifts_y.min():.3g}..{self.result.shifts_y.max():.3g}, "
                    f"dx {self.result.shifts_x.min():.3g}..{self.result.shifts_x.max():.3g}"
                )

            def failed(exc: BaseException) -> None:
                QMessageBox.critical(self, "Center Correction Failed", str(exc))
                self.status_label.setText("Failed")

            def cancelled(message: str) -> None:
                self.status_label.setText(message)

            def done() -> None:
                self.stop_button.setEnabled(False)
                self._task = None

            self.result = None
            self.apply_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self._task = start_background_task(
                self,
                task,
                cancelled_exception=CenterCorrectionCancelled,
                on_progress=progress,
                on_finished=finished,
                on_failed=failed,
                on_cancelled=cancelled,
                on_done=done,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Center Correction Failed", str(exc))
            self.stop_button.setEnabled(False)
            return

    def stop_correction(self) -> None:
        if self._task is not None:
            self._task.request_stop()
            self.status_label.setText("Stopping after current pattern...")

    def apply_to_viewer(self) -> None:
        if self.result is None:
            return
        self.viewer.replace_datacube(
            self.result.cube,
            note=(
                "center correction applied; "
                f"target=({self.result.target_y:.3f}, {self.result.target_x:.3f})"
            ),
        )
        self.accept()

    def _update_preview(self) -> None:
        import numpy as np

        if self.result is None or self.viewer.cube is None:
            return
        before = np.asarray(self.viewer.cube.data, dtype=np.float32).mean(axis=(0, 1))
        after = np.asarray(self.result.cube.data, dtype=np.float32).mean(axis=(0, 1))
        self.before_preview.setImage(before.T, autoLevels=False, levels=_levels(before))
        self.after_preview.setImage(after.T, autoLevels=False, levels=_levels(after))


def _levels(image) -> tuple[float, float]:
    import numpy as np

    values = np.asarray(image, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, (0.5, 99.5))
    if high <= low:
        low = float(finite.min())
        high = float(finite.max())
    if high <= low:
        high = low + 1.0
    return float(low), float(high)
