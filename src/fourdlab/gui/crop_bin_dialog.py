"""Crop/bin dialog for the main 4DLAB viewer."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QGridLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from fourdlab.processing.datacube_ops import crop_and_bin_datacube


class CropBinDialog(QDialog):
    """Crop and/or bin the loaded datacube and return it to the viewer."""

    def __init__(self, viewer) -> None:
        super().__init__(viewer)
        self.viewer = viewer
        self.setWindowTitle("Crop / Bin")
        self.resize(420, 560)
        self._build_ui()
        self._load_defaults()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        scan_group = QGroupBox("Real Space")
        scan_layout = QFormLayout(scan_group)
        self.enable_scan_crop = QCheckBox("Crop real space")
        self.scan_y0 = _spin()
        self.scan_y1 = _spin()
        self.scan_x0 = _spin()
        self.scan_x1 = _spin()
        self.scan_bin_y = _spin(1, 1, 1000)
        self.scan_bin_x = _spin(1, 1, 1000)
        scan_layout.addRow(self.enable_scan_crop)
        scan_layout.addRow("Y start", self.scan_y0)
        scan_layout.addRow("Y stop", self.scan_y1)
        scan_layout.addRow("X start", self.scan_x0)
        scan_layout.addRow("X stop", self.scan_x1)
        scan_layout.addRow("Bin Y", self.scan_bin_y)
        scan_layout.addRow("Bin X", self.scan_bin_x)

        diff_group = QGroupBox("Diffraction Space")
        diff_layout = QFormLayout(diff_group)
        self.enable_diff_crop = QCheckBox("Crop diffraction space")
        self.diff_y0 = _spin()
        self.diff_y1 = _spin()
        self.diff_x0 = _spin()
        self.diff_x1 = _spin()
        self.diff_bin_y = _spin(1, 1, 1000)
        self.diff_bin_x = _spin(1, 1, 1000)
        diff_layout.addRow(self.enable_diff_crop)
        diff_layout.addRow("QY start", self.diff_y0)
        diff_layout.addRow("QY stop", self.diff_y1)
        diff_layout.addRow("QX start", self.diff_x0)
        diff_layout.addRow("QX stop", self.diff_x1)
        diff_layout.addRow("Bin QY", self.diff_bin_y)
        diff_layout.addRow("Bin QX", self.diff_bin_x)

        use_roi = QPushButton("Read boxes from Viewer")
        use_roi.clicked.connect(self._load_defaults)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")

        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Apply).setText("Apply to GUI")
        buttons.clicked.connect(self._button_clicked)

        layout.addWidget(scan_group)
        layout.addWidget(diff_group)
        layout.addWidget(use_roi)
        layout.addWidget(self.summary)
        layout.addWidget(self.progress)
        layout.addWidget(buttons)

    def _load_defaults(self) -> None:
        cube = self.viewer.cube
        if cube is None:
            return
        sy, sx = cube.scan_shape
        qy, qx = cube.diffraction_shape
        self._set_ranges(self.scan_y0, self.scan_y1, sy)
        self._set_ranges(self.scan_x0, self.scan_x1, sx)
        self._set_ranges(self.diff_y0, self.diff_y1, qy)
        self._set_ranges(self.diff_x0, self.diff_x1, qx)

        if getattr(self.viewer, "real_space_selector_shape", "point") == "square":
            scan_bounds = self.viewer._real_space_roi_bounds()
            self.enable_scan_crop.setChecked(True)
        else:
            scan_bounds = (0, 0, sy, sx)
        self.scan_y0.setValue(scan_bounds[0])
        self.scan_x0.setValue(scan_bounds[1])
        self.scan_y1.setValue(scan_bounds[2])
        self.scan_x1.setValue(scan_bounds[3])

        diff_bounds = self._diffraction_bounds()
        self.diff_y0.setValue(diff_bounds[0])
        self.diff_x0.setValue(diff_bounds[1])
        self.diff_y1.setValue(diff_bounds[2])
        self.diff_x1.setValue(diff_bounds[3])
        self.enable_diff_crop.setChecked(False)
        self._update_summary()

    def apply(self) -> None:
        cube = self.viewer.cube
        if cube is None:
            QMessageBox.warning(self, "Crop / Bin", "Load a datacube first.")
            return
        self.progress.setRange(0, 0)
        self.progress.setFormat("Processing...")
        try:
            new_cube = crop_and_bin_datacube(
                cube,
                scan_bounds=self._scan_bounds() if self.enable_scan_crop.isChecked() else None,
                diffraction_bounds=self._diff_bounds() if self.enable_diff_crop.isChecked() else None,
                scan_bin=(self.scan_bin_y.value(), self.scan_bin_x.value()),
                diffraction_bin=(self.diff_bin_y.value(), self.diff_bin_x.value()),
            )
        except Exception as exc:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setFormat("Failed")
            QMessageBox.critical(self, "Crop / Bin failed", str(exc))
            return
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.progress.setFormat("Done")
        self.viewer.replace_datacube(new_cube, note=f"Crop/bin result: shape={new_cube.shape}")
        self._update_summary()

    def _button_clicked(self, button) -> None:
        box = self.sender()
        if box.standardButton(button) == QDialogButtonBox.Apply:
            self.apply()
        elif box.standardButton(button) == QDialogButtonBox.Close:
            self.close()

    def _scan_bounds(self) -> tuple[int, int, int, int]:
        return self.scan_y0.value(), self.scan_x0.value(), self.scan_y1.value(), self.scan_x1.value()

    def _diff_bounds(self) -> tuple[int, int, int, int]:
        return self.diff_y0.value(), self.diff_x0.value(), self.diff_y1.value(), self.diff_x1.value()

    def _diffraction_bounds(self) -> tuple[int, int, int, int]:
        cube = self.viewer.cube
        if cube is None or getattr(self.viewer, "_diffraction_roi", None) is None:
            return (0, 0, 1, 1)
        qy, qx = cube.diffraction_shape
        roi = self.viewer._diffraction_roi
        pos = roi.pos()
        size = roi.size()
        x0 = int(max(0, min(qx - 1, round(pos.x()))))
        y0 = int(max(0, min(qy - 1, round(pos.y()))))
        x1 = int(max(x0 + 1, min(qx, round(pos.x() + size.x()))))
        y1 = int(max(y0 + 1, min(qy, round(pos.y() + size.y()))))
        return y0, x0, y1, x1

    def _set_ranges(self, start: QSpinBox, stop: QSpinBox, size: int) -> None:
        start.setRange(0, max(0, size - 1))
        stop.setRange(1, max(1, size))
        start.setValue(0)
        stop.setValue(size)

    def _update_summary(self) -> None:
        cube = self.viewer.cube
        if cube is None:
            self.summary.setText("")
            return
        self.summary.setText(f"Current shape: {cube.shape}")


def _spin(value: int = 0, minimum: int = 0, maximum: int = 1000000) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    return spin
