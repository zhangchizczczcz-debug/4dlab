"""RAW binary import parameter dialog."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from fourdlab.io import RawLoadConfig


class RawImportDialog(QDialog):
    """Collect explicit dimensions for RAW files without embedded metadata."""

    def __init__(self, path: str | Path, parent=None) -> None:
        super().__init__(parent)
        self.path = Path(path)
        self.file_size = self.path.stat().st_size
        self.setWindowTitle("RAW Import Parameters")
        self.setMinimumWidth(420)

        self.scan_y = _spin(1, 1, 1_000_000)
        self.scan_x = _spin(_default_scan_x(self.file_size), 1, 1_000_000)
        self.diffraction_y = _spin(130, 1, 100_000)
        self.diffraction_x = _spin(128, 1, 100_000)
        self.dtype = QComboBox()
        self.dtype.addItems(["float32", "uint16", "int16", "uint32", "int32", "float64"])
        self.crop_bottom_rows = _spin(2, 0, 100_000)
        self.auto_crop = QCheckBox("EMPAD default crop when shape is 130 x 128")
        self.auto_crop.setChecked(True)
        self.status = QLabel()
        self.status.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Scan Y", self.scan_y)
        form.addRow("Scan X", self.scan_x)
        form.addRow("Diffraction Y", self.diffraction_y)
        form.addRow("Diffraction X", self.diffraction_x)
        form.addRow("Data type", self.dtype)
        form.addRow("Crop bottom rows", self.crop_bottom_rows)
        form.addRow(self.auto_crop)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.status)
        layout.addWidget(self.buttons)

        for widget in (
            self.scan_y,
            self.scan_x,
            self.diffraction_y,
            self.diffraction_x,
            self.crop_bottom_rows,
        ):
            widget.valueChanged.connect(self._update_validation)
        self.dtype.currentTextChanged.connect(self._update_validation)
        self.diffraction_y.valueChanged.connect(self._shape_changed)
        self.diffraction_x.valueChanged.connect(self._shape_changed)
        self.auto_crop.toggled.connect(self._shape_changed)
        self._update_validation()

    def config(self) -> RawLoadConfig:
        return RawLoadConfig(
            scan_y=self.scan_y.value(),
            scan_x=self.scan_x.value(),
            diffraction_y=self.diffraction_y.value(),
            diffraction_x=self.diffraction_x.value(),
            dtype=self.dtype.currentText(),
            crop_bottom_rows=self.crop_bottom_rows.value(),
        )

    def _shape_changed(self) -> None:
        if self.auto_crop.isChecked():
            if self.diffraction_y.value() == 130 and self.diffraction_x.value() == 128:
                self.crop_bottom_rows.setValue(2)
            else:
                self.crop_bottom_rows.setValue(0)
        self._update_validation()

    def _update_validation(self) -> None:
        config = self.config()
        expected = config.expected_bytes
        ok = expected == self.file_size
        crop_ok = config.crop_bottom_rows < config.diffraction_y
        ok = ok and crop_ok
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(ok)
        if not crop_ok:
            self.status.setText("Crop bottom rows must be smaller than diffraction Y.")
            return
        scan_count = config.scan_y * config.scan_x
        self.status.setText(
            f"Actual file size: {self.file_size:,} bytes\n"
            f"Expected: {expected:,} bytes from {scan_count:,} patterns "
            f"of {config.diffraction_y} x {config.diffraction_x} {config.numpy_dtype}"
        )


def _spin(value: int, minimum: int, maximum: int) -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    return box


def _default_scan_x(file_size: int) -> int:
    pattern_bytes = 130 * 128 * np.dtype(np.float32).itemsize
    if file_size % pattern_bytes == 0:
        return max(1, file_size // pattern_bytes)
    return 1
