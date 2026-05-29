"""4DLAB's own import and visualization main window."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from pathlib import Path

from fourdlab.gui.extensions import ViewerExtension, install_extensions
from fourdlab.io import DataCube, RawShapeError, load_datacube
from fourdlab.processing.center_correction import make_detector_mask
from fourdlab.visualization import default_detector


def _main_window_base():
    """Delay Qt imports so non-GUI modules remain cheap to import."""

    try:
        from PyQt5.QtWidgets import QMainWindow
    except ImportError:
        class QMainWindow:  # type: ignore[no-redef]
            pass

    return QMainWindow


def project_root() -> Path:
    """Return the source checkout root when running from this project layout."""

    return Path(__file__).resolve().parents[3]


def gui_icon_path() -> Path | None:
    """Find the 4DLAB GUI icon image."""

    path = project_root() / "4dlab.png"
    return path if path.exists() else None


@dataclass
class ViewerLaunchConfig:
    """Runtime options for the import and visualization viewer."""

    argv: Sequence[str] | None = None
    extensions: list[ViewerExtension] = field(default_factory=list)


def launch_viewer(config: ViewerLaunchConfig | None = None) -> int:
    """Launch the 4DLAB-owned import and visualization viewer."""

    config = config or ViewerLaunchConfig()
    argv = list(config.argv or [])
    if not argv:
        argv = ["4dlab-viewer"]

    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtGui import QIcon
    except ImportError as exc:
        raise RuntimeError(
            "The 4DLAB viewer requires PyQt5. "
            "Activate the 4dlab environment and install project dependencies."
        ) from exc

    app = QApplication.instance() or QApplication(argv)
    icon_path = gui_icon_path()
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))
    viewer = FourDLabViewer()
    viewer.setObjectName("fourdlab_main_viewer")
    install_extensions(viewer, config.extensions)
    viewer.show()
    return app.exec_()


class FourDLabViewer(_main_window_base()):
    """Main window for loading and browsing 4D-STEM datacubes."""

    def __init__(self) -> None:
        super().__init__()
        self.cube: DataCube | None = None
        self.virtual_image = None
        self.display_scaling = "linear"
        self.detector_shape = "annulus"
        self.detector_response = "sum"
        self.real_space_selector_shape = "point"
        self.real_space_mode = "image"
        self.fast_initial_preview = True
        self.autorange_real_space = True
        self.autorange_diffraction = True
        self._syncing_roi = False
        self._real_space_roi = None
        self._diffraction_roi = None
        self._diffraction_inner_roi = None
        self._crop_bin_dialog = None
        self._center_correction_dialog = None
        self._diffraction_analysis_window = None
        self._last_open_dir = str(Path.home())
        self._build_ui()
        self._connect_signals()
        self._set_empty_state()

    def _build_ui(self) -> None:
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QIcon
        from PyQt5.QtWidgets import (
            QAction,
            QActionGroup,
            QApplication,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QPushButton,
            QSpinBox,
            QSplitter,
            QStatusBar,
            QToolBar,
            QVBoxLayout,
            QWidget,
        )
        import pyqtgraph as pg

        if not isinstance(self, QMainWindow):
            raise TypeError("FourDLabViewer must inherit QMainWindow.")

        self.setWindowTitle("4DLAB Viewer")
        icon_path = gui_icon_path()
        if icon_path is not None:
            icon = QIcon(str(icon_path))
            self.setWindowIcon(icon)
            app = QApplication.instance()
            if app is not None:
                app.setWindowIcon(icon)
        self.resize(1280, 760)
        self._file_dialog_class = QFileDialog

        self._build_menus(QAction, QActionGroup)

        toolbar = QToolBar("Main", self)
        toolbar.setMovable(False)
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.refresh_action)
        self.addToolBar(toolbar)

        self.diffraction_view = pg.ImageView()
        self.real_space_view = pg.ImageView()
        self.diffraction_view.ui.roiBtn.hide()
        self.diffraction_view.ui.menuBtn.hide()
        self.real_space_view.ui.roiBtn.hide()
        self.real_space_view.ui.menuBtn.hide()

        self.scan_y = QSpinBox()
        self.scan_x = QSpinBox()
        self.detector_y = QDoubleSpinBox()
        self.detector_x = QDoubleSpinBox()
        self.detector_inner = QDoubleSpinBox()
        self.detector_outer = QDoubleSpinBox()
        for widget in (
            self.detector_y,
            self.detector_x,
            self.detector_inner,
            self.detector_outer,
        ):
            widget.setDecimals(2)
            widget.setSingleStep(1.0)

        self.refresh_button = QPushButton("Refresh")
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)

        controls = QWidget()
        controls_layout = QFormLayout(controls)
        controls_layout.addRow("Scan Y", self.scan_y)
        controls_layout.addRow("Scan X", self.scan_x)
        controls_layout.addRow("Detector Y", self.detector_y)
        controls_layout.addRow("Detector X", self.detector_x)
        controls_layout.addRow("Inner Radius", self.detector_inner)
        controls_layout.addRow("Outer Radius", self.detector_outer)
        controls_layout.addRow(self.refresh_button)
        controls_layout.addRow(self.info_label)

        image_splitter = QSplitter(Qt.Horizontal)
        image_splitter.addWidget(self.real_space_view)
        image_splitter.addWidget(self.diffraction_view)
        image_splitter.setStretchFactor(0, 1)
        image_splitter.setStretchFactor(1, 1)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(controls)
        main_splitter.addWidget(image_splitter)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(main_splitter)
        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar())

    def _build_menus(self, action_class, action_group_class) -> None:
        self.open_action = action_class("Open...", self)
        self.open_action.triggered.connect(self.open_file_dialog)

        self.refresh_action = action_class("Refresh Views", self)
        self.refresh_action.triggered.connect(self.update_all_views)

        exit_action = action_class("Exit", self)
        exit_action.triggered.connect(self.close)
        self.export_action = action_class("Export Datacube...", self)
        self.export_action.triggered.connect(self.export_datacube_dialog)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        scaling_menu = self.menuBar().addMenu("Scaling")
        scaling_group = action_group_class(self)
        for label, value in (
            ("Linear", "linear"),
            ("Log", "log"),
            ("Square Root", "sqrt"),
        ):
            action = action_class(label, self, checkable=True)
            action.setChecked(value == self.display_scaling)
            action.triggered.connect(lambda _checked, mode=value: self.set_display_scaling(mode))
            scaling_group.addAction(action)
            scaling_menu.addAction(action)

        autorange_menu = self.menuBar().addMenu("Autorange")
        self.autorange_real_action = action_class("Real Space", self, checkable=True)
        self.autorange_real_action.setChecked(True)
        self.autorange_real_action.toggled.connect(self.set_real_space_autorange)
        self.autorange_diffraction_action = action_class("Diffraction", self, checkable=True)
        self.autorange_diffraction_action.setChecked(True)
        self.autorange_diffraction_action.toggled.connect(self.set_diffraction_autorange)
        autorange_now_action = action_class("Apply Both Now", self)
        autorange_now_action.triggered.connect(self.autorange_now)
        autorange_menu.addAction(self.autorange_real_action)
        autorange_menu.addAction(self.autorange_diffraction_action)
        autorange_menu.addSeparator()
        autorange_menu.addAction(autorange_now_action)

        response_menu = self.menuBar().addMenu("Detector Response")
        response_group = action_group_class(self)
        for label, value in (("Sum", "sum"), ("Mean", "mean"), ("Maximum", "max")):
            action = action_class(label, self, checkable=True)
            action.setChecked(value == self.detector_response)
            action.triggered.connect(
                lambda _checked, response=value: self.set_detector_response(response)
            )
            response_group.addAction(action)
            response_menu.addAction(action)

        shape_menu = self.menuBar().addMenu("Detector Shape")
        shape_group = action_group_class(self)
        for label, value in (
            ("Point", "point"),
            ("Circle", "circle"),
            ("Annulus", "annulus"),
            ("Square", "square"),
        ):
            action = action_class(label, self, checkable=True)
            action.setChecked(value == self.detector_shape)
            action.triggered.connect(lambda _checked, shape=value: self.set_detector_shape(shape))
            shape_group.addAction(action)
            shape_menu.addAction(action)

        fft_menu = self.menuBar().addMenu("FFT View")
        fft_group = action_group_class(self)
        for label, value in (("Virtual Image", "image"), ("FFT Magnitude", "fft")):
            action = action_class(label, self, checkable=True)
            action.setChecked(value == self.real_space_mode)
            action.triggered.connect(lambda _checked, mode=value: self.set_real_space_mode(mode))
            fft_group.addAction(action)
            fft_menu.addAction(action)

        processing_menu = self.menuBar().addMenu("Processing")
        reset_detector_action = action_class("Reset Detector", self)
        reset_detector_action.triggered.connect(self.reset_detector)
        real_point_action = action_class("Real-Space Point Selector", self, checkable=True)
        real_square_action = action_class("Real-Space Square Selector", self, checkable=True)
        real_point_action.setChecked(True)
        real_selector_group = action_group_class(self)
        for action, value in (
            (real_point_action, "point"),
            (real_square_action, "square"),
        ):
            action.triggered.connect(
                lambda _checked, shape=value: self.set_real_space_selector_shape(shape)
            )
            real_selector_group.addAction(action)
        processing_menu.addAction(self.refresh_action)
        processing_menu.addAction(reset_detector_action)
        processing_menu.addSeparator()
        processing_menu.addAction(real_point_action)
        processing_menu.addAction(real_square_action)

        more_menu = self.menuBar().addMenu("More")
        crop_bin_action = action_class("Crop / Bin", self)
        crop_bin_action.triggered.connect(self.open_crop_bin)
        more_menu.addAction(crop_bin_action)
        center_correction_action = action_class("Center Correction", self)
        center_correction_action.triggered.connect(self.open_center_correction)
        more_menu.addAction(center_correction_action)
        diffraction_analysis_action = action_class("Diffraction Analysis", self)
        diffraction_analysis_action.triggered.connect(self.open_diffraction_analysis)
        more_menu.addAction(diffraction_analysis_action)

    def _connect_signals(self) -> None:
        self.scan_y.valueChanged.connect(self._scan_controls_changed)
        self.scan_x.valueChanged.connect(self._scan_controls_changed)
        self.detector_y.valueChanged.connect(self._detector_controls_changed)
        self.detector_x.valueChanged.connect(self._detector_controls_changed)
        self.detector_inner.valueChanged.connect(self._detector_controls_changed)
        self.detector_outer.valueChanged.connect(self._detector_controls_changed)
        self.refresh_button.clicked.connect(self.update_all_views)

    def _set_empty_state(self) -> None:
        self.info_label.setText("Open a 4D dataset to begin.")
        self.statusBar().showMessage("Ready")

    def open_file_dialog(self) -> None:
        filename, _ = self._file_dialog_class.getOpenFileName(
            self,
            "Open 4D-STEM dataset",
            self._last_open_dir,
            "4D datasets (*.npy *.raw *.h5 *.hdf5 *.emd *.py4dstem);;All files (*)",
        )
        if filename:
            self._last_open_dir = str(Path(filename).parent)
            self.load_file(filename)

    def load_file(self, path: str | Path) -> None:
        from PyQt5.QtWidgets import QMessageBox

        try:
            cube = load_datacube(path)
        except RawShapeError as exc:
            from fourdlab.gui.raw_import_dialog import RawImportDialog

            dialog = RawImportDialog(path, self)
            dialog.status.setText(f"{exc}\n\n{dialog.status.text()}")
            if dialog.exec_() != dialog.Accepted:
                return
            try:
                cube = load_datacube(path, raw_config=dialog.config())
            except Exception as retry_exc:
                QMessageBox.critical(self, "Load failed", str(retry_exc))
                return
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        if self.cube is not None:
            self.cube.close()
        self.cube = cube
        self._configure_controls(cube)
        self.update_diffraction_view()
        self.update_virtual_image(quick=self.fast_initial_preview)
        self.setWindowTitle(f"4DLAB Viewer - {cube.source_path.name}")
        dataset_note = f" [{cube.dataset_path}]" if cube.dataset_path else ""
        self.info_label.setText(
            f"{cube.source_path.name}{dataset_note}\n"
            f"shape={cube.shape}, dtype={cube.dtype}"
        )
        self.statusBar().showMessage(f"Loaded {cube.source_path}", 5000)

    def export_datacube_dialog(self) -> None:
        from PyQt5.QtWidgets import QFileDialog, QMessageBox

        from fourdlab.io.exporters import export_datacube

        if self.cube is None:
            QMessageBox.warning(self, "Export Datacube", "Load or create a datacube first.")
            return
        default = self.cube.source_path.with_name(f"{self.cube.source_path.stem}_4dlab.npy")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export current datacube",
            str(default),
            "NumPy array (*.npy);;HDF5 datacube (*.h5 *.hdf5);;EMD/py4DSTEM (*.emd *.py4dstem);;RAW binary (*.raw);;All files (*)",
        )
        if not path:
            return
        try:
            out = export_datacube(self.cube, path)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Exported {out}", 8000)
        QMessageBox.information(self, "Export complete", f"Exported:\n{out}")

    def replace_datacube(self, cube: DataCube, *, note: str = "") -> None:
        if self.cube is not None and self.cube is not cube:
            self.cube.close()
        self.cube = cube
        self._configure_controls(cube)
        self.update_diffraction_view()
        self.update_virtual_image(quick=self.fast_initial_preview)
        self.setWindowTitle(f"4DLAB Viewer - {cube.source_path.name} [processed]")
        dataset_note = f" [{cube.dataset_path}]" if cube.dataset_path else ""
        suffix = f"\n{note}" if note else ""
        self.info_label.setText(
            f"{cube.source_path.name}{dataset_note}\n"
            f"shape={cube.shape}, dtype={cube.dtype}{suffix}"
        )
        self.statusBar().showMessage(note or "Datacube replaced", 5000)

    def open_crop_bin(self) -> None:
        from PyQt5.QtWidgets import QMessageBox

        from fourdlab.gui.crop_bin_dialog import CropBinDialog

        if self.cube is None:
            QMessageBox.warning(self, "Crop / Bin", "Load a datacube first.")
            return
        self._crop_bin_dialog = CropBinDialog(self)
        self._crop_bin_dialog.show()

    def open_center_correction(self) -> None:
        from PyQt5.QtWidgets import QMessageBox

        from fourdlab.gui.center_correction_dialog import CenterCorrectionDialog

        if self.cube is None:
            QMessageBox.warning(self, "Center Correction", "Load a datacube first.")
            return
        self._center_correction_dialog = CenterCorrectionDialog(self)
        self._center_correction_dialog.show()

    def open_diffraction_analysis(self) -> None:
        from PyQt5.QtWidgets import QMessageBox

        from fourdlab.gui.diffraction_analysis_window import DiffractionAnalysisWindow

        if self.cube is None:
            QMessageBox.warning(self, "Diffraction Analysis", "Load a datacube first.")
            return
        self._diffraction_analysis_window = DiffractionAnalysisWindow(self)
        self._diffraction_analysis_window.show()

    def _configure_controls(self, cube: DataCube) -> None:
        sy, sx = cube.scan_shape
        qy, qx = cube.diffraction_shape
        center_y, center_x, inner, outer = default_detector(cube.diffraction_shape)

        self.scan_y.setRange(0, sy - 1)
        self.scan_x.setRange(0, sx - 1)
        self.detector_y.setRange(0, qy - 1)
        self.detector_x.setRange(0, qx - 1)
        self.detector_inner.setRange(0, max(qy, qx))
        self.detector_outer.setRange(1, max(qy, qx))

        self.scan_y.setValue(0)
        self.scan_x.setValue(0)
        self.detector_y.setValue(center_y)
        self.detector_x.setValue(center_x)
        self.detector_inner.setValue(inner)
        self.detector_outer.setValue(outer)
        self._install_real_space_roi()
        self._install_diffraction_roi()

    def update_all_views(self) -> None:
        self.update_diffraction_view()
        self.update_virtual_image()

    def update_diffraction_view(self) -> None:
        if self.cube is None:
            return
        if self.real_space_selector_shape == "square" and self._real_space_roi is not None:
            y0, x0, y1, x1 = self._real_space_roi_bounds()
            pattern = self.cube.diffraction_region(
                y0,
                x0,
                y1,
                x1,
                response=self.detector_response,
            )
            self.statusBar().showMessage(
                f"Real-space square [{y0}:{y1}, {x0}:{x1}] -> diffraction",
                1000,
            )
        else:
            pattern = self.cube.diffraction_pattern(self.scan_y.value(), self.scan_x.value())
        self.diffraction_view.setImage(
            self._scale_for_display(pattern).T,
            autoLevels=False,
            levels=self._auto_levels(pattern) if self.autorange_diffraction else None,
            autoRange=False,
        )
        if self.autorange_diffraction:
            self._fit_histogram_to_image(self.diffraction_view, pattern)

    def update_virtual_image(self, *, quick: bool = False) -> None:
        if self.cube is None:
            return
        inner = min(self.detector_inner.value(), self.detector_outer.value())
        outer = max(self.detector_inner.value(), self.detector_outer.value())
        if quick and self.detector_shape != "point":
            image = self.cube.quick_navigation_image(
                center_y=self.detector_y.value(),
                center_x=self.detector_x.value(),
            )
            self.statusBar().showMessage(
                "Fast navigation preview shown. Use Refresh Views or detector ROI changes for full VDF.",
                5000,
            )
        else:
            image = self.cube.virtual_image_chunked(
                center_y=self.detector_y.value(),
                center_x=self.detector_x.value(),
                radius_inner=inner,
                radius_outer=outer,
                detector_shape=self.detector_shape,
                detector_response=self.detector_response,
            )
        self.virtual_image = image
        display_image = self._real_space_display_image(image)
        self.real_space_view.setImage(
            self._scale_for_display(display_image).T,
            autoLevels=False,
            levels=self._auto_levels(display_image) if self.autorange_real_space else None,
            autoRange=False,
        )
        if self.autorange_real_space:
            self._fit_histogram_to_image(self.real_space_view, display_image)

    def current_diffraction_detector_mask(self):
        if self.cube is None:
            raise ValueError("No datacube is loaded.")
        inner = min(self.detector_inner.value(), self.detector_outer.value())
        outer = max(self.detector_inner.value(), self.detector_outer.value())
        return make_detector_mask(
            self.cube.diffraction_shape,
            detector_shape=self.detector_shape,
            center_y=self.detector_y.value(),
            center_x=self.detector_x.value(),
            radius_inner=inner,
            radius_outer=outer,
        )

    def _scan_controls_changed(self) -> None:
        if self._syncing_roi:
            return
        self._move_real_space_point_to_controls()
        self.update_diffraction_view()

    def _detector_controls_changed(self) -> None:
        if self._syncing_roi:
            return
        self._move_diffraction_roi_to_controls()
        self.update_virtual_image()

    def _install_real_space_roi(self) -> None:
        if self.cube is None:
            return
        import pyqtgraph as pg

        self._remove_roi(self.real_space_view.getView(), "_real_space_roi")
        sy, sx = self.cube.scan_shape
        y = min(self.scan_y.value(), sy - 1)
        x = min(self.scan_x.value(), sx - 1)
        pen = pg.mkPen("g", width=2)
        hover_pen = pg.mkPen("c", width=2)
        if self.real_space_selector_shape == "square":
            size = max(1, min(sy, sx) // 8)
            self._real_space_roi = pg.RectROI(
                [x - size / 2, y - size / 2],
                [size, size],
                pen=pen,
                hoverPen=hover_pen,
            )
            self._real_space_roi.addScaleHandle([1, 1], [0, 0])
            self._real_space_roi.sigRegionChangeFinished.connect(
                self._real_space_roi_changed
            )
        else:
            self._real_space_roi = pg.ROI(
                [x - 0.5, y - 0.5],
                [1, 1],
                pen=pen,
                hoverPen=hover_pen,
                movable=True,
            )
            self._real_space_roi.sigRegionChanged.connect(self._real_space_roi_changed)
        self.real_space_view.getView().addItem(self._real_space_roi)

    def _install_diffraction_roi(self) -> None:
        if self.cube is None:
            return
        import pyqtgraph as pg

        view = self.diffraction_view.getView()
        self._remove_roi(view, "_diffraction_roi")
        self._remove_roi(view, "_diffraction_inner_roi")
        qy, qx = self.cube.diffraction_shape
        cy = float(self.detector_y.value())
        cx = float(self.detector_x.value())
        outer = max(1.0, float(self.detector_outer.value()))
        inner = max(1.0, float(self.detector_inner.value()))
        pen = pg.mkPen("g", width=2)
        inner_pen = pg.mkPen("y", width=2)
        hover_pen = pg.mkPen("c", width=2)

        if self.detector_shape == "point":
            self._diffraction_roi = pg.ROI(
                [cx - 0.5, cy - 0.5],
                [1, 1],
                pen=pen,
                hoverPen=hover_pen,
                movable=True,
            )
            self._diffraction_roi.sigRegionChanged.connect(
                self._diffraction_roi_changed
            )
        elif self.detector_shape == "square":
            self._diffraction_roi = pg.RectROI(
                [cx - outer, cy - outer],
                [2 * outer, 2 * outer],
                pen=pen,
                hoverPen=hover_pen,
            )
            self._diffraction_roi.addScaleHandle([1, 1], [0, 0])
            self._diffraction_roi.sigRegionChangeFinished.connect(
                self._diffraction_roi_changed
            )
        elif self.detector_shape == "circle":
            self._diffraction_roi = pg.CircleROI(
                [cx - outer, cy - outer],
                [2 * outer, 2 * outer],
                pen=pen,
                hoverPen=hover_pen,
            )
            self._diffraction_roi.sigRegionChangeFinished.connect(
                self._diffraction_roi_changed
            )
        else:
            self._diffraction_roi = pg.CircleROI(
                [cx - outer, cy - outer],
                [2 * outer, 2 * outer],
                pen=pen,
                hoverPen=hover_pen,
            )
            self._diffraction_inner_roi = pg.CircleROI(
                [cx - inner, cy - inner],
                [2 * inner, 2 * inner],
                pen=inner_pen,
                hoverPen=hover_pen,
            )
            self._diffraction_roi.sigRegionChanged.connect(self._sync_annulus_inner)
            self._diffraction_roi.sigRegionChangeFinished.connect(
                self._diffraction_roi_changed
            )
            self._diffraction_inner_roi.sigRegionChangeFinished.connect(
                self._diffraction_roi_changed
            )
            view.addItem(self._diffraction_inner_roi)

        view.addItem(self._diffraction_roi)

    def _remove_roi(self, view, attr: str) -> None:
        roi = getattr(self, attr, None)
        if roi is None:
            return
        try:
            view.removeItem(roi)
        except Exception:
            pass
        setattr(self, attr, None)

    def _real_space_roi_changed(self) -> None:
        if self.cube is None or self._real_space_roi is None or self._syncing_roi:
            return
        self._syncing_roi = True
        try:
            if self.real_space_selector_shape == "point":
                y, x = self._real_space_point_from_roi()
                self.scan_y.setValue(y)
                self.scan_x.setValue(x)
        finally:
            self._syncing_roi = False
        self.update_diffraction_view()

    def _diffraction_roi_changed(self) -> None:
        if self.cube is None or self._diffraction_roi is None or self._syncing_roi:
            return
        self._syncing_roi = True
        try:
            cy, cx, inner, outer = self._detector_values_from_roi()
            self.detector_y.setValue(cy)
            self.detector_x.setValue(cx)
            self.detector_inner.setValue(inner)
            self.detector_outer.setValue(outer)
        finally:
            self._syncing_roi = False
        self.update_virtual_image()

    def _real_space_point_from_roi(self) -> tuple[int, int]:
        if self.cube is None or self._real_space_roi is None:
            return 0, 0
        sy, sx = self.cube.scan_shape
        pos = self._real_space_roi.pos()
        y = int(round(pos.y() + 0.5))
        x = int(round(pos.x() + 0.5))
        return int(max(0, min(sy - 1, y))), int(max(0, min(sx - 1, x)))

    def _real_space_roi_bounds(self) -> tuple[int, int, int, int]:
        if self.cube is None or self._real_space_roi is None:
            return 0, 0, 1, 1
        sy, sx = self.cube.scan_shape
        pos = self._real_space_roi.pos()
        size = self._real_space_roi.size()
        x0 = int(max(0, min(sx - 1, round(pos.x()))))
        y0 = int(max(0, min(sy - 1, round(pos.y()))))
        x1 = int(max(x0 + 1, min(sx, round(pos.x() + size.x()))))
        y1 = int(max(y0 + 1, min(sy, round(pos.y() + size.y()))))
        return y0, x0, y1, x1

    def _detector_values_from_roi(self) -> tuple[float, float, float, float]:
        if self._diffraction_roi is None:
            return 0.0, 0.0, 0.0, 1.0
        pos = self._diffraction_roi.pos()
        size = self._diffraction_roi.size()
        if self.detector_shape == "point":
            return pos.y() + 0.5, pos.x() + 0.5, 0.0, 1.0

        outer = max(float(size.x()), float(size.y())) / 2.0
        cy = pos.y() + float(size.y()) / 2.0
        cx = pos.x() + float(size.x()) / 2.0
        inner = 0.0
        if self.detector_shape == "annulus" and self._diffraction_inner_roi is not None:
            inner = max(
                float(self._diffraction_inner_roi.size().x()),
                float(self._diffraction_inner_roi.size().y()),
            ) / 2.0
        return cy, cx, inner, outer

    def _move_real_space_point_to_controls(self) -> None:
        if (
            self._real_space_roi is None
            or self.real_space_selector_shape != "point"
            or self.cube is None
        ):
            return
        self._syncing_roi = True
        try:
            self._real_space_roi.setPos(
                [self.scan_x.value() - 0.5, self.scan_y.value() - 0.5],
                update=False,
            )
        finally:
            self._syncing_roi = False

    def _move_diffraction_roi_to_controls(self) -> None:
        if self._diffraction_roi is None or self.cube is None:
            return
        self._syncing_roi = True
        try:
            cy = float(self.detector_y.value())
            cx = float(self.detector_x.value())
            outer = max(1.0, float(self.detector_outer.value()))
            inner = max(1.0, float(self.detector_inner.value()))
            if self.detector_shape == "point":
                self._diffraction_roi.setPos([cx - 0.5, cy - 0.5], update=False)
            else:
                self._diffraction_roi.setPos([cx - outer, cy - outer], update=False)
                self._diffraction_roi.setSize([2 * outer, 2 * outer], update=False)
                if self._diffraction_inner_roi is not None:
                    self._diffraction_inner_roi.setPos(
                        [cx - inner, cy - inner],
                        update=False,
                    )
                    self._diffraction_inner_roi.setSize(
                        [2 * inner, 2 * inner],
                        update=False,
                    )
        finally:
            self._syncing_roi = False

    def _sync_annulus_inner(self) -> None:
        if (
            self._diffraction_roi is None
            or self._diffraction_inner_roi is None
            or self._syncing_roi
        ):
            return
        cy, cx, inner, _outer = self._detector_values_from_roi()
        self._diffraction_inner_roi.setPos([cx - inner, cy - inner], update=False)

    def set_display_scaling(self, mode: str) -> None:
        self.display_scaling = mode
        self.update_all_views()

    def set_real_space_autorange(self, enabled: bool) -> None:
        self.autorange_real_space = enabled
        self.update_virtual_image()

    def set_diffraction_autorange(self, enabled: bool) -> None:
        self.autorange_diffraction = enabled
        self.update_diffraction_view()

    def autorange_now(self) -> None:
        self.update_all_views()

    def set_detector_response(self, response: str) -> None:
        self.detector_response = response
        self.update_virtual_image()

    def set_detector_shape(self, shape: str) -> None:
        self.detector_shape = shape
        if shape in {"circle", "point"}:
            self.detector_inner.setValue(0.0)
            self.detector_inner.setEnabled(False)
        else:
            self.detector_inner.setEnabled(True)
        self._install_diffraction_roi()
        self.update_virtual_image()

    def set_real_space_selector_shape(self, shape: str) -> None:
        self.real_space_selector_shape = shape
        self._install_real_space_roi()
        self.update_diffraction_view()

    def set_real_space_mode(self, mode: str) -> None:
        self.real_space_mode = mode
        self.update_virtual_image()

    def reset_detector(self) -> None:
        if self.cube is None:
            return
        center_y, center_x, inner, outer = default_detector(self.cube.diffraction_shape)
        self.detector_y.setValue(center_y)
        self.detector_x.setValue(center_x)
        self.detector_inner.setValue(inner)
        self.detector_outer.setValue(outer)
        self._install_diffraction_roi()
        self.update_virtual_image()

    def _real_space_display_image(self, image):
        if self.real_space_mode != "fft":
            return image
        import numpy as np

        return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image))))

    def _scale_for_display(self, image):
        import numpy as np

        values = np.asarray(image, dtype=np.float64)
        if self.display_scaling == "log":
            min_value = np.nanmin(values)
            if min_value < 0:
                values = values - min_value
            return np.log1p(values)
        if self.display_scaling == "sqrt":
            min_value = np.nanmin(values)
            if min_value < 0:
                values = values - min_value
            return np.sqrt(values)
        return values

    def _auto_levels(self, image) -> tuple[float, float]:
        import numpy as np

        values = self._scale_for_display(image)
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

    def _fit_histogram_to_image(self, image_view, image) -> None:
        low, high = self._auto_levels(image)
        padding = max((high - low) * 0.08, 1e-12)
        try:
            image_view.ui.histogram.setHistogramRange(low - padding, high + padding)
            image_view.ui.histogram.setLevels(low, high)
        except Exception:
            pass

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt method name
        if self.cube is not None:
            self.cube.close()
        super().closeEvent(event)
