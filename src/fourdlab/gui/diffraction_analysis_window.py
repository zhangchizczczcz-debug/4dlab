"""Nanobeam diffraction analysis window."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fourdlab.gui.vispy_views import (
    AnalysisOverlay,
    AnalysisResult,
    VispyOrientationMapView,
)
from fourdlab.gui.workers import RunningTask, start_background_task
from fourdlab.processing.orientation import (
    NoCifInPlaneConfig,
    OrientationCancelled,
    OrientationConfig,
    build_bvm_from_peaks,
    cif_zone_preview,
    build_no_cif_template,
    build_orientation_workspace,
    diagnose_cif_fit,
    generate_fit_patterns,
    load_crystal_structure,
    match_current_orientation,
    match_no_cif_in_plane,
    match_no_cif_orientation_map,
    match_orientation_map,
    orientation_color_image,
    orientation_cuda_available,
    peak_list_to_pointlist,
    peak_list_to_centered_vectors_px,
    peak_list_radial_profile,
    rotate_vectors,
    scan_q_pixel_sizes,
)
from fourdlab.processing.peak_detection import (
    PeakDetectionConfig,
    PeakDetectionCancelled,
    PeakDetectionResult,
    PeakList,
    detect_peaks_in_datacube,
    detect_peaks_in_pattern,
    gpu_peak_detection_available,
    load_peak_detection_result,
    save_peak_detection_result,
)
from fourdlab.processing.strain import (
    StrainCancelled,
    StrainConfig,
    StrainResult,
    calculate_strain_map,
    fit_lattice_transform,
    guess_basis_from_peaks,
    polar_strain_components,
    save_strain_result,
)


class DiffractionAnalysisWindow(QMainWindow):
    """Main entry point for nanobeam diffraction analysis tools."""

    def __init__(self, viewer) -> None:
        super().__init__(viewer)
        self.viewer = viewer
        self.current_peaks: PeakList | None = None
        self.all_peaks: PeakDetectionResult | None = None
        self.orientation_controls: dict[str, dict[str, object]] = {}
        self.orientation_crystals: dict[str, object] = {}
        self.orientation_workspaces: dict[str, object] = {}
        self.orientation_results: dict[str, object] = {}
        self.orientation_bvms: dict[str, np.ndarray] = {}
        self.no_cif_templates: dict[str, object] = {}
        self.strain_controls: dict[str, object] = {}
        self.strain_result: StrainResult | None = None
        self.analysis_results: dict[str, AnalysisResult] = {}
        self.current_analysis_result_key: str | None = None
        self.result_menu: QMenu | None = None
        self.results_button: QToolButton | None = None
        self._peak_task: RunningTask | None = None
        self._orientation_task: RunningTask | None = None
        self._strain_task: RunningTask | None = None
        self.scatter = pg.ScatterPlotItem(
            pen=pg.mkPen("y", width=2),
            brush=None,
            size=12,
            symbol="o",
        )
        self.setWindowTitle("Diffraction Analysis")
        self.resize(1180, 820)
        self._build_ui()
        self.load_peak_preset(auto=True)
        self._refresh_pattern_preview()

    def closeEvent(self, event) -> None:
        running = [task for task in (self._peak_task, self._orientation_task, self._strain_task) if task is not None]
        if running:
            for task in running:
                task.request_stop()
            self._log("A background task is stopping; close the window again after it stops.")
            event.ignore()
            return
        super().closeEvent(event)

    def _build_ui(self) -> None:
        self._build_analysis_toolbar()

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_peak_tab(), "Peak Detection")
        self.tabs.addTab(self._build_orientation_tab("in_plane"), "In-Plane")
        self.tabs.addTab(self._build_orientation_tab("out_of_plane"), "Out-of-Plane")
        self.tabs.addTab(self._build_strain_tab(), "Strain")
        self.setCentralWidget(self.tabs)

    def _build_analysis_toolbar(self) -> None:
        toolbar = QToolBar("Analysis", self)
        toolbar.setMovable(False)
        self.result_menu = QMenu("Result Images", self)
        self.results_button = QToolButton(self)
        self.results_button.setText("Result Images")
        self.results_button.setToolTip("Open saved analysis figures in a separate window.")
        self.results_button.setPopupMode(QToolButton.InstantPopup)
        self.results_button.setMenu(self.result_menu)
        toolbar.addWidget(self.results_button)
        self._refresh_results_menu()
        self.addToolBar(toolbar)

    def _add_analysis_result(self, result: AnalysisResult) -> None:
        self.analysis_results[result.key] = result
        self.current_analysis_result_key = result.key
        self._refresh_results_menu()
        self.statusBar().showMessage(f"Added {result.title} to Result Images.", 2500)

    def _refresh_results_menu(self) -> None:
        if self.result_menu is None:
            return
        self.result_menu.clear()
        if not self.analysis_results:
            empty = self.result_menu.addAction("No result images yet")
            empty.setEnabled(False)
            if self.results_button is not None:
                self.results_button.setEnabled(False)
            return
        if self.results_button is not None:
            self.results_button.setEnabled(True)
        for key, result in self.analysis_results.items():
            action = self.result_menu.addAction(result.title)
            action.setToolTip("Open this result in a separate image window.")
            action.triggered.connect(lambda _checked=False, result_key=key: self._show_analysis_result(result_key))
        self.result_menu.addSeparator()
        save_action = self.result_menu.addAction("Save Current Figure")
        save_action.triggered.connect(self._save_current_workbench_result)

    def _show_analysis_result(self, key: str) -> None:
        result = self.analysis_results.get(key)
        if result is None:
            return
        self.current_analysis_result_key = key
        self._open_analysis_result_window(result)
        self.statusBar().showMessage(f"Opened {result.title}", 2500)

    def _open_analysis_result_window(self, result: AnalysisResult) -> None:
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.setWindowTitle(result.title)
        aspect = _image_aspect(result.image)
        if aspect > 1.25:
            dialog.resize(1100, 720)
        else:
            dialog.resize(820, 820)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)
        view = VispyOrientationMapView(dialog)
        view.set_result(result)
        if aspect > 1.25:
            layout.addWidget(_aspect_panel(view, min_size=360, aspect=aspect))
        else:
            layout.addWidget(_square_panel(view, min_size=360))
        dialog.show()

    def _save_current_workbench_result(self) -> None:
        if not self.analysis_results:
            self.statusBar().showMessage("No analysis result to save.", 3000)
            return
        key = self.current_analysis_result_key or next(reversed(self.analysis_results))
        result = self.analysis_results[key]
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save current analysis figure",
            str(_project_root() / f"{result.key}.png"),
            "PNG image (*.png);;All files (*)",
        )
        if not path:
            return
        out = Path(path)
        if out.suffix == "":
            out = out.with_suffix(".png")
        from matplotlib import pyplot as plt

        plt.imsave(str(out), np.asarray(result.image))
        self.statusBar().showMessage(f"Saved {out}", 5000)

    def _build_peak_tab(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)

        params = QGroupBox("Custom Peak Detection")
        form = QFormLayout(params)
        self.smooth_sigma = _double_box(0.5, 0.0, 20.0, 0.1)
        self.edge_boundary = _int_box(15, 0, 512)
        self.min_relative = _double_box(0.0, 0.0, 1.0, 0.01)
        self.min_absolute = _double_box(0.0, 0.0, 1.0e12, 10.0)
        self.min_spacing = _int_box(20, 1, 512)
        self.max_peaks = _int_box(50, 1, 1000)
        self.gaussian_radius = _int_box(4, 2, 50)
        self.refine = QCheckBox("Subpixel refine")
        self.refine.setChecked(True)
        self.refine.setToolTip("Refine each local maximum with the selected subpixel method.")
        self.refine_method = QComboBox()
        self.refine_method.addItems(["Fast centroid", "Gaussian fit"])
        self.refine_method.setToolTip("Fast centroid is much faster for full scans; Gaussian is slower but more precise.")
        self.use_gpu = QCheckBox("Use GPU")
        self.use_gpu.setChecked(False)
        if not gpu_peak_detection_available():
            self.use_gpu.setEnabled(False)
            self.use_gpu.setToolTip("CuPy is not installed in the 4dlab environment.")
        self.peak_workers = _int_box(_default_worker_count(), 1, max(1, os.cpu_count() or 1))
        form.addRow("Smooth sigma", self.smooth_sigma)
        form.addRow("Edge boundary", self.edge_boundary)
        form.addRow("Min relative intensity", self.min_relative)
        form.addRow("Min absolute intensity", self.min_absolute)
        form.addRow("Min peak spacing", self.min_spacing)
        form.addRow("Max peaks", self.max_peaks)
        form.addRow("Gaussian radius", self.gaussian_radius)
        form.addRow(self.refine)
        form.addRow("Refine mode", self.refine_method)
        form.addRow("CPU workers", self.peak_workers)
        form.addRow(self.use_gpu)

        self.preview = pg.ImageView()
        self.preview.ui.roiBtn.hide()
        self.preview.ui.menuBtn.hide()
        self.preview.ui.histogram.hide()
        self.preview.getView().addItem(self.scatter)

        detect_current = QPushButton("Detect Current Pattern")
        detect_current.clicked.connect(self.detect_current_pattern)
        detect_all = QPushButton("Detect All Patterns")
        detect_all.clicked.connect(self.detect_all_patterns)
        self.stop_detection = QPushButton("Stop")
        self.stop_detection.setEnabled(False)
        self.stop_detection.clicked.connect(self.stop_peak_detection)
        refresh = QPushButton("Refresh Preview")
        refresh.clicked.connect(self._refresh_pattern_preview)
        save_preset = QPushButton("Save Preset")
        save_preset.clicked.connect(self.save_peak_preset)
        load_preset = QPushButton("Load Preset")
        load_preset.clicked.connect(self.load_peak_preset)
        save_peaks = QPushButton("Save Peaks")
        save_peaks.clicked.connect(self.save_detected_peaks)
        load_peaks = QPushButton("Load Peaks")
        load_peaks.clicked.connect(self.load_detected_peaks)
        show_bvm = QPushButton("BVM Centered")
        show_bvm.clicked.connect(self.show_centered_bvm_peak_tab)

        buttons = QWidget()
        button_layout = QVBoxLayout(buttons)
        button_layout.addWidget(refresh)
        button_layout.addWidget(detect_current)
        button_layout.addWidget(detect_all)
        button_layout.addWidget(self.stop_detection)
        button_layout.addWidget(save_preset)
        button_layout.addWidget(load_preset)
        button_layout.addWidget(save_peaks)
        button_layout.addWidget(load_peaks)
        button_layout.addWidget(show_bvm)
        button_layout.addStretch(1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(params)
        left_layout.addWidget(buttons)
        left_layout.addStretch(1)
        left_scroll = _scroll_area(left_panel)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")

        layout.addWidget(left_scroll, 0, 0, 2, 1)
        layout.addWidget(_square_panel(self.preview), 0, 1, 2, 1)
        layout.addWidget(self.progress, 2, 0, 1, 2)
        layout.addWidget(self.log, 3, 0, 1, 2)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        return page

    def _build_orientation_tab(self, mode: str) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)

        field_forms: dict[QWidget, QFormLayout] = {}

        def add_row(form: QFormLayout, label_or_widget, widget: QWidget | None = None) -> None:
            if widget is None:
                form.addRow(label_or_widget)
                if isinstance(label_or_widget, QWidget):
                    field_forms[label_or_widget] = form
            else:
                form.addRow(label_or_widget, widget)
                field_forms[widget] = form

        cube = self.viewer.cube
        scan_y_size = cube.scan_shape[0] if cube is not None else 100
        scan_x_size = cube.scan_shape[1] if cube is not None else 100

        method_group = QGroupBox("Method and CIF")
        method_form = QFormLayout(method_group)
        method = QComboBox()
        if mode == "in_plane":
            method.addItems(["CIF indexing", "No-CIF template"])
        else:
            method.addItems(["CIF indexing"])
            method.setEnabled(False)
        cif_path = QLineEdit()
        browse_cif = QPushButton("Browse CIF")
        browse_cif.clicked.connect(lambda: self.browse_orientation_cif(mode))

        add_row(method_form, "Method", method)
        add_row(method_form, "CIF", cif_path)
        add_row(method_form, browse_cif)

        calibration_group = QGroupBox("Center and Calibration")
        calibration_form = QFormLayout(calibration_group)
        q_pixel_size = _double_box(0.01, 1.0e-8, 100.0, 0.001)
        q_pixel_size.setDecimals(8)
        qx_center = _double_box(self._default_qx_center(), -100000.0, 100000.0, 0.5)
        qy_center = _double_box(self._default_qy_center(), -100000.0, 100000.0, 0.5)
        flip_qy = QCheckBox("Flip qy")
        flip_qy.setChecked(True)
        add_row(calibration_form, "q pixel size (A^-1/pixel)", q_pixel_size)
        add_row(calibration_form, "qx center (pixel)", qx_center)
        add_row(calibration_form, "qy center (pixel)", qy_center)
        add_row(calibration_form, flip_qy)

        cif_model_group = QGroupBox("CIF Zone Axis and Radial Calibration")
        cif_model_form = QFormLayout(cif_model_group)
        k_max = _double_box(2.0, 0.01, 20.0, 0.1)
        axis_h = _double_box(0.0, -20.0, 20.0, 1.0)
        axis_k = _double_box(0.0, -20.0, 20.0, 1.0)
        axis_l = _double_box(1.0, -20.0, 20.0, 1.0)
        fiber_axis = QWidget()
        fiber_axis_layout = QHBoxLayout(fiber_axis)
        fiber_axis_layout.setContentsMargins(0, 0, 0, 0)
        fiber_axis_layout.addWidget(axis_h)
        fiber_axis_layout.addWidget(axis_k)
        fiber_axis_layout.addWidget(axis_l)
        q_pixel_min = _double_box(0.01, 1.0e-8, 100.0, 0.001)
        q_pixel_min.setDecimals(8)
        q_pixel_max = _double_box(0.05, 1.0e-8, 100.0, 0.001)
        q_pixel_max.setDecimals(8)
        q_pixel_steps = _int_box(41, 2, 1000)
        zone_tol = _double_box(0.03, 0.0001, 2.0, 0.005)
        q_pixel_source = QComboBox()
        q_pixel_source.addItems(["Full BVM", "Current scan pixel"])
        q_pixel_scan_y = _int_box(max(0, self.viewer.scan_y.value()), 0, max(0, scan_y_size - 1))
        q_pixel_scan_x = _int_box(max(0, self.viewer.scan_x.value()), 0, max(0, scan_x_size - 1))
        q_pixel_scan_widget = QWidget()
        q_pixel_scan_layout = QHBoxLayout(q_pixel_scan_widget)
        q_pixel_scan_layout.setContentsMargins(0, 0, 0, 0)
        q_pixel_scan_layout.addWidget(q_pixel_scan_y)
        q_pixel_scan_layout.addWidget(q_pixel_scan_x)
        use_viewer_pixel = QPushButton("Use Viewer Pixel")
        use_viewer_pixel.clicked.connect(lambda: self.use_viewer_q_pixel(mode))
        preview_cif = QPushButton("Preview CIF Diffraction")
        preview_cif.clicked.connect(lambda: self.preview_cif_zone(mode))
        scan_pixel = QPushButton("Scan q Pixel")
        scan_pixel.clicked.connect(lambda: self.scan_orientation_q_pixel(mode))
        diagnose_fit = QPushButton("Diagnose CIF Fit")
        diagnose_fit.clicked.connect(lambda: self.diagnose_cif_fit(mode))

        add_row(cif_model_form, "k max (A^-1)", k_max)
        add_row(cif_model_form, "Zone axis / beam direction", fiber_axis)
        add_row(cif_model_form, "Zone tolerance", zone_tol)
        add_row(cif_model_form, "Radial source", q_pixel_source)
        add_row(cif_model_form, "Scan y / x", q_pixel_scan_widget)
        add_row(cif_model_form, use_viewer_pixel)
        add_row(cif_model_form, "q pixel min", q_pixel_min)
        add_row(cif_model_form, "q pixel max", q_pixel_max)
        add_row(cif_model_form, "q pixel steps", q_pixel_steps)
        add_row(cif_model_form, preview_cif)
        add_row(cif_model_form, scan_pixel)
        add_row(cif_model_form, diagnose_fit)

        oop_range_group = QGroupBox("Out-of-Plane Search Range")
        oop_range_form = QFormLayout(oop_range_group)
        range_mode = QComboBox()
        range_mode.addItems(["Center + tilt cone", "Three zone-axis vertices"])
        center_h = _double_box(0.0, -20.0, 20.0, 1.0)
        center_k = _double_box(0.0, -20.0, 20.0, 1.0)
        center_l = _double_box(1.0, -20.0, 20.0, 1.0)
        center_axis = _hkl_widget(center_h, center_k, center_l)
        cone_angle = _double_box(10.0, 0.0, 180.0, 1.0)
        vertex_boxes = [
            (
                _double_box(0.0, -20.0, 20.0, 1.0),
                _double_box(0.0, -20.0, 20.0, 1.0),
                _double_box(1.0, -20.0, 20.0, 1.0),
            ),
            (
                _double_box(0.0, -20.0, 20.0, 1.0),
                _double_box(1.0, -20.0, 20.0, 1.0),
                _double_box(1.0, -20.0, 20.0, 1.0),
            ),
            (
                _double_box(1.0, -20.0, 20.0, 1.0),
                _double_box(0.0, -20.0, 20.0, 1.0),
                _double_box(1.0, -20.0, 20.0, 1.0),
            ),
        ]
        vertex_widgets = [_hkl_widget(*boxes) for boxes in vertex_boxes]
        use_in_plane_sync = QCheckBox("Use In-Plane CIF/Calibration")
        use_in_plane_sync.setChecked(mode == "out_of_plane")
        sync_from_in_plane = QPushButton("Sync from In-Plane")
        sync_from_in_plane.clicked.connect(lambda: self.sync_out_of_plane_from_in_plane())

        add_row(oop_range_form, "Range mode", range_mode)
        add_row(oop_range_form, "Center h k l", center_axis)
        add_row(oop_range_form, "Max tilt (deg)", cone_angle)
        add_row(oop_range_form, "Vertex 1 h k l", vertex_widgets[0])
        add_row(oop_range_form, "Vertex 2 h k l", vertex_widgets[1])
        add_row(oop_range_form, "Vertex 3 h k l", vertex_widgets[2])
        add_row(oop_range_form, use_in_plane_sync)
        add_row(oop_range_form, sync_from_in_plane)
        if mode != "out_of_plane":
            oop_range_group.hide()
        range_mode.currentIndexChanged.connect(lambda _value: self._out_of_plane_range_mode_changed(mode))

        match_group = QGroupBox("Orientation Matching")
        match_form = QFormLayout(match_group)
        accel_voltage = _double_box(300000.0, 1000.0, 2000000.0, 10000.0)
        corr_kernel = _double_box(0.08, 0.001, 2.0, 0.005)
        sigma_excitation = _double_box(0.02, 0.0, 2.0, 0.005)
        zone_step = _double_box(2.0, 0.1, 30.0, 0.5)
        in_plane_step = _double_box(2.0, 0.1, 30.0, 0.5)
        min_peaks = _int_box(3, 1, 100)
        num_matches = _int_box(1, 1, 10)
        cif_workers = _int_box(_default_worker_count(), 1, max(1, os.cpu_count() or 1))
        rotation_symmetry = _int_box(1, 1, 12)
        use_cuda = QCheckBox("Use CUDA")
        use_cuda.setChecked(orientation_cuda_available())
        if not orientation_cuda_available():
            use_cuda.setEnabled(False)
            use_cuda.setToolTip("CuPy CUDA is not available in the 4dlab environment.")
        build_plan = QPushButton("Build Plan / Template")
        build_plan.clicked.connect(lambda: self.build_orientation_plan(mode))
        preview = QPushButton("Preview Current")
        preview.clicked.connect(lambda: self.preview_current_orientation(mode))
        fit_preview = QPushButton("Fit Preview")
        fit_preview.clicked.connect(lambda: self.show_fit_preview(mode))
        run_map = QPushButton("Run Map")
        run_map.clicked.connect(lambda: self.run_orientation_map(mode))
        stop = QPushButton("Stop")
        stop.setEnabled(False)
        stop.clicked.connect(self.stop_orientation)

        add_row(match_form, "Voltage (V)", accel_voltage)
        add_row(match_form, "Correlation kernel", corr_kernel)
        add_row(match_form, "Excitation sigma", sigma_excitation)
        add_row(match_form, "Zone-axis step (deg)", zone_step)
        add_row(match_form, "In-plane step (deg)", in_plane_step)
        add_row(match_form, "Min peaks", min_peaks)
        add_row(match_form, "Matches", num_matches)
        add_row(match_form, "CIF CPU workers", cif_workers)
        add_row(match_form, "Rotation symmetry", rotation_symmetry)
        add_row(match_form, use_cuda)
        add_row(match_form, build_plan)
        add_row(match_form, preview)
        add_row(match_form, fit_preview)
        add_row(match_form, run_map)
        add_row(match_form, stop)

        no_cif_group = QGroupBox("No-CIF Experimental Template")
        no_cif_form = QFormLayout(no_cif_group)
        no_cif_tolerance = _double_box(3.0, 0.05, 1000.0, 0.25)
        no_cif_center_exclusion = _double_box(2.0, 0.0, 1000.0, 0.5)
        no_cif_workers = _int_box(_default_worker_count(), 1, max(1, os.cpu_count() or 1))
        add_row(no_cif_form, "Tolerance (pixels)", no_cif_tolerance)
        add_row(no_cif_form, "Center exclude (pixels)", no_cif_center_exclusion)
        add_row(no_cif_form, "CPU workers", no_cif_workers)

        result_group = QGroupBox("Results and Display")
        result_layout = QVBoxLayout(result_group)
        bvm = QPushButton("BVM Centered")
        bvm.clicked.connect(lambda: self.show_centered_bvm(mode))
        save_image = QPushButton("Save Result Image")
        save_image.clicked.connect(lambda: self.save_orientation_image(mode))
        display = QComboBox()
        display.addItems(["Orientation color", "Correlation", "In-plane angle", "Angle 0", "Angle 1"])
        if mode == "out_of_plane":
            display.insertItem(1, "py4DSTEM orientation map")
        display.currentIndexChanged.connect(lambda: self.refresh_orientation_display(mode))
        color_wheel = QLabel()
        color_wheel.setAlignment(Qt.AlignCenter)
        color_wheel.setPixmap(_color_wheel_pixmap(128, rotation_symmetry.value()))
        color_wheel.setToolTip("In-plane rotation color wheel")
        if mode != "in_plane":
            color_wheel.hide()
        rotation_symmetry.valueChanged.connect(lambda _value: self._rotation_symmetry_changed(mode))
        method.currentIndexChanged.connect(lambda _value: self._orientation_method_changed(mode))

        result_layout.addWidget(bvm)
        result_layout.addWidget(QLabel("Map display"))
        result_layout.addWidget(display)
        result_layout.addWidget(save_image)
        result_layout.addWidget(color_wheel)

        image = pg.ImageView()
        image.ui.roiBtn.hide()
        image.ui.menuBtn.hide()
        image.ui.histogram.hide()
        plot = pg.PlotWidget()
        plot.setMinimumHeight(260)
        _style_plot(plot)
        plot.setLabel("bottom", "q")
        plot.setLabel("left", "intensity")
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setFormat("Idle")
        log = QTextEdit()
        log.setReadOnly(True)
        log.setMinimumHeight(120)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(method_group)
        left_layout.addWidget(calibration_group)
        left_layout.addWidget(cif_model_group)
        left_layout.addWidget(oop_range_group)
        left_layout.addWidget(match_group)
        left_layout.addWidget(no_cif_group)
        left_layout.addWidget(result_group)
        left_layout.addStretch(1)
        left_scroll = _scroll_area(left_panel)

        layout.addWidget(left_scroll, 0, 0, 2, 1)
        image_panel = _aspect_panel(image, aspect=2.05) if mode == "out_of_plane" else _square_panel(image)
        layout.addWidget(image_panel, 0, 1, 1, 1)
        layout.addWidget(plot, 1, 1, 1, 1)
        layout.addWidget(progress, 3, 0, 1, 2)
        layout.addWidget(log, 4, 0, 1, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnMinimumWidth(0, 300)
        layout.setRowStretch(0, 4)
        layout.setRowStretch(1, 3)
        layout.setRowStretch(4, 1)

        self.orientation_controls[mode] = {
            "method": method,
            "cif_path": cif_path,
            "q_pixel_size": q_pixel_size,
            "qx_center": qx_center,
            "qy_center": qy_center,
            "flip_qy": flip_qy,
            "k_max": k_max,
            "accel_voltage": accel_voltage,
            "corr_kernel": corr_kernel,
            "sigma_excitation": sigma_excitation,
            "zone_step": zone_step,
            "in_plane_step": in_plane_step,
            "axis_h": axis_h,
            "axis_k": axis_k,
            "axis_l": axis_l,
            "fiber_axis_widget": fiber_axis,
            "zone_tol": zone_tol,
            "q_pixel_source": q_pixel_source,
            "q_pixel_scan_y": q_pixel_scan_y,
            "q_pixel_scan_x": q_pixel_scan_x,
            "q_pixel_scan_widget": q_pixel_scan_widget,
            "q_pixel_min": q_pixel_min,
            "q_pixel_max": q_pixel_max,
            "q_pixel_steps": q_pixel_steps,
            "oop_range_group": oop_range_group,
            "range_mode": range_mode,
            "center_h": center_h,
            "center_k": center_k,
            "center_l": center_l,
            "center_axis_widget": center_axis,
            "cone_angle": cone_angle,
            "vertex_boxes": vertex_boxes,
            "vertex_widgets": vertex_widgets,
            "use_in_plane_sync": use_in_plane_sync,
            "sync_from_in_plane": sync_from_in_plane,
            "min_peaks": min_peaks,
            "num_matches": num_matches,
            "cif_workers": cif_workers,
            "rotation_symmetry": rotation_symmetry,
            "no_cif_symmetry": rotation_symmetry,
            "no_cif_tolerance": no_cif_tolerance,
            "no_cif_center_exclusion": no_cif_center_exclusion,
            "no_cif_workers": no_cif_workers,
            "use_cuda": use_cuda,
            "stop": stop,
            "display": display,
            "color_wheel": color_wheel,
            "cif_model_group": cif_model_group,
            "no_cif_group": no_cif_group,
            "image": image,
            "plot": plot,
            "plot_kind": "profile",
            "py4dstem_map_rgb": None,
            "progress": progress,
            "log": log,
            "field_forms": field_forms,
            "cif_only": [
                cif_path,
                browse_cif,
                q_pixel_size,
                k_max,
                accel_voltage,
                corr_kernel,
                sigma_excitation,
                zone_step,
                fiber_axis,
                zone_tol,
                q_pixel_source,
                q_pixel_scan_widget,
                use_viewer_pixel,
                preview_cif,
                q_pixel_min,
                q_pixel_max,
                q_pixel_steps,
                scan_pixel,
                diagnose_fit,
                num_matches,
                cif_workers,
                use_cuda,
            ],
            "no_cif_only": [
                no_cif_tolerance,
                no_cif_center_exclusion,
                no_cif_workers,
            ],
        }
        self._orientation_method_changed(mode)
        self._out_of_plane_range_mode_changed(mode)
        return page

    def _build_strain_tab(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        cube = self.viewer.cube
        scan_y = cube.scan_shape[0] if cube is not None else 100
        scan_x = cube.scan_shape[1] if cube is not None else 100

        params = QGroupBox("Lattice Strain")
        form = QFormLayout(params)
        qx_center = _double_box(self._default_qx_center(), -100000.0, 100000.0, 0.5)
        qy_center = _double_box(self._default_qy_center(), -100000.0, 100000.0, 0.5)
        flip_qy = QCheckBox("Flip qy")
        flip_qy.setChecked(True)
        q_pixel_size = _double_box(1.0, 1.0e-8, 1000.0, 0.001)
        q_pixel_size.setDecimals(8)
        g1_x = _double_box(0.0, -100000.0, 100000.0, 0.25)
        g1_y = _double_box(0.0, -100000.0, 100000.0, 0.25)
        g2_x = _double_box(0.0, -100000.0, 100000.0, 0.25)
        g2_y = _double_box(0.0, -100000.0, 100000.0, 0.25)
        match_tolerance = _double_box(3.0, 0.05, 1000.0, 0.25)
        center_exclusion = _double_box(2.0, 0.0, 1000.0, 0.5)
        max_index = _int_box(3, 1, 12)
        min_peaks = _int_box(4, 2, 200)
        strain_workers = _int_box(_default_worker_count(), 1, max(1, os.cpu_count() or 1))
        use_weights = QCheckBox("Intensity weighted fit")
        use_weights.setChecked(True)
        ref_y0 = _int_box(0, 0, 100000)
        ref_y1 = _int_box(scan_y, 0, 100000)
        ref_x0 = _int_box(0, 0, 100000)
        ref_x1 = _int_box(scan_x, 0, 100000)
        rotate_angle = _double_box(0.0, -360.0, 360.0, 1.0)
        flip_theta = QCheckBox("Flip theta sign")
        polar_y = _double_box((scan_y - 1) / 2.0, -100000.0, 100000.0, 0.5)
        polar_x = _double_box((scan_x - 1) / 2.0, -100000.0, 100000.0, 0.5)

        form.addRow("qx center (pixel)", qx_center)
        form.addRow("qy center (pixel)", qy_center)
        form.addRow(flip_qy)
        form.addRow("q pixel size", q_pixel_size)
        form.addRow("g1 x (pixels)", g1_x)
        form.addRow("g1 y (pixels)", g1_y)
        form.addRow("g2 x (pixels)", g2_x)
        form.addRow("g2 y (pixels)", g2_y)
        form.addRow("Match tolerance (pixels)", match_tolerance)
        form.addRow("Center exclude (pixels)", center_exclusion)
        form.addRow("Max h/k index", max_index)
        form.addRow("Min matched peaks", min_peaks)
        form.addRow("CPU workers", strain_workers)
        form.addRow(use_weights)
        form.addRow("Reference y start", ref_y0)
        form.addRow("Reference y stop", ref_y1)
        form.addRow("Reference x start", ref_x0)
        form.addRow("Reference x stop", ref_x1)
        form.addRow("Output rotate (deg)", rotate_angle)
        form.addRow(flip_theta)
        form.addRow("Polar center y", polar_y)
        form.addRow("Polar center x", polar_x)

        guess_basis = QPushButton("Guess Basis From Current")
        guess_basis.clicked.connect(self.guess_strain_basis)
        use_reference = QPushButton("Use Current As Reference")
        use_reference.clicked.connect(self.use_current_as_strain_reference)
        preview_fit = QPushButton("Preview Fit")
        preview_fit.clicked.connect(self.preview_strain_fit)
        run = QPushButton("Run Strain")
        run.clicked.connect(self.run_strain_map)
        stop = QPushButton("Stop")
        stop.setEnabled(False)
        stop.clicked.connect(self.stop_strain)
        save_image = QPushButton("Save Result Image")
        save_image.clicked.connect(self.save_strain_image)
        save_data = QPushButton("Save Result Data")
        save_data.clicked.connect(self.save_strain_data)
        display = QComboBox()
        display.addItems(
            [
                "e_xx",
                "e_yy",
                "e_xy",
                "theta (deg)",
                "e_rr",
                "e_tt",
                "e_rt",
                "error",
                "matched peaks",
                "mask",
            ]
        )
        display.currentIndexChanged.connect(self.refresh_strain_display)

        buttons = QWidget()
        button_layout = QVBoxLayout(buttons)
        button_layout.addWidget(guess_basis)
        button_layout.addWidget(use_reference)
        button_layout.addWidget(preview_fit)
        button_layout.addWidget(run)
        button_layout.addWidget(stop)
        button_layout.addWidget(save_image)
        button_layout.addWidget(save_data)
        button_layout.addWidget(display)
        button_layout.addStretch(1)

        image = pg.ImageView()
        image.ui.roiBtn.hide()
        image.ui.menuBtn.hide()
        image.ui.histogram.hide()
        _set_image_gradient(image, "e_xx")
        plot = pg.PlotWidget()
        plot.setMinimumHeight(260)
        _style_fit_plot(plot)
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setFormat("Idle")
        log = QTextEdit()
        log.setReadOnly(True)
        log.setMinimumHeight(120)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(params)
        left_layout.addWidget(buttons)
        left_layout.addStretch(1)
        left_scroll = _scroll_area(left_panel)

        layout.addWidget(left_scroll, 0, 0, 2, 1)
        layout.addWidget(_square_panel(image), 0, 1, 1, 1)
        layout.addWidget(plot, 1, 1, 1, 1)
        layout.addWidget(progress, 2, 0, 1, 2)
        layout.addWidget(log, 3, 0, 1, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnMinimumWidth(0, 300)
        layout.setRowStretch(0, 4)
        layout.setRowStretch(1, 3)
        layout.setRowStretch(3, 1)

        self.strain_controls = {
            "qx_center": qx_center,
            "qy_center": qy_center,
            "flip_qy": flip_qy,
            "q_pixel_size": q_pixel_size,
            "g1_x": g1_x,
            "g1_y": g1_y,
            "g2_x": g2_x,
            "g2_y": g2_y,
            "match_tolerance": match_tolerance,
            "center_exclusion": center_exclusion,
            "max_index": max_index,
            "min_peaks": min_peaks,
            "strain_workers": strain_workers,
            "use_weights": use_weights,
            "ref_y0": ref_y0,
            "ref_y1": ref_y1,
            "ref_x0": ref_x0,
            "ref_x1": ref_x1,
            "rotate_angle": rotate_angle,
            "flip_theta": flip_theta,
            "polar_y": polar_y,
            "polar_x": polar_x,
            "display": display,
            "image": image,
            "plot": plot,
            "progress": progress,
            "log": log,
            "stop": stop,
        }
        return page

    def _placeholder(self, text: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        return page

    def _config(self) -> PeakDetectionConfig:
        return PeakDetectionConfig(
            smooth_sigma=self.smooth_sigma.value(),
            edge_boundary=self.edge_boundary.value(),
            min_relative_intensity=self.min_relative.value(),
            min_absolute_intensity=self.min_absolute.value(),
            min_peak_spacing=self.min_spacing.value(),
            max_num_peaks=self.max_peaks.value(),
            gaussian_radius=self.gaussian_radius.value(),
            refine=self.refine.isChecked(),
            refine_method="gaussian" if self.refine_method.currentText() == "Gaussian fit" else "centroid",
            use_gpu=self.use_gpu.isChecked() and self.use_gpu.isEnabled(),
            num_workers=self.peak_workers.value(),
        )

    def save_peak_preset(self) -> None:
        path = _preset_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._preset_data(), indent=2), encoding="utf-8")
        self._log(f"Saved peak detection preset: {path}")

    def load_peak_preset(self, auto: bool = False) -> None:
        path = _preset_path()
        if not path.exists():
            if not auto:
                self._log(f"No preset found: {path}")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._apply_preset_data(data)
        except Exception as exc:
            self._log(f"Failed to load preset {path}: {exc}")
            return
        self._log(f"Loaded peak detection preset: {path}")

    def _preset_data(self) -> dict[str, float | int | bool]:
        return {
            "smooth_sigma": self.smooth_sigma.value(),
            "edge_boundary": self.edge_boundary.value(),
            "min_relative_intensity": self.min_relative.value(),
            "min_absolute_intensity": self.min_absolute.value(),
            "min_peak_spacing": self.min_spacing.value(),
            "max_num_peaks": self.max_peaks.value(),
            "gaussian_radius": self.gaussian_radius.value(),
            "refine": self.refine.isChecked(),
            "refine_method": "gaussian" if self.refine_method.currentText() == "Gaussian fit" else "centroid",
            "num_workers": self.peak_workers.value(),
            "use_gpu": self.use_gpu.isChecked() and self.use_gpu.isEnabled(),
        }

    def _apply_preset_data(self, data: dict) -> None:
        self.smooth_sigma.setValue(float(data.get("smooth_sigma", self.smooth_sigma.value())))
        self.edge_boundary.setValue(int(data.get("edge_boundary", self.edge_boundary.value())))
        self.min_relative.setValue(
            float(data.get("min_relative_intensity", self.min_relative.value()))
        )
        self.min_absolute.setValue(
            float(data.get("min_absolute_intensity", self.min_absolute.value()))
        )
        self.min_spacing.setValue(int(data.get("min_peak_spacing", self.min_spacing.value())))
        self.max_peaks.setValue(int(data.get("max_num_peaks", self.max_peaks.value())))
        self.gaussian_radius.setValue(
            int(data.get("gaussian_radius", self.gaussian_radius.value()))
        )
        self.refine.setChecked(bool(data.get("refine", self.refine.isChecked())))
        method = str(data.get("refine_method", "centroid"))
        self.refine_method.setCurrentText("Gaussian fit" if method == "gaussian" else "Fast centroid")
        self.peak_workers.setValue(int(data.get("num_workers", self.peak_workers.value())))
        self.use_gpu.setChecked(bool(data.get("use_gpu", False)) and self.use_gpu.isEnabled())

    def _refresh_pattern_preview(self) -> None:
        cube = self.viewer.cube
        if cube is None:
            self._log("No datacube loaded.")
            return
        pattern = cube.diffraction_pattern(self.viewer.scan_y.value(), self.viewer.scan_x.value())
        self.preview.setImage(pattern.T, autoLevels=False, levels=_levels(pattern))
        self.scatter.setData([])
        self._add_analysis_result(
            AnalysisResult(
                key="current_pattern",
                title="Current Pattern",
                kind="diffraction",
                image=pattern,
            )
        )

    def detect_current_pattern(self) -> None:
        cube = self.viewer.cube
        if cube is None:
            self._log("No datacube loaded.")
            return
        pattern = cube.diffraction_pattern(self.viewer.scan_y.value(), self.viewer.scan_x.value())
        peaks = detect_peaks_in_pattern(pattern, self._config())
        self.current_peaks = peaks
        self._show_peaks(peaks)
        points = np.column_stack((peaks.qx, peaks.qy)) if peaks.count else np.zeros((0, 2))
        self._add_analysis_result(
            AnalysisResult(
                key="detected_peaks",
                title="Detected Peaks",
                kind="peaks",
                image=pattern,
                overlays=[AnalysisOverlay(points=points)],
            )
        )
        self._log(
            f"Current pattern ({self.viewer.scan_y.value()}, {self.viewer.scan_x.value()}): "
            f"{peaks.count} peaks"
        )

    def stop_peak_detection(self) -> None:
        if self._peak_task is not None:
            self._peak_task.request_stop()
        self.progress.setFormat("Stopping after current pattern...")
        self._log("Stop requested for full-scan peak detection.")

    def detect_all_patterns(self) -> None:
        cube = self.viewer.cube
        if cube is None:
            self._log("No datacube loaded.")
            return
        if self._peak_task is not None:
            self._log("Peak detection is already running.")
            return
        config = self._config()

        def progress(done: int, total: int) -> None:
            self.progress.setRange(0, total)
            self.progress.setValue(done)
            self.progress.setFormat(f"Detecting peaks {done}/{total}")

        def task(progress_emit, stop_requested):
            return detect_peaks_in_datacube(
                cube,
                config,
                progress=progress_emit,
                stop_requested=stop_requested,
            )

        def finished(result: PeakDetectionResult) -> None:
            self.all_peaks = result
            count_map = self.all_peaks.peak_count_map()
            self.progress.setValue(self.progress.maximum())
            self.progress.setFormat("Peak detection complete")
            self._log(
                "Finished all-pattern peak detection. "
                f"count range {count_map.min()}..{count_map.max()}, "
                f"mean {count_map.mean():.2f}"
            )
            self._show_current_peaks_from_all()

        def failed(exc: BaseException) -> None:
            self.progress.setFormat("Peak detection failed")
            self._log(f"Peak detection failed: {exc}")

        def cancelled(message: str) -> None:
            self.progress.setFormat("Peak detection stopped")
            self._log(message)

        def done() -> None:
            self.stop_detection.setEnabled(False)
            self._peak_task = None

        self._log("Detecting peaks for all scan positions...")
        if self.use_gpu.isChecked() and not self.use_gpu.isEnabled():
            self._log("GPU peak detection requested but CuPy is unavailable; using CPU.")
        self.stop_detection.setEnabled(True)
        self.progress.setValue(0)
        self.progress.setFormat("Starting peak detection...")
        self._peak_task = start_background_task(
            self,
            task,
            cancelled_exception=PeakDetectionCancelled,
            on_progress=progress,
            on_finished=finished,
            on_failed=failed,
            on_cancelled=cancelled,
            on_done=done,
        )

    def save_detected_peaks(self) -> None:
        if self.all_peaks is None:
            self._log("No full-scan peak result to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save detected peaks",
            str(_project_root() / "configs" / "detected_peaks.npz"),
            "Peak detection result (*.npz)",
        )
        if not path:
            return
        save_peak_detection_result(self.all_peaks, path)
        self._log(f"Saved detected peaks: {path}")

    def load_detected_peaks(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load detected peaks",
            str(_project_root() / "configs"),
            "Peak detection result (*.npz)",
        )
        if not path:
            return
        try:
            result = load_peak_detection_result(path)
        except Exception as exc:
            self._log(f"Failed to load detected peaks {path}: {exc}")
            return
        self.all_peaks = result
        count_map = result.peak_count_map()
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.progress.setFormat("Loaded peak result")
        self._log(
            f"Loaded detected peaks: {path}. "
            f"count range {count_map.min()}..{count_map.max()}, "
            f"mean {count_map.mean():.2f}"
        )
        self._show_current_peaks_from_all()

    def _show_current_peaks_from_all(self) -> None:
        if self.all_peaks is None:
            return
        y = self.viewer.scan_y.value()
        x = self.viewer.scan_x.value()
        if y >= self.all_peaks.scan_shape[0] or x >= self.all_peaks.scan_shape[1]:
            self._log("Loaded peak result does not match the current scan coordinates.")
            return
        peaks = self.all_peaks.peaks[y][x]
        self.current_peaks = peaks
        self._show_peaks(peaks)
        self._log(f"Showing loaded peaks at ({y}, {x}): {peaks.count} peaks")

    def show_centered_bvm_peak_tab(self) -> None:
        if self.all_peaks is None:
            self._log("No full-scan peak result. Run or load peak detection first.")
            return
        bvm = build_bvm_from_peaks(
            self.all_peaks,
            qx_center=self._default_qx_center(),
            qy_center=self._default_qy_center(),
        )
        self.preview.setImage(bvm.T, autoLevels=True)
        self.scatter.setData([])
        self._log("Displayed BVM from centered peak data.")

    def browse_orientation_cif(self, mode: str) -> None:
        controls = self.orientation_controls[mode]
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CIF",
            str(_project_root()),
            "CIF files (*.cif);;All files (*)",
        )
        if path:
            controls["cif_path"].setText(path)

    def show_centered_bvm(self, mode: str) -> None:
        if self.all_peaks is None:
            self._orientation_log(mode, "No full-scan peak result. Run or load peak detection first.")
            return
        controls = self.orientation_controls[mode]
        bvm = build_bvm_from_peaks(
            self.all_peaks,
            qx_center=controls["qx_center"].value(),
            qy_center=controls["qy_center"].value(),
        )
        self.orientation_bvms[mode] = bvm
        controls["image"].setImage(bvm.T, autoLevels=True)
        self._plot_bvm_radial_profile(mode, bvm)
        self._add_analysis_result(
            AnalysisResult(
                key=f"{mode}_bvm",
                title="BVM" if mode == "in_plane" else "Out BVM",
                kind="bvm",
                image=bvm,
            )
        )
        self._orientation_log(mode, "Displayed BVM from centered data.")

    def use_viewer_q_pixel(self, mode: str) -> None:
        controls = self.orientation_controls[mode]
        y, x = self._viewer_scan_position()
        controls["q_pixel_scan_y"].setValue(y)
        controls["q_pixel_scan_x"].setValue(x)
        controls["q_pixel_source"].setCurrentText("Current scan pixel")
        self._orientation_log(mode, f"q-pixel radial source set to viewer scan ({y}, {x}).")

    def preview_cif_zone(self, mode: str) -> None:
        if self._is_no_cif_mode(mode):
            self._orientation_log(mode, "CIF diffraction preview is available in CIF indexing mode.")
            return
        if mode == "out_of_plane":
            self._sync_out_of_plane_if_requested()
        controls = self.orientation_controls[mode]
        crystal = self._orientation_crystal(mode)
        if crystal is None:
            return
        try:
            preview = cif_zone_preview(
                crystal,
                controls["k_max"].value(),
                zone_axis=self._orientation_zone_axis(mode),
                zone_z_tol=controls["zone_tol"].value(),
                sigma_excitation_error=controls["sigma_excitation"].value(),
            )
        except Exception as exc:
            self._orientation_log(mode, f"CIF diffraction preview failed: {exc}")
            return
        self._plot_cif_zone_preview(mode, preview)
        self._add_analysis_result(
            AnalysisResult(
                key=f"{mode}_cif_preview",
                title="CIF Diffraction Preview",
                kind="cif_preview",
                image=_spot_raster(preview.qx, preview.qy, preview.intensity),
            )
        )
        h, k, l = self._orientation_zone_axis(mode)
        self._orientation_log(
            mode,
            f"CIF diffraction preview for zone [{h:.3g} {k:.3g} {l:.3g}] "
            f"with {preview.qx.size} spots.",
        )

    def scan_orientation_q_pixel(self, mode: str) -> None:
        if self._is_no_cif_mode(mode):
            self._orientation_log(mode, "q pixel scan uses CIF structure factors; switch to CIF indexing first.")
            return
        if mode == "out_of_plane":
            self._sync_out_of_plane_if_requested()
        controls = self.orientation_controls[mode]
        source_name = controls["q_pixel_source"].currentText()
        if source_name == "Current scan pixel":
            peaks = self._orientation_peaks_at(
                controls["q_pixel_scan_y"].value(),
                controls["q_pixel_scan_x"].value(),
            )
            if peaks is None:
                self._orientation_log(mode, "No peaks are available for the selected scan pixel.")
                return
            radial_source = peak_list_radial_profile(
                peaks,
                qx_center=controls["qx_center"].value(),
                qy_center=controls["qy_center"].value(),
            )
        else:
            bvm = self.orientation_bvms.get(mode)
            if bvm is None:
                if self.all_peaks is None:
                    self._orientation_log(mode, "No full-scan peak result. Run or load peak detection first.")
                    return
                bvm = build_bvm_from_peaks(
                    self.all_peaks,
                    qx_center=controls["qx_center"].value(),
                    qy_center=controls["qy_center"].value(),
                )
                self.orientation_bvms[mode] = bvm
            radial_source = bvm
        crystal = self._orientation_crystal(mode)
        if crystal is None:
            return
        try:
            scan = scan_q_pixel_sizes(
                radial_source,
                crystal,
                k_max=controls["k_max"].value(),
                pixel_min=controls["q_pixel_min"].value(),
                pixel_max=controls["q_pixel_max"].value(),
                steps=controls["q_pixel_steps"].value(),
                zone_axis=self._orientation_zone_axis(mode),
                zone_z_tol=controls["zone_tol"].value(),
                sigma_excitation_error=controls["sigma_excitation"].value(),
            )
        except Exception as exc:
            self._orientation_log(mode, f"q pixel scan failed: {exc}")
            return
        controls["q_pixel_size"].setValue(scan.best_pixel_size)
        self._plot_q_pixel_scan(mode, scan)
        suffix = ""
        if source_name == "Current scan pixel":
            suffix = f" using scan ({controls['q_pixel_scan_y'].value()}, {controls['q_pixel_scan_x'].value()})"
        self._orientation_log(
            mode,
            f"Best q pixel size from {source_name} radial/SF scan{suffix}: "
            f"{scan.best_pixel_size:.6g} A^-1/pixel",
        )

    def diagnose_cif_fit(self, mode: str) -> None:
        if self._is_no_cif_mode(mode):
            self._orientation_log(mode, "CIF fit diagnostics are available in CIF indexing mode.")
            return
        if mode == "out_of_plane":
            self._sync_out_of_plane_if_requested()
        peaks = self._current_orientation_peaks()
        if peaks is None:
            self._orientation_log(mode, "No current peaks. Detect or load peaks first.")
            return
        controls = self.orientation_controls[mode]
        crystal = self._orientation_crystal(mode)
        if crystal is None:
            return
        try:
            diagnostic = diagnose_cif_fit(
                peaks,
                crystal,
                qx_center=controls["qx_center"].value(),
                qy_center=controls["qy_center"].value(),
                flip_qy=controls["flip_qy"].isChecked(),
                q_pixel_size=controls["q_pixel_size"].value(),
                k_max=controls["k_max"].value(),
                zone_axis=self._orientation_zone_axis(mode),
                zone_z_tol=controls["zone_tol"].value(),
                sigma_excitation_error=controls["sigma_excitation"].value(),
            )
        except Exception as exc:
            self._orientation_log(mode, f"CIF fit diagnostic failed: {exc}")
            return
        self._plot_cif_fit_diagnostic(mode, diagnostic)
        self._add_analysis_result(
            AnalysisResult(
                key=f"{mode}_fit_diagnostic",
                title="CIF Fit Diagnostic",
                kind="fit_diagnostic",
                image=_diagnostic_raster(diagnostic),
            )
        )
        y, x = self._viewer_scan_position()
        self._orientation_log(
            mode,
            f"CIF diagnostic at ({y}, {x}): matched {diagnostic.matched_count} peaks, "
            f"median residual {diagnostic.median_residual:.4g} A^-1, "
            f"mean residual {diagnostic.mean_residual:.4g} A^-1.",
        )
        for suggestion in diagnostic.suggestions:
            self._orientation_log(mode, f"Diagnostic suggestion: {suggestion}.")

    def build_orientation_plan(self, mode: str) -> None:
        if self._is_no_cif_mode(mode):
            self.build_no_cif_template(mode)
            return
        if mode == "out_of_plane":
            self._sync_out_of_plane_if_requested()
        controls = self.orientation_controls[mode]
        try:
            config = self._orientation_config(mode)
            controls["progress"].setRange(0, 0)
            controls["progress"].setFormat("Building orientation plan...")
            workspace = build_orientation_workspace(config)
        except Exception as exc:
            controls["progress"].setRange(0, 100)
            controls["progress"].setValue(0)
            controls["progress"].setFormat("Plan failed")
            self._orientation_log(mode, f"Failed to build orientation plan: {exc}")
            return
        controls["progress"].setRange(0, 100)
        controls["progress"].setValue(100)
        controls["progress"].setFormat("Plan ready")
        self.orientation_workspaces[mode] = workspace
        self.orientation_crystals[mode] = workspace.crystal
        self._orientation_log(mode, f"Orientation plan ready for {mode.replace('_', '-')}.")

    def build_no_cif_template(self, mode: str) -> None:
        controls = self.orientation_controls[mode]
        peaks = self._current_orientation_peaks()
        if peaks is None:
            self._orientation_log(mode, "No current peaks. Detect or load peaks first.")
            return
        y = self.viewer.scan_y.value()
        x = self.viewer.scan_x.value()
        try:
            template = build_no_cif_template(
                peaks,
                self._no_cif_config(mode),
                source_scan=(y, x),
            )
        except Exception as exc:
            controls["progress"].setFormat("Template failed")
            self._orientation_log(mode, f"Failed to build no-CIF template: {exc}")
            return
        self.no_cif_templates[mode] = template
        controls["progress"].setRange(0, 100)
        controls["progress"].setValue(100)
        controls["progress"].setFormat("Template ready")
        self._show_no_cif_fit_overlay(mode, template, peaks, 0.0)
        self._orientation_log(
            mode,
            f"No-CIF template ready from scan ({y}, {x}) with {template.vectors_px.shape[0]} peaks.",
        )

    def preview_current_orientation(self, mode: str) -> None:
        if self._is_no_cif_mode(mode):
            self.preview_current_no_cif(mode)
            return
        workspace = self._orientation_workspace(mode)
        if workspace is None:
            return
        peaks = self._current_orientation_peaks()
        if peaks is None:
            self._orientation_log(mode, "No current peaks. Detect or load peaks first.")
            return
        try:
            orientation = match_current_orientation(workspace, peaks)
        except Exception as exc:
            self._orientation_log(mode, f"Preview orientation failed: {exc}")
            return
        corr = float(orientation.corr[0]) if orientation.corr.size else 0.0
        angles = np.rad2deg(orientation.angles[0]) if orientation.angles.size else np.zeros(3)
        self._orientation_log(
            mode,
            "Current orientation: "
            f"corr {corr:.4g}, angles deg "
            f"[{angles[0]:.2f}, {angles[1]:.2f}, {angles[2]:.2f}]",
        )
        self._show_fit_overlay(mode, workspace, peaks, orientation)

    def show_fit_preview(self, mode: str) -> None:
        if self._is_no_cif_mode(mode):
            self.show_no_cif_fit_preview(mode)
            return
        workspace = self._orientation_workspace(mode)
        if workspace is None:
            return
        peaks = self._current_orientation_peaks()
        if peaks is None:
            self._orientation_log(mode, "No current peaks. Detect or load peaks first.")
            return
        y = self.viewer.scan_y.value()
        x = self.viewer.scan_x.value()
        orientation = None
        result = self.orientation_results.get(mode)
        if result is not None and y < result.raw.num_x and x < result.raw.num_y:
            orientation = result.raw.get_orientation(y, x)
        if orientation is None:
            try:
                orientation = match_current_orientation(workspace, peaks)
            except Exception as exc:
                self._orientation_log(mode, f"Fit preview failed: {exc}")
                return
        self._show_fit_overlay(mode, workspace, peaks, orientation)
        corr = float(orientation.corr[0]) if orientation.corr.size else 0.0
        self._orientation_log(mode, f"Fit preview at ({y}, {x}) corr {corr:.4g}.")

    def run_orientation_map(self, mode: str) -> None:
        if self._is_no_cif_mode(mode):
            self.run_no_cif_orientation_map(mode)
            return
        workspace = self._orientation_workspace(mode)
        if workspace is None:
            return
        if self.all_peaks is None:
            self._orientation_log(mode, "No full-scan peak result. Run or load peak detection first.")
            return
        if self._orientation_task is not None:
            self._orientation_log(mode, "An orientation map is already running.")
            return
        controls = self.orientation_controls[mode]
        peaks = self.all_peaks

        def progress(done: int, total: int) -> None:
            controls["progress"].setRange(0, total)
            controls["progress"].setValue(done)
            controls["progress"].setFormat(f"Matching orientations {done}/{total}")

        def task(progress_emit, stop_requested):
            return match_orientation_map(
                workspace,
                peaks,
                progress=progress_emit,
                stop_requested=stop_requested,
            )

        def finished(result) -> None:
            self.orientation_results[mode] = result
            controls["progress"].setValue(controls["progress"].maximum())
            controls["progress"].setFormat("Orientation map complete")
            self.refresh_orientation_display(mode)
            self.show_fit_preview(mode)
            self._orientation_log(
                mode,
                f"Orientation map complete. corr range "
                f"{np.nanmin(result.corr):.4g}..{np.nanmax(result.corr):.4g}",
            )

        def failed(exc: BaseException) -> None:
            controls["progress"].setFormat("Orientation failed")
            self._orientation_log(mode, f"Orientation map failed: {exc}")

        def cancelled(message: str) -> None:
            controls["progress"].setFormat("Orientation stopped")
            self._orientation_log(mode, message)

        def done() -> None:
            controls["stop"].setEnabled(False)
            self._orientation_task = None

        controls["stop"].setEnabled(True)
        controls["progress"].setValue(0)
        controls["progress"].setFormat("Starting orientation map...")
        self._orientation_task = start_background_task(
            self,
            task,
            cancelled_exception=OrientationCancelled,
            on_progress=progress,
            on_finished=finished,
            on_failed=failed,
            on_cancelled=cancelled,
            on_done=done,
        )

    def preview_current_no_cif(self, mode: str) -> None:
        template = self._no_cif_template(mode)
        if template is None:
            return
        peaks = self._current_orientation_peaks()
        if peaks is None:
            self._orientation_log(mode, "No current peaks. Detect or load peaks first.")
            return
        angle, corr = match_no_cif_in_plane(template, peaks, self._no_cif_config(mode))
        self._show_no_cif_fit_overlay(mode, template, peaks, angle)
        self._orientation_log(mode, f"No-CIF current rotation {angle:.2f} deg, score {corr:.4g}.")

    def show_no_cif_fit_preview(self, mode: str) -> None:
        template = self._no_cif_template(mode)
        if template is None:
            return
        peaks = self._current_orientation_peaks()
        if peaks is None:
            self._orientation_log(mode, "No current peaks. Detect or load peaks first.")
            return
        y = self.viewer.scan_y.value()
        x = self.viewer.scan_x.value()
        result = self.orientation_results.get(mode)
        if result is not None and y < result.corr.shape[0] and x < result.corr.shape[1]:
            angle = float(result.in_plane_angle_deg[y, x])
            corr = float(result.corr[y, x])
        else:
            angle, corr = match_no_cif_in_plane(template, peaks, self._no_cif_config(mode))
        self._show_no_cif_fit_overlay(mode, template, peaks, angle)
        self._orientation_log(mode, f"No-CIF fit preview at ({y}, {x}): {angle:.2f} deg, score {corr:.4g}.")

    def run_no_cif_orientation_map(self, mode: str) -> None:
        template = self._no_cif_template(mode)
        if template is None:
            return
        if self.all_peaks is None:
            self._orientation_log(mode, "No full-scan peak result. Run or load peak detection first.")
            return
        if self._orientation_task is not None:
            self._orientation_log(mode, "An orientation map is already running.")
            return
        controls = self.orientation_controls[mode]
        peaks = self.all_peaks
        config = self._no_cif_config(mode)

        def progress(done: int, total: int) -> None:
            controls["progress"].setRange(0, total)
            controls["progress"].setValue(done)
            controls["progress"].setFormat(f"No-CIF matching {done}/{total}")

        def task(progress_emit, stop_requested):
            return match_no_cif_orientation_map(
                template,
                peaks,
                config,
                progress=progress_emit,
                stop_requested=stop_requested,
            )

        def finished(result) -> None:
            self.orientation_results[mode] = result
            controls["progress"].setValue(controls["progress"].maximum())
            controls["progress"].setFormat("No-CIF map complete")
            self.refresh_orientation_display(mode)
            self.show_no_cif_fit_preview(mode)
            self._orientation_log(
                mode,
                f"No-CIF map complete. score range "
                f"{np.nanmin(result.corr):.4g}..{np.nanmax(result.corr):.4g}",
            )

        def failed(exc: BaseException) -> None:
            controls["progress"].setFormat("No-CIF failed")
            self._orientation_log(mode, f"No-CIF map failed: {exc}")

        def cancelled(message: str) -> None:
            controls["progress"].setFormat("No-CIF stopped")
            self._orientation_log(mode, message)

        def done() -> None:
            controls["stop"].setEnabled(False)
            self._orientation_task = None

        controls["stop"].setEnabled(True)
        controls["progress"].setValue(0)
        controls["progress"].setFormat("Starting no-CIF map...")
        self._orientation_task = start_background_task(
            self,
            task,
            cancelled_exception=OrientationCancelled,
            on_progress=progress,
            on_finished=finished,
            on_failed=failed,
            on_cancelled=cancelled,
            on_done=done,
        )

    def stop_orientation(self) -> None:
        if self._orientation_task is not None:
            self._orientation_task.request_stop()

    def save_orientation_image(self, mode: str) -> None:
        controls = self.orientation_controls[mode]
        default_name = f"{mode}_{controls.get('plot_kind', 'result')}.png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save result image",
            str(_project_root() / default_name),
            "PNG image (*.png);;All files (*)",
        )
        if not path:
            return
        out = Path(path)
        if out.suffix == "":
            out = out.with_suffix(".png")
        try:
            if controls.get("plot_kind") == "py4dstem_map" and controls.get("py4dstem_map_rgb") is not None:
                from matplotlib import pyplot as plt

                plt.imsave(str(out), controls["py4dstem_map_rgb"])
                self._orientation_log(mode, f"Saved result image: {out}")
                return
            if controls.get("plot_kind") == "image":
                from pyqtgraph.exporters import ImageExporter

                exporter = ImageExporter(controls["image"].getView())
                exporter.parameters()["width"] = 1400
                exporter.export(str(out))
                self._orientation_log(mode, f"Saved result image: {out}")
                return
            from pyqtgraph.exporters import ImageExporter

            exporter = ImageExporter(controls["plot"].plotItem)
            exporter.parameters()["width"] = 1400
            exporter.export(str(out))
        except Exception as exc:
            self._orientation_log(mode, f"Failed to save result image: {exc}")
            return
        self._orientation_log(mode, f"Saved result image: {out}")

    def refresh_orientation_display(self, mode: str) -> None:
        result = self.orientation_results.get(mode)
        if result is None:
            return
        controls = self.orientation_controls[mode]
        label = controls["display"].currentText()
        if label == "py4DSTEM orientation map":
            self._show_py4dstem_orientation_map(mode, result)
            return
        if label == "Orientation color":
            image = orientation_color_image(
                result,
                mode,
                symmetry_order=self._rotation_symmetry(mode),
            )
            if mode == "out_of_plane":
                image = self._fallback_orientation_map_with_legend(mode, result)
            controls["image"].setImage(np.transpose(image, (1, 0, 2)), autoLevels=False)
            controls["plot_kind"] = "image"
            self._add_analysis_result(
                AnalysisResult(
                    key=f"{mode}_orientation_color",
                    title="In-Plane Orientation Map" if mode == "in_plane" else "Out-of-Plane Orientation Map",
                    kind="orientation_map",
                    image=image,
                )
            )
            return
        if label == "In-plane angle":
            image = result.in_plane_angle_deg
        elif label == "Angle 0":
            image = result.angle_0_deg
        elif label == "Angle 1":
            image = result.angle_1_deg
        else:
            image = result.corr
        controls["image"].setImage(image.T, autoLevels=True)
        controls["plot_kind"] = "image"

    def _show_py4dstem_orientation_map(self, mode: str, result) -> None:
        controls = self.orientation_controls[mode]
        workspace = self.orientation_workspaces.get(mode)
        if workspace is None or result.raw is None:
            self._orientation_log(mode, "py4DSTEM orientation map needs a built CIF plan and raw orientation result.")
            image = self._fallback_orientation_map_with_legend(mode, result)
        else:
            try:
                rendered = workspace.crystal.plot_orientation_maps(
                    result.raw,
                    orientation_ind=0,
                    corr_range=np.array([1, 4]),
                    camera_dist=10,
                    show_axes=False,
                    show_legend=True,
                    returnfig=True,
                    progress_bar=False,
                )
                fig = self._figure_from_py4dstem_return(rendered)
                image = self._figure_to_rgb(fig)
                self._close_figure(fig)
                image = self._ensure_out_of_plane_legend(mode, result, image)
            except Exception as exc:
                self._orientation_log(mode, f"py4DSTEM orientation map failed; using fallback legend: {exc}")
                image = self._fallback_orientation_map_with_legend(mode, result)
        controls["py4dstem_map_rgb"] = image
        controls["image"].setImage(np.transpose(image, (1, 0, 2)), autoLevels=False)
        controls["plot_kind"] = "py4dstem_map"
        self._add_analysis_result(
            AnalysisResult(
                key=f"{mode}_py4dstem_orientation_map",
                title="Out-of-Plane Orientation Map",
                kind="orientation_map",
                image=image,
            )
        )

    def _figure_from_py4dstem_return(self, rendered):
        if hasattr(rendered, "savefig"):
            return rendered
        if isinstance(rendered, tuple):
            for item in rendered:
                if hasattr(item, "savefig"):
                    return item
        raise ValueError("py4DSTEM did not return a matplotlib figure.")

    def _figure_to_rgb(self, fig) -> np.ndarray:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        return np.ascontiguousarray(rgba[:, :, :3])

    def _close_figure(self, fig) -> None:
        try:
            from matplotlib import pyplot as plt

            plt.close(fig)
        except Exception:
            pass

    def _fallback_orientation_map_with_legend(self, mode: str, result) -> np.ndarray:
        from matplotlib import pyplot as plt

        image = orientation_color_image(result, mode, symmetry_order=self._rotation_symmetry(mode))
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [3.0, 1.15]})
        axes[0].imshow(np.transpose(image, (1, 0, 2)), origin="lower")
        axes[0].set_axis_off()
        axes[0].set_title("Orientation map")
        legend, labels = self._orientation_triangle_legend_image(mode)
        axes[1].imshow(legend, origin="upper")
        axes[1].set_axis_off()
        axes[1].text(0.50, 0.02, labels[0], ha="center", va="bottom", transform=axes[1].transAxes, fontsize=8)
        axes[1].text(0.05, 0.93, labels[1], ha="left", va="top", transform=axes[1].transAxes, fontsize=8)
        axes[1].text(0.95, 0.93, labels[2], ha="right", va="top", transform=axes[1].transAxes, fontsize=8)
        axes[1].set_title("Zone color")
        fig.tight_layout(pad=0.4)
        rgb = self._figure_to_rgb(fig)
        self._close_figure(fig)
        return rgb

    def _orientation_triangle_legend_image(self, mode: str) -> tuple[np.ndarray, list[str]]:
        size = 240
        yy, xx = np.indices((size, size), dtype=np.float64)
        p0 = np.asarray([size / 2.0, size * 0.12])
        p1 = np.asarray([size * 0.12, size * 0.86])
        p2 = np.asarray([size * 0.88, size * 0.86])
        denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        w0 = ((p1[1] - p2[1]) * (xx - p2[0]) + (p2[0] - p1[0]) * (yy - p2[1])) / denom
        w1 = ((p2[1] - p0[1]) * (xx - p2[0]) + (p0[0] - p2[0]) * (yy - p2[1])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
        colors = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.3, 1.0]], dtype=np.float64)
        legend = np.ones((size, size, 3), dtype=np.float64)
        legend[inside] = (
            w0[inside][:, None] * colors[0]
            + w1[inside][:, None] * colors[1]
            + w2[inside][:, None] * colors[2]
        )
        labels = self._orientation_triangle_labels(mode)
        return legend, labels

    def _orientation_triangle_labels(self, mode: str) -> list[str]:
        workspace = self.orientation_workspaces.get(mode)
        values = None
        if workspace is not None:
            values = getattr(workspace.crystal, "orientation_zone_axis_range", None)
        if values is None:
            controls = self.orientation_controls[mode]
            values = np.asarray(
                [
                    (boxes[0].value(), boxes[1].value(), boxes[2].value())
                    for boxes in controls["vertex_boxes"]
                ],
                dtype=np.float64,
            )
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (3, 3):
            return ["v1", "v2", "v3"]
        return [_format_zone_label(row) for row in arr]

    def _ensure_out_of_plane_legend(self, mode: str, result, image: np.ndarray) -> np.ndarray:
        if mode != "out_of_plane":
            return image
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[0] == 0:
            return self._fallback_orientation_map_with_legend(mode, result)
        aspect = float(rgb.shape[1]) / max(float(rgb.shape[0]), 1.0)
        if aspect >= 1.8:
            return image
        self._orientation_log(mode, "py4DSTEM map did not include a wide legend panel; using 4DLAB triangle legend.")
        return self._fallback_orientation_map_with_legend(mode, result)

    def guess_strain_basis(self) -> None:
        peaks = self._current_orientation_peaks()
        if peaks is None:
            self._strain_log("No current peaks. Detect or load peaks first.")
            return
        try:
            g1, g2 = guess_basis_from_peaks(peaks, self._strain_config())
        except Exception as exc:
            self._strain_log(f"Basis guess failed: {exc}")
            return
        controls = self.strain_controls
        controls["g1_x"].setValue(float(g1[0]))
        controls["g1_y"].setValue(float(g1[1]))
        controls["g2_x"].setValue(float(g2[0]))
        controls["g2_y"].setValue(float(g2[1]))
        self._strain_log(
            "Guessed basis from current pattern: "
            f"g1=({g1[0]:.3f}, {g1[1]:.3f}), "
            f"g2=({g2[0]:.3f}, {g2[1]:.3f}) pixels."
        )
        self.preview_strain_fit()

    def use_current_as_strain_reference(self) -> None:
        y, x = self._viewer_scan_position()
        controls = self.strain_controls
        controls["ref_y0"].setValue(y)
        controls["ref_y1"].setValue(y + 1)
        controls["ref_x0"].setValue(x)
        controls["ref_x1"].setValue(x + 1)
        if not self._strain_basis_is_valid():
            self.guess_strain_basis()
        self._strain_log(f"Reference region set to current scan point ({y}, {x}).")

    def preview_strain_fit(self) -> None:
        peaks = self._current_orientation_peaks()
        if peaks is None:
            self._strain_log("No current peaks. Detect or load peaks first.")
            return
        try:
            g1, g2 = self._strain_basis_or_guess(peaks)
            fit = fit_lattice_transform(peaks, g1, g2, self._strain_config())
        except Exception as exc:
            self._strain_log(f"Strain fit preview failed: {exc}")
            return
        self._show_strain_fit(fit)
        y, x = self._viewer_scan_position()
        self._strain_log(
            f"Fit preview at ({y}, {x}): {fit.matched_count} peaks, "
            f"RMS error {fit.error_px:.3g} px."
        )

    def run_strain_map(self) -> None:
        if self.all_peaks is None:
            self._strain_log("No full-scan peak result. Run or load peak detection first.")
            return
        if self._strain_task is not None:
            self._strain_log("A strain map is already running.")
            return
        current = self._current_orientation_peaks()
        if current is None:
            self._strain_log("No current peaks available for basis validation.")
            return
        try:
            g1, g2 = self._strain_basis_or_guess(current)
            config = self._strain_config()
        except Exception as exc:
            self._strain_log(f"Strain setup failed: {exc}")
            return
        controls = self.strain_controls
        peaks = self.all_peaks

        def progress(done: int, total: int) -> None:
            controls["progress"].setRange(0, total)
            controls["progress"].setValue(done)
            controls["progress"].setFormat(f"Fitting strain {done}/{total}")

        def task(progress_emit, stop_requested):
            return calculate_strain_map(
                peaks,
                g1,
                g2,
                config,
                progress=progress_emit,
                stop_requested=stop_requested,
            )

        def finished(result: StrainResult) -> None:
            self.strain_result = result
            controls["progress"].setValue(controls["progress"].maximum())
            controls["progress"].setFormat("Strain map complete")
            self.refresh_strain_display()
            self.preview_strain_fit()
            self._strain_log(
                "Strain map complete. "
                f"valid {int(np.count_nonzero(result.mask))}/{result.mask.size}; "
                f"e_xx {self._array_range_text(result.e_xx)}, "
                f"e_yy {self._array_range_text(result.e_yy)}, "
                f"theta {self._array_range_text(np.rad2deg(result.theta))} deg."
            )

        def failed(exc: BaseException) -> None:
            controls["progress"].setFormat("Strain failed")
            self._strain_log(f"Strain map failed: {exc}")

        def cancelled(message: str) -> None:
            controls["progress"].setFormat("Strain stopped")
            self._strain_log(message)

        def done() -> None:
            controls["stop"].setEnabled(False)
            self._strain_task = None

        controls["stop"].setEnabled(True)
        controls["progress"].setValue(0)
        controls["progress"].setFormat("Starting strain map...")
        self._strain_task = start_background_task(
            self,
            task,
            cancelled_exception=StrainCancelled,
            on_progress=progress,
            on_finished=finished,
            on_failed=failed,
            on_cancelled=cancelled,
            on_done=done,
        )

    def stop_strain(self) -> None:
        if self._strain_task is not None:
            self._strain_task.request_stop()
        self.strain_controls["progress"].setFormat("Stopping after current pattern...")
        self._strain_log("Stop requested for strain fitting.")

    def refresh_strain_display(self, *_args) -> None:
        if self.strain_result is None:
            return
        controls = self.strain_controls
        label = controls["display"].currentText()
        image = self._strain_display_array(label)
        display = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        _set_image_gradient(controls["image"], label)
        controls["image"].setImage(
            display.T,
            autoLevels=False,
            levels=_strain_levels(display, label),
        )

    def save_strain_image(self) -> None:
        if self.strain_result is None:
            self._strain_log("No strain result image to save.")
            return
        label = self.strain_controls["display"].currentText().replace(" ", "_")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save strain image",
            str(_project_root() / f"strain_{label}.png"),
            "PNG image (*.png);;All files (*)",
        )
        if not path:
            return
        out = Path(path)
        if out.suffix == "":
            out = out.with_suffix(".png")
        try:
            from pyqtgraph.exporters import ImageExporter

            exporter = ImageExporter(self.strain_controls["image"].getView())
            exporter.parameters()["width"] = 1400
            exporter.export(str(out))
        except Exception as exc:
            self._strain_log(f"Failed to save strain image: {exc}")
            return
        self._strain_log(f"Saved strain image: {out}")

    def save_strain_data(self) -> None:
        if self.strain_result is None:
            self._strain_log("No strain result data to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save strain data",
            str(_project_root() / "strain_result.npz"),
            "Strain result (*.npz)",
        )
        if not path:
            return
        out = Path(path)
        if out.suffix == "":
            out = out.with_suffix(".npz")
        try:
            save_strain_result(self.strain_result, out)
        except Exception as exc:
            self._strain_log(f"Failed to save strain data: {exc}")
            return
        self._strain_log(f"Saved strain data: {out}")

    def _show_strain_fit(self, fit) -> None:
        controls = self.strain_controls
        config = self._strain_config()
        plot = controls["plot"]
        plot.clear()
        _style_fit_plot(plot)
        plot.getViewBox().setAspectLocked(True, ratio=1.0)
        _ensure_legend(plot)

        x_arrays: list[np.ndarray] = []
        y_arrays: list[np.ndarray] = []
        if fit.observed_px.size:
            exp_x = fit.observed_px[:, 0]
            exp_y = _viewer_display_qy(fit.observed_px[:, 1], config.flip_qy)
            plot.plot(
                exp_x,
                exp_y,
                pen=None,
                symbol="o",
                symbolSize=12,
                symbolBrush=pg.mkBrush(0, 170, 255, 150),
                symbolPen=pg.mkPen("#08aeea", width=1),
                name="observed",
            )
            x_arrays.append(exp_x)
            y_arrays.append(exp_y)
        if fit.predicted_px.size:
            ref_x = fit.predicted_px[:, 0]
            ref_y = _viewer_display_qy(fit.predicted_px[:, 1], config.flip_qy)
            plot.plot(
                ref_x,
                ref_y,
                pen=None,
                symbol="+",
                symbolSize=12,
                symbolPen=pg.mkPen("#1d1d1f", width=1.6),
                name="reference hk",
            )
            x_arrays.append(ref_x)
            y_arrays.append(ref_y)
        if fit.fitted_px.size:
            fit_x = fit.fitted_px[:, 0]
            fit_y = _viewer_display_qy(fit.fitted_px[:, 1], config.flip_qy)
            plot.plot(
                fit_x,
                fit_y,
                pen=None,
                symbol="+",
                symbolSize=17,
                symbolPen=pg.mkPen("#ff3333", width=2.4),
                name="fitted",
            )
            x_arrays.append(fit_x)
            y_arrays.append(fit_y)
            for idx in range(min(fit.hk.shape[0], 30)):
                label = f"{int(fit.hk[idx, 0])},{int(fit.hk[idx, 1])}"
                text = pg.TextItem(label, color="#ff3333", anchor=(0.5, 1.15))
                text.setPos(float(fit_x[idx]), float(fit_y[idx]))
                plot.addItem(text)
        self._set_square_plot_range(plot, x_arrays, y_arrays, fallback_radius=10.0, margin=0.12)

    def _strain_config(self) -> StrainConfig:
        controls = self.strain_controls
        y0 = controls["ref_y0"].value()
        y1 = controls["ref_y1"].value()
        x0 = controls["ref_x0"].value()
        x1 = controls["ref_x1"].value()
        if y1 < y0:
            y0, y1 = y1, y0
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 == y0:
            y1 += 1
        if x1 == x0:
            x1 += 1
        return StrainConfig(
            qx_center=controls["qx_center"].value(),
            qy_center=controls["qy_center"].value(),
            flip_qy=controls["flip_qy"].isChecked(),
            q_pixel_size=controls["q_pixel_size"].value(),
            match_tolerance_px=controls["match_tolerance"].value(),
            min_number_peaks=controls["min_peaks"].value(),
            max_index=controls["max_index"].value(),
            center_exclusion_px=controls["center_exclusion"].value(),
            reference_bounds=(y0, y1, x0, x1),
            rotate_angle_deg=controls["rotate_angle"].value(),
            flip_theta=controls["flip_theta"].isChecked(),
            use_intensity_weights=controls["use_weights"].isChecked(),
            num_workers=controls["strain_workers"].value(),
        )

    def _strain_basis(self) -> tuple[np.ndarray, np.ndarray]:
        controls = self.strain_controls
        g1 = np.asarray([controls["g1_x"].value(), controls["g1_y"].value()], dtype=np.float64)
        g2 = np.asarray([controls["g2_x"].value(), controls["g2_y"].value()], dtype=np.float64)
        return g1, g2

    def _strain_basis_is_valid(self) -> bool:
        g1, g2 = self._strain_basis()
        return (
            np.linalg.norm(g1) > 1.0e-8
            and np.linalg.norm(g2) > 1.0e-8
            and abs(float(g1[0] * g2[1] - g1[1] * g2[0])) > 1.0e-8
        )

    def _strain_basis_or_guess(self, peaks: PeakList) -> tuple[np.ndarray, np.ndarray]:
        if self._strain_basis_is_valid():
            return self._strain_basis()
        g1, g2 = guess_basis_from_peaks(peaks, self._strain_config())
        controls = self.strain_controls
        controls["g1_x"].setValue(float(g1[0]))
        controls["g1_y"].setValue(float(g1[1]))
        controls["g2_x"].setValue(float(g2[0]))
        controls["g2_y"].setValue(float(g2[1]))
        self._strain_log("Basis was empty; guessed it from the current pattern.")
        return g1, g2

    def _strain_display_array(self, label: str) -> np.ndarray:
        result = self.strain_result
        if result is None:
            return np.zeros((1, 1), dtype=np.float64)
        if label == "e_xx":
            return result.e_xx
        if label == "e_yy":
            return result.e_yy
        if label == "e_xy":
            return result.e_xy
        if label == "theta (deg)":
            return np.rad2deg(result.theta)
        if label == "error":
            return result.error_px
        if label == "matched peaks":
            return result.matched_count.astype(np.float64)
        if label == "mask":
            return result.mask.astype(np.float64)
        e_rr, e_tt, e_rt = polar_strain_components(
            result.e_xx,
            result.e_yy,
            result.e_xy,
            center_y=self.strain_controls["polar_y"].value(),
            center_x=self.strain_controls["polar_x"].value(),
        )
        if label == "e_rr":
            return e_rr
        if label == "e_tt":
            return e_tt
        return e_rt

    def _strain_log(self, message: str) -> None:
        self.strain_controls["log"].append(message)

    def _array_range_text(self, values: np.ndarray) -> str:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return "nan"
        return f"{float(np.nanmin(finite)):.4g}..{float(np.nanmax(finite)):.4g}"

    def _orientation_config(self, mode: str) -> OrientationConfig:
        controls = self.orientation_controls[mode]
        return OrientationConfig(
            cif_path=Path(controls["cif_path"].text()),
            mode=mode,
            q_pixel_size=controls["q_pixel_size"].value(),
            qx_center=controls["qx_center"].value(),
            qy_center=controls["qy_center"].value(),
            flip_qy=controls["flip_qy"].isChecked(),
            k_max=controls["k_max"].value(),
            accel_voltage=controls["accel_voltage"].value(),
            corr_kernel_size=controls["corr_kernel"].value(),
            sigma_excitation_error=controls["sigma_excitation"].value(),
            angle_step_zone_axis=controls["zone_step"].value(),
            angle_step_in_plane=controls["in_plane_step"].value(),
            num_matches=controls["num_matches"].value(),
            min_number_peaks=controls["min_peaks"].value(),
            use_cuda=controls["use_cuda"].isChecked() and controls["use_cuda"].isEnabled(),
            num_workers=(
                1
                if controls["use_cuda"].isChecked() and controls["use_cuda"].isEnabled()
                else controls["cif_workers"].value()
            ),
            fiber_axis=(
                controls["axis_h"].value(),
                controls["axis_k"].value(),
                controls["axis_l"].value(),
            ),
            zone_range_mode=(
                "three_vertices"
                if mode == "out_of_plane"
                and controls["range_mode"].currentText() == "Three zone-axis vertices"
                else "center_cone"
                if mode == "out_of_plane"
                else "auto"
            ),
            zone_axis_center=(
                controls["center_h"].value(),
                controls["center_k"].value(),
                controls["center_l"].value(),
            ),
            zone_angle_deg=controls["cone_angle"].value(),
            zone_axis_vertices=tuple(
                (boxes[0].value(), boxes[1].value(), boxes[2].value())
                for boxes in controls["vertex_boxes"]
            ),
        )

    def _no_cif_config(self, mode: str) -> NoCifInPlaneConfig:
        controls = self.orientation_controls[mode]
        return NoCifInPlaneConfig(
            qx_center=controls["qx_center"].value(),
            qy_center=controls["qy_center"].value(),
            flip_qy=controls["flip_qy"].isChecked(),
            angle_step_deg=controls["in_plane_step"].value(),
            symmetry_order=controls["no_cif_symmetry"].value(),
            match_tolerance_px=controls["no_cif_tolerance"].value(),
            min_number_peaks=controls["min_peaks"].value(),
            center_exclusion_px=controls["no_cif_center_exclusion"].value(),
            num_workers=controls["no_cif_workers"].value(),
        )

    def _rotation_symmetry(self, mode: str) -> int:
        controls = self.orientation_controls.get(mode, {})
        widget = controls.get("rotation_symmetry")
        if widget is None:
            return 1
        return max(1, int(widget.value()))

    def _rotation_symmetry_changed(self, mode: str) -> None:
        controls = self.orientation_controls.get(mode)
        if controls is None:
            return
        wheel = controls.get("color_wheel")
        if wheel is not None:
            wheel.setPixmap(_color_wheel_pixmap(128, self._rotation_symmetry(mode)))
        self.refresh_orientation_display(mode)
        self._orientation_log(mode, f"Rotation symmetry set to {self._rotation_symmetry(mode)}.")

    def _orientation_method_changed(self, mode: str) -> None:
        controls = self.orientation_controls.get(mode)
        if controls is None:
            return
        no_cif = self._is_no_cif_mode(mode)
        self._set_orientation_widgets_visible(controls, controls["cif_only"], not no_cif)
        self._set_orientation_widgets_visible(controls, controls["no_cif_only"], no_cif)
        controls["cif_model_group"].setVisible(not no_cif)
        controls["no_cif_group"].setVisible(no_cif)
        controls["color_wheel"].setVisible(mode == "in_plane")
        self._orientation_log(mode, f"Method: {controls['method'].currentText()}")

    def _out_of_plane_range_mode_changed(self, mode: str) -> None:
        controls = self.orientation_controls.get(mode)
        if controls is None or mode != "out_of_plane":
            return
        three_vertices = controls["range_mode"].currentText() == "Three zone-axis vertices"
        controls["center_axis_widget"].setVisible(not three_vertices)
        controls["cone_angle"].setVisible(not three_vertices)
        for widget in controls["vertex_widgets"]:
            widget.setVisible(three_vertices)
        field_forms = controls.get("field_forms", {})
        for widget in [controls["center_axis_widget"], controls["cone_angle"], *controls["vertex_widgets"]]:
            form = field_forms.get(widget)
            if form is None:
                continue
            label = form.labelForField(widget)
            if label is not None:
                label.setVisible(widget.isVisible())
        self._orientation_log(mode, f"Out-of-plane range: {controls['range_mode'].currentText()}")

    def sync_out_of_plane_from_in_plane(self) -> None:
        if "in_plane" not in self.orientation_controls or "out_of_plane" not in self.orientation_controls:
            return
        source = self.orientation_controls["in_plane"]
        target = self.orientation_controls["out_of_plane"]
        for key in (
            "cif_path",
            "q_pixel_size",
            "qx_center",
            "qy_center",
            "flip_qy",
            "k_max",
            "accel_voltage",
            "corr_kernel",
            "sigma_excitation",
            "min_peaks",
            "num_matches",
        ):
            self._copy_orientation_widget_value(source[key], target[key])
        self.orientation_crystals.pop("out_of_plane", None)
        self.orientation_workspaces.pop("out_of_plane", None)
        self.orientation_results.pop("out_of_plane", None)
        self._orientation_log("out_of_plane", "Copied CIF/calibration settings from In-Plane.")

    def _sync_out_of_plane_if_requested(self) -> None:
        controls = self.orientation_controls.get("out_of_plane")
        if controls is not None and controls["use_in_plane_sync"].isChecked():
            self.sync_out_of_plane_from_in_plane()

    def _copy_orientation_widget_value(self, source: QWidget, target: QWidget) -> None:
        if isinstance(source, QLineEdit) and isinstance(target, QLineEdit):
            target.setText(source.text())
        elif isinstance(source, QDoubleSpinBox) and isinstance(target, QDoubleSpinBox):
            target.setValue(source.value())
        elif isinstance(source, QSpinBox) and isinstance(target, QSpinBox):
            target.setValue(source.value())
        elif isinstance(source, QCheckBox) and isinstance(target, QCheckBox):
            target.setChecked(source.isChecked())

    def _set_orientation_widgets_visible(
        self,
        controls: dict[str, object],
        widgets: list[QWidget],
        visible: bool,
    ) -> None:
        field_forms = controls.get("field_forms", {})
        for widget in widgets:
            widget.setVisible(visible)
            form = field_forms.get(widget)
            if form is None:
                continue
            label = form.labelForField(widget)
            if label is not None:
                label.setVisible(visible)

    def _is_no_cif_mode(self, mode: str) -> bool:
        controls = self.orientation_controls[mode]
        return mode == "in_plane" and controls["method"].currentText() == "No-CIF template"

    def _no_cif_template(self, mode: str):
        template = self.no_cif_templates.get(mode)
        if template is None:
            self._orientation_log(mode, "Build a no-CIF template first from the current pattern.")
        return template

    def _orientation_workspace(self, mode: str):
        workspace = self.orientation_workspaces.get(mode)
        if workspace is None:
            self._orientation_log(mode, "Build the orientation plan first.")
        return workspace

    def _orientation_crystal(self, mode: str):
        crystal = self.orientation_crystals.get(mode)
        if crystal is not None:
            return crystal
        controls = self.orientation_controls[mode]
        try:
            crystal = load_crystal_structure(
                Path(controls["cif_path"].text()),
                controls["k_max"].value(),
            )
        except Exception as exc:
            self._orientation_log(mode, f"Failed to load CIF structure factors: {exc}")
            return None
        self.orientation_crystals[mode] = crystal
        self._orientation_log(mode, "CIF structure factors ready.")
        return crystal

    def _orientation_zone_axis(self, mode: str) -> tuple[float, float, float]:
        controls = self.orientation_controls[mode]
        return (
            controls["axis_h"].value(),
            controls["axis_k"].value(),
            controls["axis_l"].value(),
        )

    def _plot_bvm_radial_profile(self, mode: str, bvm: np.ndarray) -> None:
        from fourdlab.processing.orientation import radial_q_profile

        q, intensity = radial_q_profile(bvm)
        controls = self.orientation_controls[mode]
        plot = controls["plot"]
        plot.clear()
        _style_plot(plot)
        plot.getViewBox().setAspectLocked(False)
        plot.setLabel("bottom", "q (pixels)")
        plot.setLabel("left", "intensity * q")
        plot.plot(q, intensity * q, pen=pg.mkPen("#5b8def", width=2))
        controls["plot_kind"] = "profile"

    def _plot_q_pixel_scan(self, mode: str, scan) -> None:
        controls = self.orientation_controls[mode]
        plot = controls["plot"]
        pixel_size = controls["q_pixel_size"].value()
        q_exp = scan.q_pixels * pixel_size
        weighted = np.asarray(scan.weighted_intensity, dtype=np.float64)
        sf = np.asarray(scan.intensity_sf, dtype=np.float64)
        scale = float(np.nanmax(weighted)) if np.isfinite(weighted).any() else 1.0
        if scale <= 0:
            scale = 1.0
        plot.clear()
        _style_plot(plot)
        plot.getViewBox().setAspectLocked(False)
        plot.setLabel("bottom", "q (A^-1)")
        plot.setLabel("left", "intensity * q")
        _ensure_legend(plot)
        plot.plot(q_exp, weighted, pen=pg.mkPen("#5b8def", width=2), name="experiment")
        plot.plot(scan.q_sf, sf * scale, pen=pg.mkPen("#ff4d4f", width=2), name="structure factors")
        score_scaled = scan.scores / max(float(np.nanmax(scan.scores)), 1.0) * scale
        q_score = np.interp(scan.pixel_sizes, [scan.pixel_sizes.min(), scan.pixel_sizes.max()], [0, controls["k_max"].value()])
        plot.plot(q_score, score_scaled, pen=pg.mkPen("#26a269", width=1), name="pixel score")
        controls["plot_kind"] = "profile"

    def _plot_cif_zone_preview(self, mode: str, preview) -> None:
        controls = self.orientation_controls[mode]
        plot = controls["plot"]
        plot.clear()
        _style_fit_plot(plot)
        plot.getViewBox().setAspectLocked(True, ratio=1.0)
        _ensure_legend(plot)
        if preview.qx.size:
            intensity = np.asarray(preview.intensity, dtype=np.float64)
            scale = intensity / max(float(np.nanmax(intensity)), 1.0)
            sizes = np.clip(6.0 + 18.0 * np.sqrt(scale), 6.0, 24.0)
            spots = [
                {
                    "pos": (float(qx), float(qy)),
                    "size": float(size),
                    "brush": pg.mkBrush(255, 77, 79, 90 + int(130 * min(1.0, s))),
                    "pen": pg.mkPen("#ff4d4f", width=1.1),
                }
                for qx, qy, size, s in zip(preview.qx, preview.qy, sizes, scale)
            ]
            scatter = pg.ScatterPlotItem(spots=spots)
            plot.addItem(scatter)
            self._add_preview_hkl_labels(plot, preview)
        self._set_square_plot_range(
            plot,
            [np.asarray(preview.qx, dtype=np.float64)],
            [np.asarray(preview.qy, dtype=np.float64)],
            fallback_radius=controls["k_max"].value(),
            margin=0.12,
        )
        plot.setLabel("bottom", "q projection")
        plot.setLabel("left", "q projection")
        controls["plot_kind"] = "fit"

    def _plot_cif_fit_diagnostic(self, mode: str, diagnostic) -> None:
        controls = self.orientation_controls[mode]
        plot = controls["plot"]
        plot.clear()
        _style_fit_plot(plot)
        plot.getViewBox().setAspectLocked(True, ratio=1.0)
        _ensure_legend(plot)
        if diagnostic.sim_qx.size:
            plot.plot(
                diagnostic.sim_qx,
                diagnostic.sim_qy,
                pen=None,
                symbol="+",
                symbolSize=14,
                symbolPen=pg.mkPen("#ff4d4f", width=2),
                name="CIF simulated",
            )
        if diagnostic.exp_qx.size:
            plot.plot(
                diagnostic.exp_qx,
                diagnostic.exp_qy,
                pen=None,
                symbol="o",
                symbolSize=10,
                symbolBrush=pg.mkBrush(0, 170, 255, 140),
                symbolPen=pg.mkPen("#08aeea", width=1),
                name="experimental",
            )
        for exp_idx, sim_idx in zip(diagnostic.matched_exp_indices, diagnostic.matched_sim_indices):
            plot.plot(
                [float(diagnostic.exp_qx[exp_idx]), float(diagnostic.sim_qx[sim_idx])],
                [float(diagnostic.exp_qy[exp_idx]), float(diagnostic.sim_qy[sim_idx])],
                pen=pg.mkPen("#26a269", width=1),
            )
        self._set_square_plot_range(
            plot,
            [diagnostic.exp_qx, diagnostic.sim_qx],
            [diagnostic.exp_qy, diagnostic.sim_qy],
            fallback_radius=controls["k_max"].value(),
            margin=0.12,
        )
        plot.setLabel("bottom", "qx (A^-1)")
        plot.setLabel("left", "qy (A^-1)")
        controls["plot_kind"] = "fit"

    def _add_preview_hkl_labels(self, plot, preview) -> None:
        hkl = np.asarray(preview.hkl)
        if hkl.shape[0] == 0:
            return
        order = np.argsort(np.asarray(preview.intensity, dtype=np.float64))[::-1]
        for idx in order[: min(30, order.size)]:
            h, k, l = hkl[idx]
            if h == 0 and k == 0 and l == 0:
                continue
            text = pg.TextItem(f"{int(h)}{int(k)}{int(l)}", color="#9a3412", anchor=(0.5, 1.2))
            text.setPos(float(preview.qx[idx]), float(preview.qy[idx]))
            plot.addItem(text)

    def _show_fit_overlay(self, mode: str, workspace, peaks: PeakList, orientation) -> None:
        controls = self.orientation_controls[mode]
        plot = controls["plot"]
        plot.clear()
        _style_fit_plot(plot)
        plot.getViewBox().setAspectLocked(True, ratio=1.0)
        _ensure_legend(plot)
        experimental = peak_list_to_pointlist(peaks, workspace.config)
        qx_values = []
        qy_values = []
        if experimental.length:
            exp_x = np.asarray(experimental.data["qx"], dtype=np.float64)
            exp_y = _viewer_display_qy(
                np.asarray(experimental.data["qy"], dtype=np.float64),
                workspace.config.flip_qy,
            )
            qx_values.append(exp_x)
            qy_values.append(exp_y)
            plot.plot(
                exp_x,
                exp_y,
                pen=None,
                symbol="o",
                symbolSize=12,
                symbolBrush=pg.mkBrush(0, 170, 255, 160),
                symbolPen=pg.mkPen("#08aeea", width=1),
                name="experimental",
            )
        patterns = generate_fit_patterns(workspace, orientation)
        if patterns:
            patterns = patterns[:1]
        for pattern in patterns:
            data = pattern.data
            color = "#ff3333"
            fit_x = np.asarray(data["qx"], dtype=np.float64)
            fit_y = _viewer_display_qy(np.asarray(data["qy"], dtype=np.float64), workspace.config.flip_qy)
            qx_values.append(fit_x)
            qy_values.append(fit_y)
            plot.plot(
                fit_x,
                fit_y,
                pen=None,
                symbol="+",
                symbolSize=16,
                symbolPen=pg.mkPen(color, width=2.4),
                name="fit",
            )
            self._add_hkl_labels(plot, data, color, fit_x, fit_y)
        self._set_square_plot_range(
            plot,
            qx_values,
            qy_values,
            fallback_radius=float(workspace.config.k_max) + 0.1,
            margin=0.08,
        )
        controls["plot_kind"] = "fit"

    def _show_no_cif_fit_overlay(self, mode: str, template, peaks: PeakList, angle_deg: float) -> None:
        controls = self.orientation_controls[mode]
        plot = controls["plot"]
        plot.clear()
        _style_fit_plot(plot)
        plot.getViewBox().setAspectLocked(True, ratio=1.0)
        _ensure_legend(plot)
        config = self._no_cif_config(mode)
        vectors, _intensity = peak_list_to_centered_vectors_px(peaks, config)
        if vectors.size:
            radius = np.linalg.norm(vectors, axis=1)
            keep = radius >= float(config.center_exclusion_px)
            vectors = vectors[keep]
        if vectors.size:
            exp_x = vectors[:, 0]
            exp_y = _viewer_display_qy(vectors[:, 1], config.flip_qy)
            plot.plot(
                exp_x,
                exp_y,
                pen=None,
                symbol="o",
                symbolSize=12,
                symbolBrush=pg.mkBrush(0, 170, 255, 150),
                symbolPen=pg.mkPen("#08aeea", width=1),
                name="experimental",
            )
        fitted = rotate_vectors(template.vectors_px, angle_deg)
        if fitted.size:
            fit_x = fitted[:, 0]
            fit_y = _viewer_display_qy(fitted[:, 1], config.flip_qy)
            plot.plot(
                fit_x,
                fit_y,
                pen=None,
                symbol="+",
                symbolSize=16,
                symbolPen=pg.mkPen("#ff3333", width=2.4),
                name=f"template {angle_deg:.1f} deg",
            )
        x_values = []
        y_values = []
        if vectors.size:
            x_values.append(exp_x)
            y_values.append(exp_y)
        if fitted.size:
            x_values.append(fit_x)
            y_values.append(fit_y)
        self._set_square_plot_range(plot, x_values, y_values, fallback_radius=10.0, margin=0.12)
        controls["plot_kind"] = "fit"

    def _add_hkl_labels(
        self,
        plot,
        data,
        color: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
    ) -> None:
        fields = data.dtype.names or ()
        if not all(field in fields for field in ("h", "k", "l")):
            return
        max_labels = min(int(data.shape[0]), 30)
        for idx in range(max_labels):
            label = f"{int(data['h'][idx])}{int(data['k'][idx])}{int(data['l'][idx])}"
            text = pg.TextItem(label, color=color, anchor=(0.5, 1.2))
            text.setPos(float(x_values[idx]), float(y_values[idx]))
            plot.addItem(text)

    def _set_square_plot_range(
        self,
        plot,
        x_arrays: list[np.ndarray],
        y_arrays: list[np.ndarray],
        *,
        fallback_radius: float,
        margin: float,
    ) -> None:
        x = np.concatenate([np.asarray(arr, dtype=np.float64).ravel() for arr in x_arrays if arr.size])
        y = np.concatenate([np.asarray(arr, dtype=np.float64).ravel() for arr in y_arrays if arr.size])
        if x.size == 0 or y.size == 0:
            radius = float(fallback_radius)
            plot.setXRange(-radius, radius, padding=0.02)
            plot.setYRange(-radius, radius, padding=0.02)
            return
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            radius = float(fallback_radius)
            plot.setXRange(-radius, radius, padding=0.02)
            plot.setYRange(-radius, radius, padding=0.02)
            return
        x = x[finite]
        y = y[finite]
        cx = float((np.nanmin(x) + np.nanmax(x)) / 2.0)
        cy = float((np.nanmin(y) + np.nanmax(y)) / 2.0)
        half = max(float(np.nanmax(np.abs(x - cx))), float(np.nanmax(np.abs(y - cy))), 1.0)
        half *= 1.0 + float(margin)
        plot.setXRange(cx - half, cx + half, padding=0.0)
        plot.setYRange(cy - half, cy + half, padding=0.0)

    def _current_orientation_peaks(self) -> PeakList | None:
        peaks = self._viewer_orientation_peaks()
        if peaks is not None:
            return peaks
        return self.current_peaks

    def _viewer_orientation_peaks(self) -> PeakList | None:
        if self.all_peaks is None:
            return None
        y, x = self._viewer_scan_position()
        if y >= self.all_peaks.scan_shape[0] or x >= self.all_peaks.scan_shape[1]:
            return None
        return self.all_peaks.peaks[y][x]

    def _orientation_peaks_at(self, y: int, x: int) -> PeakList | None:
        if self.all_peaks is not None:
            if 0 <= y < self.all_peaks.scan_shape[0] and 0 <= x < self.all_peaks.scan_shape[1]:
                return self.all_peaks.peaks[y][x]
            return None
        viewer_y, viewer_x = self._viewer_scan_position()
        if self.current_peaks is not None and y == viewer_y and x == viewer_x:
            return self.current_peaks
        return None

    def _viewer_scan_position(self) -> tuple[int, int]:
        return int(self.viewer.scan_y.value()), int(self.viewer.scan_x.value())

    def _orientation_log(self, mode: str, message: str) -> None:
        self.orientation_controls[mode]["log"].append(message)

    def _default_qx_center(self) -> float:
        cube = self.viewer.cube
        if cube is None:
            return 64.0
        return (cube.diffraction_shape[1] - 1) / 2.0

    def _default_qy_center(self) -> float:
        cube = self.viewer.cube
        if cube is None:
            return 64.0
        return (cube.diffraction_shape[0] - 1) / 2.0


    def _show_peaks(self, peaks: PeakList) -> None:
        spots = [
            {"pos": (float(qx), float(qy)), "data": idx}
            for idx, (qx, qy) in enumerate(zip(peaks.qx, peaks.qy))
        ]
        self.scatter.setData(spots)

    def _log(self, message: str) -> None:
        self.log.append(message)


def _scroll_area(widget: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(widget)
    scroll.setMinimumWidth(260)
    scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
    scroll.setFrameShape(QScrollArea.NoFrame)
    scroll.setStyleSheet(
        "QScrollArea { background: transparent; }"
        "QScrollBar:vertical { width: 10px; margin: 0px; }"
    )
    return scroll


def _square_panel(widget: QWidget, min_size: int = 320) -> QWidget:
    return _AspectPanel(widget, min_size=min_size, aspect=1.0, object_name="square_image_panel")


def _aspect_panel(widget: QWidget, min_size: int = 320, aspect: float = 1.0) -> QWidget:
    return _AspectPanel(widget, min_size=min_size, aspect=aspect, object_name="aspect_image_panel")


class _AspectPanel(QWidget):
    """Center one child in a fixed-aspect viewport that follows available space."""

    def __init__(
        self,
        child: QWidget,
        *,
        min_size: int = 320,
        aspect: float = 1.0,
        object_name: str = "aspect_image_panel",
    ) -> None:
        super().__init__()
        self.setObjectName(object_name)
        self.aspect = max(0.1, float(aspect))
        self.child = child
        self.child.setParent(self)
        self.child.setMinimumSize(0, 0)
        self.child.show()
        self.setMinimumSize(min_size, max(1, int(round(min_size / self.aspect))))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def resizeEvent(self, event) -> None:
        available_w = max(0, self.width())
        available_h = max(0, self.height())
        width_from_height = int(round(available_h * self.aspect))
        if width_from_height <= available_w:
            child_w = width_from_height
            child_h = available_h
        else:
            child_w = available_w
            child_h = int(round(available_w / self.aspect))
        x = (available_w - child_w) // 2
        y = (available_h - child_h) // 2
        self.child.setGeometry(x, y, max(0, child_w), max(0, child_h))
        super().resizeEvent(event)


def _image_aspect(image: np.ndarray) -> float:
    arr = np.asarray(image)
    if arr.ndim < 2 or arr.shape[0] == 0:
        return 1.0
    return max(0.1, float(arr.shape[1]) / float(arr.shape[0]))


def _spot_raster(qx: np.ndarray, qy: np.ndarray, intensity: np.ndarray, size: int = 420) -> np.ndarray:
    x = np.asarray(qx, dtype=np.float64)
    y = np.asarray(qy, dtype=np.float64)
    values = np.asarray(intensity, dtype=np.float64)
    image = np.zeros((size, size), dtype=np.float32)
    if x.size == 0 or y.size == 0:
        return image
    radius = max(float(np.nanmax(np.abs(x))), float(np.nanmax(np.abs(y))), 1.0)
    px = np.clip(np.rint((x / (2 * radius) + 0.5) * (size - 1)).astype(np.int64), 0, size - 1)
    py = np.clip(np.rint((0.5 - y / (2 * radius)) * (size - 1)).astype(np.int64), 0, size - 1)
    weights = values / max(float(np.nanmax(values)), 1.0)
    for x_i, y_i, weight in zip(px, py, weights):
        rr = max(2, int(3 + 5 * np.sqrt(max(float(weight), 0.0))))
        y0 = max(0, y_i - rr)
        y1 = min(size, y_i + rr + 1)
        x0 = max(0, x_i - rr)
        x1 = min(size, x_i + rr + 1)
        image[y0:y1, x0:x1] = np.maximum(image[y0:y1, x0:x1], float(weight))
    return image


def _diagnostic_raster(diagnostic, size: int = 420) -> np.ndarray:
    sim = _spot_raster(diagnostic.sim_qx, diagnostic.sim_qy, np.ones_like(diagnostic.sim_qx), size=size)
    exp = _spot_raster(diagnostic.exp_qx, diagnostic.exp_qy, np.ones_like(diagnostic.exp_qx), size=size)
    rgb = np.dstack((sim, exp, np.zeros_like(sim))).astype(np.float32)
    return np.clip(rgb, 0.0, 1.0)


def _style_plot(plot: pg.PlotWidget) -> None:
    plot.setBackground("#fbfcfe")
    plot.showGrid(x=True, y=True, alpha=0.22)
    item = plot.getPlotItem()
    item.showAxis("left")
    item.showAxis("bottom")
    item.setLabel("left", "")
    item.setLabel("bottom", "")
    item.getViewBox().setBackgroundColor("#fbfcfe")
    item.getViewBox().setBorder(None)
    for axis_name in ("left", "bottom"):
        axis = item.getAxis(axis_name)
        axis.setPen(pg.mkPen("#7b8190", width=1))
        axis.setTextPen(pg.mkPen("#2f3542"))


def _style_fit_plot(plot: pg.PlotWidget) -> None:
    plot.setBackground("#ffffff")
    plot.showGrid(x=False, y=False)
    item = plot.getPlotItem()
    item.hideAxis("left")
    item.hideAxis("bottom")
    item.getViewBox().setBackgroundColor("#ffffff")
    item.getViewBox().setBorder(pg.mkPen("#111111", width=1.8))


def _ensure_legend(plot: pg.PlotWidget) -> None:
    legend = plot.plotItem.legend
    if legend is None:
        plot.addLegend(offset=(10, 10))
    else:
        legend.clear()


def _viewer_display_qy(qy_values: np.ndarray, flip_qy: bool) -> np.ndarray:
    qy = np.asarray(qy_values, dtype=np.float64)
    return qy if flip_qy else -qy


def _color_wheel_pixmap(size: int, symmetry_order: int = 1) -> QPixmap:
    yy, xx = np.indices((size, size), dtype=np.float64)
    center = (size - 1) / 2.0
    dx = xx - center
    dy = center - yy
    radius = np.sqrt(dx**2 + dy**2)
    outer = center * 0.92
    inner = center * 0.34
    symmetry = max(1, int(symmetry_order))
    hue = ((np.arctan2(dy, dx) / (2.0 * np.pi)) * symmetry) % 1.0
    sat = np.ones_like(hue)
    val = np.ones_like(hue)
    rgb = _hsv_to_rgb(hue, sat, val)
    alpha = ((radius <= outer) & (radius >= inner)).astype(np.float64)
    edge = np.clip((outer - radius) / 2.0, 0.0, 1.0) * np.clip((radius - inner) / 2.0, 0.0, 1.0)
    alpha *= edge
    rgba = np.dstack((rgb, alpha))
    rgba8 = np.ascontiguousarray(np.clip(rgba * 255.0, 0, 255).astype(np.uint8))
    image = QImage(rgba8.data, size, size, 4 * size, QImage.Format_RGBA8888).copy()
    return QPixmap.fromImage(image)


def _hsv_to_rgb(hue: np.ndarray, sat: np.ndarray, val: np.ndarray) -> np.ndarray:
    h = np.mod(hue, 1.0) * 6.0
    i = np.floor(h).astype(np.int64)
    f = h - i
    p = val * (1.0 - sat)
    q = val * (1.0 - sat * f)
    t = val * (1.0 - sat * (1.0 - f))
    rgb = np.zeros(hue.shape + (3,), dtype=np.float64)
    choices = i % 6
    masks = [choices == n for n in range(6)]
    rgb[masks[0]] = np.stack((val, t, p), axis=-1)[masks[0]]
    rgb[masks[1]] = np.stack((q, val, p), axis=-1)[masks[1]]
    rgb[masks[2]] = np.stack((p, val, t), axis=-1)[masks[2]]
    rgb[masks[3]] = np.stack((p, q, val), axis=-1)[masks[3]]
    rgb[masks[4]] = np.stack((t, p, val), axis=-1)[masks[4]]
    rgb[masks[5]] = np.stack((val, p, q), axis=-1)[masks[5]]
    return np.clip(rgb, 0.0, 1.0)


def _double_box(value: float, minimum: float, maximum: float, step: float) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    box.setSingleStep(step)
    box.setDecimals(4)
    return box


def _int_box(value: int, minimum: int, maximum: int) -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    return box


def _hkl_widget(h_box: QDoubleSpinBox, k_box: QDoubleSpinBox, l_box: QDoubleSpinBox) -> QWidget:
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(h_box)
    layout.addWidget(k_box)
    layout.addWidget(l_box)
    return widget


def _format_zone_label(values: np.ndarray) -> str:
    parts = []
    for value in np.asarray(values, dtype=np.float64):
        if abs(value - round(value)) < 1.0e-6:
            parts.append(str(int(round(value))))
        else:
            parts.append(f"{value:.2g}")
    return "[" + " ".join(parts) + "]"


def _default_worker_count() -> int:
    cpus = max(1, os.cpu_count() or 1)
    return max(1, min(4, cpus - 1 if cpus > 1 else 1))


def _levels(image) -> tuple[float, float]:
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


def _strain_levels(image: np.ndarray, label: str) -> tuple[float, float]:
    values = np.asarray(image, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    if label == "mask":
        return 0.0, 1.0
    if label == "matched peaks":
        return 0.0, max(1.0, float(np.nanmax(finite)))
    if label == "error":
        low, high = np.percentile(finite, (1.0, 99.0))
        if high <= low:
            high = low + 1.0
        return float(low), float(high)
    max_abs = float(np.nanpercentile(np.abs(finite), 99.0))
    if max_abs <= 0:
        max_abs = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))), 1.0)
    return -max_abs, max_abs


def _set_image_gradient(image_view: pg.ImageView, label: str) -> None:
    if label == "mask":
        preset = "grey"
    elif label in {"error", "matched peaks"}:
        preset = "viridis"
    else:
        preset = "bipolar"
    try:
        image_view.ui.histogram.gradient.loadPreset(preset)
    except Exception:
        return


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _preset_path() -> Path:
    return _project_root() / "configs" / "peak_detection_preset.json"
