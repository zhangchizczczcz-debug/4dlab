# 4DLAB Work Log

This file records project progress so future sessions can continue without losing context.

## 2026-05-27 19:32 Asia/Shanghai

Author: Codex session

Goal:
- Establish the first long-term rules for the 4DLAB 4DSTEM processing library.

Files changed:
- `docs/PROJECT_RULES.md`
- `docs/WORK_LOG.md`

Details:
- Created a project rules document that records four foundational rules:
  1. Every code change must be logged in detail.
  2. Different feature areas should remain isolated and avoid interfering with each other.
  3. Code and related files must be placed in matching, organized folders.
  4. The project should stay portable across computers after environment setup.
- Added a suggested folder layout for future source code, tests, examples, scripts, configs, data, and docs.
- Added a reusable checklist for future work log entries.

Verification:
- Documentation files were created in the project `docs` folder.

Known limitations:
- No library source code has been created yet.
- Environment files such as `pyproject.toml`, `environment.yml`, or `requirements.txt` have not been selected yet.

Suggested next step:
- Decide the first implementation target and create the initial portable Python package structure.

## 2026-05-27 19:36 Asia/Shanghai

Author: Codex session

Goal:
- Create the dedicated conda environment for the 4DLAB 4DSTEM processing library.

Files changed:
- `environment.yml`
- `docs/ENVIRONMENT.md`
- `docs/WORK_LOG.md`

Details:
- Added `environment.yml` at the project root.
- Created a conda environment named `4dlab`.
- Chose Python 3.11 as the initial runtime because it is a stable scientific Python target and is broadly compatible with common microscopy, numerical, and packaging tools.
- Included `pip` so packages that are not available through conda can still be installed later.
- Added `docs/ENVIRONMENT.md` with basic create, activate, and update commands.

Verification:
- Ran `conda env create -f environment.yml` successfully.
- Ran `conda run -n 4dlab python --version`.
- Result: `Python 3.11.15`.
- Ran `conda run -n 4dlab pip --version`.
- Result: `pip 26.1.1` from `C:\ProgramData\Miniconda3\envs\4dlab\Lib\site-packages\pip`.

Known limitations:
- Only the base environment has been created.
- No scientific or 4DSTEM-specific dependencies have been installed yet.
- PowerShell prints a profile script execution-policy warning before commands, but the conda commands still run successfully.

Suggested next step:
- Create the initial Python package structure under `src/` and decide the first processing module.

## 2026-05-27 20:03 Asia/Shanghai

Author: Codex session

Goal:
- Add the first feature: data import and real-time visualization as the main 4DLAB interface.

Files changed:
- `pyproject.toml`
- `environment.yml`
- `src/fourdlab/__init__.py`
- `src/fourdlab/gui/__init__.py`
- `src/fourdlab/gui/extensions.py`
- `src/fourdlab/gui/viewer.py`
- `src/fourdlab/gui/main.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/ENVIRONMENT.md`
- `docs/WORK_LOG.md`

Details:
- Inspected the existing local `py4dstem` conda environment.
- Found that the `py4DGUI.exe` command comes from `py4D_browser`.
- Found the known-good local GUI version: `py4D_browser 1.2.1`.
- Created the installable project package `fourdlab`.
- Added the console command `4dlab-viewer`.
- Implemented `fourdlab.gui.viewer.launch_viewer`, which launches `py4D_browser.DataViewer` inside the dedicated `4dlab` environment.
- Added `ViewerLaunchConfig` and a lightweight `ViewerExtension` protocol so later processing features can attach to the viewer without rewriting the import and visualization shell.
- Added `docs/FEATURE_IMPORT_VISUALIZATION.md` to document the feature boundary and usage.
- Updated `environment.yml` to install `py4D-browser==1.2.1` through pip and install the local package with `pip install -e .`.

Dependency notes:
- An initial conda install brought in `py4D-browser 1.5.0`.
- Attempting to downgrade with `conda install py4d-browser=1.2.1` stalled during dependency solving.
- The stalled conda/python solver process was stopped.
- Replaced only the GUI package with `conda run -n 4dlab python -m pip install --no-deps py4D-browser==1.2.1`, which completed quickly because dependencies were already installed.

Verification:
- Verified installed versions in `4dlab`:
  - `fourdlab 0.1.0`
  - `py4D-browser 1.2.1`
  - `py4DSTEM 0.14.18`
  - `PyQt5 5.15.11`
  - `pyqtgraph 0.14.0`
- Verified `fourdlab.gui` imports.
- Verified console entry point:
  - `4dlab-viewer -> fourdlab.gui.main:main`
- Verified `py4D_browser.DataViewer` can be constructed inside the `4dlab` environment.

Known limitations:
- This first version delegates the main viewer behavior to upstream `py4D_browser.DataViewer`.
- No custom 4DLAB processing panels are attached yet.
- No sample dataset has been opened interactively in this session.
- PowerShell still prints a profile execution-policy warning, but commands complete successfully.

Suggested next step:
- Launch `4dlab-viewer` interactively, drag in a real 4D-STEM dataset, and confirm the browsing speed and controls match the known py4DGUI workflow.

## 2026-05-27 20:24 Asia/Shanghai

Author: Codex session

Goal:
- Replace the `py4D_browser` wrapper with a 4DLAB-owned import and visualization implementation.
- Add RAW import support for `float32` files with the usual diffraction shape `130 x 128`.

Files changed:
- `pyproject.toml`
- `environment.yml`
- `src/fourdlab/io/__init__.py`
- `src/fourdlab/io/datacube.py`
- `src/fourdlab/io/loaders.py`
- `src/fourdlab/visualization/__init__.py`
- `src/fourdlab/visualization/rendering.py`
- `src/fourdlab/gui/viewer.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Removed direct project dependency on `py4D-browser`.
- Uninstalled `py4D-browser` from the active `4dlab` environment.
- Kept the fast GUI stack idea, but implemented the viewer in 4DLAB source code using PyQt5 and pyqtgraph.
- Added a `DataCube` container in `src/fourdlab/io/datacube.py`.
- Added `load_datacube` in `src/fourdlab/io/loaders.py`.
- Added support for:
  - `.npy`
  - `.h5`
  - `.hdf5`
  - `.emd`
  - `.py4dstem`
  - `.raw`
- `.raw` import assumes:
  - dtype: `float32`
  - diffraction shape: `130 x 128`
  - scan shape: inferred as square if possible, otherwise `1 x N`
- Added `DataCube.close()` to release file-backed `numpy.memmap` resources, which avoids Windows file-lock cleanup issues.
- Rebuilt `fourdlab.gui.viewer.FourDLabViewer` as a custom 4DLAB main window with:
  - file open action
  - real-space virtual image display
  - diffraction pattern display
  - scan Y/X controls
  - circular or annular detector controls
  - real-time view refresh through pyqtgraph

Dependency notes:
- Current direct runtime dependencies are:
  - `h5py`
  - `numpy`
  - `PyQt5`
  - `pyqtgraph`
- `py4D-browser` is no longer a direct dependency and is not importable in the `4dlab` environment.

Verification:
- Confirmed `importlib.util.find_spec("py4D_browser")` returns `None`.
- Confirmed `fourdlab.gui.viewer.FourDLabViewer` imports successfully.
- Confirmed the self-owned GUI window can be constructed:
  - `FourDLabViewer 4DLAB Viewer`
- Created temporary `.npy`, `.h5`, and `.raw` files and loaded them through `load_datacube`.
- Verified output:
  - `.npy`: shape `(2, 3, 4, 5)`, virtual image `(2, 3)`, diffraction pattern `(4, 5)`
  - `.h5`: shape `(2, 3, 4, 5)`, dataset path `/entry/data`
  - `.raw`: shape `(4, 4, 130, 128)`, dataset path `raw:4x4x130x128`
- Confirmed `DataCube.close()` releases the RAW memmap and temporary files clean up successfully.
- Verified current dependency versions:
  - `fourdlab 0.1.0`
  - `PyQt5 5.15.11`
  - `pyqtgraph 0.14.0`
  - `numpy 1.26.4`
  - `h5py 3.16.0`

Known limitations:
- RAW import currently has fixed default diffraction shape `130 x 128`; no GUI parameter dialog yet.
- HDF5-like import selects the first 4D dataset found.
- The custom viewer currently implements the core browse workflow only; advanced py4DGUI tools such as detector shape menus, FFT/cepstrum, exports, calibration dialogs, and specialized vendor importers are not implemented yet.

Suggested next step:
- Add a RAW import dialog for scan shape and diffraction shape overrides, then test with a real `.raw` dataset.

## 2026-05-27 20:37 Asia/Shanghai

Author: Codex session

Goal:
- Improve RAW import for EMPAD data.
- Add the main py4DGUI-style top menu categories, excluding Help.

Files changed:
- `src/fourdlab/io/datacube.py`
- `src/fourdlab/io/loaders.py`
- `src/fourdlab/gui/viewer.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Updated RAW import assumptions:
  - file dtype remains `float32`
  - raw diffraction shape remains `130 x 128`
  - the final two diffraction rows are treated as EMPAD bad rows and cropped on import
  - loaded RAW diffraction patterns become `128 x 128`
- Added top-level menus:
  - `File`
  - `Scaling`
  - `Autorange`
  - `Detector Response`
  - `Detector Shape`
  - `FFT View`
  - `Processing`
- Did not add `Help`, per user request.
- Added display scaling modes:
  - linear
  - log
  - square root
- Added detector response modes:
  - sum
  - mean
  - maximum
- Added detector shape modes:
  - circle
  - annulus
  - square
- Added FFT view switching:
  - virtual image
  - FFT magnitude of virtual image
- Added processing actions:
  - refresh views
  - reset detector

Verification:
- Created a temporary RAW file with shape-equivalent data for `(4, 4, 130, 128)`.
- Loaded it through `load_datacube`.
- Verified loaded shape: `(4, 4, 128, 128)`.
- Verified diffraction shape: `(128, 128)`.
- Verified dataset note: `raw:4x4x130x128 cropped_to_128x128`.
- Verified virtual image calculation still returns scan shape `(4, 4)`.
- Verified GUI menu bar contains:
  - `File`
  - `Scaling`
  - `Autorange`
  - `Detector Response`
  - `Detector Shape`
  - `FFT View`
  - `Processing`
- Loaded temporary RAW data in `FourDLabViewer` and exercised:
  - log scaling
  - circle detector
  - mean detector response
  - FFT view
  - autorange

Known limitations:
- RAW crop is currently always applied for `.raw`; there is no toggle yet for non-EMPAD RAW data.
- Detector shape controls are still numeric spin boxes, not graphical draggable ROIs.
- FFT view currently shows FFT magnitude only, not cepstrum or other py4DGUI FFT modes.

Suggested next step:
- Add draggable real-space and diffraction-space selectors so the UI feels closer to py4DGUI during live browsing.

## 2026-05-27 20:55 Asia/Shanghai

Author: Codex session

Goal:
- Make the custom viewer's core interaction closer to py4DGUI.
- Add image-based ROI selectors instead of relying mainly on spin boxes.

Files changed:
- `src/fourdlab/io/datacube.py`
- `src/fourdlab/gui/viewer.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Added `DataCube.diffraction_region()` to calculate a diffraction pattern from
  a real-space rectangular region.
- Extended `DataCube.virtual_image()` with a point detector mode.
- Added real-space selector state to `FourDLabViewer`.
- Added real-space point ROI:
  - Dragging the point changes scan Y/X.
  - The diffraction-space image updates from that scan position.
- Added real-space square ROI:
  - Dragging/resizing the square selects a scan region.
  - The diffraction-space image updates from the selected region using the
    active detector response mode.
- Added diffraction-space ROI detector shapes:
  - point
  - circle
  - annulus
  - square
- Added synchronization between ROI movement and numeric controls.
- Updated `Detector Shape` menu to include `Point`.
- Added `Processing` menu actions to switch real-space selector mode:
  - `Real-Space Point Selector`
  - `Real-Space Square Selector`

Verification:
- Loaded temporary RAW data and confirmed EMPAD crop still produces
  diffraction shape `(128, 128)`.
- Verified GUI menus still contain:
  - `File`
  - `Scaling`
  - `Autorange`
  - `Detector Response`
  - `Detector Shape`
  - `FFT View`
  - `Processing`
- Verified initial ROI objects are created:
  - real-space point ROI
  - diffraction annulus ROI
- Verified switching real-space selector to square creates a `RectROI`.
- Verified switching detector shapes creates the expected ROI types:
  - point: `ROI`
  - circle: `CircleROI`
  - annulus: `CircleROI` outer and inner rings
  - square: `RectROI`
- Simulated moving the real-space ROI and confirmed scan controls update.
- Simulated moving a diffraction point ROI and confirmed detector controls and
  virtual image update.
- Ran `python -m compileall -q src` successfully in the `4dlab` environment.

Known limitations:
- Real-space square selector currently uses the same response choices as
  diffraction detector response: sum, mean, maximum.
- Annulus inner and outer rings are independent except for center-following
  when the outer ring moves; the polish is not yet identical to py4DGUI.
- Keyboard nudging shortcuts are not implemented yet.
- ROI styling is functional but still simpler than py4DGUI.

Suggested next step:
- Add keyboard nudging and improve annulus inner/outer linked resizing to more
  closely match py4DGUI's live feel.

## 2026-05-27 21:19 Asia/Shanghai

Author: Codex session

Goal:
- Add the first nanobeam diffraction analysis entry point under a new `More`
  menu.
- Implement COM-based center-disk correction based on the user's
  `S:\codex\SOFTWARE\PY4D\strain` code.

Files changed:
- `pyproject.toml`
- `environment.yml`
- `src/fourdlab/processing/__init__.py`
- `src/fourdlab/processing/center_correction.py`
- `src/fourdlab/gui/center_correction_dialog.py`
- `src/fourdlab/gui/viewer.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Added a top-level `More` menu after `Processing`.
- Added `More -> Center Correction`.
- Added an independent center correction dialog.
- The dialog reads the current viewer diffraction-space ROI as the selected
  center disk region.
- The correction algorithm follows the COM alignment idea from the strain code:
  - subtract local minimum inside the selected mask
  - calculate COM for each diffraction pattern
  - calculate shift from COM to target diffraction center
  - shift each diffraction pattern
  - return a corrected datacube to the viewer
- Added direct dependency on `scipy` because subpixel correction uses
  `scipy.ndimage.shift`.
- Added `replace_datacube()` to the viewer so processing modules can return
  new datacubes without coupling to file import.
- Added `current_diffraction_detector_mask()` to the viewer so processing
  modules can reuse the currently drawn detector ROI.

Verification:
- Refreshed editable install with `python -m pip install -e .`.
- Verified `scipy` is available in the `4dlab` environment.
- Ran `python -m compileall -q src` successfully.
- Verified menu bar contains:
  - `File`
  - `Scaling`
  - `Autorange`
  - `Detector Response`
  - `Detector Shape`
  - `FFT View`
  - `Processing`
  - `More`
- Created synthetic 4D data with shifted Gaussian center disks.
- Ran `align_center_disks()` and verified:
  - corrected datacube shape `(3, 3, 32, 32)`
  - measured shift ranges around the synthetic offsets
  - maximum post-correction COM L1 error about `0.0838` pixels
- Ran an end-to-end GUI-level check:
  - loaded synthetic `.npy` data in `FourDLabViewer`
  - set a circular diffraction ROI
  - opened `Center Correction`
  - ran correction
  - applied result back to viewer
  - verified viewer datacube path note is `center_corrected`

Known limitations:
- The correction currently runs synchronously in the dialog, so very large
  datacubes may temporarily block the UI.
- The result replaces the viewer datacube in memory; there is not yet an
  undo stack or save prompt.
- The target defaults to the geometric center of the diffraction image.
- Progress is text-only for now.

Suggested next step:
- Move center correction into a background worker with a progress bar and add a
  save/export option for the corrected datacube.

## 2026-05-27 21:29 Asia/Shanghai

Author: Codex session

Goal:
- Set `4dlab.png` as the GUI icon.
- Add a one-click Windows startup script.

Files changed:
- `src/fourdlab/gui/viewer.py`
- `start_4dlab.bat`
- `docs/ENVIRONMENT.md`
- `docs/WORK_LOG.md`

Details:
- Added `gui_icon_path()` to locate `4dlab.png` from the project root.
- Set the icon on both the `QApplication` and the main viewer window.
- Added `start_4dlab.bat` at the project root.
- The batch file changes to the project directory, activates the `4dlab`
  conda environment, and starts the viewer with `python -m fourdlab.gui.main`.
- The batch file pauses only if conda activation or viewer startup fails.

Verification:
- Confirmed `4dlab.png` exists at `S:\codex\SOFTWARE\4DLAB\4dlab.png`.
- Constructed `FourDLabViewer` in the `4dlab` environment.
- Verified:
  - `window_icon_null False`
  - `app_icon_null False`
- Read back `start_4dlab.bat` content successfully.
- Ran `python -m compileall -q src` successfully after the icon changes.

Known limitations:
- The `.bat` file assumes conda is available on PATH or Miniconda is installed
  at `%ProgramData%\Miniconda3`.

Suggested next step:
- Optionally add a `.ico` file later for a nicer Windows shortcut icon.

## 2026-05-27 21:45 Asia/Shanghai

Author: Codex session

Goal:
- Improve file opening behavior, image contrast autoranging, and center
  correction review.

Files changed:
- `src/fourdlab/gui/viewer.py`
- `src/fourdlab/gui/center_correction_dialog.py`
- `src/fourdlab/processing/center_correction.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- File open dialog now starts from the user's home directory instead of a fixed
  folder.
- After opening a file, the viewer remembers that folder for the next open
  dialog in the same session.
- Image contrast levels now use a percentile-based range instead of relying on
  pyqtgraph defaults:
  - low: 0.5 percentile
  - high: 99.5 percentile
- The histogram/contrast bar is also fit to the same adaptive range.
- Center correction shifted edges now use zero fill:
  - subpixel path uses `scipy.ndimage.shift(..., mode="constant", cval=0.0)`
  - integer path uses a custom zero-fill shift instead of `np.roll`
- Center Correction dialog now includes a preview area:
  - left: mean diffraction pattern before correction
  - right: mean diffraction pattern after correction
- Preview updates automatically after pressing `Run`; applying to the main
  viewer is still a separate explicit step.

Verification:
- Ran `python -m compileall -q src` successfully.
- Verified integer shift no longer wraps around:
  - shifted exposed edge values are `0`.
- Verified viewer default open directory exists.
- Verified adaptive levels and histogram fitting run without errors.
- Ran Center Correction dialog on synthetic data and verified:
  - result datacube shape `(2, 2, 32, 32)`
  - Apply button becomes enabled
  - before preview image is populated
  - after preview image is populated

Known limitations:
- The file dialog remembers the last folder only during the current app
  session; it does not persist across restarts yet.
- Contrast autorange uses fixed percentiles for now; there is no UI for
  changing percentile bounds.

Suggested next step:
- Persist the last opened folder in a small config file and add optional
  contrast percentile settings if needed.

## 2026-05-27 22:04 Asia/Shanghai

Author: Codex session

Goal:
- Add `More -> Diffraction Analysis` as the nanobeam diffraction analysis main
  window.
- Add the first diffraction analysis function: custom peak detection based on
  the user's `S:\codex\SOFTWARE\PY4D\strain` algorithm.
- Add a progress bar for full-datacube peak detection.

Files changed:
- `src/fourdlab/processing/peak_detection.py`
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `src/fourdlab/gui/viewer.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Added `More -> Diffraction Analysis` after `Center Correction`.
- Added a standalone `DiffractionAnalysisWindow`.
- Added analysis tabs:
  - `Peak Detection`
  - `In-Plane`
  - `Out-of-Plane`
  - `Strain`
- The orientation and strain tabs are placeholders for now.
- Added custom peak detection module:
  - optional Gaussian smoothing
  - edge exclusion
  - relative and absolute intensity thresholds
  - minimum peak spacing
  - maximum peak count
  - Gaussian subpixel refinement with COM fallback
- Added current-pattern detection with peak markers overlaid on the diffraction
  preview.
- Added full-scan detection with a `QProgressBar`.
- Full-scan detection stores peak lists in a `PeakDetectionResult` and can
  produce a peak-count map.

Verification:
- Ran `python -m compileall -q src` successfully.
- Tested custom detector on a synthetic 64x64 diffraction pattern with two
  Gaussian peaks.
- Verified detected peak count: `2`.
- Verified refined peak positions near:
  - qx `[18.0, 45.0]`
  - qy `[20.0, 42.0]`
- Tested the GUI entry on a synthetic 2x2 datacube.
- With edge boundary set to 2 for the small 32x32 synthetic patterns, verified:
  - current-pattern peak count: `2`
  - full-scan progress: `4/4`
  - peak-count map: `[[2, 2], [2, 2]]`

Known limitations:
- Full-scan peak detection currently runs synchronously while processing Qt
  events; a background worker would be better for large datasets.
- In-plane orientation, out-of-plane orientation, and strain tabs are
  placeholders only.
- Peak results are stored in memory but not yet saved/exported.

Suggested next step:
- Add background worker support and an export format for detected peak lists,
  then connect orientation and strain workflows to the peak detection result.

## 2026-05-27 22:18 Asia/Shanghai

Author: Codex session

Goal:
- Add save/load presets for Diffraction Analysis peak detection parameters.
- Read the existing `PY4D/strain` code path for in-plane/out-of-plane
  orientation and strain analysis planning.

Files changed:
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Added `Save Preset` and `Load Preset` buttons to the Peak Detection tab.
- Presets are stored as JSON at `configs/peak_detection_preset.json`.
- On opening `Diffraction Analysis`, the window automatically loads the preset
  if the JSON file exists.
- Saved fields:
  - smooth sigma
  - edge boundary
  - minimum relative intensity
  - minimum absolute intensity
  - minimum peak spacing
  - maximum number of peaks
  - Gaussian radius
  - refine on/off

Verification:
- Ran `python -m compileall -q src` successfully.
- Tested preset save/load with a temporary preset path.
- Verified saved values load back into controls:
  - edge boundary `3`
  - minimum spacing `9`
  - refine `False`

Notes From Reading `PY4D/strain`:
- Orientation and strain workflow currently wraps py4DSTEM's diffraction
  crystal tools.
- Main sequence:
  1. Detect Bragg peaks.
  2. Optionally refine peaks.
  3. Fit/measure Bragg origin and calibrate Bragg vectors.
  4. Load CIF with `Crystal.from_CIF`.
  5. Calculate structure factors up to `k_max`.
  6. Build orientation plan:
     - fiber mode for constrained in-plane style workflows
     - auto mode for broader out-of-plane/zone-axis searches
  7. Match a single pattern for preview/debug.
  8. Match orientations over the full scan.
  9. Plot orientation maps.
  10. Calculate strain from crystal, Bragg peaks, and orientation map.
  11. Extract `e_xx`, `e_yy`, `e_xy`, and `theta`.
  12. Optionally transform strain to polar components.

Known limitations:
- Preset path is project-local, not per-user. This is intentional for now so
  settings travel with the project folder.
- Orientation and strain tabs are still placeholders.

Suggested next step:
- Decide whether `In-Plane` and `Out-of-Plane` should be separate UI workflows
  or two modes of one orientation-plan panel.

## 2026-05-27 22:42 Asia/Shanghai

Author: Codex session

Goal:
- Improve Diffraction Analysis peak detection for long runs and reuse:
  stopping, saving/loading detected peaks, and preparing optional GPU use.

Files changed:
- `src/fourdlab/processing/peak_detection.py`
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Added `PeakDetectionCancelled` and a `stop_requested` callback to
  `detect_peaks_in_datacube`.
- Added a `Stop` button to the Peak Detection tab. During full-scan detection,
  it requests cancellation and the detector stops cleanly between diffraction
  patterns.
- Added `Save Peaks` and `Load Peaks` buttons for full-scan peak results.
- Peak results are saved as compressed `.npz` archives with:
  - scan shape
  - diffraction shape
  - scan indices
  - peak `qx`, `qy`, intensity, and refined flags
- Added `use_gpu` to `PeakDetectionConfig` and to the preset format.
- Added optional CuPy-backed candidate peak search for the local-maximum filter.
  If CuPy is unavailable, detection automatically falls back to CPU.
- The GUI disables the `Use GPU` checkbox when the `4dlab` environment cannot
  import CuPy/CuPy ndimage.

Verification:
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Confirmed `gpu_peak_detection_available()` returns `False` in the current
  `4dlab` environment.
- Ran a synthetic 3x3 scan smoke test:
  - full-scan peak detection found 2 peaks at every scan position
  - saved detected peaks to `.npz`
  - loaded the `.npz` and recovered the same peak-count map
  - stop callback raised `PeakDetectionCancelled` after `2/9` patterns

Known limitations:
- GPU acceleration currently covers local-maximum candidate search only; Gaussian
  refinement and scan iteration still run on CPU.
- The current `4dlab` environment does not have CuPy installed, so GPU mode is
  disabled until an appropriate CUDA/CuPy package is installed.

## 2026-05-27 22:58 Asia/Shanghai

Author: Codex session

Goal:
- Enable and verify GPU support in the `4dlab` environment.

Files changed:
- `environment.yml`
- `pyproject.toml`
- `docs/WORK_LOG.md`

Environment changes:
- Installed `cupy-cuda12x==13.6.0` into the `4dlab` conda environment.
- Restored/pinned `numpy==1.26.4` after `cupy-cuda12x==14.1.0` initially
  upgraded NumPy to `2.4.6`, which conflicted with the installed `py4DSTEM`
  package.
- Updated `environment.yml`:
  - `numpy=1.26`
  - `cupy-cuda12x==13.6.0`
- Updated `pyproject.toml`:
  - constrained core NumPy dependency to `<2`
  - added optional `gpu` dependency for `cupy-cuda12x==13.6.0`

Hardware check:
- `nvidia-smi` found:
  - GPU: NVIDIA GeForce RTX 3090
  - Driver: 591.86
  - CUDA driver capability: 13.1

Verification:
- Verified core versions inside `4dlab`:
  - `numpy 1.26.4`
  - `scipy 1.17.1`
  - `cupy 13.6.0`
- Verified CuPy can access the GPU:
  - device count `1`
  - GPU array sum returned `45`
- Verified `py4DSTEM` imports successfully after restoring NumPy 1.26.
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Verified 4DLAB peak detection GPU path:
  - `gpu_peak_detection_available()` returned `True`
  - synthetic 128x128 pattern with two peaks returned:
    - count `2`
    - `qx [50.0, 90.0]`
    - `qy [40.0, 80.0]`

Known limitations:
- Current GPU acceleration covers the local-maximum candidate search in peak
  detection. Gaussian refinement still uses CPU SciPy.
- A future speed pass should move full-scan work into a background worker and
  benchmark CPU/GPU on real EMPAD datacubes before expanding GPU coverage.

## 2026-05-27 23:31 Asia/Shanghai

Author: Codex session

Goal:
- Add the first usable orientation-analysis stage after peak detection.

Files changed:
- `src/fourdlab/processing/orientation.py`
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `environment.yml`
- `pyproject.toml`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Environment changes:
- Installed `pymatgen` in the `4dlab` environment because py4DSTEM
  `Crystal.from_CIF` requires it for CIF loading.
- Added `pymatgen` to both `environment.yml` and `pyproject.toml`.

Details:
- Added isolated processing module `fourdlab.processing.orientation`.
- Added py4DSTEM ACOM wrapper functions:
  - load CIF with `Crystal.from_CIF`
  - calculate structure factors with `calculate_structure_factors`
  - build orientation plans with `orientation_plan`
  - convert 4DLAB `PeakList` data to py4DSTEM `PointList`
  - match current-pattern orientation
  - match full-scan orientation maps with progress and stop callbacks
- Added `OrientationConfig`, `OrientationWorkspace`, `OrientationResult`, and
  `OrientationCancelled`.
- Added real `In-Plane` and `Out-of-Plane` tabs in Diffraction Analysis.
- Orientation GUI controls include:
  - CIF path and browse button
  - reciprocal pixel size in `A^-1/pixel`
  - q center in detector pixels
  - qy flip option
  - k max, voltage, kernel, excitation sigma
  - zone-axis and in-plane angular steps
  - minimum peak count and match count
  - CUDA toggle
- In-plane mode uses py4DSTEM fiber orientation mode around the default
  `[0, 0, 1]` fiber axis.
- Out-of-plane mode uses py4DSTEM `auto` zone-axis range from CIF symmetry.
- Full-scan orientation maps can display:
  - correlation
  - in-plane angle
  - angle 0
  - angle 1

Verification:
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Ran a temporary synthetic CIF smoke test:
  - converted a 4DLAB `PeakList` to py4DSTEM `PointList`
  - built an in-plane orientation plan
  - ran single-pattern orientation matching
- Ran an offscreen Qt smoke test:
  - instantiated `DiffractionAnalysisWindow`
  - verified orientation tabs are registered as `in_plane` and `out_of_plane`
  - verified CUDA checkbox is enabled in the current environment

Known limitations:
- Orientation physical correctness depends strongly on the user-entered
  reciprocal pixel size and diffraction center.
- The first implementation runs synchronously while processing Qt events; large
  orientation maps should be moved to a background worker.
- Orientation results are held in memory only; save/load for orientation maps is
  not implemented yet.
- Strain analysis is not connected yet.

## 2026-05-27 23:58 Asia/Shanghai

Author: Codex session

Goal:
- Extend orientation analysis using the notebook workflow instead of the old
  GUI flow.

Notebook/source reference:
- `S:\codex\SOFTWARE\PY4D\strain\strain 1.0 best.ipynb`
- Key notebook cells reproduced:
  - `bragg_peaks.get_bvm()` for BVM from centered data
  - `radial_integral` style radial profile
  - q-pixel-size test with CIF structure-factor overlay
  - `match_single_pattern`
  - `generate_diffraction_pattern`
  - `plot_diffraction_pattern` style fit comparison
  - `plot_orientation_maps` style color orientation output
- py4DSTEM source was used to preserve algorithm meaning where needed.

Files changed:
- `src/fourdlab/processing/orientation.py`
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Added BVM from centered detected peaks:
  - available on the Peak Detection tab after full-scan peak detection
  - available on each orientation tab using that tab's q-center settings
- Added notebook-style radial profile from BVM.
- Added CIF structure-factor profile calculation.
- Added q-pixel-size scan:
  - scans a user-defined q-pixel-size range
  - compares BVM radial profile against CIF structure-factor positions
  - writes the best q-pixel size back into the orientation controls
  - plots experiment, structure-factor overlay, and score trend
- Added fit preview:
  - uses current scan position
  - overlays experimental peaks with simulated Bragg peaks from the best fit
  - uses generated diffraction patterns from py4DSTEM's crystal model
  - shows hkl labels when the generated pattern includes h/k/l fields
- After `Run Map`, the GUI now automatically shows a fit preview at the current
  scan position so the user can quickly inspect matching quality.
- Added RGB orientation display:
  - in-plane mode maps fitted in-plane angle to hue and correlation to value
  - out-of-plane mode maps the fitted zone-axis vector to RGB and correlation
    to brightness

Verification:
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Ran a temporary processing smoke test:
  - built a synthetic BVM from detected peaks
  - loaded a small synthetic CIF
  - scanned q-pixel size against structure factors
  - built an orientation plan
  - generated fit diffraction patterns
  - rendered an RGB orientation image
- Ran an offscreen Qt smoke test:
  - instantiated `DiffractionAnalysisWindow`
  - verified new q-pixel controls exist
  - generated and displayed a centered BVM in the in-plane tab

Known limitations:
- The q-pixel scan is a helpful calibration assistant, not a final calibration
  authority; real confirmation should still use the 2D fit overlay.
- Structure-factor overlay currently uses the notebook's radial/z-tolerance
  approach. More advanced zone-specific overlays can be added after testing on
  real data.

## 2026-05-28 00:34 Asia/Shanghai

Author: Codex session

Goal:
- Add an optional no-CIF in-plane orientation route and improve fit-result
  inspection.

Files changed:
- `src/fourdlab/processing/orientation.py`
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Added no-CIF experimental-template in-plane matching:
  - In-Plane `Method` now supports `CIF indexing` and `No-CIF template`
  - `Build Plan/Template` builds an experimental template from the main
    Viewer's current real-space scan point
  - `Preview Current`, `Fit Preview`, and `Run Map` are routed to either CIF
    matching or no-CIF template matching based on the selected method
  - no-CIF controls include symmetry order, pixel tolerance, center exclusion,
    angular step, and minimum peak count
- Added no-CIF map output using the same orientation-result display path:
  - correlation score map
  - in-plane rotation map
  - color orientation image
- Changed orientation fit preview to prefer peaks from the main Viewer's
  current real-space point whenever a full peak result is available. This makes
  fit inspection follow the point the user selects in the main Viewer.
- Improved fit-overlay display:
  - qx/qy plot aspect is locked to 1:1
  - display range is automatically fit to a square bounding box around
    experimental and fitted/template peaks
  - radial/profile plots unlock the aspect ratio again
- Added `Save Result Image` on orientation tabs. It exports the current lower
  result/profile/fit plot as a PNG.

Verification:
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Ran a no-CIF processing smoke test:
  - built a synthetic template
  - matched a rotated synthetic peak set
  - generated a one-row no-CIF orientation map
  - rendered the color orientation image
- Ran an offscreen Qt smoke test:
  - instantiated `DiffractionAnalysisWindow`
  - switched In-Plane to `No-CIF template`
  - verified new no-CIF controls and plot state exist

Known limitations:
- No-CIF mode reports relative in-plane rotation only. It does not identify hkl
  labels, absolute orientation, or out-of-plane tilt.
- The current no-CIF matcher is CPU based and scans discrete angles. It is
  intentionally simple first, so it can be tested against real data before
  adding acceleration or more advanced scoring.

## 2026-05-28 01:07 Asia/Shanghai

Author: Codex session

Goal:
- Make orientation fit-preview coordinates visually match the main Viewer
  diffraction display.

Files changed:
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `docs/WORK_LOG.md`

Details:
- The main Viewer displays diffraction with screen-horizontal `qx` and
  screen-vertical `qy`.
- The fit overlay had been using the notebook-style axes:
  - horizontal `q_y`
  - vertical `q_x`
  This made the experimental peaks look rotated relative to the Viewer.
- Updated CIF fit overlay to draw:
  - horizontal `q_x`
  - vertical `q_y` in the Viewer direction
  - generated pattern labels at the transformed display coordinates
- Updated no-CIF fit overlay with the same Viewer-like display convention.
- Matching math still uses the configured `Flip qy`; only the preview display
  is transformed back to the Viewer direction.

Verification:
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Ran an offscreen no-CIF fit-preview smoke test and verified the plot axes are
  `q_x (pixels)` and `q_y (pixels, viewer direction)`.

## 2026-05-28 01:31 Asia/Shanghai

Author: Codex session

Goal:
- Add diffraction rotation symmetry to in-plane color mapping and clean up the
  orientation GUI proportions/preview style.

Files changed:
- `src/fourdlab/processing/orientation.py`
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Added an In-Plane `Rotation symmetry` control.
  - CIF and no-CIF color maps use this value.
  - no-CIF matching also uses this value as its angular equivalence period.
  - Example: with 4-fold symmetry, rotations separated by 90 degrees map to
    the same hue family.
- Updated `orientation_color_image` to accept `symmetry_order`.
- Updated the in-plane color wheel so it repeats according to the selected
  symmetry order.
- Improved GUI proportions:
  - larger default Diffraction Analysis window
  - wider but bounded scrollable parameter panels
  - larger main preview/image area
  - balanced image/fit-preview row stretch
- Cleaned the lower fit preview:
  - hidden q-axis tick labels and axis labels for fit overlays
  - white fit canvas
  - black border around the preview
  - retained 1:1 aspect ratio and square auto-fit range
  - profile/q-pixel scan plots still show axes and grids

Verification:
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Ran an offscreen Qt smoke test:
  - set In-Plane `Rotation symmetry` to 4
  - verified the color wheel pixmap exists
  - verified no-CIF fit preview hides both axes
  - verified no-CIF fit preview keeps a locked 1:1 aspect ratio

Known limitations:
- The color wheel is a compact visual legend only; it does not yet include
  numeric angle labels. The fit preview intentionally hides numeric axes for a
  cleaner inspection view.

## 2026-05-28 02:08 Asia/Shanghai

Author: Codex session

Goal:
- Address first batch of follow-up issues:
  - fit-preview y direction still looked mirrored
  - CIF/no-CIF parameter pages were too mixed
  - large datasets loaded too slowly
  - add a More-menu crop/bin utility
  - clarify peak Gaussian fitting

Files changed:
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `src/fourdlab/gui/viewer.py`
- `src/fourdlab/gui/crop_bin_dialog.py`
- `src/fourdlab/io/loaders.py`
- `src/fourdlab/processing/datacube_ops.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Corrected fit-preview y display direction:
  - previous fix restored qx/qy axis order but still mirrored y because
    `PlotWidget` and `ImageView` use different visual y conventions
  - `_viewer_display_qy` now maps the internal qy convention back to the
    Viewer-like diffraction display
- Renamed the peak refine control to `Subpixel Gaussian fit`.
  - The existing peak detector already performs a local 2D Gaussian refinement
    when this is enabled.
  - More advanced atom-finding-style models are still a future improvement.
- Added CIF/no-CIF parameter filtering:
  - `CIF indexing` hides no-CIF-only tolerance/center-exclusion controls
  - `No-CIF template` hides CIF/q-pixel/structure-factor controls
  - common controls remain visible
- Added CIF in-plane zone/fiber axis inputs as h/k/l, default `[0, 0, 1]`.
  These feed `OrientationConfig.fiber_axis`.
- Added `More -> Crop / Bin` as the first More-menu entry.
  - real-space crop can come from the real-space square ROI or manual values
  - diffraction crop can come from the diffraction ROI bounding box or manual
    values
  - scan and diffraction axes can be binned independently
  - processed data returns directly to the main Viewer through
    `replace_datacube`
- Changed the default loader mode to lazy/memory-mapped access where supported.
  - `.npy` now opens as `np.memmap` by default
  - HDF5/EMD/PY4DSTEM keeps the dataset file-backed by default
  - RAW was already memmapped

Verification:
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Ran an offscreen Crop / Bin GUI smoke test:
  - synthetic `(4,4,8,8)` datacube
  - real-space crop/bin and diffraction crop/bin
  - returned `(2,2,4,4)` datacube to the viewer
- Ran an offscreen orientation GUI smoke test:
  - verified CIF/no-CIF parameter visibility switches correctly
- Verified `.npy` loading now returns `memmap` by default.
- Ran `conda run -n 4dlab python -m pip check` successfully.

Known limitations:
- Lazy loading avoids the initial full RAM copy, but virtual-image and some
  analysis paths may still need chunked computation for very large datasets.
- Crop/bin currently returns an in-memory datacube. For very large crop/bin
  results, a future save-to-file/memmap output option would be safer.

## 2026-05-28 02:32 Asia/Shanghai

Author: Codex session

Goal:
- Re-check py4DGUI/py4DSTEM-style loading behavior and improve 4DLAB viewer
  load speed for large datasets.

Reference inspected:
- `S:\codex\SOFTWARE\PY4D\strain\strain_core.py`
  - RAW loading uses `np.memmap`.
  - Heavy overview/VDF calculations are explicit tasks, not required for
    immediate object creation.
- `C:\ProgramData\Miniconda3\envs\4dlab\Lib\site-packages\py4DSTEM\io\legacy\legacy12\read_v0_6.py`
  - legacy reader exposes `mem="RAM"` vs `mem="MEMMAP"`.
  - `MEMMAP` keeps HDF5 `g["data"]` file-backed instead of copying into RAM.

Files changed:
- `src/fourdlab/io/datacube.py`
- `src/fourdlab/gui/viewer.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Kept the existing lazy/memmap default from the previous pass and aligned the
  reasoning with py4DSTEM's `MEMMAP` path.
- Added `DataCube.quick_navigation_image`.
  - It returns a cheap real-space image from one detector pixel near the current
    detector center.
  - Viewer load/replace now uses this for the initial real-space display instead
    of computing a full virtual detector image.
- Added `DataCube.virtual_image_chunked`.
  - Computes virtual detector images scan-row chunk by scan-row chunk.
  - Reads only the detector mask bounding box for each chunk.
  - Avoids the previous large temporary allocation from `data[:, :, mask]`.
- Updated `FourDLabViewer.update_virtual_image`.
  - `quick=True` gives immediate first display.
  - normal refresh/ROI changes calculate the full VDF with chunking.

Verification:
- Ran a numerical comparison:
  - original `virtual_image` vs new `virtual_image_chunked`
  - annulus/sum and circle/mean both matched with `np.allclose`.
- Ran an offscreen Viewer load smoke test:
  - saved a temporary `.npy`
  - loaded it through `FourDLabViewer.load_file`
  - verified the data object is `memmap`
  - verified the initial virtual/navigation image has the expected scan shape.
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.

Known limitations:
- Full virtual detector images still require reading the full scan, just in
  safer chunks. For very large files, the next step is background-thread VDF
  calculation with progress/cancel and optional downsampled preview.

## 2026-05-28 02:55 Asia/Shanghai

Author: Codex session

Goal:
- Improve non-RAW file import compatibility and add export for processed
  datacubes.

Reference inspected:
- `S:\codex\SOFTWARE\PY4D\strain\strain_core.py`
  - `read_datacube` uses `py4DSTEM.read(filepath, root=root)`.
  - `save_py4dstem_object` uses `py4DSTEM.save(...)`.
- `py4DSTEM.io.read`
  - uses a datapath/root concept for HDF5/EMD trees.
  - legacy readers can expose root groups and datacube paths.
- `py4DSTEM.io.save`
  - wraps `emdfile.save`.

Files changed:
- `src/fourdlab/io/loaders.py`
- `src/fourdlab/io/datacube.py`
- `src/fourdlab/io/exporters.py`
- `src/fourdlab/gui/viewer.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- RAW import was intentionally left unchanged.
- Improved HDF5/EMD/PY4DSTEM import:
  - now scans all 4D datasets instead of taking the first one encountered
  - ranks candidates and prefers py4DSTEM-like datacube paths such as
    `/datacube/data` or paths containing `datacube/datacubes`
  - error messages now include a short list of datasets found when no 4D
    dataset can be loaded
- Added `DataCube.close` handling for HDF5 datasets, closing the backing file
  handle when a file-backed dataset is replaced or explicitly closed.
- Added `src/fourdlab/io/exporters.py`.
  - `.npy`: chunked `open_memmap` export
  - `.h5/.hdf5/.emd/.py4dstem`: writes `/datacube/data` with gzip compression
    and simple 4DLAB metadata
  - `.raw`: writes C-order binary data and a `.raw.json` sidecar with shape,
    dtype, source path, and dataset path
- Added `File -> Export Datacube...` in the main Viewer.
  - exports the current datacube exactly as currently loaded/processed,
    including crop/bin or center-correction results.

Verification:
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Ran export/import smoke test:
  - exported a synthetic datacube to `.npy`, `.h5`, and `.raw`
  - reloaded `.npy` and `.h5`
  - verified shape, memmap loading, HDF5 dataset path, and raw sidecar metadata
  - verified HDF5 file handles close cleanly on Windows
- Ran offscreen Viewer smoke test confirming `Export Datacube...` action exists.
- Ran `conda run -n 4dlab python -m pip check` successfully.

Known limitations:
- The HDF5/EMD/PY4DSTEM importer still chooses one best 4D dataset
  automatically. If a file contains multiple valid 4D datacubes, a future
  dataset-selection dialog would be the next improvement.
- RAW export has no embedded shape metadata by design; the generated JSON
  sidecar must be kept with the raw file.

## 2026-05-28 00:52 Asia/Shanghai

Author: Codex session

Goal:
- Make growing parameter panels usable and improve orientation plot readability.

Files changed:
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Wrapped Peak Detection and Orientation left-side control panels in
  scrollable containers. This keeps all parameters/buttons reachable as more
  controls are added.
- Added an in-plane rotation color wheel to the In-Plane orientation panel.
- Applied a consistent light plot style:
  - light background
  - subtle grid
  - darker axis text
  - clearer legend handling
- Improved orientation fit plots:
  - fit overlays keep a locked 1:1 aspect ratio
  - profile/q-pixel plots unlock aspect ratio again
  - q-pixel scan now shows a legend for experiment, structure factors, and
    score trend
  - fit markers are slightly larger and easier to distinguish

Verification:
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Ran an offscreen Qt smoke test:
  - instantiated `DiffractionAnalysisWindow`
  - verified the In-Plane color wheel exists
  - verified In-Plane still exposes both CIF and no-CIF methods
- Ran a no-CIF fit-preview smoke test:
  - built a synthetic template
  - opened the fit preview
  - verified the plot kind is `fit` and the qx/qy view has a locked 1:1 aspect

## 2026-05-28 03:10 Asia/Shanghai

Author: Codex session

Goal:
- Add the strain-analysis stage from the strain notebook workflow and polish
  the diffraction-analysis GUI so it can continue as the central nanobeam
  analysis surface.

Reference inspected:
- `S:\codex\SOFTWARE\PY4D\strain\strain 1.0 best.ipynb`
  - Bragg peak workflow centers peaks, calibrates reciprocal pixels, maps
    orientations, then calculates strain.
  - The notebook extracts and plots `e_xx`, `e_yy`, `e_xy`, and `theta`, and
    later converts them to polar components `e_rr`, `e_tt`, and `e_rt`.
- `py4DSTEM.process.strain.latticevectors`
  - The key strain conversion uses a fitted reciprocal-lattice transform:
    `e_xx = 1 - beta[0,0]`, `e_yy = 1 - beta[1,1]`,
    `e_xy = -(beta[0,1] + beta[1,0]) / 2`, and
    `theta = (beta[0,1] - beta[1,0]) / 2`.

Files changed:
- `src/fourdlab/processing/strain.py`
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Added an independent strain-processing module that works directly from
  4DLAB detected peak results.
- The strain module:
  - guesses two non-collinear reciprocal basis vectors from the current peak
    list
  - enumerates h/k lattice spots up to a configurable max index
  - matches predicted lattice spots to observed peaks with a configurable
    pixel tolerance
  - fits a 2D lattice transform per scan position with optional intensity
    weighting
  - uses the median fitted transform in a user-selected reference scan region
    as the zero-strain reference
  - calculates `e_xx`, `e_yy`, `e_xy`, `theta`, fit error, matched peak count,
    and valid mask
  - supports output tensor rotation, theta sign flip, and notebook-style polar
    strain components
  - saves full strain results as compressed `.npz`
- Replaced the previous placeholder `Strain` tab with a complete GUI page:
  - scrollable parameter panel
  - diffraction center and qy flip controls
  - manual `g1/g2` lattice-vector controls
  - buttons for guessing basis, setting the current scan point as reference,
    previewing the current fit, running the full map, stopping, saving PNG,
    and saving result data
  - progress bar and log output
  - result selector for strain components, rotation, error, match count, mask,
    and polar strain components
  - current-point fit preview with observed peaks, reference h/k points, and
    fitted lattice points on a square axis-free plot
  - strain maps use diverging color gradients; error/count maps use sequential
    gradients
- Kept the implementation isolated from import/export, center correction,
  crop/bin, peak detection, and orientation code except for explicitly reused
  peak result structures.

Verification:
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Ran a synthetic strain smoke test:
  - created a 2x2 peak-result grid with known reciprocal-lattice transforms
  - verified the reference point reports near-zero strain
  - verified a known transform reports `e_xx ~= 0.01` and `e_yy ~= -0.02`
  - verified lattice matching returned 10 matched spots for the test pattern
- Ran an offscreen Qt smoke test:
  - instantiated `DiffractionAnalysisWindow`
  - verified the fourth tab is `Strain`
  - verified the strain controls and display selector are present
- Verified the `ImageView` PNG export path used by the Strain tab works in an
  offscreen PyQtGraph smoke test.

Known limitations:
- The new strain path is a direct lattice-vector fit from detected peaks, not a
  full wrapper around `crystal.calculate_strain`. This keeps it usable with or
  without CIF, but a later CIF-assisted mode could use orientation-map matches
  to auto-pick reference basis vectors.
- Reference selection is numeric/current-point based in this pass. A future
  improvement could bind it directly to the main Viewer's real-space square ROI
  for multi-pixel reference regions.
- Full-scan strain fitting still runs synchronously in the GUI thread with
  progress and cancel checks between scan positions. For very large scans, the
  next polish step is moving peak detection/orientation/strain long jobs to a
  worker thread.

## 2026-05-28 03:20 Asia/Shanghai

Author: Codex session

Goal:
- Do a global polish pass over the current codebase and fix the highest-impact
  rough edges discovered during earlier work.

Files changed:
- `src/fourdlab/gui/workers.py`
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `src/fourdlab/gui/center_correction_dialog.py`
- `src/fourdlab/processing/center_correction.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Added a reusable Qt background-task helper:
  - `Worker` runs a cooperative task inside a `QThread`
  - `CallbackRelay` keeps GUI updates on the main Qt thread
  - `RunningTask` owns the thread/worker and exposes `request_stop`
  - callers pass a task function that receives `progress(done, total)` and
    `stop_requested()` callbacks
- Migrated long diffraction-analysis jobs away from the GUI thread:
  - full-scan peak detection
  - CIF orientation-map matching
  - no-CIF orientation-map matching
  - full-scan strain fitting
- Preserved the existing progress bars, logs, completion summaries, and Stop
  buttons while replacing the previous `QApplication.processEvents()` pattern.
- Added guards so a second long task cannot be started while one of the same
  class is already running.
- Added close-event protection for long-running analysis windows:
  - closing requests stop
  - the window stays open until the worker exits cleanly
  - this avoids destroying active `QThread` objects
- Updated center correction:
  - processing now accepts `stop_requested`
  - added `CenterCorrectionCancelled`
  - the dialog now runs correction in the same background worker system
  - added a Stop button and close-event protection
- Removed the now-redundant manual `_cancel_*` state flags from diffraction
  analysis because cancellation is owned by `RunningTask`.

Verification:
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Confirmed no old `QApplication.processEvents()` long-task pattern remains in
  `src/fourdlab` with `rg`.
- Ran an offscreen asynchronous GUI smoke test covering:
  - background full-scan peak detection
  - background no-CIF orientation map
  - background strain map
  - background center correction
  - result objects and GUI state are written back after worker completion

Known limitations:
- CIF-backed py4DSTEM orientation work now runs off the GUI thread, but the
  underlying py4DSTEM/CuPy/HDF5 stack may still have its own thread-safety
  limits depending on the exact data source. If a vendor file reader proves
  thread-sensitive, the next step is to queue only CPU-bound matching in the
  worker after copying the needed peak data.
- Background jobs are still one task at a time per analysis window. This is
  intentional for now because peak/orientation/strain outputs depend on shared
  GUI state.

## 2026-05-28 11:38 Asia/Shanghai

Author: Codex session

Goal:
- Improve actual computation speed after moving long jobs to background
  threads made the GUI responsive but did not reduce runtime enough.

Files changed:
- `src/fourdlab/processing/peak_detection.py`
- `src/fourdlab/processing/orientation.py`
- `src/fourdlab/processing/strain.py`
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Added a faster peak-refinement path:
  - new `refine_peak_centroid` performs local baseline-subtracted centroid
    refinement
  - Gaussian `curve_fit` refinement remains available for highest precision
  - peak detection presets now store `refine_method`
  - GUI label changed from `Subpixel Gaussian fit` to `Subpixel refine`
  - GUI now exposes `Refine mode` with `Fast centroid` and `Gaussian fit`
- Added configurable CPU workers:
  - peak detection `PeakDetectionConfig.num_workers`
  - no-CIF orientation `NoCifInPlaneConfig.num_workers`
  - strain fitting `StrainConfig.num_workers`
  - GUI exposes worker count controls for each relevant page
  - default worker count is conservative: up to 4, leaving one CPU free when
    possible
- Added ThreadPool-based scan-position parallelization for independent tasks:
  - full-scan peak detection
  - no-CIF full-scan orientation matching
  - full-scan strain lattice fitting
- Kept CIF/py4DSTEM orientation sequential internally because py4DSTEM/CuPy
  internals may own their own threading and state.

Verification:
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Ran a synthetic correctness/speed smoke test:
  - sequential and parallel peak detection produced identical peak-count maps
  - sequential and parallel no-CIF orientation maps matched
  - sequential and parallel strain maps matched
  - on the synthetic test, fast centroid refinement was about 32x faster than
    Gaussian refinement for peak detection
  - small synthetic data showed parallel overhead can exceed benefit, so the
    worker control remains user-adjustable
- Ran an offscreen GUI smoke test confirming:
  - peak refinement mode initializes
  - peak, no-CIF, and strain worker controls are present

Known limitations:
- Thread parallelism helps most on medium/large scans. For tiny scans, thread
  scheduling overhead can be slower than one worker.
- HDF5-backed datasets may not scale as much as RAW/NPY/memmap files because
  h5py can serialize file reads internally.
- A future speed pass could add chunked/process-based execution for RAW/NPY
  data only, but that should stay opt-in because Windows multiprocessing has
  more startup and serialization overhead.

## 2026-05-28 12:00 Asia/Shanghai

Author: Codex session

Goal:
- Further optimize map-stage runtime after fit preview became fast but full
  maps were still slow.

Files changed:
- `src/fourdlab/processing/orientation.py`
- `src/fourdlab/processing/strain.py`
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Optimized no-CIF orientation maps:
  - added `PreparedNoCifMatcher`
  - all rotated template vectors are now generated once at map start
  - each scan point performs one batched KDTree query across all rotated
    templates instead of looping over angles and rotating every time
  - single-point preview still uses the same public `match_no_cif_in_plane`
    API, now backed by the prepared matcher
- Optimized CIF-backed orientation maps:
  - added `OrientationConfig.num_workers`
  - GUI now exposes `CIF CPU workers`
  - both in-plane CIF and out-of-plane CIF maps can run scan-point matching in
    parallel when CUDA is not active
  - when CUDA is active, worker count is forced to 1 to avoid multiple threads
    competing for the same GPU/py4DSTEM state
- Optimized strain maps:
  - added `PreparedStrainMatcher`
  - h/k index generation, predicted lattice vectors, radius filtering, and
    tolerance settings are cached once per map
  - preview fit still uses the same public `fit_lattice_transform` API, now
    backed by the prepared matcher
  - full strain maps reuse the cached matcher for every scan point

Verification:
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Ran a map optimization smoke test:
  - no-CIF single-point matching returns a finite angle and positive score
  - no-CIF sequential and parallel map outputs match
  - strain preview fit still returns enough matched peaks
  - strain sequential and parallel maps match for `e_xx`, `e_yy`, and `e_xy`
  - GUI initializes CIF worker controls for both In-Plane and Out-of-Plane tabs

Known limitations:
- CIF map parallelism depends on py4DSTEM's internal matching being stable for
  concurrent read-only use of the prepared crystal/orientation plan. If a
  specific CIF or CUDA path behaves badly, set `CIF CPU workers` to 1.
- Full CIF map speed still depends heavily on py4DSTEM's orientation-plan size.
  Reducing zone/in-plane angular step density or narrowing the zone-axis search
  remains the biggest algorithmic speed lever.

## 2026-05-28 13:38 Asia/Shanghai

Author: Codex session

Goal:
- Make the Diffraction Analysis orientation tabs easier to understand by
  grouping parameters by function.
- Add finer q-pixel calibration and CIF plane preview controls.

Files changed:
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `src/fourdlab/processing/orientation.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Reorganized the In-Plane and Out-of-Plane orientation left panels into
  separate functional boxes:
  - Method and CIF
  - Center and Calibration
  - CIF Plane and Radial Calibration
  - Orientation Matching
  - No-CIF Experimental Template
  - Results and Display
- Added q-pixel radial source selection:
  - Full BVM
  - Current scan pixel
- Added scan-y/scan-x controls plus a `Use Viewer Pixel` shortcut so one
  selected scan position can be used for radial q-pixel calibration.
- Added zone/plane tolerance and reused the selected h/k/l zone for CIF radial
  structure-factor matching.
- Added CIF diffraction preview for the selected zone/plane, displaying
  projected reciprocal spots with hkl labels.
- Added processing helpers for:
  - radial profiles from one `PeakList`
  - CIF zone/plane structure-factor profiles
  - CIF zone/plane projected diffraction previews

Verification:
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Ran an offscreen Qt smoke test that instantiated
  `DiffractionAnalysisWindow` and confirmed the new orientation groups and
  q-pixel source controls exist.
- Ran a lightweight synthetic processing test for:
  - one-scan-pixel peak radial profile
  - CIF zone diffraction preview
  - q-pixel scan against a selected zone

Known limitations:
- The CIF preview is currently a reciprocal-space projection of CIF
  structure-factor spots, not a full 3D atomic crystal model viewer.
- The selected h/k/l zone is interpreted using the same practical coordinate
  convention already used by the orientation workflow; more crystallographic
  plane/zone helpers can be added later if needed.

Suggested next step:
- Add a dedicated 3D CIF structure viewer or a richer plane/zone selector if
  users need to inspect the atomic model rather than only the predicted
  diffraction for the selected zone.

## 2026-05-28 16:53 Asia/Shanghai

Author: Codex session

Goal:
- Make RAW import recoverable when files do not match the default EMPAD
  `float32 130 x 128` layout.
- Replace the simplified CIF zone projection with py4DSTEM-generated
  diffraction and add a first CIF fit diagnostic.

Files changed:
- `src/fourdlab/io/__init__.py`
- `src/fourdlab/io/loaders.py`
- `src/fourdlab/gui/raw_import_dialog.py`
- `src/fourdlab/gui/viewer.py`
- `src/fourdlab/processing/orientation.py`
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Added `RawLoadConfig` and `RawShapeError`.
- `load_datacube(..., raw_config=...)` can now load non-default RAW files with
  explicit scan shape, diffraction shape, dtype, and bottom-row crop settings.
- Default RAW loading still uses `float32 130 x 128` and crops the final two
  EMPAD rows to `128 x 128`.
- If default RAW loading fails in the Viewer, a new `RawImportDialog` opens and
  validates expected bytes against actual file size before enabling OK.
- Reworded the CIF controls from plane wording to zone-axis / beam-direction
  wording.
- Replaced the previous hand-projected CIF preview with
  `Crystal.generate_diffraction_pattern(zone_axis_lattice=...)` so the preview
  follows py4DSTEM's orientation and excitation-error conventions.
- q-pixel radial scan now uses the same generated CIF diffraction pattern as
  the CIF preview.
- Added `Diagnose CIF Fit` for the current scan pixel:
  - converts experimental peaks to reciprocal units using current center,
    q-pixel size, and qy flip settings
  - compares them to generated CIF spots by nearest-neighbor matching
  - plots experimental peaks, simulated peaks, and match lines
  - logs matched count, mean/median residual, and suggestions
  - checks current settings, flipped qy, and a small radial q-pixel scan

Verification:
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Created a temporary default EMPAD-style RAW file and verified it loads as
  `(2, 2, 128, 128)` after crop.
- Created a temporary non-default RAW file and verified explicit
  `RawLoadConfig(4, 5, 64, 64, "float32", 0)` loads as `(4, 5, 64, 64)`.
- Verified non-default RAW loading without config raises `RawShapeError`.
- Ran an offscreen `RawImportDialog` validation smoke test and confirmed OK is
  enabled only for matching dimensions.
- Ran a synthetic CIF-processing smoke test for generated diffraction preview,
  q-pixel scan, and CIF fit diagnostic.
- Ran an offscreen `DiffractionAnalysisWindow` smoke test confirming the
  updated CIF zone-axis controls initialize.

Known limitations:
- The RAW dialog does not persist per-file import settings yet.
- CIF fit diagnostic gives suggestions and visual evidence, but does not
  automatically change q-pixel size, qy flip, center, or zone axis.
- The CIF preview is diffraction-focused. It does not yet show a 3D atomic
  structure model.

Suggested next step:
- Test `Preview CIF Diffraction` and `Diagnose CIF Fit` with a real CIF/raw
  pair and tune the default zone tolerance / excitation sigma if needed for the
  user's material.

## 2026-05-28 17:25 Asia/Shanghai

Author: Codex session

Goal:
- Make Out-of-Plane orientation search user-controlled instead of only using
  py4DSTEM `auto`.
- Add a py4DSTEM-style out-of-plane orientation map with triangular color
  legend.
- Let Out-of-Plane reuse In-Plane CIF/calibration settings without overwriting
  the Out-of-Plane search range.

Files changed:
- `src/fourdlab/processing/orientation.py`
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Extended `OrientationConfig` with Out-of-Plane range settings:
  - `zone_range_mode`
  - `zone_axis_center`
  - `zone_angle_deg`
  - `zone_axis_vertices`
- Added Out-of-Plane range generation:
  - `center_cone` maps to py4DSTEM `zone_axis_range="fiber"` with
    `fiber_axis=center` and `fiber_angles=[max_tilt, 360]`
  - `three_vertices` maps to a 3x3 py4DSTEM `zone_axis_range`
  - In-Plane remains unchanged and still uses fiber mode around its configured
    axis.
- Added an `Out-of-Plane Search Range` GUI group with:
  - `Center + tilt cone`
  - `Three zone-axis vertices`
  - center h/k/l and max tilt controls
  - three h/k/l vertex rows
  - `Use In-Plane CIF/Calibration`
  - `Sync from In-Plane`
- `Sync from In-Plane` copies CIF path, q-pixel size, q center, qy flip, k max,
  voltage, correlation kernel, excitation sigma, min peaks, and matches.
- Sync intentionally invalidates any existing Out-of-Plane crystal/workspace
  because the CIF/calibration inputs changed, but it keeps the Out-of-Plane
  search range untouched.
- Added `py4DSTEM orientation map` to the Out-of-Plane display selector.
- The py4DSTEM map path calls `crystal.plot_orientation_maps(...,
  show_legend=True, returnfig=True)` and renders the full figure, including the
  triangular orientation color legend, into the GUI image pane.
- Added a fallback orientation-map renderer that combines the existing 4DLAB RGB
  map with a simple triangular zone-color legend if py4DSTEM figure rendering is
  unavailable.
- `Save Result Image` now saves the current image pane for map displays and
  writes the rendered py4DSTEM map image directly when selected.

Verification:
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Verified `center_cone` config produces py4DSTEM fiber-search arguments.
- Verified `three_vertices` config produces a 3x3 zone-axis range and no fiber
  arguments.
- Ran an offscreen Qt smoke test confirming:
  - Out-of-Plane range controls exist
  - switching to three-vertices mode hides the center/tilt controls and shows
    vertex controls
  - `py4DSTEM orientation map` appears in the Out-of-Plane display selector
  - In-Plane sync copies calibration fields and preserves the range mode
- Ran an offscreen fallback legend render test and confirmed it produces an RGB
  image.
- Ran an offscreen py4DSTEM-figure mock render test and confirmed the rendered
  image path works.

Known limitations:
- The py4DSTEM orientation map uses fixed plotting defaults for now:
  `orientation_ind=0`, `corr_range=[1, 4]`, `camera_dist=10`, and hidden axes.
- The fallback legend labels the three range vertices numerically; py4DSTEM's
  own legend is preferred whenever available.

Suggested next step:
- Run Out-of-Plane on a real CIF/raw dataset, compare the py4DSTEM orientation
  map legend against expected zone-axis colors, then expose `corr_range` and
  `orientation_ind` as optional controls if needed.

## 2026-05-29 00:00 Asia/Shanghai

Author: Codex session

Goal:
- Redesign `Diffraction Analysis` toward a professional Viewer-style
  orientation workbench.
- Add VisPy as the primary interactive drawing layer for the orientation
  analysis chain while keeping existing pyqtgraph tabs available.

Files changed:
- `environment.yml`
- `pyproject.toml`
- `src/fourdlab/gui/vispy_views.py`
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Added `vispy` to the conda and Python project dependencies.
- Added `gui/vispy_views.py` with lightweight result structures and reusable
  Qt widgets:
  - `AnalysisResult`
  - `AnalysisOverlay`
  - `VispyImageView`
  - `VispyOrientationMapView`
  - thumbnail and triangle-legend image helpers
- Reworked `DiffractionAnalysisWindow` into a workbench layout while preserving
  the existing tab implementations:
  - top analysis toolbar
  - left task toolbox
  - horizontal Result Strip
  - central VisPy-backed result view
  - existing Peak Detection, In-Plane, Out-of-Plane, and Strain tabs below the
    new result view
- Registered orientation-chain outputs into the Result Strip:
  - current diffraction pattern
  - detected peaks
  - centered BVM
  - CIF diffraction preview
  - CIF fit diagnostic
  - In-Plane orientation color map
  - Out-of-Plane orientation map / py4DSTEM reference map
- Clicking a Result Strip card now switches the central result view, and
  `Save Current Figure` saves the currently selected result instead of an
  incidental latest insertion.
- Offscreen Qt sessions intentionally use the QLabel fallback instead of trying
  to create a VisPy OpenGL canvas; normal desktop sessions still use VisPy.

Verification:
- Installed VisPy into the `4dlab` conda environment from conda-forge.
- Ran `conda run -n 4dlab python -c "import vispy; print(vispy.__version__)"`
  and confirmed `vispy 0.16.2`.
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Ran an offscreen `DiffractionAnalysisWindow` smoke test confirming:
  - top toolbar exists
  - left toolbox has five panels
  - Result Strip creates a card for the current pattern
  - selecting a card updates the current central result key
- Ran a lightweight helper test for `AnalysisResult`, `AnalysisOverlay`,
  orientation triangle legend composition, and thumbnail generation.

Known limitations:
- This first pass focuses the orientation analysis chain. Peak detection and
  strain still use their existing pyqtgraph display panes.
- The standalone VisPy triangle helper draws the color triangle; labeled
  out-of-plane legends still come from the existing py4DSTEM figure path or the
  4DLAB Matplotlib fallback.
- The Result Strip is session-local and does not persist across application
  restarts.

Suggested next step:
- Run the redesigned workbench on a real Out-of-Plane dataset, compare the
  central VisPy map with the py4DSTEM reference figure, then expose additional
  display controls such as `corr_range`, selected orientation index, and map
  contrast presets.

## 2026-05-29 00:30 Asia/Shanghai

Author: Codex session

Goal:
- Correct the Diffraction Analysis result display after user feedback: result
  figures should be hidden behind a top-bar list and opened on demand, not
  consume a large permanent area above the analysis tabs.
- Reduce UI lag by avoiding always-on VisPy result canvases in the main window.

Files changed:
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Removed the permanent horizontal Result Strip from the main layout.
- Removed the permanent central VisPy result view from the main layout so the
  Peak / In-Plane / Out-of-Plane / Strain tabs regain the main workspace.
- Added a compact `Result Images` toolbar menu.
- Analysis outputs are now registered in that menu and do not automatically
  redraw a large canvas.
- Selecting a result from the menu opens it in a separate non-modal VisPy result
  window.
- `Save Current Figure` still saves the currently selected/generated result.

Verification:
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Ran an offscreen Qt smoke test confirming:
  - toolbar exists
  - toolbox still has five panels
  - main tabs remain present
  - `Result Images` becomes enabled once the current pattern result is
    registered
  - no permanent `workbench_view` exists in the main window
- Ran an offscreen pop-out smoke test confirming a selected result opens one
  separate dialog and updates the current result key.

Known limitations:
- Result windows are session-local and open on demand. They are not persisted
  across application restarts.

## 2026-05-29 01:00 Asia/Shanghai

Author: Codex session

Goal:
- Remove the redundant Diffraction Analysis command surfaces and make embedded
  analysis images square and more resize-friendly.

Files changed:
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Removed the outer left `QToolBox` command panel.
- Removed duplicate toolbar actions for refresh, detect, CIF preview,
  diagnostics, map runs, strain runs, and save.
- Kept tab-local buttons as the only primary operation entry point.
- Kept the top toolbar lightweight with only the `Result Images` menu.
- Made `QTabWidget` the direct central widget, without an extra outer splitter.
- Added a reusable square image panel wrapper for embedded result images.
- Wrapped Peak, In-Plane, Out-of-Plane, Strain, and popped-out result displays
  in square panels.
- Removed the old long rectangular image size hints and changed result pop-out
  windows to default to `820 x 820`.
- Relaxed the left parameter scroll area width so it can adapt better to
  smaller and larger windows.

Verification:
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Ran an offscreen Qt smoke test confirming:
  - the central widget is the main `QTabWidget`
  - all four analysis tabs still exist
  - no outer `QToolBox` remains
  - the toolbar has only the `Result Images` widget action
  - no `analysis_toolbox` attribute is created
  - embedded square image panels are present
  - `Result Images` is enabled after the current pattern is registered
- Ran an offscreen pop-out smoke test confirming:
  - selecting a result opens one separate dialog
  - the dialog defaults to `820 x 820`
  - the dialog uses a square image panel

## 2026-05-29 01:25 Asia/Shanghai

Author: Codex session

Goal:
- Fix Out-of-Plane display for two-part orientation images and make the
  triangular orientation legend reliably visible.
- Simplify fit previews so only one final fitted pattern is drawn.

Files changed:
- `src/fourdlab/gui/diffraction_analysis_window.py`
- `src/fourdlab/gui/vispy_views.py`
- `docs/FEATURE_IMPORT_VISUALIZATION.md`
- `docs/WORK_LOG.md`

Details:
- Added a reusable fixed-aspect image panel in addition to the square panel.
- Kept normal single-image previews square, but changed the Out-of-Plane image
  display to a wide aspect panel for map + legend layouts.
- Result pop-out windows now choose their layout from the result image aspect:
  square results open as `820 x 820`, wide results open as `1100 x 720`.
- Out-of-Plane orientation color display now renders the map with the 4DLAB
  triangular zone-color legend.
- The py4DSTEM orientation map path now checks whether a wide legend panel is
  present; if not, it falls back to the 4DLAB map + labeled triangular legend.
- VisPy fallback labels now scale their pixmap to the result view instead of
  clipping large/wide result images.
- CIF fit preview now draws only one final fitted diffraction pattern and
  labels it `fit`, instead of drawing `fit 1`, `fit 2`, etc.

Verification:
- Ran `conda run -n 4dlab python -m compileall -q src` successfully.
- Ran an offscreen wide-result pop-out smoke test confirming:
  - wide result dialog opens as `1100 x 720`
  - the dialog uses an aspect image panel
  - the Out-of-Plane tab includes wide aspect panels
- Ran a synthetic Out-of-Plane legend smoke test confirming:
  - fallback legend image shape is `(320, 720, 3)`
  - fallback aspect is `2.25`
  - the py4DSTEM-legend guard falls back to the same wide legend image when
    given a square map without a legend panel
