# Import and Visualization

This is the first user-facing feature in 4DLAB. It is the main window for
loading 4D-STEM data, inspecting diffraction patterns, creating virtual images,
and serving as the attachment point for later processing modules.

## Current Design

- The viewer is implemented in 4DLAB's own source code and does not import or
  depend on `py4D_browser`.
- The GUI uses PyQt5 and pyqtgraph for fast interaction and image updates.
- `Diffraction Analysis` now also uses VisPy for detached orientation/result
  figure windows and keeps pyqtgraph for the embedded peak/strain panes.
- File loading lives in `src/fourdlab/io/`.
- Rendering helpers live in `src/fourdlab/visualization/`.
- 4DLAB owns the launcher and extension boundary in `src/fourdlab/gui/`, so
  later modules can attach menus, dock panels, actions, or callbacks without
  modifying unrelated features.

## Entry Point

```powershell
conda activate 4dlab
4dlab-viewer
```

For development without installing the package:

```powershell
conda run -n 4dlab python -m fourdlab.gui.main
```

## Extension Boundary

Future features should implement the `ViewerExtension` protocol in
`src/fourdlab/gui/extensions.py` and attach themselves to the viewer through
`launch_viewer(ViewerLaunchConfig(extensions=[...]))`.

This keeps the import and visualization shell stable while allowing processing
features to be added independently.

## Notes

- This is now a clean 4DLAB implementation, not a wrapper around `py4D_browser`.
- The first loader supports `.npy`, `.raw`, `.h5`, `.hdf5`, `.emd`, and
  `.py4dstem` files that contain a 4D dataset.
- Supported loaders use lazy/memory-mapped access by default where possible, so
  large datasets do not need to be copied into RAM immediately on open.
- HDF5/EMD/PY4DSTEM import ranks all 4D datasets and prefers py4DSTEM-like
  datacube paths such as `/datacube/data` or `.../datacubes/.../data`.
- Initial display uses a fast detector-center navigation preview instead of
  immediately calculating a full virtual detector image. Full VDF images are
  calculated when the user refreshes or changes detector settings.
- Virtual detector images are computed in scan-row chunks and only read the
  detector mask bounding box, avoiding large one-shot `data[:, :, mask]`
  temporary arrays.
- `.raw` import currently assumes `float32` data and a default diffraction
  shape of `130 x 128`.
- Because the common RAW input is EMPAD data, the last two diffraction rows are
  cropped on import. Loaded RAW diffraction patterns therefore become
  `128 x 128`.
- If a RAW file does not fit the default `float32 130 x 128` layout, the GUI
  opens a RAW import parameter dialog instead of failing immediately. The
  dialog lets the user set scan Y/X, diffraction Y/X, dtype, and bottom-row
  crop, and validates the expected byte count before loading.
- The scan shape is inferred as square when possible; otherwise it is loaded as
  `1 x N`.
- The top menu currently includes:
  - `File`: open data, export the current datacube, and exit.
  - `Scaling`: linear, log, or square-root display scaling.
  - `Autorange`: toggle real-space/diffraction auto levels or apply both once.
  - `Detector Response`: sum, mean, or maximum detector response.
  - `Detector Shape`: circle, annulus, or square detector masks.
  - `FFT View`: show virtual image or FFT magnitude.
  - `Processing`: refresh views, reset detector, and choose real-space point
    or square selection.
  - `More`: analysis and processing feature entry points.

## py4DGUI-Style ROI Interaction

- The real-space view has an interactive selector:
  - Point selector: one scan position drives the displayed diffraction pattern.
  - Square selector: a scan-region average/sum/max drives the displayed
    diffraction pattern.
- The diffraction-space view has an interactive virtual detector:
  - Point detector: one diffraction pixel drives the virtual image.
  - Circle detector: circular integration drives the virtual image.
  - Annulus detector: ring integration drives the virtual image.
  - Square detector: rectangular integration drives the virtual image.
- Numeric controls remain available for precise adjustment, but the primary
  interaction is now through draggable ROIs on the images.

## More Menu

- `Crop / Bin` is the first More-menu entry.
- It can crop real space from the current real-space square ROI or manual
  bounds.
- It can crop diffraction space from the current diffraction ROI bounding box
  or manual bounds.
- It can bin real-space and diffraction-space axes independently, then return
  the processed datacube directly to the main GUI.
- `File -> Export Datacube...` writes the current datacube, including crop/bin
  or center-corrected results:
  - `.npy` using chunked `open_memmap` output
  - `.h5`, `.hdf5`, `.emd`, or `.py4dstem` as `/datacube/data`
  - `.raw` as C-order binary data plus a `.raw.json` sidecar containing shape,
    dtype, source path, and dataset path
- `Center Correction` opens a separate correction window.
- The correction uses the current diffraction-space detector ROI as the
  approximate center disk selection.
- It computes the center of mass inside that ROI for every diffraction pattern.
- Each diffraction pattern is shifted so the selected disk COM lands at the
  target diffraction center.
- Shifted-in edges are filled with 0.
- Correction runs in a background Qt worker, so the dialog can keep updating
  status and the operation can be stopped between diffraction patterns.
- After running correction, the dialog previews mean diffraction patterns
  before and after correction.
- Applying the result replaces the viewer datacube with the corrected datacube
  and refreshes the preview.
- `Diffraction Analysis` opens the nanobeam diffraction analysis main window.
- The Diffraction Analysis window is organized as a workbench:
  - a `Result Images` toolbar menu that hides generated intermediate figures in
    a compact list and opens the selected figure in a separate VisPy window
  - one main command surface: each tab owns its own operation buttons, avoiding
    duplicated toolbar/toolbox actions
  - square embedded image panels for single diffraction previews, orientation
    maps, and strain maps
  - a wider Out-of-Plane image panel for the two-part orientation map plus
    triangular zone-color legend
  - the main area remains dedicated to the analysis tabs instead of an outer
    toolbox, permanent result strip, or large always-on result canvas
  - Matplotlib remains available only for py4DSTEM reference figures and
    fallback legend rendering
- The first analysis tab is custom peak detection:
  - detect peaks in the current diffraction pattern
  - detect peaks across the full scan
  - show peak markers on the preview
  - refine peak centers with either fast subpixel centroid refinement or a
    slower 2D Gaussian fit
  - show progress during full-scan detection
  - run full-scan peak detection in a background worker
  - use configurable CPU workers for large full-scan peak detection jobs
  - stop full-scan peak detection after the current diffraction pattern
  - save and reload peak detection parameter presets from
    `configs/peak_detection_preset.json`
  - save and reload detected full-scan peak results as compressed `.npz` files
  - optionally use a CuPy GPU path for local-maximum candidate search when CuPy
    is installed; otherwise the GUI disables the GPU checkbox and uses CPU
- `In-Plane` and `Out-of-Plane` tabs now provide first-pass py4DSTEM-backed
  orientation analysis:
  - organize controls into separate functional boxes for method/CIF selection,
    center calibration, CIF zone-axis and q-pixel calibration, orientation matching,
    No-CIF template settings, and result display
  - choose a CIF file
  - set an in-plane CIF zone/fiber axis as h k l, interpreted as the beam
    direction / zone axis rather than a single plane reflection
  - preview the selected CIF zone-axis diffraction using py4DSTEM
    `generate_diffraction_pattern`, with hkl labels before running a full
    orientation plan
  - set reciprocal pixel calibration and diffraction center
  - build a py4DSTEM orientation plan
  - preview the current scan position orientation
  - run a full orientation map from saved or newly detected peaks
  - run full-scan orientation matching in a background worker
  - use configurable CPU workers for CIF-backed in-plane and out-of-plane maps
    when CUDA is not active
  - use configurable CPU workers for no-CIF full-scan matching
  - no-CIF map matching precomputes all rotated templates once per map instead
    of rebuilding them at every scan point
  - stop full-scan orientation matching between scan positions
  - display centered BVM from detected peaks
  - scan q-pixel size against CIF structure-factor radial peaks using either
    the full BVM or one user-selected scan pixel
  - compare q-pixel radial calibration against the currently selected
    CIF zone-axis simulation instead of a hand-projected reciprocal lattice
  - diagnose the current CIF fit by plotting experimental peaks, simulated CIF
    peaks, nearest-neighbor match lines, residuals, and suggestions for q-pixel
    size, qy flip, center, or zone-axis issues
  - preview experimental peaks overlaid with the final fitted simulated Bragg
    pattern only
  - save the currently displayed result/fit plot as a PNG image
  - display correlation, fitted angle maps, or color orientation maps
  - display an out-of-plane orientation map with a visible triangular
    orientation color legend; if py4DSTEM does not return a wide legend panel,
    4DLAB uses its own labeled triangular legend fallback
  - set in-plane rotation symmetry so equivalent diffraction rotations share
    the same color
  - show an in-plane rotation color wheel that updates with the selected
    symmetry
  - use scrollable control panels so longer parameter sets remain accessible
  - use clean fit overlays with hidden axes, black plot borders, and square
    auto-fit qx/qy ranges
- `In-Plane` uses py4DSTEM fiber mode around the configured fiber axis.
- `In-Plane` can also run a no-CIF experimental-template mode:
  - build a template from the main Viewer's current real-space scan point
  - match relative in-plane rotation against detected peaks
  - use symmetry order, angular step, peak tolerance, and center-exclusion
    controls to tune the match
  - preview the current Viewer's point with experimental peaks overlaid against
    the rotated template
  - hide no-CIF parameters when `CIF indexing` is selected, and hide CIF-only
    parameters when `No-CIF template` is selected
- `Out-of-Plane` can search a user-defined zone-axis range instead of only
  py4DSTEM `auto`:
  - center zone axis plus maximum tilt cone
  - three zone-axis vertices defining the py4DSTEM `zone_axis_range`
  - optional copy of In-Plane CIF/calibration settings while keeping the
    Out-of-Plane search range independent
- `Strain` now provides notebook-style lattice-vector strain analysis from
  detected Bragg peaks:
  - use the main Viewer's current scan point to preview fit quality
  - guess reciprocal basis vectors from the current diffraction pattern
  - manually refine `g1` and `g2` in centered diffraction-pixel coordinates
  - set diffraction center, qy flip, center exclusion, h/k search range,
    matching tolerance, and minimum matched peaks
  - choose a reference scan region; strain is reported relative to the median
    fitted lattice transform in that region
  - run a full scan in a background worker with progress and stop support
  - use configurable CPU workers for full-scan lattice fitting
  - cache h/k lattice predictions once per map instead of rebuilding them at
    every scan point
  - display `e_xx`, `e_yy`, `e_xy`, `theta`, fit error, matched-peak count,
    valid mask, and polar components `e_rr`, `e_tt`, `e_rt`
  - preview observed peaks against reference/fitted lattice points with a
    square, axis-free fit plot
  - save the displayed strain map as PNG and save all strain arrays as `.npz`
- More vendor-specific importers should be added as isolated loaders under
  `src/fourdlab/io/`.
