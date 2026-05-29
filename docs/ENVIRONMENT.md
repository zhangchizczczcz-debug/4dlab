# Environment Setup

The project conda environment is named `4dlab`.

## Create

Run this from the project root.

```powershell
conda env create -f environment.yml
```

## Activate

```powershell
conda activate 4dlab
```

## One-Click Start

Double-click `start_4dlab.bat` in the project root to activate `4dlab` and open
the viewer.

## Update After Adding Dependencies

When new packages are needed, install them into `4dlab` and update
`environment.yml` so the project remains portable.

```powershell
conda activate 4dlab
conda install -c conda-forge PACKAGE_NAME
```

If a package is installed with `pip`, record it under the `pip:` section in
`environment.yml`.

The local `fourdlab` package is installed in editable mode by `environment.yml`
using `pip install -e .`.
