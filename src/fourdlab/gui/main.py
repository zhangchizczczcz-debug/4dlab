"""Command-line entry point for the 4DLAB viewer."""

from __future__ import annotations

import sys

from fourdlab.gui.viewer import ViewerLaunchConfig, launch_viewer


def main() -> int:
    """Run the 4DLAB import and visualization main window."""

    return launch_viewer(ViewerLaunchConfig(argv=sys.argv))


if __name__ == "__main__":
    raise SystemExit(main())

