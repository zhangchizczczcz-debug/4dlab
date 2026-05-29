"""Extension hooks for the 4DLAB viewer shell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ViewerExtension(Protocol):
    """Small interface for future processing modules that attach to the viewer."""

    name: str

    def install(self, viewer: object) -> None:
        """Attach menus, actions, docks, or callbacks to an existing viewer."""


@dataclass(frozen=True)
class ExtensionInstallResult:
    """Result of installing one viewer extension."""

    name: str
    ok: bool
    message: str = ""


def install_extensions(
    viewer: object, extensions: list[ViewerExtension] | None
) -> list[ExtensionInstallResult]:
    """Install optional viewer extensions without coupling them to the GUI backend."""

    results: list[ExtensionInstallResult] = []
    for extension in extensions or []:
        name = getattr(extension, "name", extension.__class__.__name__)
        try:
            extension.install(viewer)
        except Exception as exc:  # pragma: no cover - GUI safety net
            results.append(ExtensionInstallResult(name=name, ok=False, message=str(exc)))
        else:
            results.append(ExtensionInstallResult(name=name, ok=True))
    return results

