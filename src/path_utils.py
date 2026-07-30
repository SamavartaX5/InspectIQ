"""Portable, deliberately strict resolution of report-relative artifact paths."""
from __future__ import annotations

from pathlib import Path, PureWindowsPath


class RelativePathError(ValueError):
    """A report path is absolute or escapes the declared artifact base."""


def resolve_report_path(value: str | Path, base_directory: Path) -> Path:
    """Resolve a slash-agnostic relative report path under ``base_directory``.

    Reports are portable when they store relative artifact paths.  Absolute
    paths are rejected (including Windows drive paths on Unix), as are parent
    traversals that would escape the supplied base directory.
    """
    text = str(value).strip().replace("\\", "/")
    if not text:
        raise RelativePathError("Artifact path is empty.")
    normalized = Path(text)
    if normalized.is_absolute() or PureWindowsPath(text).is_absolute():
        raise RelativePathError("Absolute artifact paths are not permitted in reports.")
    base = base_directory.resolve()
    candidate = (base / normalized).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise RelativePathError("Artifact path escapes its report base directory.") from error
    return candidate
