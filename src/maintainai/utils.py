"""Small IO and plotting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def ensure_dir(path: Path) -> Path:
    """Create *path* and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Write stable, human-readable JSON."""

    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def save_figure(path: Path) -> None:
    """Save and close the active Matplotlib figure."""

    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def soft_grid(ax, axis: str = "both") -> None:
    """Add restrained separators without competing with the data marks."""

    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#D8DEE6", linewidth=0.65, alpha=0.55)
