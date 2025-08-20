"""Visualization style configuration for consistent plots."""

import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any

# WCAG 2.1 AA compliant color palette
VALID_COLORS: Dict[str, str] = {
    "blue": "#1f77b4",  # Dark blue
    "orange": "#ff7f0e",  # Dark orange
    "green": "#2ca02c",  # Dark green
    "red": "#d62728",  # Dark red
    "purple": "#9467bd",  # Dark purple
    "brown": "#8c564b",  # Dark brown
    "pink": "#e377c2",  # Dark pink
    "yellow": "#bcbd22",  # Dark yellow
    "black": "#000000",  # Black
    "white": "#ffffff",  # White
}

#: dict: Publication-ready matplotlib style configuration
# Contains parameters for:
# - Font sizes and families
# - Line widths and marker sizes
# - Figure and axes styling
PUBLICATION_STYLE: Dict[str, Any] = {
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def set_publication_style() -> None:
    """Apply publication-ready style to matplotlib plots."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(PUBLICATION_STYLE)


def check_color_contrast(color1: str, color2: str, ratio: float = 4.5) -> bool:
    """Check if two colors meet contrast requirements.

    Args:
        color1: First color in hex format
        color2: Second color in hex format
        ratio: Minimum contrast ratio (default 4.5)

    Returns:
        bool: True if contrast meets ratio
    """

    def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
        """Convert hex color to RGB values in 0-1 range."""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) != 6:
            raise ValueError(f"Invalid hex color: {hex_color}")
        r: float = int(hex_color[0:2], 16) / 255
        g: float = int(hex_color[2:4], 16) / 255
        b: float = int(hex_color[4:6], 16) / 255
        return (r, g, b)

    def luminance(r: float, g: float, b: float) -> float:
        """Calculate relative luminance per WCAG 2.1 specification."""
        rs: float = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        gs: float = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        bs: float = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
        return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs

    rgb1: Tuple[float, float, float] = hex_to_rgb(color1)
    rgb2: Tuple[float, float, float] = hex_to_rgb(color2)

    l1: float = luminance(*rgb1)
    l2: float = luminance(*rgb2)

    contrast: float = (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)
    return contrast >= ratio
