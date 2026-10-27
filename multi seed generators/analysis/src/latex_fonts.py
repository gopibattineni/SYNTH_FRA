"""Shared Times New Roman / Times-compatible serif font

Prefers Times New Roman when installed; otherwise Liberation Serif
(metric-compatible Times substitute used for LaTeX documents on Linux),
then TeX Gyre Termes / Nimbus Roman / FreeSerif.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager, pyplot as plt
from matplotlib.font_manager import FontProperties

_CANDIDATES = (
    "Times New Roman",
    "TimesNewRoman",
    "Tinos",
    "TeXGyreTermes",
    "TeX Gyre Termes",
    "Liberation Serif",
    "Nimbus Roman",
    "Nimbus Roman No9 L",
    "FreeSerif",
    "DejaVu Serif",
)

# Prefer known TTF paths so the face is embedded even if family lookup is flaky.
_FILE_CANDIDATES = (
    # Times New Roman (msttcorefonts)
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
    # User-local Times New Roman
    str(Path.home() / ".local/share/fonts/Times_New_Roman.ttf"),
    str(Path.home() / ".local/share/fonts/times.ttf"),
    # Tinos — free Times New Roman metric-compatible face
    str(Path.home() / ".local/share/fonts/tinos/Tinos-Regular.ttf"),
    "/usr/share/fonts/truetype/tinos/Tinos-Regular.ttf",
    # Liberation Serif (Times-metric compatible)
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    # TeX Gyre Termes (LaTeX Times clone)
    "/usr/share/fonts/opentype/texgyre/texgyretermes-regular.otf",
    "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf",
)

_ITALIC_FILES = {
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf":
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/times.ttf":
        "/usr/share/fonts/truetype/msttcorefonts/timesi.ttf",
    str(Path.home() / ".local/share/fonts/tinos/Tinos-Regular.ttf"):
        str(Path.home() / ".local/share/fonts/tinos/Tinos-Italic.ttf"),
    "/usr/share/fonts/truetype/tinos/Tinos-Regular.ttf":
        "/usr/share/fonts/truetype/tinos/Tinos-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf":
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf":
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Italic.ttf",
}

_BOLD_FILES = {
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf":
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/times.ttf":
        "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
    str(Path.home() / ".local/share/fonts/tinos/Tinos-Regular.ttf"):
        str(Path.home() / ".local/share/fonts/tinos/Tinos-Bold.ttf"),
    "/usr/share/fonts/truetype/tinos/Tinos-Regular.ttf":
        "/usr/share/fonts/truetype/tinos/Tinos-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf":
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf":
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
}


def _resolve_font_file() -> str | None:
    for path in _FILE_CANDIDATES:
        if Path(path).is_file():
            return path
    return None


def configure_times_font() -> str:
    """Set global matplotlib rcParams to a Times-like serif face. Returns chosen name."""
    font_file = _resolve_font_file()
    if font_file is not None:
        font_manager.fontManager.addfont(font_file)
        italic = _ITALIC_FILES.get(font_file)
        if italic and Path(italic).is_file():
            font_manager.fontManager.addfont(italic)
        chosen = FontProperties(fname=font_file).get_name()
    else:
        available = {f.name for f in font_manager.fontManager.ttflist}
        chosen = next((c for c in _CANDIDATES if c in available), "DejaVu Serif")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [chosen, "Times New Roman", "Tinos", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return chosen


def times_fontproperties(*, size: float | None = None, style: str = "normal", weight: str = "normal") -> FontProperties:
    """FontProperties bound to the Times-compatible face (by file when possible)."""
    font_file = _resolve_font_file()
    if font_file is not None:
        use_file = font_file
        if style == "italic":
            italic = _ITALIC_FILES.get(font_file)
            if italic and Path(italic).is_file():
                use_file = italic
        elif str(weight).lower() in {"bold", "heavy"}:
            bold = _BOLD_FILES.get(font_file)
            if bold and Path(bold).is_file():
                use_file = bold
                # register bold face for embedding
                font_manager.fontManager.addfont(bold)
        kwargs: dict = {"fname": use_file, "style": style, "weight": weight}
        if size is not None:
            kwargs["size"] = size
        return FontProperties(**kwargs)

    name = configure_times_font()
    kwargs = {"family": name, "style": style, "weight": weight}
    if size is not None:
        kwargs["size"] = size
    return FontProperties(**kwargs)


def apply_font_to_figure(fig: plt.Figure, font_name: str | None = None) -> None:
    """Force every text artist on a figure onto the chosen serif face."""
    if font_name is None:
        font_name = configure_times_font()

    def _apply(text) -> None:
        if text is None:
            return
        style = (text.get_fontstyle() or "normal").lower()
        weight = str(text.get_fontweight() or "normal").lower()
        size = text.get_fontsize()
        fp = times_fontproperties(size=size, style=style, weight=weight)
        text.set_fontproperties(fp)

    for ax in fig.axes:
        for text in ax.texts:
            _apply(text)
        if ax.title is not None:
            _apply(ax.title)
        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            _apply(label)
        _apply(ax.xaxis.label)
        _apply(ax.yaxis.label)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                _apply(text)
            if legend.get_title() is not None:
                _apply(legend.get_title())
    for text in fig.texts:
        _apply(text)
