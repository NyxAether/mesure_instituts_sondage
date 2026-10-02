"""Généré par rr-style/build.py depuis tokens.json — ne pas éditer.

Thème matplotlib rr/.

    import rr_mpl
    rr_mpl.use()             # ou rr_mpl.use("dark")
    ax.set_title("Titre")    # titre en Newsreader : rr_mpl.title(ax, "Titre")
    ax.yaxis.set_major_formatter(rr_mpl.fr_number(1))   # 1 234,5
"""
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.style  # noqa: F401  (mpl.style n'est pas chargé par défaut)
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

THEMES = {
  "light": {
    "colors": {
      "bg-primary": "#E4E7DD",
      "bg-secondary": "#D7DCCD",
      "bg-card": "#EEF0E8",
      "paper": "#FFFFFF",
      "text-primary": "#1A1F19",
      "text-secondary": "#444C42",
      "text-muted": "#596256",
      "line": "rgba(26, 31, 25, 0.18)",
      "line-strong": "rgba(26, 31, 25, 0.42)",
      "rule": "#BBBEB6",
      "accent": "#7A2E4F",
      "sage": "#3A6849",
      "dusk": "#3F6283",
      "ochre": "#835527",
      "warning": "#805400",
      "ascii-ink": "rgba(26, 31, 25, 0.22)",
      "overlay": "rgba(26, 31, 25, 0.45)"
    },
    "chart": {
      "surface": "#EEF0E8",
      "grid": "#D9DED1",
      "axis": "#A9B0A1",
      "mark-muted": "#BFC6B6",
      "mark-ink": "#6F736A",
      "series": [
        "#9E5468",
        "#4670A8",
        "#854F14",
        "#8368A8",
        "#1A653A"
      ],
      "sequential": {
        "prune": [
          "#C58292",
          "#B26B7D",
          "#9E5568",
          "#814152",
          "#652E3D"
        ],
        "dusk": [
          "#759BCD",
          "#5D85BB",
          "#4770A7",
          "#34598A",
          "#23436D"
        ]
      },
      "diverging": [
        "#34598A",
        "#5D85BB",
        "#BFC6B6",
        "#B26B7D",
        "#814152"
      ]
    }
  },
  "dark": {
    "colors": {
      "bg-primary": "#161A16",
      "bg-secondary": "#1C211C",
      "bg-card": "#242A24",
      "paper": "#161A16",
      "text-primary": "#E4EADF",
      "text-secondary": "#BAC4B5",
      "text-muted": "#8C9788",
      "line": "rgba(228, 234, 223, 0.14)",
      "line-strong": "rgba(228, 234, 223, 0.38)",
      "rule": "#3C443C",
      "accent": "#E39CBB",
      "sage": "#A6CDB0",
      "dusk": "#9FBEDA",
      "ochre": "#D9B488",
      "warning": "#E0B050",
      "ascii-ink": "rgba(228, 234, 223, 0.16)",
      "overlay": "rgba(7, 9, 7, 0.65)"
    },
    "chart": {
      "surface": "#242A24",
      "grid": "#313831",
      "axis": "#4A534A",
      "mark-muted": "#4F584F",
      "mark-ink": "#8F948B",
      "series": [
        "#BA5F78",
        "#5E97CD",
        "#B98749",
        "#9E83C5",
        "#3F936E"
      ],
      "sequential": {
        "prune": [
          "#8B3D54",
          "#AC546C",
          "#C57288",
          "#DE91A4",
          "#F6B1C1"
        ],
        "dusk": [
          "#2C5E8C",
          "#4179AC",
          "#6294C4",
          "#83B0DB",
          "#A4CCF3"
        ]
      },
      "diverging": [
        "#83B0DB",
        "#4179AC",
        "#4F584F",
        "#AC546C",
        "#DE91A4"
      ]
    }
  }
}
# Polices : $RR_DESIGN_FONTS, sinon rr-fonts/ à côté de ce fichier (rr-design pull), sinon ../fonts (dépôt de la charte).
_HERE = Path(__file__).resolve().parent
FONTS = next((d for d in (Path(os.environ.get("RR_DESIGN_FONTS", "")), _HERE / "rr-fonts", _HERE.parent / "fonts")
              if str(d) not in ("", ".") and d.is_dir()), None)


def use(mode: str = "light") -> None:
    if FONTS:
        for f in FONTS.glob("*.ttf"):
            font_manager.fontManager.addfont(str(f))
    mpl.style.use(str(Path(__file__).with_name("rr.mplstyle")))
    if mode == "dark":
        c, ch = THEMES["dark"]["colors"], THEMES["dark"]["chart"]
        mpl.rcParams.update({
            "figure.facecolor": c["bg-card"], "axes.facecolor": c["bg-card"], "savefig.facecolor": c["bg-card"],
            "axes.edgecolor": ch["axis"], "axes.labelcolor": c["text-secondary"], "axes.titlecolor": c["text-primary"],
            "grid.color": ch["grid"], "xtick.color": c["text-muted"], "ytick.color": c["text-muted"],
            "text.color": c["text-primary"], "patch.edgecolor": ch["surface"],
            "axes.prop_cycle": mpl.cycler(color=ch["series"]),
        })


def title(ax, text: str, size: float = 15) -> None:
    """Titre aligné à gauche en Newsreader (serif)."""
    ax.set_title(text, loc="left", fontfamily="serif", fontsize=size, pad=12)


def fr_number(digits: int = 0, suffix: str = "") -> FuncFormatter:
    """Format français : espace fine insécable des milliers, virgule décimale."""
    def fmt(x, _pos=None):
        s = f"{x:,.{digits}f}".replace(",", "\u202f").replace(".", ",")
        return s + suffix
    return FuncFormatter(fmt)


def series(mode: str = "light") -> list[str]:
    return THEMES[mode]["chart"]["series"]


def sequential(name: str = "prune", mode: str = "light") -> list[str]:
    """Rampe de 5 pas, de la plus faible à la plus forte valeur : "prune" ou "dusk" (crépuscule)."""
    return THEMES[mode]["chart"]["sequential"][name]


def diverging(mode: str = "light") -> list[str]:
    """5 pas : crépuscule (−), −, neutre, +, prune (+)."""
    return THEMES[mode]["chart"]["diverging"]


def cmap(name: str = "prune", mode: str = "light"):
    """Colormap discrète pour imshow/pcolormesh : une rampe séquentielle, ou "diverging"."""
    from matplotlib.colors import ListedColormap
    colors = diverging(mode) if name == "diverging" else sequential(name, mode)
    return ListedColormap(colors, name=f"rr-{name}-{mode}")


INK = {m: THEMES[m]["chart"]["mark-ink"] for m in THEMES}  # gris de contexte foncé (marque, pas texte)
AREA_ALPHA = 0.15  # opacité des aplats (zones, intervalles)
