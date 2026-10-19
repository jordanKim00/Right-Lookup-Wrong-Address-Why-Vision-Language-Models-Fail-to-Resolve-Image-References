"""Paper figure style (knowledge/FIGURE_CONVENTIONS.md): ROME/filter-heads heatmap conventions, cross-modal-flow line-plot
conventions, validated colour roles. Import `apply()` before drawing."""
import matplotlib as mpl

TARGET, SURFACE, ACCENT, INK, MUTED, RULE = "#008f7e", "#c0571a", "#3a52c4", "#18201c", "#7d8780", "#d8ddd8"
GREYS = ["#9aa39d", "#b4bbb6", "#7d8780", "#c9cec9"]
ONE, TWO = 3.25, 6.75                                                         # column widths (in)


def apply(korean=False):
    rc = {
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.linewidth": .6, "xtick.major.width": .5, "ytick.major.width": .5,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": .02, "pdf.fonttype": 42,
    }
    if korean:
        from matplotlib import font_manager as fm
        for f in ("/hostfonts/opentype/noto/NotoSansCJK-Regular.ttc", "/hostfonts/opentype/noto/NotoSerifCJK-Regular.ttc"):
            try:
                fm.fontManager.addfont(f)
            except Exception:
                pass
        rc["font.family"] = "sans-serif"; rc["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK KR", "DejaVu Sans"]
    mpl.rcParams.update(rc)


def save(fig, path):
    fig.savefig(path + ".pdf"); fig.savefig(path + ".png", dpi=300)
