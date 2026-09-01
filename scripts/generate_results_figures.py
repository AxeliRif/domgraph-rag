"""
Génère les figures de RÉSULTATS (pas les figures de méthode déjà produites par
generate_paper_figures.py) : les trois ablations de experiments.tex, le detail
de coût de l'appendice, l'écart de synthèse de réponse end-to-end, et les
statistiques du dataset -- à partir des tables déjà publiées dans le papier et,
quand ils existent, des JSON de résultats bruts (pour montrer une distribution
plutôt que la seule moyenne).

Toutes les figures utilisent la palette catégorielle validée du skill dataviz
(slots 1-3 : bleu / orange / aqua, l'ordre qui passe les seuils CVD all-pairs)
et sont écrites dans graphiques/ en PNG 300dpi.

Usage :
    python3 -m scripts.generate_results_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "graphiques"
DATA_DIR = ROOT / "data"

# --- Palette (references/palette.md du skill dataviz, mode clair) ----------
BLUE = "#2a78d6"    # slot 1
ORANGE = "#eb6834"  # slot 2
AQUA = "#1baf7a"    # slot 3
RED = "#e34948"     # slot 8 -- réservé aux annotations de contraste (écart)
GRAY_TEXT = "#52514e"
INK = "#0b0b0b"
SURFACE = "#fcfcfb"
GRID = "#e4e2dc"

# Rampe séquentielle bleue (une teinte, clair -> foncé) pour les figures
# ordinales (étapes d'un pipeline, gravité croissante d'un même phénomène).
SEQ_BLUE = ["#86b6ef", "#2a78d6", "#104281"]  # steps 250 / 450 / 650

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "xtick.color": GRAY_TEXT,
    "ytick.color": GRAY_TEXT,
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _style_axes(ax, ygrid=True):
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)


def _bar_labels(ax, bars, fmt="{:.2f}", dy=0.01, color=INK):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                 ha="center", va="bottom", fontsize=9.5, color=color)


# ---------------------------------------------------------------------------
# Ablation (a) -- DOM tiling vs. blind grid vs. flat text chunks
# ---------------------------------------------------------------------------

def fig_ablation_a_quality() -> None:
    conditions = ["DOM\n(ours)", "Grid", "Text"]
    colors = [BLUE, ORANGE, AQUA]
    ns = [211, 105, 231]
    attach = [51.6, 25.7, 56.5]
    r1 = [0.550, 0.762, 0.602]
    pctl = [0.869, 0.854, 0.864]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.4))

    ax = axes[0]
    bars = ax.bar(conditions, attach, color=colors, width=0.62, zorder=3)
    _bar_labels(ax, bars, fmt="{:.1f}%", dy=1.2)
    ax.set_title("Attach rate")
    ax.set_ylabel("% of 409 questions\nwith an answer-bearing unit")
    ax.set_ylim(0, 68)
    _style_axes(ax)

    ax = axes[1]
    bars = ax.bar(conditions, r1, color=colors, width=0.62, zorder=3)
    _bar_labels(ax, bars, dy=0.015)
    ax.set_title("Recall@1")
    ax.set_ylabel("R@1 of attached questions")
    ax.set_ylim(0, 0.98)
    for i, n in enumerate(ns):
        ax.annotate(f"n={n}", xy=(i, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -30), textcoords="offset points",
                    ha="center", va="top", fontsize=8.5, color=GRAY_TEXT)
    _style_axes(ax)

    ax = axes[2]
    bars = ax.bar(conditions, pctl, color=colors, width=0.62, zorder=3)
    _bar_labels(ax, bars, dy=0.012)
    ax.set_title("Percentile rank")
    ax.set_ylabel("pool-size-normalized rank\n(1 = best, 0.5 = random)")
    ax.set_ylim(0, 0.98)
    ax.axhline(0.5, color=GRAY_TEXT, linewidth=1, linestyle=(0, (3, 2)), zorder=2)
    ax.text(2.55, 0.5, "chance", fontsize=8, color=GRAY_TEXT, va="center", ha="left")
    _style_axes(ax)

    fig.suptitle(
        "Ablation (a) — DOM-guided tiling vs. blind grid vs. flat text chunks\n"
        "same TF-IDF scorer, 18 pages / 409 questions",
        fontsize=11.5, fontweight="bold", y=1.05,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT_DIR / "ablation_a_tiling_quality.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("écrit graphiques/ablation_a_tiling_quality.png")


def fig_ablation_a_pixel_budget() -> None:
    conditions = ["DOM tiling\n(ours)", "Blind grid"]
    colors = [BLUE, ORANGE]
    total_px = [3_829_395, 10_213_983]
    tiles_per_page = [24.0, 4.9]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4))

    ax = axes[0]
    bars = ax.bar(conditions, [v / 1e6 for v in total_px], color=colors, width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.1f}M px", dy=0.15)
    ax.set_title("Total pixel budget / page")
    ax.set_ylabel("million px / page")
    ax.set_ylim(0, 12.5)
    ax.annotate(
        "2.7× smaller", xy=(0, total_px[0] / 1e6), xytext=(0.5, 9.2),
        ha="center", fontsize=10, color=RED, fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": RED, "linewidth": 1.2,
                    "shrinkA": 2, "shrinkB": 8},
    )
    _style_axes(ax)

    ax = axes[1]
    bars = ax.bar(conditions, tiles_per_page, color=colors, width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.1f}", dy=0.4)
    ax.set_title("Tiles / page")
    ax.set_ylabel("mean tiles per page")
    ax.set_ylim(0, 29)
    _style_axes(ax)

    fig.suptitle(
        "Ablation (a) cost detail — same 18 pages (Appendix, Table 5)",
        fontsize=11.5, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_a_pixel_budget.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("écrit graphiques/ablation_a_pixel_budget.png")


def fig_ablation_a_distribution() -> None:
    """Distribution (pas seulement la moyenne) du percentile rank par
    condition, à partir des résultats bruts par page/question."""
    path = DATA_DIR / "dom_vs_grid_results.json"
    if not path.exists():
        print(f"  (sauté : {path} introuvable)")
        return
    pages = json.loads(path.read_text())

    by_cond: dict[str, list[float]] = {"dom": [], "grid": [], "text": []}
    for page in pages:
        for cond, items in page.get("results", {}).items():
            for item in items:
                by_cond[cond].append(item["percentile_rank"])

    order = ["dom", "grid", "text"]
    labels = ["DOM\n(ours)", "Grid", "Text"]
    colors = [BLUE, ORANGE, AQUA]
    data = [by_cond[c] for c in order]
    ns = [len(d) for d in data]

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    bp = ax.boxplot(
        data, positions=range(len(order)), widths=0.5, patch_artist=True,
        showfliers=False, medianprops={"color": INK, "linewidth": 1.6},
        whiskerprops={"color": GRAY_TEXT}, capprops={"color": GRAY_TEXT},
        zorder=3,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)

    rng = np.random.default_rng(0)
    for i, (d, color) in enumerate(zip(data, colors)):
        jitter = rng.uniform(-0.14, 0.14, size=len(d))
        ax.scatter(np.full(len(d), i) + jitter, d, s=10, color=color, alpha=0.35,
                   linewidths=0, zorder=2)

    ax.axhline(0.5, color=GRAY_TEXT, linewidth=1, linestyle=(0, (3, 2)), zorder=1)
    ax.text(len(order) - 0.42, 0.505, "chance", fontsize=8, color=GRAY_TEXT)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(labels, ns)])
    ax.set_ylabel("percentile rank of the\nanswer-bearing unit (1 = best)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        "Ablation (a) — per-question percentile rank distribution\n"
        "(same data as Table 1, not aggregated)",
        fontsize=11, fontweight="bold",
    )
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_a_percentile_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("écrit graphiques/ablation_a_percentile_distribution.png")


# ---------------------------------------------------------------------------
# Ablation (b) -- fine-tuned reader vs. base model vs. TF-IDF
# ---------------------------------------------------------------------------

def fig_ablation_b_reader() -> None:
    metrics = ["R@1", "R@3", "R@5", "MRR"]
    series = {
        "TF-IDF (lexical)": ([0.461, 0.618, 0.724, 0.579], BLUE),
        "Base VLM (no LoRA)": ([0.433, 0.645, 0.752, 0.578], ORANGE),
        "LoRA fine-tuned": ([0.785, 0.915, 0.952, 0.856], AQUA),
    }

    x = np.arange(len(metrics))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for i, (name, (vals, color)) in enumerate(series.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width=width * 0.92, color=color, label=name, zorder=3)
        _bar_labels(ax, bars, dy=0.015)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Ablation (b) — held-out tile retrieval, val split (330 questions)",
        fontsize=11.5, fontweight="bold",
    )
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_b_reader_retrieval.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("écrit graphiques/ablation_b_reader_retrieval.png")


# ---------------------------------------------------------------------------
# Ablation (c) -- evidence controller vs. static top-k, cross-page multi-hop
# ---------------------------------------------------------------------------

def fig_ablation_c_controller() -> None:
    seed_widths = ["$k_{seed}=3$\n(budget 5)", "$k_{seed}=5$\n(= budget)"]
    controller = [39.6, 40.7]
    static = [40.7, 40.7]
    # 95% Wilson intervals, k_seed=3 only (paper reports none for k_seed=5,
    # where controller and static coincide exactly).
    controller_err = [[39.6 - 30.1], [49.8 - 39.6]]
    static_err = [[40.7 - 31.1], [50.9 - 40.7]]

    x = np.arange(len(seed_widths))
    width = 0.32
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    b1 = ax.bar(x - width / 2, controller, width=width * 0.92, color=BLUE,
                label="Controller (cross-doc expansion)", zorder=3)
    ax.errorbar(x[0] - width / 2, controller[0], yerr=[[controller_err[0][0]], [controller_err[1][0]]],
                fmt="none", ecolor=INK, elinewidth=1.2, capsize=4, zorder=4)

    b2 = ax.bar(x + width / 2, static, width=width * 0.92, color=ORANGE,
                label="Static top-$k$ (same TF-IDF scores)", zorder=3)
    ax.errorbar(x[0] + width / 2, static[0], yerr=[[static_err[0][0]], [static_err[1][0]]],
                fmt="none", ecolor=INK, elinewidth=1.2, capsize=4, zorder=4)

    _bar_labels(ax, b1, fmt="{:.1f}%", dy=0.6)
    _bar_labels(ax, b2, fmt="{:.1f}%", dy=0.6)

    ax.axhline(29.7, color=AQUA, linewidth=1.6, linestyle=(0, (4, 2)), zorder=2)
    ax.text(1.42, 29.7, "reading_order only\n(before cross-doc expansion): 29.7%",
            fontsize=8.3, color=AQUA, va="center", ha="left")

    ax.set_xticks(x)
    ax.set_xticklabels(seed_widths)
    ax.set_ylabel("% of 91 cross-page questions with\nboth positive tiles covered")
    ax.set_ylim(0, 62)
    ax.set_xlim(-0.6, 2.05)
    ax.set_title(
        "Ablation (c) — evidence coverage, cross-page multi-hop (n=91)\n"
        "error bars = 95% Wilson interval",
        fontsize=11, fontweight="bold",
    )
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_c_controller_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("écrit graphiques/ablation_c_controller_coverage.png")


# ---------------------------------------------------------------------------
# End-to-end multi-hop: evidence coverage vs. final answer EM/F1
# ---------------------------------------------------------------------------

def fig_e2e_gap() -> None:
    path = DATA_DIR / "qa_dataset" / "multihop_e2e_eval.json"
    per_question_f1 = None
    if path.exists():
        d = json.loads(path.read_text())
        per_question_f1 = [r["f1"] for r in d.get("results", [])]

    stages = ["Evidence coverage\n(ablation c, n=91)", "End-to-end EM\n(n=20 sample)", "End-to-end F1\n(n=20 sample)"]
    values = [39.6, 5.0, 5.7]
    colors = SEQ_BLUE

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    bars = ax.bar(stages, values, color=colors, width=0.6, zorder=3)
    _bar_labels(ax, bars, fmt="{:.1f}%", dy=1.0)
    ax.set_ylabel("%")
    ax.set_ylim(0, 48)
    ax.set_title("Evidence reaches the tile,\nanswer synthesis mostly fails", fontsize=10.5)
    _style_axes(ax)
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(9)

    ax = axes[1]
    if per_question_f1 is not None:
        bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
        counts, edges = np.histogram(per_question_f1, bins=bins)
        centers = [f"[{edges[i]:.1f},{edges[i+1]:.1f})" for i in range(len(edges) - 1)]
        bars = ax.bar(centers, counts, color=SEQ_BLUE[1], width=0.65, zorder=3)
        _bar_labels(ax, bars, fmt="{:.0f}", dy=0.15)
        ax.set_xlabel("per-question F1 bucket")
        ax.set_ylabel("# questions (of 20)")
        ax.set_title("Per-question F1, n=20 sample\n(seed=0)", fontsize=10.5)
        ax.set_ylim(0, max(counts) + 2)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8.5)
        _style_axes(ax)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "per-question F1\nnot available", ha="center", va="center")

    fig.suptitle(
        "End-to-end multi-hop gap — evidence coverage vs. final answer correctness",
        fontsize=11.5, fontweight="bold", y=1.04,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e2e_answer_synthesis_gap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("écrit graphiques/e2e_answer_synthesis_gap.png")


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def fig_dataset_stats() -> None:
    labels = [
        "Distinct Wikipedia\narticles",
        "Page units\n(articles + sections)",
        "Tiles extracted",
        "QA pairs generated",
        "Contrastive examples\n(QA + hard negatives)",
    ]
    values = [19, 185, 1784, 1717, 1339]
    colors = [SEQ_BLUE[0], SEQ_BLUE[1], SEQ_BLUE[2], SEQ_BLUE[2], SEQ_BLUE[1]]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, height=0.6, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(10, 3000)
    for b, v in zip(bars, values):
        ax.text(b.get_width() * 1.05, b.get_y() + b.get_height() / 2, f"{v:,}",
                va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("count (log scale)")
    ax.set_title(
        "Contrastive dataset size — 19 articles, 12 split into sections",
        fontsize=11.5, fontweight="bold",
    )
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, which="both", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dataset_pipeline_stats.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("écrit graphiques/dataset_pipeline_stats.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_ablation_a_quality()
    fig_ablation_a_pixel_budget()
    fig_ablation_a_distribution()
    fig_ablation_b_reader()
    fig_ablation_c_controller()
    fig_e2e_gap()
    fig_dataset_stats()
    print("\nTerminé.")


if __name__ == "__main__":
    main()
