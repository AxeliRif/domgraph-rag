"""
Visualisation du graphe (Phase 1d) — deux vues complémentaires :

  - `plot_page_overlay`  : les tuiles dessinées par-dessus la capture d'écran
    réelle de la page, avec leur ordre de lecture. C'est la vue la plus
    directement lisible : elle montre "ce que le pipeline a vu".
  - `plot_graph`         : le graphe page/éléments en tant que tel (noeuds,
    arêtes `contains` et `reading_order`), mais positionné selon les
    coordonnées spatiales réelles des tuiles plutôt qu'une disposition
    arbitraire — pour que la forme du graphe reste lisible par rapport à la
    page d'origine.

Les éléments sont regroupés par catégorie de contenu (titre / texte / tableau
/ liste / média / citation-code / autre) plutôt que par balise brute : avec
seulement 7-8 couleurs lisibles disponibles, mieux vaut coder la catégorie
(peu de valeurs, bien distinguables) et réserver la balise exacte (h1 vs h2,
par ex.) à l'étiquette textuelle de chaque noeud.
"""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

from .graph_builder import compute_reading_order
from .tiling import Tile

# Palette catégorielle (ordre fixe, jamais permuté) — cf. skill dataviz.
_CATEGORY_COLORS: dict[str, str] = {
    "heading": "#2a78d6",     # bleu
    "text": "#1baf7a",        # aqua
    "table": "#eda100",       # jaune
    "list": "#008300",        # vert
    "media": "#4a3aa7",       # violet
    "quote_code": "#e34948",  # rouge
    "other": "#e87ba4",       # magenta
}
_CATEGORY_LABELS: dict[str, str] = {
    "heading": "Titre (h1-h4)",
    "text": "Texte (p / fusionné)",
    "table": "Tableau",
    "list": "Liste",
    "media": "Média (img / figure)",
    "quote_code": "Citation / code",
    "other": "Autre",
}
_ROOT_COLOR = "#0b0b0b"  # noeud "page" : encre primaire, pas une couleur de série
_CONTAINS_COLOR = "#c3c2b7"  # arêtes "contains" : recessives (cf. baseline/axis)
_ORDER_COLOR = "#0b0b0b"  # arêtes "reading_order" : encre primaire, ce sont le sujet


def _tag_category(tag: str) -> str:
    if tag in {"h1", "h2", "h3", "h4"}:
        return "heading"
    if tag in {"p", "merged"}:
        return "text"
    if tag == "table":
        return "table"
    if tag in {"ul", "ol"}:
        return "list"
    if tag in {"img", "figure"}:
        return "media"
    if tag in {"blockquote", "pre"}:
        return "quote_code"
    return "other"


def _draw_category_legend(ax: plt.Axes, categories: set[str]) -> None:
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                    markerfacecolor=_CATEGORY_COLORS[cat], markeredgecolor="none",
                    label=_CATEGORY_LABELS[cat])
        for cat in _CATEGORY_COLORS
        if cat in categories
    ]
    if handles:
        ax.legend(
            handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
            frameon=False, fontsize=9, title="Catégorie", title_fontsize=9,
            labelcolor="#52514e",
        )


def plot_page_overlay(tiles: list[Tile], screenshot_bytes: bytes, ax: plt.Axes | None = None) -> plt.Axes:
    """Superpose les tuiles (couleur = catégorie, numéro = ordre de lecture) sur
    la capture d'écran réelle — la vue la plus directe pour vérifier le tuilage."""
    page_image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")

    if ax is None:
        # Une page longue (Wikipedia...) donnerait sinon une figure de centaines
        # de pouces de haut : on borne la hauteur, matplotlib sous-échantillonne
        # l'image pour tenir dans la figure (les rectangles/numéros restent nets,
        # eux, puisqu'ils sont définis en points, pas en pixels image).
        height = min(page_image.height / page_image.width * 9 + 1, 28)
        _, ax = plt.subplots(figsize=(9, height))

    ax.imshow(page_image)
    categories_seen: set[str] = set()

    for order, tile in enumerate(compute_reading_order(tiles), start=1):
        category = _tag_category(tile.dom_tag)
        categories_seen.add(category)
        color = _CATEGORY_COLORS[category]

        ax.add_patch(Rectangle(
            (tile.x, tile.y), tile.width, tile.height,
            linewidth=2, edgecolor=color, facecolor="none",
        ))
        ax.text(
            tile.x + 3, tile.y + 3, str(order),
            fontsize=8, fontweight="bold", color="white", va="top", ha="left",
            bbox={"boxstyle": "circle,pad=0.25", "facecolor": color, "edgecolor": "none"},
        )

    ax.set_title("Tuiles et ordre de lecture, superposés à la page")
    ax.axis("off")
    _draw_category_legend(ax, categories_seen)
    return ax


def plot_graph(graph: nx.MultiDiGraph, ax: plt.Axes | None = None) -> plt.Axes:
    """Dessine le graphe page/éléments avec les noeuds placés à leur position
    spatiale réelle sur la page (issue des coordonnées des tuiles), plutôt
    qu'une disposition arbitraire — la topologie reste lisible par rapport à
    la mise en page d'origine.

    Arêtes `contains` : recessives (structure), `reading_order` : encre
    primaire + numérotées (c'est l'information que ce graphe met en avant).
    """
    # Élément dans l'ordre de lecture : `build_graph` insère les noeuds dans cet
    # ordre, et networkx conserve l'ordre d'insertion — pas besoin de le recalculer.
    element_nodes = [n for n in graph.nodes if graph.nodes[n].get("type") == "element"]

    if ax is None:
        height = max(6.0, 0.42 * len(element_nodes) + 1.5)
        _, ax = plt.subplots(figsize=(7, height))

    # Layout hybride : x = position horizontale réelle sur la page (révèle les
    # éléments décalés/indentés), y = rang dans l'ordre de lecture, espacé
    # uniformément (évite l'écrasement vertical d'une page longue avec une
    # disposition 1:1 en pixels réels).
    xs = [data["x"] + data["width"] / 2 for _, data in graph.nodes(data=True) if data.get("type") == "element"]
    x_min, x_max = (min(xs), max(xs)) if xs else (0.0, 1.0)
    x_span = max(x_max - x_min, 1.0)

    pos: dict[str, tuple[float, float]] = {}
    for rank, n in enumerate(element_nodes):
        data = graph.nodes[n]
        cx = data["x"] + data["width"] / 2
        x_norm = (cx - x_min) / x_span * 6.0
        pos[n] = (x_norm, -rank)

    if pos:
        mean_x = sum(x for x, _ in pos.values()) / len(pos)
        pos["page"] = (mean_x, 1.2)
    else:
        pos["page"] = (0, 0)

    contains_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["relation"] == "contains"]
    order_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["relation"] == "reading_order"]

    nx.draw_networkx_edges(
        graph, pos, ax=ax, edgelist=contains_edges, edge_color=_CONTAINS_COLOR,
        width=1, alpha=0.6, arrows=False,
    )
    for order, (u, v) in enumerate(order_edges, start=1):
        arrow = FancyArrowPatch(
            pos[u], pos[v], connectionstyle="arc3,rad=0.05",
            arrowstyle="-|>", mutation_scale=12, linewidth=1.5,
            color=_ORDER_COLOR, zorder=2,
        )
        ax.add_patch(arrow)

    categories_seen: set[str] = set()
    for n in element_nodes:
        category = _tag_category(graph.nodes[n]["tag"])
        categories_seen.add(category)
        x, y = pos[n]
        ax.scatter([x], [y], s=140, color=_CATEGORY_COLORS[category], edgecolors="white", linewidths=1.2, zorder=3)
        ax.annotate(
            graph.nodes[n]["tag"], (x, y), xytext=(12, 0), textcoords="offset points",
            ha="left", va="center", fontsize=8, color="#52514e", zorder=4,
        )

    px, py = pos["page"]
    ax.scatter([px], [py], s=200, marker="D", color=_ROOT_COLOR, edgecolors="white", linewidths=1.2, zorder=3)
    ax.annotate(
        "page", (px, py), xytext=(11, 0), textcoords="offset points",
        ha="left", va="center", fontsize=8, color="#52514e", zorder=4,
    )

    ax.set_title("Graphe page/éléments — x = position horizontale réelle, y = ordre de lecture")
    ax.axis("off")
    ax.margins(x=0.35, y=0.03)
    _draw_category_legend(ax, categories_seen)
    return ax