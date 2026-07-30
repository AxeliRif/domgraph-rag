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
    "heading": "Heading (h1-h4)",
    "text": "Text (p / merged)",
    "table": "Table",
    "list": "List",
    "media": "Media (img / figure)",
    "quote_code": "Quote / code",
    "other": "Other",
}
_ROOT_COLOR = "#0b0b0b"  # noeud "page" : encre primaire, pas une couleur de série
_CONTAINS_COLOR = "#c3c2b7"  # arêtes "contains" : recessives (cf. baseline/axis)
_ORDER_COLOR = "#0b0b0b"  # arêtes "reading_order" : encre primaire, ce sont le sujet
_ADJACENCY_COLOR = "#2a78d6"  # arêtes "layout_adjacency" : bleu, pointillé (voisinage 2D, secondaire à l'ordre)
_HIERARCHY_COLOR = "#e34948"  # arêtes "section_hierarchy" : rouge, distinct de "contains" (arbre de sections, pas de tuiles)


def _tag_category(tag: str) -> str:
    # Tuiles composées ("h2+p", cf. tiling.py, appariement titre+contenu) :
    # catégoriser par le contenu qui suit le titre, pas par le titre lui-même,
    # sans quoi la quasi-totalité des tuiles d'une page bien structurée
    # finirait dans "other" (une tuile de titre seul, sans appariement, garde
    # son tag brut et reste donc catégorisée "heading" normalement).
    if "+" in tag:
        tag = tag.split("+")[-1]
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


def _draw_category_legend(ax: plt.Axes, categories: set[str]):
    """Retourne la Legend créée (ou None) : quand un second `ax.legend(...)` est
    appelé ensuite (cf. `_draw_relation_legend`), matplotlib remplace le
    premier au lieu de l'empiler -- l'appelant doit donc le réattacher via
    `ax.add_artist(...)` s'il veut garder les deux."""
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                    markerfacecolor=_CATEGORY_COLORS[cat], markeredgecolor="none",
                    label=_CATEGORY_LABELS[cat])
        for cat in _CATEGORY_COLORS
        if cat in categories
    ]
    if not handles:
        return None
    return ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
        frameon=False, fontsize=9, title="Category", title_fontsize=9,
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

    ax.set_title("Tiles and reading order, overlaid on the page")
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
    adjacency_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["relation"] == "layout_adjacency"]
    hierarchy_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["relation"] == "section_hierarchy"]

    nx.draw_networkx_edges(
        graph, pos, ax=ax, edgelist=contains_edges, edge_color=_CONTAINS_COLOR,
        width=1, alpha=0.6, arrows=False,
    )
    # layout_adjacency et section_hierarchy dessinées avant reading_order pour
    # que l'ordre de lecture (le sujet de cette vue) reste au premier plan.
    for u, v in adjacency_edges:
        if u not in pos or v not in pos:
            continue
        arrow = FancyArrowPatch(
            pos[u], pos[v], connectionstyle="arc3,rad=0.15",
            arrowstyle="-", linewidth=0.8, linestyle=(0, (2, 2)),
            color=_ADJACENCY_COLOR, alpha=0.6, zorder=1,
        )
        ax.add_patch(arrow)
    for u, v in hierarchy_edges:
        if u not in pos or v not in pos:
            continue
        arrow = FancyArrowPatch(
            pos[u], pos[v], connectionstyle="arc3,rad=-0.15",
            arrowstyle="-|>", mutation_scale=10, linewidth=1.0, linestyle=(0, (5, 2)),
            color=_HIERARCHY_COLOR, alpha=0.8, zorder=1,
        )
        ax.add_patch(arrow)
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

    ax.set_title("Page/element graph -- x = real horizontal position, y = reading order")
    ax.axis("off")
    ax.margins(x=0.35, y=0.03)
    category_legend = _draw_category_legend(ax, categories_seen)
    _draw_relation_legend(ax, has_adjacency=bool(adjacency_edges), has_hierarchy=bool(hierarchy_edges))
    if category_legend is not None:
        ax.add_artist(category_legend)  # le second ax.legend() ci-dessus l'aurait sinon remplacé
    return ax


def plot_page_overview(
    graph: nx.MultiDiGraph, ax: plt.Axes | None = None,
    max_external_per_page: int = 2, max_pages: int = 6,
) -> plt.Axes:
    """Vue "page de pages" d'un graphe qui contient plusieurs noeuds "page" --
    une page démesurée découpée en sections (relation "continues", cf.
    sectioning.py) et/ou un corpus multi-pages (relation "links_to", cf.
    corpus_builder.py) -- sans le détail des tuiles, qui noierait justement
    la structure qu'on veut montrer ici : comment les pages/sections
    s'enchaînent, pas ce que contient chacune.

    Les arêtes "links_to" sont portées par des noeuds élément dans le graphe
    complet ; cette vue les fait remonter au niveau page (quel noeud "page"
    contient l'élément source) pour rester à la même granularité que
    "continues". `max_pages` borne le nombre de sections effectivement
    dessinées (une chaîne "continues" réelle peut compter des dizaines de
    sections, illisible en une figure) ; le reste est résumé par une seule
    annotation "+N more sections" plutôt que dessiné entassé."""
    page_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "page"]

    continues_next = {u: v for u, v, d in graph.edges(data=True) if d.get("relation") == "continues"}
    continues_prev = {v: u for u, v in continues_next.items()}
    starts = [n for n in page_nodes if n not in continues_prev]
    full_order: list[str] = []
    for start in starts:
        cur = start
        while cur is not None:
            full_order.append(cur)
            cur = continues_next.get(cur)
    full_order += [n for n in page_nodes if n not in full_order]  # pages isolées (pas de chaîne continues)

    ordered = full_order[:max_pages]
    n_hidden = len(full_order) - len(ordered)

    if ax is None:
        _, ax = plt.subplots(figsize=(1.7 * max(len(ordered), 1) + 1.5, 3.6))

    pos = {n: (1.3 * i, 0.0) for i, n in enumerate(ordered)}

    page_of_element: dict[str, str] = {}
    for page in ordered:
        for _, el, d in graph.out_edges(page, data=True):
            if d.get("relation") == "contains":
                page_of_element[el] = page

    external_by_page: dict[str, list[str]] = {p: [] for p in ordered}
    for u, v, d in graph.edges(data=True):
        if d.get("relation") == "links_to" and u in page_of_element:
            targets = external_by_page[page_of_element[u]]
            if v not in targets:
                targets.append(v)

    for u, v in continues_next.items():
        if u not in pos or v not in pos:
            continue
        ax.add_patch(FancyArrowPatch(
            pos[u], pos[v], arrowstyle="-|>", mutation_scale=14, linewidth=2,
            color=_ORDER_COLOR, zorder=2,
        ))

    for page, targets in external_by_page.items():
        px, py = pos[page]
        shown = targets[:max_external_per_page]
        for j, ext in enumerate(shown):
            ex = px + (j - (len(shown) - 1) / 2) * 0.55
            ey = py - 1.3
            ax.add_patch(FancyArrowPatch(
                (px, py), (ex, ey), arrowstyle="-|>", mutation_scale=10, linewidth=1,
                color=_CATEGORY_COLORS["other"], alpha=0.8, zorder=1,
                connectionstyle="arc3,rad=0.1",
            ))
            title = str(graph.nodes[ext].get("url", ext)).rsplit("/", 1)[-1].replace("_", " ")
            ax.scatter([ex], [ey], s=90, marker="o", color=_CATEGORY_COLORS["other"], edgecolors="white", linewidths=1, zorder=3)
            ax.annotate(title[:16], (ex, ey), xytext=(0, -4), textcoords="offset points",
                        ha="center", va="top", fontsize=6.5, color="#52514e", zorder=4, rotation=-30)
        if len(targets) > max_external_per_page:
            ax.annotate(f"+{len(targets) - max_external_per_page}", (px, py - 1.3),
                        xytext=(0, -22), textcoords="offset points", ha="center", fontsize=7, color="#8a8a86")

    for n in ordered:
        x, y = pos[n]
        ax.scatter([x], [y], s=260, marker="D", color=_ROOT_COLOR, edgecolors="white", linewidths=1.4, zorder=3)
        label = str(graph.nodes[n].get("title") or n)
        if label.startswith("http://") or label.startswith("https://"):
            label = label.split("#", 1)[0].rstrip("/").rsplit("/", 1)[-1]  # préambule sans titre : nom de page, pas l'URL brute
        label = label.replace("_", " ")
        ax.annotate(label[:20], (x, y), xytext=(-8, 12), textcoords="offset points",
                    ha="left", va="bottom", fontsize=7.5, fontweight="bold", color="#0b0b0b",
                    rotation=25, zorder=4)

    if n_hidden > 0:
        last_x = pos[ordered[-1]][0] if ordered else 0.0
        ax.annotate(f"+{n_hidden} more sections\n(continues)", (last_x + 1.3, 0.0),
                    ha="left", va="center", fontsize=8, color="#8a8a86", style="italic")

    handles = [
        plt.Line2D([0], [0], color=_ORDER_COLOR, linewidth=2, label="continues"),
        plt.Line2D([0], [0], color=_CATEGORY_COLORS["other"], linewidth=1, label="links_to"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9,
              title="Relation", title_fontsize=9, labelcolor="#52514e")

    ax.set_title("Page-of-pages view: sections (continues) and external links (links_to)")
    ax.axis("off")
    ax.margins(x=0.15, y=0.6)
    return ax


def _draw_relation_legend(ax: plt.Axes, has_adjacency: bool, has_hierarchy: bool) -> None:
    handles = [
        plt.Line2D([0], [0], color=_ORDER_COLOR, linewidth=1.5, label="reading_order"),
        plt.Line2D([0], [0], color=_CONTAINS_COLOR, linewidth=1, label="contains"),
    ]
    if has_adjacency:
        handles.append(plt.Line2D([0], [0], color=_ADJACENCY_COLOR, linewidth=0.8, linestyle=(0, (2, 2)), label="layout_adjacency"))
    if has_hierarchy:
        handles.append(plt.Line2D([0], [0], color=_HIERARCHY_COLOR, linewidth=1.0, linestyle=(0, (5, 2)), label="section_hierarchy"))
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.01, 0.55),
        frameon=False, fontsize=9, title="Relation", title_fontsize=9,
        labelcolor="#52514e",
    )