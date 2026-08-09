"""
Génère les figures de données du papier (paper/figures/*.pdf) à partir de
pages Wikipédia réelles -- extraction DOM + tuilage + graphe uniquement, pas
d'appel VLM, donc rapide même sur une page démesurée.

Deux exemples :
  - George_P._Gunn : page de taille normale, pour illustrer le tuilage
    (tuiles superposées à la capture réelle) et le graphe page/éléments
    (contains, reading_order, layout_adjacency, section_hierarchy).
  - The_Beatles : page démesurée (cf. sectioning.py), pour illustrer le
    découpage en sections et la relation "continues" via la vue
    "page-de-pages" (plot_page_overview).

Usage :
    python3 -m scripts.generate_paper_figures
"""
from __future__ import annotations

import asyncio
import io
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from scripts.grid_baseline import build_grid_tiles
from src.corpus_builder import build_page_graph_from_elements, slug_for_url
from src.dom_extraction import extract_dom_elements_async
from src.graph_builder import build_graph
from src.tiling import build_tiles
from src.visualization import plot_graph, plot_page_overlay, plot_page_overview

FIGURES_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"

NORMAL_PAGE_URL = "https://en.wikipedia.org/wiki/George_P._Gunn"
OVERSIZED_PAGE_URL = "https://en.wikipedia.org/wiki/The_Beatles"

# Hauteur (px) sur laquelle recadrer la comparaison DOM vs grille : assez pour
# montrer l'infobox (le cas qui vend l'argument -- une grille la coupe à mi-
# hauteur, le DOM jamais) sans une figure de la hauteur de la page entière.
COMPARISON_CROP_HEIGHT = 1500


def _plot_grid_overlay(grid_tiles, screenshot_bytes: bytes, ax: plt.Axes) -> None:
    """Pendant minimal de `plot_page_overlay` pour les tuiles de la grille
    fixe (scripts/grid_baseline.py) : pas de notion de catégorie de contenu
    ni d'ordre de lecture calculé, juste les bandes numérées dans l'ordre où
    `build_grid_tiles` les produit (= ordre vertical, trivialement)."""
    page_image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
    ax.imshow(page_image)
    color = "#e34948"
    for order, tile in enumerate(grid_tiles, start=1):
        ax.add_patch(Rectangle(
            (tile.x, tile.y), tile.width, tile.height,
            linewidth=2, edgecolor=color, facecolor="none",
        ))
        ax.text(
            tile.x + 3, tile.y + 3, str(order),
            fontsize=8, fontweight="bold", color="white", va="top", ha="left",
            bbox={"boxstyle": "circle,pad=0.25", "facecolor": color, "edgecolor": "none"},
        )
    ax.set_title("Blind fixed-height grid (PixelRAG-style)")
    ax.axis("off")


async def generate_dom_vs_grid_comparison_figure(url: str) -> None:
    print(f"[0/2] {url} — comparaison tuilage DOM vs grille...")
    elements, screenshot = await extract_dom_elements_async(url, wait_until="load")
    dom_tiles = build_tiles(elements, screenshot)
    grid_tiles = build_grid_tiles(elements, screenshot)

    # Recadre aussi horizontalement sur l'étendue réelle du contenu (pas la
    # largeur de viewport 2048px, bien plus large que la colonne de texte) --
    # sans ça, une bonne moitié de la figure n'est que marge grise vide.
    x_max = max((t.x + t.width for t in dom_tiles), default=1200) + 20

    fig, (ax_dom, ax_grid) = plt.subplots(1, 2, figsize=(7.2, 3.6))
    plot_page_overlay(dom_tiles, screenshot, ax=ax_dom)
    ax_dom.set_title("DOM-guided tiling (ours)", fontsize=10)
    _plot_grid_overlay(grid_tiles, screenshot, ax_grid)
    for ax in (ax_dom, ax_grid):
        ax.set_ylim(COMPARISON_CROP_HEIGHT, 0)
        ax.set_xlim(0, x_max)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dom_vs_grid_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"      écrit {FIGURES_DIR / 'dom_vs_grid_comparison.pdf'}")


async def generate_tiling_and_graph_figures(url: str) -> None:
    print(f"[1/2] {url} — extraction + tuilage + graphe...")
    elements, screenshot = await extract_dom_elements_async(url, wait_until="load")
    tiles = build_tiles(elements, screenshot)
    graph = build_graph(tiles, page_url=url, page_title=url)
    print(f"      {len(elements)} éléments -> {len(tiles)} tuiles")

    plot_page_overlay(tiles, screenshot)
    plt.savefig(FIGURES_DIR / "tiling_overlay.pdf", bbox_inches="tight")
    plt.close()
    print(f"      écrit {FIGURES_DIR / 'tiling_overlay.pdf'}")

    plot_graph(graph)
    plt.savefig(FIGURES_DIR / "page_graph.pdf", bbox_inches="tight")
    plt.close()
    print(f"      écrit {FIGURES_DIR / 'page_graph.pdf'}")


async def generate_section_split_figure(url: str) -> None:
    print(f"[2/2] {url} — extraction + découpage en sections...")
    elements, screenshot = await extract_dom_elements_async(url, wait_until="load")
    graph = build_page_graph_from_elements(elements, screenshot, url=url, slug=slug_for_url(url))
    n_sections = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "page")
    print(f"      {len(elements)} éléments -> {n_sections} section(s)")

    plot_page_overview(graph)
    plt.savefig(FIGURES_DIR / "section_split_overview.pdf", bbox_inches="tight")
    plt.close()
    print(f"      écrit {FIGURES_DIR / 'section_split_overview.pdf'}")


async def main_async() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    await generate_dom_vs_grid_comparison_figure(NORMAL_PAGE_URL)
    await generate_tiling_and_graph_figures(NORMAL_PAGE_URL)
    await generate_section_split_figure(OVERSIZED_PAGE_URL)
    print("\nTerminé.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
