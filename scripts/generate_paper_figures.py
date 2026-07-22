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
from pathlib import Path

import matplotlib.pyplot as plt

from src.corpus_builder import build_page_graph_from_elements, slug_for_url
from src.dom_extraction import extract_dom_elements_async
from src.graph_builder import build_graph
from src.tiling import build_tiles
from src.visualization import plot_graph, plot_page_overlay, plot_page_overview

FIGURES_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"

NORMAL_PAGE_URL = "https://en.wikipedia.org/wiki/George_P._Gunn"
OVERSIZED_PAGE_URL = "https://en.wikipedia.org/wiki/The_Beatles"


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
    await generate_tiling_and_graph_figures(NORMAL_PAGE_URL)
    await generate_section_split_figure(OVERSIZED_PAGE_URL)
    print("\nTerminé.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
