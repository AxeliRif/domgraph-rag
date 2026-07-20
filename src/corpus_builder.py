"""
Phase 1e — Construction d'un graphe multi-pages ("corpus"), en suivant les
liens hypertextes extraits par build_graph (relation "links_to").

Chaque page a son propre graphe 2 niveaux (page/éléments), construit par le
même pipeline que pour une page seule (dom_extraction -> tiling ->
graph_builder). Pour les assembler en un seul graphe sans collision
d'identifiants (chaque page utilise indépendamment "page", "tile_0000", ...),
les noeuds de chaque sous-graphe sont préfixés par un slug dérivé de son URL
avant fusion.

Le crawl est volontairement peu profond (1 saut : la page de départ, puis les
`max_linked_pages` pages qu'elle mentionne dans ses premières tuiles de texte)
et non récursif : suivre les liens de proche en proche ferait exploser le
nombre de pages à charger (cf. discussion dans README.md).
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

import networkx as nx

from .config import RENDER_WIDTH
from .dom_extraction import extract_dom_elements_async
from .graph_builder import build_graph
from .tiling import build_tiles


def slug_for_url(url: str) -> str:
    """Identifiant court et stable dérivé de l'URL, pour préfixer les noeuds
    d'une page (ex. "https://en.wikipedia.org/wiki/Retrieval..." -> "Retrieval...")."""
    path = urlparse(url).path.rstrip("/")
    tail = path.rsplit("/", 1)[-1] or urlparse(url).netloc
    return re.sub(r"[^A-Za-z0-9_.-]", "_", tail)


def _namespace_graph(G: nx.MultiDiGraph, prefix: str) -> nx.MultiDiGraph:
    """Copie de G où chaque noeud est préfixé par `prefix`, pour permettre la
    fusion de plusieurs graphes de page sans collision d'ids."""
    return nx.relabel_nodes(G, {n: f"{prefix}::{n}" for n in G.nodes}, copy=True)


async def build_page_graph(url: str, render_width: int = RENDER_WIDTH) -> nx.MultiDiGraph:
    """Pipeline complet (Phase 1a-1d) pour une seule page : DOM -> tuiles -> graphe."""
    elements, screenshot = await extract_dom_elements_async(url, render_width=render_width, wait_until="load")
    tiles = build_tiles(elements, screenshot)
    return build_graph(tiles, page_url=url, page_title=url)


async def build_corpus_graph(seed_url: str, max_linked_pages: int = 3) -> nx.MultiDiGraph:
    """Construit un graphe multi-pages : la page de départ + les
    `max_linked_pages` premières pages qu'elle mentionne (relation
    "links_to", cf. build_graph), reliées entre elles par cette même relation.

    Ne suit les liens que sur 1 saut (pas de crawl récursif) : le but est de
    relier explicitement un petit voisinage autour de la page de départ, pas
    de crawler tout Wikipedia.
    """
    seed_graph = await build_page_graph(seed_url)
    seed_slug = slug_for_url(seed_url)

    # URLs mentionnées par la page de départ, dans l'ordre où build_graph les
    # a rencontrées (= ordre de lecture des tuiles de texte), sans doublons.
    linked_urls: list[str] = []
    for _, data in seed_graph.nodes(data=True):
        if data.get("type") == "external_page" and data["url"] not in linked_urls:
            linked_urls.append(data["url"])
    linked_urls = linked_urls[:max_linked_pages]

    corpus = _namespace_graph(seed_graph, seed_slug)
    used_slugs = {seed_slug}

    for url in linked_urls:
        target_slug = slug_for_url(url)
        while target_slug in used_slugs:
            target_slug += "_2"
        used_slugs.add(target_slug)

        target_graph = await build_page_graph(url)
        corpus = nx.compose(corpus, _namespace_graph(target_graph, target_slug))

        # Rebrancher l'arête "links_to" (qui pointait vers le noeud
        # placeholder "ext::<url>") sur le vrai noeud "page" de la page
        # maintenant crawlée, puis retirer le placeholder devenu inutile.
        placeholder = f"{seed_slug}::ext::{url}"
        real_page_node = f"{target_slug}::page"
        for source, _, data in list(corpus.in_edges(placeholder, data=True)):
            corpus.add_edge(source, real_page_node, **data)
        corpus.remove_node(placeholder)

    return corpus
