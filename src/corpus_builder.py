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

Une page démesurément longue (cf. sectioning.py) n'est pas tuilée d'un bloc :
elle est découpée en sections, chacune promue à son propre noeud "page" --
comme s'il s'agissait de pages distinctes -- reliées entre elles par une
relation "continues" pour garder trace de l'ordre d'origine (distincte de
"links_to", un vrai hyperlien vers un *autre* document). Le graphe d'une page
(découpée ou non) expose toujours `G.graph["entry_page"]` : le noeud "page" à
utiliser comme point d'entrée (la première section si la page a été
découpée, sinon l'unique noeud "page") -- c'est ce que `build_corpus_graph`
rebranche quand un lien hypertexte est suivi.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

import networkx as nx

from .config import RENDER_WIDTH
from .dom_extraction import DOMElement, extract_dom_elements_async
from .graph_builder import build_graph
from .sectioning import is_oversized_page, split_into_sections
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


def build_page_graph_from_elements(
    elements: list[DOMElement], screenshot: bytes, url: str, slug: str,
    use_semantic_similarity: bool = False,
) -> nx.MultiDiGraph:
    """Partie synchrone (sans réseau ni navigateur) du pipeline page -> graphe :
    à partir d'éléments DOM et d'une capture déjà extraits, tuile puis
    construit le graphe -- découpé en sections si la page est démesurément
    longue (cf. module docstring). Factorisée hors de `build_page_graph` pour
    rester testable sans Playwright.

    `use_semantic_similarity` : transmis tel quel à `build_graph` (cf.
    graph_builder.py) -- désactivé par défaut, pour ne rien changer au
    comportement existant tant qu'on ne l'active pas explicitement."""
    if not is_oversized_page(elements):
        G = build_graph(
            build_tiles(elements, screenshot), page_url=url, page_title=url,
            use_semantic_similarity=use_semantic_similarity,
        )
        G.graph["entry_page"] = "page"
        return G

    combined = nx.MultiDiGraph()
    previous_page_node: str | None = None
    entry_page_node: str | None = None
    for section in split_into_sections(elements):
        section_tiles = build_tiles(section.elements, screenshot)
        section_url = url if section.index == 0 else f"{url}#section-{section.index}"
        section_title = section.title or url
        section_graph = build_graph(
            section_tiles, page_url=section_url, page_title=section_title,
            use_semantic_similarity=use_semantic_similarity,
        )

        section_slug = f"{slug}__sec{section.index}"
        namespaced = _namespace_graph(section_graph, section_slug)
        combined = nx.compose(combined, namespaced)

        page_node = f"{section_slug}::page"
        entry_page_node = entry_page_node or page_node
        if previous_page_node is not None:
            combined.add_edge(previous_page_node, page_node, relation="continues")
        previous_page_node = page_node

    combined.graph["entry_page"] = entry_page_node
    return combined


async def build_page_graph(
    url: str, render_width: int = RENDER_WIDTH, use_semantic_similarity: bool = False,
) -> nx.MultiDiGraph:
    """Pipeline complet (Phase 1a-1d) pour une seule page : DOM -> tuiles -> graphe."""
    elements, screenshot = await extract_dom_elements_async(url, render_width=render_width, wait_until="load")
    return build_page_graph_from_elements(
        elements, screenshot, url=url, slug=slug_for_url(url), use_semantic_similarity=use_semantic_similarity,
    )


async def build_corpus_graph(
    seed_url: str, max_linked_pages: int = 3, use_semantic_similarity: bool = False,
) -> nx.MultiDiGraph:
    """Construit un graphe multi-pages : la page de départ + les
    `max_linked_pages` premières pages qu'elle mentionne (relation
    "links_to", cf. build_graph), reliées entre elles par cette même relation.

    Ne suit les liens que sur 1 saut (pas de crawl récursif) : le but est de
    relier explicitement un petit voisinage autour de la page de départ, pas
    de crawler tout Wikipedia.

    `use_semantic_similarity` : transmis à chaque `build_page_graph` (une par
    page du corpus) -- désactivé par défaut. Pensé pour comparer, à structure
    de crawl égale, un corpus avec et sans la relation "semantic_neighbor".
    """
    seed_graph = await build_page_graph(seed_url, use_semantic_similarity=use_semantic_similarity)
    seed_slug = slug_for_url(seed_url)

    # URLs mentionnées par la page de départ, dans l'ordre où build_graph les
    # a rencontrées (= ordre de lecture des tuiles de texte), sans doublons.
    linked_urls: list[str] = []
    for _, data in seed_graph.nodes(data=True):
        if data.get("type") == "external_page" and data["url"] not in linked_urls:
            linked_urls.append(data["url"])
    linked_urls = linked_urls[:max_linked_pages]

    corpus = _namespace_graph(seed_graph, seed_slug)
    corpus.graph["entry_page"] = f"{seed_slug}::{seed_graph.graph['entry_page']}"
    used_slugs = {seed_slug}

    for url in linked_urls:
        target_slug = slug_for_url(url)
        while target_slug in used_slugs:
            target_slug += "_2"
        used_slugs.add(target_slug)

        target_graph = await build_page_graph(url, use_semantic_similarity=use_semantic_similarity)
        target_entry_node = f"{target_slug}::{target_graph.graph['entry_page']}"
        corpus = nx.compose(corpus, _namespace_graph(target_graph, target_slug))

        # Rebrancher l'arête "links_to" (qui pointait vers le noeud
        # placeholder "ext::<url>") sur le vrai point d'entrée de la page
        # maintenant crawlée -- sa première section si elle a dû être
        # découpée (cf. build_page_graph_from_elements) -- puis retirer le
        # placeholder devenu inutile.
        placeholder = f"{seed_slug}::ext::{url}"
        for source, _, data in list(corpus.in_edges(placeholder, data=True)):
            corpus.add_edge(source, target_entry_node, **data)
        corpus.remove_node(placeholder)

    return corpus
