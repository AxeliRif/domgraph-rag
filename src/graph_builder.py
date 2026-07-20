"""
Phase 1d — Construction de la topologie du graphe.

Construit un graphe multi-granulaire à 2 niveaux (inspiré de MAGE-RAG,
arXiv:2606.15906) : un noeud "page" et des noeuds "élément" (une par tuile).

Cinq relations sont prévues par MAGE-RAG ; quatre sont déjà instanciées ici
(la cinquième, "semantic_neighbor", suppose des embeddings et reste une tâche
de Phase 2/3) :
  - "contains"          : page -> élément (niveau page / niveau élément)
  - "reading_order"     : élément -> élément suivant, dans l'ordre de lecture
  - "links_to"          : élément -> page externe (lien hypertexte), limité
    aux premières tuiles de texte (cf. build_graph, MAX_LINKED_TEXT_TILES)
  - "layout_adjacency"  : élément <-> élément voisin dans la mise en page 2D
    (voisin de "ligne" ou de "colonne"), indépendamment de l'ordre de lecture
    linéaire — cf. compute_layout_adjacency
  - "section_hierarchy" : titre -> sous-titre / contenu, respectant
    l'imbrication des niveaux h1-h4 (arbre de sections, distinct du
    rattachement plat page -> élément de "contains") — cf.
    compute_section_hierarchy

L'ordre de lecture n'est PAS l'ordre du DOM, mais une heuristique spatiale de
tri visuel : les tuiles sont regroupées par "lignes" (chevauchement vertical),
puis triées de gauche à droite au sein d'une ligne, et les lignes sont
parcourues de haut en bas. C'est une version simplifiée de la détection
d'ordre de lecture utilisée en analyse de mise en page de documents.
"layout_adjacency" réutilise ce même regroupement en lignes, mais garde le
voisinage 2D (gauche/droite/dessus/dessous) que l'ordre linéaire aplatit.

Chaque noeud élément reçoit un attribut `state`, initialisé à "inactive" :
c'est la machine à états (Inactive/Active/Opened/Pruned) que le contrôleur
d'évidence en ligne de la Phase 3 fera évoluer — elle n'est pas utilisée ici,
mais le graphe est déjà prêt à l'accueillir.
"""
from __future__ import annotations

from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

import networkx as nx

from .config import MAX_LINKED_TEXT_TILES
from .tiling import Tile


def _group_into_rows(tiles: list[Tile], row_tolerance: float) -> list[list[Tile]]:
    """Regroupe les tuiles en "lignes" de lecture (chevauchement vertical à
    `row_tolerance` près), chaque ligne étant triée gauche -> droite.

    Factorisé car `compute_reading_order` (aplati en une seule séquence) et
    `compute_layout_adjacency` (qui a besoin de conserver les lignes
    distinctes pour détecter les voisins verticaux) partagent ce regroupement.
    """
    remaining = sorted(tiles, key=lambda t: t.y)
    rows: list[list[Tile]] = []
    used_ids: set[str] = set()

    for anchor in remaining:
        if anchor.id in used_ids:
            continue
        row = [t for t in remaining if t.id not in used_ids and abs(t.y - anchor.y) <= row_tolerance]
        row.sort(key=lambda t: t.x)
        rows.append(row)
        used_ids.update(t.id for t in row)

    return rows


def compute_reading_order(tiles: list[Tile], row_tolerance: float = 20.0) -> list[Tile]:
    """Heuristique spatiale de tri visuel (haut -> bas, gauche -> droite par 'ligne').

    `row_tolerance` : écart vertical (en pixels) toléré pour considérer deux
    tuiles comme appartenant à la même "ligne" de lecture.
    """
    ordered: list[Tile] = []
    for row in _group_into_rows(tiles, row_tolerance):
        ordered.extend(row)
    return ordered


def compute_layout_adjacency(tiles: list[Tile], row_tolerance: float = 20.0) -> list[tuple[str, str, str]]:
    """Heuristique de voisinage spatial 2D, distincte de l'ordre de lecture linéaire.

    Deux types de voisinage, réciproques (une arête dans chaque sens) :
      - voisins "gauche"/"droite" : tuiles consécutives d'une même ligne ;
      - voisins "dessus"/"dessous" : une tuile et la tuile la plus proche
        (par recouvrement horizontal, puis proximité des centres) de la ligne
        suivante — capture les voisins qu'une mise en page multi-colonnes fait
        "sauter" dans l'ordre de lecture linéaire.

    Renvoie des triplets (id_source, id_cible, direction) avec direction dans
    {"left", "right", "above", "below"}.
    """
    rows = _group_into_rows(tiles, row_tolerance)
    edges: list[tuple[str, str, str]] = []

    for row in rows:
        for left, right in zip(row, row[1:]):
            edges.append((left.id, right.id, "right"))
            edges.append((right.id, left.id, "left"))

    for row_above, row_below in zip(rows, rows[1:]):
        for tile in row_above:
            x0, x1 = tile.x, tile.x + tile.width
            overlapping = [t for t in row_below if t.x < x1 and t.x + t.width > x0]
            if not overlapping:
                continue
            center = tile.x + tile.width / 2
            nearest = min(overlapping, key=lambda t: abs((t.x + t.width / 2) - center))
            edges.append((tile.id, nearest.id, "below"))
            edges.append((nearest.id, tile.id, "above"))

    return edges


_HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}


def _heading_level(dom_tag: str) -> int | None:
    """Niveau de titre d'une tuile, à partir de son `dom_tag`.

    Une tuile de titre est presque toujours fusionnée avec l'élément suivant
    par le tuilage (cf. tiling.py, `pending_heading`), d'où un tag composé
    comme "h2+p" : seul le premier composant porte le niveau de titre.
    """
    return _HEADING_LEVELS.get(dom_tag.split("+", 1)[0])


def compute_section_hierarchy(ordered_tiles: list[Tile]) -> list[tuple[str, str]]:
    """Arêtes "section_hierarchy" : rattache chaque tuile de contenu à la
    section (titre h1-h4) la plus proche qui la précède en ordre de lecture,
    et chaque titre à son titre parent immédiat (niveau juste supérieur),
    en respectant l'imbrication (un h3 est sous le dernier h2 ouvert, pas sous
    le dernier h1). Forme un arbre de sections, distinct du rattachement plat
    page -> tuile de "contains".

    Le contenu précédant le tout premier titre n'a pas de section parente et
    ne produit donc aucune arête.
    """
    edges: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []  # (niveau, id tuile), du plus englobant au plus profond

    for tile in ordered_tiles:
        level = _heading_level(tile.dom_tag)
        if level is not None:
            while stack and stack[-1][0] >= level:
                stack.pop()
            if stack:
                edges.append((stack[-1][1], tile.id))
            stack.append((level, tile.id))
        elif stack:
            edges.append((stack[-1][1], tile.id))

    return edges


def build_graph(
    tiles: list[Tile],
    page_url: str = "",
    page_title: str = "",
    max_linked_text_tiles: int = MAX_LINKED_TEXT_TILES,
) -> nx.MultiDiGraph:
    """Construit le graphe multi-granulaire (1 noeud page + N noeuds élément).

    Les liens hypertextes (arêtes "links_to", élément -> page externe) ne sont
    conservés que pour les `max_linked_text_tiles` premières tuiles de texte
    en ordre de lecture qui portent effectivement des liens — seules les
    tuiles issues d'un <p> peuvent en porter (cf. dom_extraction.py), donc les
    tableaux et les listes de références en bas de page en sont exclus
    d'office. Au-delà de la limite, les liens sont ignorés pour éviter
    l'explosion combinatoire du crawl multi-pages.
    """
    ordered_tiles = compute_reading_order(tiles)

    G = nx.MultiDiGraph()
    G.add_node("page", type="page", url=page_url, title=page_title, state="active")

    previous_id = None
    linked_text_tile_count = 0
    for tile in ordered_tiles:
        G.add_node(
            tile.id,
            type="element",
            tag=tile.dom_tag,
            x=tile.x,
            y=tile.y,
            width=tile.width,
            height=tile.height,
            text_preview=tile.text_preview,
            state="inactive",
        )
        G.add_edge("page", tile.id, relation="contains")
        if previous_id is not None:
            G.add_edge(previous_id, tile.id, relation="reading_order")
        previous_id = tile.id

        if tile.links and linked_text_tile_count < max_linked_text_tiles:
            linked_text_tile_count += 1
            for link in tile.links:
                target_id = f"ext::{link.href}"
                G.add_node(target_id, type="external_page", url=link.href)
                G.add_edge(tile.id, target_id, relation="links_to", anchor_text=link.anchor_text)

    for src_id, dst_id, direction in compute_layout_adjacency(ordered_tiles):
        G.add_edge(src_id, dst_id, relation="layout_adjacency", direction=direction)

    for parent_id, child_id in compute_section_hierarchy(ordered_tiles):
        G.add_edge(parent_id, child_id, relation="section_hierarchy")

    return G


def graph_to_xml(G: nx.MultiDiGraph, root_node: str = "page") -> str:
    """Rend le (sous-)graphe en XML lisible par le modèle lecteur (cf. MAGE-RAG,
    étape 4 : "Rendu structuré multimodal")."""
    page_attrs = G.nodes[root_node]
    root = Element(
        "page",
        {"url": str(page_attrs.get("url", "")), "title": str(page_attrs.get("title", ""))},
    )

    for _, node_id, data in G.out_edges(root_node, data=True):
        if data.get("relation") != "contains":
            continue
        node_data = G.nodes[node_id]
        el = SubElement(
            root,
            "element",
            {
                "id": str(node_id),
                "tag": str(node_data.get("tag", "")),
                "state": str(node_data.get("state", "")),
            },
        )
        el.text = str(node_data.get("text_preview", ""))

    return minidom.parseString(tostring(root)).toprettyxml(indent="  ")
