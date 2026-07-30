"""
Tests de la relation "semantic_neighbor" (cf. graph_builder.py) : optionnelle
(build_graph(..., use_semantic_similarity=False) par défaut), pensée pour
comparer, benchmark à l'appui, un graphe avec et sans cette relation.

Usage : python3 -m pytest tests/test_graph_builder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_builder import build_graph, compute_semantic_neighbor_edges  # noqa: E402
from src.tiling import Tile  # noqa: E402


def _make_tile(tile_id: str, y: int, text: str) -> Tile:
    return Tile(id=tile_id, x=0, y=y, width=100, height=50, image=Image.new("RGB", (10, 10)), dom_tag="p", text_preview=text)


# tile_0000 et tile_0002 parlent d'éléphants, tile_0001 et tile_0003 de tomates
# -- non adjacentes en ordre de lecture, pour bien distinguer "semantic_neighbor"
# de "reading_order"/"layout_adjacency". TF-IDF ne fait aucune lemmatisation
# (un choix volontairement simple, cf. hard_negative_mining.py) : les deux
# phrases d'un même sujet doivent donc littéralement répéter les mêmes formes
# de mots ("elephant", pas un mélange singulier/pluriel) pour obtenir un score
# de similarité significatif.
TILES = [
    _make_tile("tile_0000", 0, "The elephant is a large animal with a trunk and big ears."),
    _make_tile("tile_0001", 60, "Tomatoes need sunlight and water to grow well in soil."),
    _make_tile("tile_0002", 120, "The elephant uses its trunk to drink water and eat food."),
    _make_tile("tile_0003", 180, "Tomatoes grow well with plenty of sunlight and water."),
]


def _pairs_from_edges(edges: list[tuple[str, str, float]]) -> set[frozenset[str]]:
    return {frozenset((src, dst)) for src, dst, _ in edges}


def test_semantically_similar_tiles_are_linked():
    edges = compute_semantic_neighbor_edges(TILES, threshold=0.1, max_neighbors_per_tile=3)
    pairs = _pairs_from_edges(edges)

    assert frozenset(("tile_0000", "tile_0002")) in pairs  # les deux tuiles "éléphants"
    assert frozenset(("tile_0001", "tile_0003")) in pairs  # les deux tuiles "tomates"
    assert frozenset(("tile_0000", "tile_0001")) not in pairs  # sujets différents
    assert frozenset(("tile_0000", "tile_0003")) not in pairs


def test_high_threshold_keeps_no_edges():
    edges = compute_semantic_neighbor_edges(TILES, threshold=0.99, max_neighbors_per_tile=3)
    assert edges == []


def test_max_neighbors_per_tile_caps_degree():
    # Quatre tuiles toutes très proches entre elles : avec max_neighbors_per_tile=1,
    # la sélection gloutonne garantit qu'aucune tuile ne dépasse ce degré,
    # même si elle serait la meilleure correspondance de plusieurs autres.
    similar_tiles = [
        _make_tile("tile_0000", 0, "Elephants are large grey mammals with trunks and big ears."),
        _make_tile("tile_0001", 60, "Elephants are large grey mammals with trunks and big ears too."),
        _make_tile("tile_0002", 120, "Elephants are large grey mammals with trunks, big ears, also."),
        _make_tile("tile_0003", 180, "Elephants are large grey mammals with trunks and big ears indeed."),
    ]

    edges = compute_semantic_neighbor_edges(similar_tiles, threshold=0.1, max_neighbors_per_tile=1)
    degree: dict[str, int] = {}
    for src, dst, _ in edges:
        degree[src] = degree.get(src, 0) + 1
        degree[dst] = degree.get(dst, 0) + 1

    assert len(edges) > 0
    assert all(count <= 1 for count in degree.values())


def test_single_tile_has_no_neighbors():
    assert compute_semantic_neighbor_edges(TILES[:1]) == []


def test_build_graph_omits_semantic_neighbor_by_default():
    graph = build_graph(TILES, page_url="https://example.org/test", page_title="test")
    relations = {data.get("relation") for _, _, data in graph.edges(data=True)}
    assert "semantic_neighbor" not in relations


def test_build_graph_adds_semantic_neighbor_when_enabled():
    graph = build_graph(
        TILES, page_url="https://example.org/test", page_title="test",
        use_semantic_similarity=True, semantic_neighbor_threshold=0.1,
    )
    semantic_pairs = {
        frozenset((u, v)) for u, v, data in graph.edges(data=True) if data.get("relation") == "semantic_neighbor"
    }
    assert frozenset(("tile_0000", "tile_0002")) in semantic_pairs
    assert frozenset(("tile_0001", "tile_0003")) in semantic_pairs


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
