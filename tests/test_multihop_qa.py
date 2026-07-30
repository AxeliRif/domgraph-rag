"""
Tests de src/multihop_qa.py -- la partie testable sans réseau ni Playwright
(la construction bout en bout d'un MultiHopExample fait un vrai rendu de
deux pages Wikipédia distinctes, cf. scripts/build_multihop_dataset.py pour
l'exercer réellement).

Usage : python3 -m pytest tests/test_multihop_qa.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.multihop_qa import (  # noqa: E402
    _sentence_overlaps_tile,
    find_positive_tiles,
    mine_cross_page_hard_negatives,
    slug_for_title,
    url_for_title,
)
from src.tiling import Tile  # noqa: E402


def _make_tile(tile_id: str, text: str) -> Tile:
    return Tile(id=tile_id, x=0, y=0, width=10, height=10, image=Image.new("RGB", (10, 10)), text_preview=text)


def test_slug_and_url_for_title():
    assert slug_for_title("Kiss and Tell (1945 film)") == "Kiss_and_Tell_(1945_film)"
    assert url_for_title("Ed Wood") == "https://en.wikipedia.org/wiki/Ed_Wood"


def test_sentence_overlaps_tile_tolerates_truncation_and_rewording():
    long_sentence = (
        "Shirley Temple was an American actress who later became a diplomat, "
        "serving as Chief of Protocol of the United States."
    )
    # tronqué (comme Tile.text_preview, 200 caractères) ET légèrement reformulé --
    # un recouvrement de tokens (pas une sous-chaîne exacte) doit encore matcher.
    reworded_and_truncated_tile_text = "Shirley Temple, American actress, later served as Chief of Protocol for the United Sta"
    assert _sentence_overlaps_tile(long_sentence, reworded_and_truncated_tile_text)


def test_sentence_overlaps_tile_rejects_unrelated_text():
    assert not _sentence_overlaps_tile(
        "Shirley Temple served as Chief of Protocol.", "The Eiffel Tower is located in Paris, France.",
    )


def test_find_positive_tiles_matches_only_overlapping_tiles():
    tiles = [
        _make_tile("tile_0000", "Shirley Temple was an American actress and diplomat."),
        _make_tile("tile_0001", "She served as Chief of Protocol of the United States from 1976 to 1977."),
        _make_tile("tile_0002", "Unrelated paragraph about a different topic entirely."),
    ]
    positives = find_positive_tiles(tiles, ["Chief of Protocol of the United States"])
    assert positives == ["tile_0001"]


def test_mine_cross_page_hard_negatives_pools_across_pages_and_excludes_positives():
    page_a_tiles = [
        _make_tile("tile_0000", "Shirley Temple was an American actress and diplomat."),
        _make_tile("tile_0001", "She served as Chief of Protocol of the United States."),
    ]
    page_b_tiles = [
        _make_tile("tile_0000", "Corliss Archer was portrayed in the 1945 film Kiss and Tell."),
        _make_tile("tile_0001", "The film also inspired a later television series adaptation."),
    ]
    positive_ids = {("Shirley_Temple", "tile_0001"), ("Kiss_and_Tell", "tile_0000")}

    negatives = mine_cross_page_hard_negatives(
        question="What government position was held by the actress who portrayed Corliss Archer?",
        pages=[("Shirley_Temple", page_a_tiles), ("Kiss_and_Tell", page_b_tiles)],
        positive_ids=positive_ids,
        n_negatives=2,
    )

    assert set(negatives).isdisjoint(positive_ids)  # jamais un positif reproposé comme négatif
    assert {"Shirley_Temple", "Kiss_and_Tell"} >= {slug for slug, _ in negatives}  # vient bien des deux pages en pool


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
