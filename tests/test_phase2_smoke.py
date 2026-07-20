"""
Test de bout en bout des parties de la Phase 2 qui ne nécessitent ni Playwright
(pas de vraie page à rendre) ni torch/transformers/peft (pas de VLM réel ni de
LoRA) : génération de QA (avec un client VLM factice), parsing robuste de la
réponse du VLM, hard-negative mining, et assemblage du Dataset contrastif.

Usage : python3 -m pytest tests/test_phase2_smoke.py   (depuis la racine du projet)
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contrastive_dataset import ContrastiveTileDataset, contrastive_collate_fn  # noqa: E402
from src.hard_negative_mining import mine_hard_negatives, save_contrastive_examples  # noqa: E402
from src.qa_generation import (  # noqa: E402
    _parse_qa_response,
    append_tiles_manifest,
    generate_qa_pair_for_tile,
    save_tile_images,
)
from src.tiling import Tile  # noqa: E402


def _make_tile(tile_id: str, tag: str, text: str, color: tuple[int, int, int]) -> Tile:
    return Tile(
        id=tile_id, x=0, y=0, width=10, height=10,
        image=Image.new("RGB", (10, 10), color=color),
        dom_tag=tag, text_preview=text,
    )


TILES = [
    _make_tile("tile_0000", "h2+p", "Population: the town had 4200 inhabitants in 2020.", (200, 0, 0)),
    _make_tile("tile_0001", "table", "Coordinates 45.5N 3.2E, area 12 km2, population 4200.", (0, 200, 0)),
    _make_tile("tile_0002", "img", "", (0, 0, 200)),
    _make_tile("tile_0003", "ol", "References: [1] city archives [2] census bureau", (100, 100, 100)),
]


def test_parse_qa_response_handles_plain_and_fenced_json_and_skip():
    assert _parse_qa_response('{"question": "How many inhabitants?", "answer": "4200"}') == (
        "How many inhabitants?", "4200",
    )
    fenced = '```json\n{"question": "Area?", "answer": "12 km2"}\n```'
    assert _parse_qa_response(fenced) == ("Area?", "12 km2")
    assert _parse_qa_response("SKIP") is None
    assert _parse_qa_response("not json at all") is None


def test_generate_qa_pair_for_tile_uses_client_response(tmp_path):
    tile = TILES[0]
    image_paths = save_tile_images([tile], "testpage", images_dir=tmp_path / "images")

    class _StubClient:
        def ask(self, image, question):
            return '{"question": "What was the population in 2020?", "answer": "4200"}'

    qa = generate_qa_pair_for_tile(
        tile, _StubClient(), "https://example.org/town", "testpage", image_paths, images_root=tmp_path,
    )
    assert qa is not None
    assert qa.tile_id == "tile_0000"
    assert qa.answer == "4200"
    assert (tmp_path / "images" / image_paths["tile_0000"].name).exists()


def test_generate_qa_pair_for_tile_returns_none_on_skip(tmp_path):
    tile = TILES[2]
    image_paths = save_tile_images([tile], "testpage", images_dir=tmp_path / "images")

    class _SkipClient:
        def ask(self, image, question):
            return "SKIP"

    assert generate_qa_pair_for_tile(tile, _SkipClient(), "https://example.org/town", "testpage", image_paths) is None


def test_mine_hard_negatives_excludes_positive_and_answer_leaking_tiles():
    from src.qa_generation import QAPair

    qa_pairs = [
        QAPair(
            id="testpage__tile_0000__qa", page_url="https://example.org/town", page_slug="testpage",
            tile_id="tile_0000", question="How many inhabitants does the town have?", answer="4200",
            image_path="images/testpage__tile_0000.png",
        )
    ]

    examples = mine_hard_negatives(qa_pairs, TILES, n_negatives=2)
    assert len(examples) == 1
    example = examples[0]
    assert example.positive_tile_id == "tile_0000"
    # tile_0001 mentionne aussi "4200" (la réponse) -> doit être exclu comme faux négatif.
    assert "tile_0001" not in example.hard_negative_tile_ids
    assert "tile_0000" not in example.hard_negative_tile_ids
    assert set(example.hard_negative_tile_ids).issubset({"tile_0002", "tile_0003"})


def test_contrastive_dataset_round_trip(tmp_path):
    from src.qa_generation import QAPair

    images_dir = tmp_path / "images"
    manifest_path = tmp_path / "tiles_manifest.jsonl"
    examples_path = tmp_path / "contrastive_examples.jsonl"

    image_paths = save_tile_images(TILES, "testpage", images_dir=images_dir)
    append_tiles_manifest(
        TILES, "https://example.org/town", "testpage", image_paths, manifest_path, images_root=tmp_path,
    )

    qa_pairs = [
        QAPair(
            id="testpage__tile_0000__qa", page_url="https://example.org/town", page_slug="testpage",
            tile_id="tile_0000", question="How many inhabitants does the town have?", answer="4200",
            image_path="images/testpage__tile_0000.png",
        )
    ]
    contrastive_examples = mine_hard_negatives(qa_pairs, TILES, n_negatives=2)
    save_contrastive_examples(contrastive_examples, examples_path)

    dataset = ContrastiveTileDataset(
        contrastive_examples_path=examples_path, tiles_manifest_path=manifest_path, images_root=tmp_path,
    )
    assert len(dataset) == 1
    item = dataset[0]
    assert item.question == "How many inhabitants does the town have?"
    assert item.positive_image.size == (10, 10)
    assert len(item.negative_images) == 2

    batch = contrastive_collate_fn([item])
    assert batch["questions"] == [item.question]
    assert len(batch["tile_images"]) == 3  # 1 positif + 2 négatifs
    assert batch["positive_indices"] == [0]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
