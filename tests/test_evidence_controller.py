"""
Tests du contrôleur d'évidence (Phase 3), en particulier de la parallélisation
des lectures VLM (étape 4 de `run_evidence_controller`) : le choix des noeuds
ouverts ne doit pas changer (il ne dépend que des scores TF-IDF, figés avant
toute lecture), et les appels VLM doivent effectivement se chevaucher plutôt
que de rester séquentiels.

Usage : python3 -m pytest tests/test_evidence_controller.py
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import vlm_client as vlm_client_module  # noqa: E402
from src.evidence_controller import (  # noqa: E402
    EvidenceControllerConfig,
    collect_evidence,
    run_evidence_controller,
    state_counts,
)
from src.graph_builder import build_graph  # noqa: E402
from src.tiling import Tile  # noqa: E402


def _make_tile(tile_id: str, y: int, text: str) -> Tile:
    return Tile(
        id=tile_id, x=0, y=y, width=100, height=50,
        image=Image.new("RGB", (10, 10), color=(y % 255, 0, 0)),
        dom_tag="p", text_preview=text,
    )


TILES = [
    _make_tile("tile_0000", 0, "Introduction to elephants and their habitat."),
    _make_tile("tile_0001", 60, "Elephants can weigh up to 6000 kilograms."),
    _make_tile("tile_0002", 120, "Unrelated paragraph about tomatoes and soil."),
    _make_tile("tile_0003", 180, "More about tomato farming techniques."),
]

QUERY = "How much do elephants weigh?"


def _build_test_graph():
    return build_graph(TILES, page_url="https://example.org/test", page_title="test")


def test_run_evidence_controller_without_vlm_opens_most_relevant_tile():
    graph = _build_test_graph()
    config = EvidenceControllerConfig(budget=2, top_k_seed=2, use_vlm=False)
    run_evidence_controller(graph, TILES, QUERY, config)

    assert state_counts(graph)["opened"] >= 1
    assert "6000" in collect_evidence(graph)  # tuile la plus pertinente ouverte, evidence = text_preview


def test_run_evidence_controller_with_vlm_fills_evidence_from_vlm_response(monkeypatch):
    class _FakeVLMClient:
        def ask(self, image, question, think=None):
            return f"lu par le VLM ({question[:10]}...)"

    monkeypatch.setattr(vlm_client_module, "VLMClient", _FakeVLMClient)

    graph = _build_test_graph()
    config = EvidenceControllerConfig(budget=2, top_k_seed=2, use_vlm=True)
    run_evidence_controller(graph, TILES, QUERY, config)

    opened = [d for _, d in graph.nodes(data=True) if d.get("type") == "element" and d.get("state") == "opened"]
    assert opened
    for data in opened:
        assert data["evidence"].startswith("lu par le VLM")


def test_vlm_reads_of_opened_tiles_run_concurrently(monkeypatch):
    """Vérifie que les lectures VLM des tuiles ouvertes se chevauchent
    réellement dans le temps (pas juste que le code compile) : chaque appel
    factice dort 50ms après s'être signalé "en cours", donc si deux appels ne
    se recouvrent jamais, `max_concurrent` restera à 1."""
    concurrency = {"current": 0, "max": 0}
    lock = threading.Lock()

    class _FakeVLMClient:
        def ask(self, image, question, think=None):
            with lock:
                concurrency["current"] += 1
                concurrency["max"] = max(concurrency["max"], concurrency["current"])
            time.sleep(0.05)
            with lock:
                concurrency["current"] -= 1
            return "evidence"

    monkeypatch.setattr(vlm_client_module, "VLMClient", _FakeVLMClient)

    graph = _build_test_graph()
    # budget=4 (toutes les tuiles), seuils à 0 pour forcer l'ouverture des 4.
    config = EvidenceControllerConfig(
        budget=4, top_k_seed=4, open_threshold=0.0, activate_threshold=0.0,
        use_vlm=True, max_concurrent_vlm_calls=4,
    )
    run_evidence_controller(graph, TILES, QUERY, config)

    assert concurrency["max"] >= 2


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
