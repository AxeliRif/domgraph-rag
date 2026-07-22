"""
Test du cache disque de VLMClient (data/cache/) — ajouté avec le cache
lui-même : sans lui, un bug qui ferait toujours (ou jamais) un cache-hit
passerait inaperçu, puisque tous les autres tests de Phase 2 utilisent un
client VLM factice (`_StubClient`) qui contourne VLMClient entièrement.

Usage : python3 -m pytest tests/test_vlm_client.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import vlm_client as vlm_client_module  # noqa: E402
from src.vlm_client import VLMClient  # noqa: E402


def _make_client(monkeypatch, tmp_path) -> tuple[VLMClient, list[int]]:
    monkeypatch.setattr(vlm_client_module, "CACHE_DIR", tmp_path)
    client = VLMClient(backend="ollama")
    calls: list[int] = []

    def _fake_ask_ollama(image, question, think=None):
        calls.append(1)
        return f"answer #{len(calls)}"

    def _fake_ask_text_ollama(question):
        calls.append(1)
        return f"answer #{len(calls)}"

    monkeypatch.setattr(client, "_ask_ollama", _fake_ask_ollama)
    monkeypatch.setattr(client, "_ask_text_ollama", _fake_ask_text_ollama)
    return client, calls


def test_ask_hits_cache_for_identical_image_and_question(monkeypatch, tmp_path):
    client, calls = _make_client(monkeypatch, tmp_path)
    image_a = Image.new("RGB", (10, 10), color=(200, 0, 0))
    image_b = Image.new("RGB", (10, 10), color=(200, 0, 0))  # même contenu, autre objet

    first = client.ask(image_a, "What color?")
    second = client.ask(image_b, "What color?")

    assert first == second == "answer #1"
    assert len(calls) == 1  # le second appel n'a pas atteint _ask_ollama


def test_ask_misses_cache_when_image_or_question_differ(monkeypatch, tmp_path):
    client, calls = _make_client(monkeypatch, tmp_path)
    red = Image.new("RGB", (10, 10), color=(200, 0, 0))
    blue = Image.new("RGB", (10, 10), color=(0, 0, 200))

    client.ask(red, "What color?")
    client.ask(blue, "What color?")  # image différente -> pas de cache-hit
    client.ask(red, "What shape?")   # question différente -> pas de cache-hit

    assert len(calls) == 3


def test_ask_text_hits_cache_for_identical_question(monkeypatch, tmp_path):
    client, calls = _make_client(monkeypatch, tmp_path)

    first = client.ask_text("Synthesize: A and B")
    second = client.ask_text("Synthesize: A and B")

    assert first == second == "answer #1"
    assert len(calls) == 1


def test_cache_persists_across_client_instances(monkeypatch, tmp_path):
    monkeypatch.setattr(vlm_client_module, "CACHE_DIR", tmp_path)

    client1 = VLMClient(backend="ollama")
    calls: list[int] = []
    monkeypatch.setattr(client1, "_ask_ollama", lambda image, question, think=None: calls.append(1) or "cached-answer")
    image = Image.new("RGB", (10, 10), color=(1, 2, 3))
    client1.ask(image, "Question?")

    client2 = VLMClient(backend="ollama")
    monkeypatch.setattr(client2, "_ask_ollama", lambda image, question, think=None: calls.append(1) or "should-not-be-called")
    result = client2.ask(image, "Question?")

    assert result == "cached-answer"
    assert len(calls) == 1


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
