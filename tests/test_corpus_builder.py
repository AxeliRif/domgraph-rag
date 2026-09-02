import io
import sys
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.corpus_builder import build_page_graph_from_elements  # noqa: E402
from src.dom_extraction import DOMElement  # noqa: E402


def _el(id_, tag, y, height, text="", depth=1) -> DOMElement:
    return DOMElement(id=id_, tag=tag, x=0, y=y, width=100, height=height, text_preview=text, depth=depth)


def _screenshot(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(120, 120, 120)).save(buf, format="PNG")
    return buf.getvalue()


class BuildPageGraphFromElementsTests(unittest.TestCase):
    def test_small_page_stays_a_single_page_node(self):
        elements = [
            _el("e0", "h1", 0, 50, "Small page"),
            _el("e1", "p", 60, 40, "Just one short paragraph."),
        ]

        G = build_page_graph_from_elements(
            elements, _screenshot(200, 200), page_url="https://example.org/small", page_slug="small",
        )

        page_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "page"]
        self.assertEqual(page_nodes, ["page"])
        self.assertEqual(G.graph["entry_page"], "page")
        self.assertFalse(any(d.get("relation") == "continues" for _, _, d in G.edges(data=True)))

    def test_oversized_page_is_split_into_linked_section_pages(self):
        # Hauteur totale (24950px) > MAX_PAGE_HEIGHT_BEFORE_SPLIT (20480px) ;
        # chaque section individuelle reste sous le seuil (pas de sous-découpe
        # par budget de hauteur nécessaire).
        elements = [
            _el("e0", "h1", 0, 50, "The Beatles"),
            _el("e1", "p", 60, 40, "Intro paragraph."),
            _el("e2", "h2", 5000, 50, "History"),
            _el("e3", "p", 5100, 19850, "Long history section."),
        ]

        G = build_page_graph_from_elements(
            elements, _screenshot(200, 25000), page_url="https://example.org/beatles", page_slug="beatles",
        )

        page_nodes = {n for n, d in G.nodes(data=True) if d.get("type") == "page"}
        self.assertEqual(page_nodes, {"beatles__sec0::page", "beatles__sec1::page"})
        self.assertEqual(G.graph["entry_page"], "beatles__sec0::page")

        continues_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "continues"]
        self.assertEqual(continues_edges, [("beatles__sec0::page", "beatles__sec1::page")])

        self.assertEqual(G.nodes["beatles__sec1::page"]["title"], "History")
        # Chaque section garde bien ses propres éléments (pas de fuite entre sections).
        sec0_elements = {v for _, v, d in G.out_edges("beatles__sec0::page", data=True) if d.get("relation") == "contains"}
        sec1_elements = {v for _, v, d in G.out_edges("beatles__sec1::page", data=True) if d.get("relation") == "contains"}
        self.assertTrue(all(n.startswith("beatles__sec0::") for n in sec0_elements))
        self.assertTrue(all(n.startswith("beatles__sec1::") for n in sec1_elements))


if __name__ == "__main__":
    unittest.main()
