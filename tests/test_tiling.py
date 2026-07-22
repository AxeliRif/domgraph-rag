import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dom_extraction import DOMElement  # noqa: E402
from src.tiling import prune_nested_elements  # noqa: E402


def _el(id_, tag, x, y, w, h, depth) -> DOMElement:
    return DOMElement(id=id_, tag=tag, x=x, y=y, width=w, height=h, text_preview="", depth=depth)


class PruneNestedElementsTests(unittest.TestCase):
    def test_img_inside_figure_is_pruned(self):
        figure = _el("el_0", "figure", x=0, y=0, w=300, h=220, depth=3)
        img = _el("el_1", "img", x=0, y=0, w=300, h=200, depth=4)  # contenu dans figure

        kept = prune_nested_elements([figure, img])

        self.assertEqual([e.id for e in kept], ["el_0"])

    def test_sibling_elements_are_not_pruned(self):
        first = _el("el_0", "p", x=0, y=0, w=300, h=100, depth=3)
        second = _el("el_1", "p", x=0, y=120, w=300, h=100, depth=3)  # ne se chevauchent pas

        kept = prune_nested_elements([first, second])

        self.assertEqual({e.id for e in kept}, {"el_0", "el_1"})

    def test_partially_overlapping_elements_are_both_kept(self):
        left = _el("el_0", "p", x=0, y=0, w=200, h=100, depth=3)
        right = _el("el_1", "p", x=150, y=0, w=200, h=100, depth=3)  # chevauchement partiel seulement

        kept = prune_nested_elements([left, right])

        self.assertEqual({e.id for e in kept}, {"el_0", "el_1"})


if __name__ == "__main__":
    unittest.main()
