import io
import sys
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dom_extraction import DOMElement  # noqa: E402
from src.tiling import build_tiles, group_overlapping_elements, prune_nested_elements  # noqa: E402


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


class GroupOverlappingElementsTests(unittest.TestCase):
    """`prune_nested_elements` garde volontairement les deux éléments d'un
    chevauchement partiel (cf. test ci-dessus) -- c'est `group_overlapping_elements`
    qui doit ensuite s'assurer qu'ils sont tuilés ensemble plutôt que
    séparément, pour ne pas dupliquer la même région de pixels dans deux
    tuiles (cf. bug rapporté : un morceau de page se retrouvait dans une
    tuile alors qu'il appartenait visuellement à une autre)."""

    def test_significantly_overlapping_siblings_are_grouped_together(self):
        paragraph = _el("el_0", "p", x=0, y=0, w=300, h=100, depth=3)
        infobox = _el("el_1", "table", x=200, y=10, w=150, h=80, depth=3)  # chevauche largement `paragraph`

        groups = group_overlapping_elements([paragraph, infobox])

        self.assertEqual(len(groups), 1)
        self.assertEqual({e.id for e in groups[0]}, {"el_0", "el_1"})

    def test_barely_touching_elements_are_not_grouped(self):
        left = _el("el_0", "p", x=0, y=0, w=200, h=100, depth=3)
        right = _el("el_1", "p", x=199, y=0, w=200, h=100, depth=3)  # 1px de recouvrement seulement

        groups = group_overlapping_elements([left, right])

        self.assertEqual(len(groups), 2)

    def test_transitively_overlapping_elements_form_a_single_group(self):
        a = _el("el_0", "p", x=0, y=0, w=100, h=100, depth=3)
        b = _el("el_1", "p", x=80, y=0, w=100, h=100, depth=3)   # chevauche a
        c = _el("el_2", "p", x=160, y=0, w=100, h=100, depth=3)  # chevauche b, pas a directement

        groups = group_overlapping_elements([a, b, c])

        self.assertEqual(len(groups), 1)
        self.assertEqual({e.id for e in groups[0]}, {"el_0", "el_1", "el_2"})

    def test_headings_are_never_grouped_even_if_they_overlap(self):
        heading = _el("el_0", "h2", x=0, y=0, w=300, h=100, depth=2)
        paragraph = _el("el_1", "p", x=0, y=0, w=300, h=100, depth=3)  # chevauchement total

        groups = group_overlapping_elements([heading, paragraph])

        self.assertEqual(len(groups), 2)
        self.assertIn([heading], groups)


class BuildTilesOverlapTests(unittest.TestCase):
    def test_overlapping_elements_produce_one_tile_instead_of_two_duplicated_crops(self):
        paragraph = _el("el_0", "p", x=0, y=0, w=300, h=100, depth=3)
        paragraph.text_preview = "Paragraph text"
        infobox = _el("el_1", "table", x=200, y=10, w=150, h=80, depth=3)  # partage l'espace visuel du paragraphe
        infobox.text_preview = "Infobox text"

        buf = io.BytesIO()
        Image.new("RGB", (400, 150), color=(255, 255, 255)).save(buf, format="PNG")

        tiles = build_tiles([paragraph, infobox], buf.getvalue())

        self.assertEqual(len(tiles), 1)  # pas deux tuiles dont les crops se recouvriraient
        tile = tiles[0]
        self.assertIn("p", tile.dom_tag)
        self.assertIn("table", tile.dom_tag)
        self.assertEqual(tile.source_element_ids, ["el_0", "el_1"])
        self.assertEqual((tile.x, tile.y), (0, 0))
        self.assertEqual((tile.width, tile.height), (350, 100))  # union des deux bounding boxes


if __name__ == "__main__":
    unittest.main()
