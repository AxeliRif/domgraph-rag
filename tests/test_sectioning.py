import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dom_extraction import DOMElement  # noqa: E402
from src.sectioning import is_oversized_page, page_height, split_into_sections  # noqa: E402


def _el(id_, tag, y, height, text="", depth=1) -> DOMElement:
    return DOMElement(id=id_, tag=tag, x=0, y=y, width=100, height=height, text_preview=text, depth=depth)


class SplitIntoSectionsTests(unittest.TestCase):
    def test_no_headings_yields_single_section(self):
        elements = [_el("e0", "p", 0, 50, "a"), _el("e1", "p", 60, 50, "b")]

        sections = split_into_sections(elements)

        self.assertEqual(len(sections), 1)
        self.assertEqual({e.id for e in sections[0].elements}, {"e0", "e1"})

    def test_splits_at_h2_boundaries_with_preamble(self):
        elements = [
            _el("e0", "p", 0, 50, "intro"),
            _el("e1", "h2", 60, 30, "Section A"),
            _el("e2", "p", 100, 50, "content A"),
            _el("e3", "h2", 160, 30, "Section B"),
            _el("e4", "p", 200, 50, "content B"),
        ]

        sections = split_into_sections(elements)

        self.assertEqual(len(sections), 3)
        self.assertEqual(sections[0].title, "")
        self.assertEqual({e.id for e in sections[0].elements}, {"e0"})
        self.assertEqual(sections[1].title, "Section A")
        self.assertEqual({e.id for e in sections[1].elements}, {"e1", "e2"})
        self.assertEqual(sections[2].title, "Section B")
        self.assertEqual({e.id for e in sections[2].elements}, {"e3", "e4"})

    def test_page_starting_with_heading_has_no_empty_preamble(self):
        elements = [_el("e0", "h1", 0, 30, "Title"), _el("e1", "p", 30, 50, "content")]

        sections = split_into_sections(elements)

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].title, "Title")

    def test_oversized_section_is_rechunked_by_height_budget(self):
        elements = [
            _el("e0", "h2", 0, 10, "Long section"),
            _el("e1", "p", 20, 100, "part 1"),
            _el("e2", "p", 130, 100, "part 2"),
        ]

        sections = split_into_sections(elements, max_section_height=100)

        self.assertGreater(len(sections), 1)
        self.assertEqual(sections[0].title, "Long section")
        self.assertTrue(sections[1].title.startswith("Long section (suite"))
        all_ids = {e.id for s in sections for e in s.elements}
        self.assertEqual(all_ids, {"e0", "e1", "e2"})

    def test_empty_elements_yields_no_sections(self):
        self.assertEqual(split_into_sections([]), [])


class IsOversizedPageTests(unittest.TestCase):
    def test_threshold(self):
        elements = [_el("e0", "p", 0, 50), _el("e1", "p", 9950, 50)]

        self.assertEqual(page_height(elements), 10000)
        self.assertFalse(is_oversized_page(elements, max_height=10000))
        self.assertTrue(is_oversized_page(elements, max_height=9999))


if __name__ == "__main__":
    unittest.main()
