import builtins
import importlib
import sys
import unittest
from unittest.mock import patch


class HardNegativeMiningFallbackTests(unittest.TestCase):
    def test_import_and_mine_without_sklearn(self):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith("sklearn"):
                raise ImportError("simulated sklearn import failure")
            return real_import(name, *args, **kwargs)

        sys.modules.pop("src.hard_negative_mining", None)
        with patch("builtins.__import__", side_effect=guarded_import):
            module = importlib.import_module("src.hard_negative_mining")

        qa = type(
            "QA",
            (),
            {
                "id": "qa_1",
                "page_url": "https://example.com",
                "page_slug": "example",
                "tile_id": "tile_1",
                "question": "What is the title of the page?",
                "answer": "Example",
            },
        )()
        tiles = [
            type("Tile", (), {"id": "tile_1", "text_preview": "This page title is Example"})(),
            type("Tile", (), {"id": "tile_2", "text_preview": "A different section about cats"})(),
        ]

        examples = module.mine_hard_negatives([qa], tiles, n_negatives=1)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].positive_tile_id, "tile_1")
        self.assertEqual(examples[0].hard_negative_tile_ids, ["tile_2"])


if __name__ == "__main__":
    unittest.main()
