import unittest


class ImportSmokeTests(unittest.TestCase):
    def test_import_qa_generation_without_optional_training_dependencies(self):
        from src.qa_generation import generate_qa_dataset

        self.assertTrue(callable(generate_qa_dataset))


if __name__ == "__main__":
    unittest.main()
