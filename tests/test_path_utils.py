import tempfile
import unittest
from pathlib import Path

from src.path_utils import RelativePathError, resolve_report_path


class ReportPathResolutionTests(unittest.TestCase):
    def test_backslash_and_forward_slash_relative_paths_resolve_under_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "data" / "processed" / "snapshot" / "file.csv"
            self.assertEqual(resolve_report_path(r"data\processed\snapshot\file.csv", root), expected)
            self.assertEqual(resolve_report_path("data/processed/snapshot/file.csv", root), expected)

    def test_temporary_native_relative_path_is_portable_and_cleanup_is_owned_by_fixture(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        self.assertEqual(resolve_report_path(Path("cache") / "manifest.json", root), root / "cache" / "manifest.json")
        temporary.cleanup()
        self.assertFalse(root.exists())

    def test_absolute_and_escaping_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in ("/tmp/outside.csv", r"C:\Users\person\artifact.csv", "../outside.csv"):
                with self.assertRaises(RelativePathError):
                    resolve_report_path(value, root)
