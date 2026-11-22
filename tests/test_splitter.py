"""Unit tests for mapreduce.splitter."""
from __future__ import annotations

import os
import tempfile
import unittest

from mapreduce.splitter import input_size, list_input_files, make_splits, read_split


class TestSplitter(unittest.TestCase):
    def _collect_all(self, paths, split_size: int):
        splits = make_splits(paths, split_size)
        records = []
        for s in splits:
            for offset, line in read_split(s):
                records.append((os.path.basename(s.path), offset, line))
        return splits, records

    def test_every_line_seen_exactly_once_across_split_sizes(self):
        lines = [
            "alpha beta gamma",
            "",
            "single",
            "a much longer line with numbers 12345 and punctuation!?",
            "short",
            "final line with trailing newline",
        ]
        content = ("\n".join(lines) + "\n").encode("utf-8")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sample.txt")
            with open(path, "wb") as f:
                f.write(content)

            expectedOffsets = []
            pos = 0
            for line in lines:
                expectedOffsets.append(("sample.txt", pos, line))
                pos += len(line.encode("utf-8")) + 1

            for split_size in list(range(1, len(content) + 5)) + [len(content) * 2, 1024]:
                with self.subTest(split_size=split_size):
                    _, records = self._collect_all(path, split_size)
                    self.assertEqual(records, expectedOffsets)

    def test_boundary_exactly_on_newline(self):
        # Each line is exactly 5 bytes including '\n': "abcd\n", "efgh\n", "ijkl\n"
        content = b"abcd\nefgh\nijkl\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "aligned.txt")
            with open(path, "wb") as f:
                f.write(content)

            for split_size in (4, 5, 6, 10, 15):
                with self.subTest(split_size=split_size):
                    _, records = self._collect_all(path, split_size)
                    self.assertEqual(
                        records,
                        [
                            ("aligned.txt", 0, "abcd"),
                            ("aligned.txt", 5, "efgh"),
                            ("aligned.txt", 10, "ijkl"),
                        ],
                    )

    def test_no_trailing_newline(self):
        content = b"first line\nsecond line\nthird without newline"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "no_eof_nl.txt")
            with open(path, "wb") as f:
                f.write(content)

            expected = [
                ("no_eof_nl.txt", 0, "first line"),
                ("no_eof_nl.txt", 11, "second line"),
                ("no_eof_nl.txt", 23, "third without newline"),
            ]
            for split_size in [1, 2, 5, 10, 11, 12, 22, 23, 24, len(content), len(content) + 10]:
                with self.subTest(split_size=split_size):
                    _, records = self._collect_all(path, split_size)
                    self.assertEqual(records, expected)

    def test_crlf_line_endings(self):
        content = b"line one\r\nline two\r\nline three\r\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "crlf.txt")
            with open(path, "wb") as f:
                f.write(content)

            expected = [
                ("crlf.txt", 0, "line one"),
                ("crlf.txt", 10, "line two"),
                ("crlf.txt", 20, "line three"),
            ]
            # Test split boundaries landing on '\r', on '\n', and inside text
            for split_size in range(1, len(content) + 3):
                with self.subTest(split_size=split_size):
                    _, records = self._collect_all(path, split_size)
                    self.assertEqual(records, expected)

    def test_empty_files_skipped_and_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_path = os.path.join(tmpdir, "00_empty.txt")
            file_a = os.path.join(tmpdir, "01_a.txt")
            file_b = os.path.join(tmpdir, "02_b.txt")
            hidden = os.path.join(tmpdir, ".gitkeep")

            with open(empty_path, "wb"):
                pass
            with open(hidden, "wb") as f:
                f.write(b"ignored")
            with open(file_a, "wb") as f:
                f.write(b"hello world\nfrom file a\n")
            with open(file_b, "wb") as f:
                f.write(b"only one line in b")

            listed = list_input_files(tmpdir)
            self.assertEqual(
                [os.path.basename(p) for p in listed],
                ["00_empty.txt", "01_a.txt", "02_b.txt"],
            )
            self.assertEqual(input_size(tmpdir), 24 + 18)

            for split_size in (1, 5, 12, 24, 100):
                with self.subTest(split_size=split_size):
                    splits, records = self._collect_all(tmpdir, split_size)
                    self.assertEqual([s.split_id for s in splits], list(range(len(splits))))
                    self.assertTrue(all(os.path.basename(s.path) != "00_empty.txt" for s in splits))
                    self.assertEqual(
                        records,
                        [
                            ("01_a.txt", 0, "hello world"),
                            ("01_a.txt", 12, "from file a"),
                            ("02_b.txt", 0, "only one line in b"),
                        ],
                    )

    def test_invalid_split_size_and_missing_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                make_splits(tmpdir, 0)
            with self.assertRaises(FileNotFoundError):
                list_input_files(os.path.join(tmpdir, "does_not_exist.txt"))


if __name__ == "__main__":
    unittest.main()
