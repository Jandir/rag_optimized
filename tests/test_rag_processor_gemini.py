import unittest
import os
import re
from rag_processor import (
    clean_srt_content,
    extract_metadata_from_filename,
    enforce_terminology,
    _parse_srt_blocks,
    _deduplicate_srt_lines
)

class TestRagProcessor(unittest.TestCase):
    def test_clean_srt_content_simple(self):
        srt_content = """1
00:00:01,000 --> 00:00:02,000
Hello world

2
00:00:02,000 --> 00:00:03,000
Hello world
How are you
"""
        # Rollup logic: if 2nd block starts with 1st, it's deduplicated
        cleaned = clean_srt_content(srt_content)
        self.assertIn("Hello world How are you", cleaned)

    def test_extract_metadata(self):
        filename = "MasterMind Abr 2024 Transcrição.txt"
        meta = extract_metadata_from_filename(filename)
        self.assertEqual(meta["title"], "MasterMind Abril 2024")
        self.assertEqual(meta["event_date"], "Abril de 2024")
        self.assertEqual(meta["video_id"], "N/A")

    def test_extract_metadata_with_id(self):
        filename = "Aula_Ekklezia [abc123_-XYZ].srt"
        meta = extract_metadata_from_filename(filename)
        self.assertEqual(meta["video_id"], "abc123_-XYZ")

    def test_enforce_terminology(self):
        rules = [
            {"original": "Sete Montes", "replacement": "7 Montes", "is_regex": False},
            {"original": r"vixe\s+maria", "compiled_pattern": re.compile(r"vixe\s+maria"), "replacement": "caramba", "is_regex": True}
        ]
        text = "Falamos sobre Sete Montes e vixe  maria que aula."
        result = enforce_terminology(text, rules)
        self.assertIn("7 Montes", result)
        self.assertIn("caramba", result)

    def test_parse_srt_blocks(self):
        srt = "1\n00:00:00,000 --> 00:00:01,000\n<b>Bold</b> text\n\n"
        blocks = _parse_srt_blocks(srt)
        self.assertEqual(blocks, ["Bold text"])

if __name__ == '__main__':
    unittest.main()
