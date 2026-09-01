import pytest
import os
from rag_processor import (
    clean_srt_content,
    extract_metadata_from_filename,
    enforce_terminology,
    _parse_srt_blocks,
    _handle_simple_repetition,
    _handle_partial_overlap
)

def test_clean_srt_content_simple():
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
    assert "Hello world How are you" in cleaned

def test_extract_metadata_mastermind():
    filename = "MasterMind Abr 2024 Transcrição.txt"
    meta = extract_metadata_from_filename(filename)
    assert meta["title"] == "MasterMind Abril 2024"
    assert meta["event_date"] == "Abril de 2024"
    assert meta["video_id"] == "N/A"

def test_extract_metadata_with_id():
    filename = "Aula_Ekklezia [abc123_-XYZ].srt"
    meta = extract_metadata_from_filename(filename)
    assert meta["video_id"] == "abc123_-XYZ"

def test_enforce_terminology():
    import re
    rules = [
        {"original": "Sete Montes", "replacement": "7 Montes", "is_regex": False},
        {"original": r"vixe\s+maria", "replacement": "caramba", "is_regex": True, "compiled_pattern": re.compile(r"vixe\s+maria")}
    ]
    text = "Falamos sobre Sete Montes e vixe  maria que aula."
    result = enforce_terminology(text, rules)
    assert "7 Montes" in result
    assert "caramba" in result

def test_parse_srt_blocks():
    srt = "1\n00:00:00,000 --> 00:00:01,000\n<b>Bold</b> text\n\n"
    blocks = _parse_srt_blocks(srt)
    assert blocks == ["Bold text"]

def test_handle_simple_repetition():
    prev = "O reino de Deus"
    curr = "O reino de Deus está próximo"
    result = _handle_simple_repetition(prev, curr)
    assert result == "está próximo"

def test_handle_partial_overlap():
    prev = "Linha 1\nLinha 2"
    curr = "Linha 2\nLinha 3"
    prev_lines = [line.strip() for line in prev.split('\n') if line.strip()]
    curr_lines = [line.strip() for line in curr.split('\n') if line.strip()]
    result = _handle_partial_overlap(prev_lines, curr_lines)
    assert result == ["Linha 3"]
