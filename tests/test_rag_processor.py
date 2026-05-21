import pytest
import os
import sys
from unittest.mock import MagicMock

# Mock google.genai and dotenv before importing rag_processor
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['dotenv'] = MagicMock()

os.environ['GEMINI_API_KEY'] = 'dummy_key'

# Add parent directory to path so we can import from rag_processor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_processor import format_duration

def test_format_duration_seconds():
    assert format_duration(30.0) == "30.00 segundos"
    assert format_duration(59.99) == "59.99 segundos"
    assert format_duration(0.0) == "0.00 segundos"

def test_format_duration_minutes():
    assert format_duration(60.0) == "1m 0s"
    assert format_duration(90.0) == "1m 30s"
    assert format_duration(3599.0) == "59m 59s"

def test_format_duration_hours():
    assert format_duration(3600.0) == "1h 0m 0s"
    assert format_duration(3665.0) == "1h 1m 5s"
    assert format_duration(7200.0) == "2h 0m 0s"
    assert format_duration(7325.0) == "2h 2m 5s"
