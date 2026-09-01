import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from lora_colab.dataset.captioning.gemini_api import SYSTEM_PROMPTS

def test_captioning_prompts():
    assert "character" in SYSTEM_PROMPTS
    assert "style" in SYSTEM_PROMPTS
    assert "skin_enhancement" in SYSTEM_PROMPTS
    assert len(SYSTEM_PROMPTS["character"]) > 20
