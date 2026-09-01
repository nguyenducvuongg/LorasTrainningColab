import os
import re
from typing import List, Set, Optional, Dict
from ..core.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_BLACKLIST_TAGS = {
    "monochrome", "greyscale", "grayscale", "lowres", "low quality", "worst quality",
    "bad anatomy", "bad hands", "error", "missing fingers", "extra digit",
    "fewer digits", "cropped", "jpeg artifacts", "signature", "watermark",
    "username", "blurry", "artist name", "censored"
}

class CaptionCleaner:
    """Cleans, formats, dedupes, and prepends trigger words to caption text."""

    @staticmethod
    def clean_tag_list(
        tags: List[str],
        trigger_word: Optional[str] = None,
        blacklist: Optional[Set[str]] = None,
        replacements: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Cleans a list of comma-separated tags, filters blacklist, applies replacements,
        and prepends the trigger word.
        """
        active_blacklist = (blacklist or DEFAULT_BLACKLIST_TAGS)
        active_blacklist = {t.lower().strip().replace("_", " ") for t in active_blacklist}
        active_replacements = replacements or {}

        seen = set()
        cleaned_tags: List[str] = []

        if trigger_word and trigger_word.strip():
            cleaned_trigger = trigger_word.strip()
            cleaned_tags.append(cleaned_trigger)
            seen.add(cleaned_trigger.lower())

        for raw_tag in tags:
            tag = raw_tag.strip().replace("_", " ")
            if not tag:
                continue

            # Apply custom replacements
            if tag in active_replacements:
                tag = active_replacements[tag]

            tag_lower = tag.lower()

            if tag_lower in active_blacklist:
                continue

            if tag_lower not in seen:
                seen.add(tag_lower)
                cleaned_tags.append(tag)

        return ", ".join(cleaned_tags)

    @classmethod
    def clean_text(
        cls,
        text: str,
        trigger_word: Optional[str] = None,
        blacklist: Optional[Set[str]] = None,
        is_danbooru_tags: bool = True
    ) -> str:
        """Cleans either comma-separated tags or natural language paragraphs."""
        if is_danbooru_tags:
            tags = [t.strip() for t in text.split(",") if t.strip()]
            return cls.clean_tag_list(tags, trigger_word=trigger_word, blacklist=blacklist)
        else:
            # Natural language mode (JoyCaption / Gemini / DeepSeek)
            cleaned = text.strip()
            # Remove excessive newlines/spaces
            cleaned = re.sub(r"\s+", " ", cleaned)
            if trigger_word and trigger_word.strip():
                tw = trigger_word.strip()
                if not cleaned.lower().startswith(tw.lower()):
                    cleaned = f"{tw}, {cleaned}"
            return cleaned

    @classmethod
    def clean_dataset_directory(
        cls,
        dataset_dir: str,
        trigger_word: Optional[str] = None,
        blacklist: Optional[Set[str]] = None,
        is_danbooru_tags: bool = True
    ):
        """Processes and cleans all .txt files in a dataset directory."""
        count = 0
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith(".txt"):
                    txt_path = os.path.join(root, file)
                    with open(txt_path, "r", encoding="utf-8") as f:
                        raw_content = f.read()
                    
                    cleaned_content = cls.clean_text(
                        raw_content,
                        trigger_word=trigger_word,
                        blacklist=blacklist,
                        is_danbooru_tags=is_danbooru_tags
                    )
                    
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(cleaned_content)
                    count += 1
                    
        logger.info(f"Cleaned {count} caption files in {dataset_dir}")
