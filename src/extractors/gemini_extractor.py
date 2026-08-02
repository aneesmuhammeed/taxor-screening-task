"""Gemini 3.6 Flash extractor using the Google GenAI SDK."""

import os
import logging
from pathlib import Path

from google import genai
from google.genai import types

from extractors.base import BaseExtractor, EXTRACTION_PROMPT, get_image_media_type
from schema import TokenUsage

logger = logging.getLogger(__name__)


class GeminiExtractor(BaseExtractor):
    """Extract bill data using Google's Gemini 3.6 Flash."""

    MODEL_ID = "gemini-3.6-flash"

    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment")
        self.client = genai.Client(api_key=api_key)

    @property
    def model_name(self) -> str:
        return "gemini-3.6-flash"

    def _call_api(self, image_path: str) -> tuple[str, TokenUsage]:
        # Read image bytes and send inline
        image_bytes = Path(image_path).read_bytes()
        media_type = get_image_media_type(image_path)

        response = self.client.models.generate_content(
            model=self.MODEL_ID,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=media_type),
                EXTRACTION_PROMPT,
            ],
        )

        raw_text = response.text

        # Extract token usage from usage_metadata
        usage_meta = response.usage_metadata
        usage = TokenUsage(
            input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
        )

        logger.info(
            f"[Gemini] Tokens: {usage.input_tokens} in / {usage.output_tokens} out"
        )
        return raw_text, usage
