"""Gemini extractor using the google-genai SDK."""

import os
import logging
from pathlib import Path

from google import genai
from google.genai import types

from extractors.base import BaseExtractor, EXTRACTION_PROMPT, get_image_media_type
from schema import TokenUsage

logger = logging.getLogger(__name__)

class GeminiExtractor(BaseExtractor):
    """Extract bill data using Gemini 3.6 Flash."""

    MODEL_ID = "gemini-3.6-flash"

    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment")
        self.client = genai.Client(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self.MODEL_ID

    def _call_api(self, image_path: str) -> tuple[str, TokenUsage]:
        media_type = get_image_media_type(image_path)
        image_bytes = Path(image_path).read_bytes()

        response = self.client.models.generate_content(
            model=self.MODEL_ID,
            contents=[
                EXTRACTION_PROMPT,
                types.Part.from_bytes(data=image_bytes, mime_type=media_type),
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
            )
        )

        raw_text = response.text
        usage = TokenUsage(
            input_tokens=response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
            output_tokens=response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
        )

        logger.info(
            f"[Gemini] Tokens: {usage.input_tokens} in / {usage.output_tokens} out"
        )
        return raw_text, usage
