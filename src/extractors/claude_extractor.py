"""Claude 3.5 Sonnet extractor using the Anthropic SDK."""

import os
import logging

import anthropic

from extractors.base import (
    BaseExtractor,
    EXTRACTION_PROMPT,
    load_image_as_base64,
    get_image_media_type,
)
from schema import TokenUsage

logger = logging.getLogger(__name__)


class ClaudeExtractor(BaseExtractor):
    """Extract bill data using Anthropic's Claude 3.5 Sonnet."""

    MODEL_ID = "claude-sonnet-4-20250514"

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        self.client = anthropic.Anthropic(api_key=api_key)

    @property
    def model_name(self) -> str:
        return "claude-sonnet-4-20250514"

    def _call_api(self, image_path: str) -> tuple[str, TokenUsage]:
        image_b64 = load_image_as_base64(image_path)
        media_type = get_image_media_type(image_path)

        response = self.client.messages.create(
            model=self.MODEL_ID,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": EXTRACTION_PROMPT,
                        },
                    ],
                }
            ],
        )

        raw_text = response.content[0].text
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        logger.info(
            f"[Claude] Tokens: {usage.input_tokens} in / {usage.output_tokens} out"
        )
        return raw_text, usage
