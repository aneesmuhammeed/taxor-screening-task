"""OpenRouter extractor using the OpenAI SDK."""

import os
import logging

import openai

from extractors.base import (
    BaseExtractor,
    EXTRACTION_PROMPT,
    load_image_as_base64,
    get_image_media_type,
)
from schema import TokenUsage

logger = logging.getLogger(__name__)


class OpenRouterExtractor(BaseExtractor):
    """Extract bill data using OpenRouter's OpenAI GPT-4o."""

    MODEL_ID = "openai/gpt-4o"

    def __init__(self):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment")
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    @property
    def model_name(self) -> str:
        return "openai/gpt-4o"

    def _call_api(self, image_path: str) -> tuple[str, TokenUsage]:
        image_b64 = load_image_as_base64(image_path)
        media_type = get_image_media_type(image_path)

        response = self.client.chat.completions.create(
            model=self.MODEL_ID,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}",
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

        raw_text = response.choices[0].message.content
        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

        logger.info(
            f"[OpenRouter] Tokens: {usage.input_tokens} in / {usage.output_tokens} out"
        )
        return raw_text, usage
