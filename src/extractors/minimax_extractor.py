"""Minimaxai extractor using the OpenAI SDK (NIM compatible)."""

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


class MinimaxExtractor(BaseExtractor):
    """Extract bill data using minimaxai/minimax-m3."""

    MODEL_ID = "minimaxai/minimax-m3"

    def __init__(self):
        api_key = os.environ.get("MINIMAX_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("MINIMAX_API_KEY or NVIDIA_API_KEY not set in environment")
            
        base_url = os.environ.get("MINIMAX_BASE_URL", "https://integrate.api.nvidia.com/v1")
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    @property
    def model_name(self) -> str:
        return self.MODEL_ID

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
                            "type": "text",
                            "text": EXTRACTION_PROMPT,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}",
                            },
                        },
                    ],
                }
            ],
        )

        raw_text = response.choices[0].message.content
        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

        logger.info(
            f"[Minimax] Tokens: {usage.input_tokens} in / {usage.output_tokens} out"
        )
        return raw_text, usage
