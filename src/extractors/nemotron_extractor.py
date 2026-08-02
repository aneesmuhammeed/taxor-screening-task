"""NVIDIA Nemotron OCR v2 extractor using a 2-step pipeline."""

import os
import logging
import base64
import requests
import json
import openai

from extractors.base import (
    BaseExtractor,
    EXTRACTION_PROMPT,
    load_image_as_base64,
    get_image_media_type,
)
from schema import TokenUsage

logger = logging.getLogger(__name__)


class NemotronExtractor(BaseExtractor):
    """Extract bill data using NVIDIA's nemotron-ocr-v2 and a text LLM for formatting."""

    MODEL_ID = "nvidia/nemotron-ocr-v2"
    TEXT_MODEL_ID = "meta/llama-3.1-8b-instruct"

    def __init__(self):
        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not set in environment")
        
        # Client for the 2nd step (formatting the OCR text)
        self.text_client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )

    @property
    def model_name(self) -> str:
        return self.MODEL_ID

    def _call_api(self, image_path: str) -> tuple[str, TokenUsage]:
        image_b64 = load_image_as_base64(image_path)
        media_type = get_image_media_type(image_path)

        # STEP 1: Get raw OCR text from Nemotron-OCR
        invoke_url = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        payload = {
            "input": [
                {
                    "type": "image_url",
                    "url": f"data:{media_type};base64,{image_b64}"
                }
            ]
        }
        
        logger.info("[Nemotron] Calling OCR endpoint...")
        response = requests.post(invoke_url, headers=headers, json=payload)
        response.raise_for_status()
        
        ocr_result = response.json()
        
        # The OCR result is structured. We dump it to string so the text LLM can read it.
        raw_ocr_text = json.dumps(ocr_result)
        
        # STEP 2: Use a fast text model to format it as JSON
        logger.info(f"[Nemotron] Formatting OCR output using {self.TEXT_MODEL_ID}...")
        
        # Modify the prompt slightly to work with text instead of an image
        modified_prompt = EXTRACTION_PROMPT.replace("attached bill image", "provided OCR text of a bill")
        modified_prompt += f"\n\nHere is the raw OCR data extracted from the image:\n{raw_ocr_text}"

        chat_response = self.text_client.chat.completions.create(
            model=self.TEXT_MODEL_ID,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": modified_prompt,
                }
            ],
        )

        final_json_text = chat_response.choices[0].message.content
        usage = TokenUsage(
            input_tokens=chat_response.usage.prompt_tokens if chat_response.usage else 0,
            output_tokens=chat_response.usage.completion_tokens if chat_response.usage else 0,
        )

        logger.info(
            f"[Nemotron] Tokens: {usage.input_tokens} in / {usage.output_tokens} out"
        )
        return final_json_text, usage
