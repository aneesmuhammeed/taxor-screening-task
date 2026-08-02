"""Base extractor interface and shared extraction prompt.

All provider-specific extractors inherit from BaseExtractor and use the
same EXTRACTION_PROMPT for a fair comparison across models.
"""

import base64
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path

from schema import BillExtraction, ExtractionResult, TokenUsage

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Shared prompt — identical across all models for fair comparison
# --------------------------------------------------------------------------- #
EXTRACTION_PROMPT = """You are an expert at reading Indian bills and receipts, including handwritten ones.

Extract the following fields from the attached bill image and return ONLY a JSON object (no markdown fences, no extra text):

{
  "vendor_name": "Name of the shop/vendor",
  "invoice_number": "Bill/invoice number, or null if not visible",
  "date": "Date in YYYY-MM-DD format if you can parse it, or null",
  "date_raw": "Date exactly as written on the bill, or null if no date visible",
  "amount": 123.45,
  "currency": "INR",
  "gst_details": "GSTIN number, tax amount, tax rate if visible, or null"
}

Rules:
- Return ONLY valid JSON, no explanation or markdown.
- If a field is not visible or not present, set it to null.
- For amount, use a number (float), not a string.
- For currency, default to "INR" unless the bill explicitly shows another currency.
- For gst_details, include any GST-related info you can read: GSTIN number, CGST/SGST amounts, tax rate percentages. If none visible, set to null.
- For date, try to parse into YYYY-MM-DD even if written as DD/MM/YYYY or DD-MM-YY. Indian bills typically use DD/MM/YYYY format.
- Be accurate — do not guess. If handwriting is illegible for a field, set it to null rather than guessing."""


def load_image_as_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded content."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """Infer MIME type from file extension."""
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    return mime_map.get(ext, "image/png")


def parse_model_json(raw_text: str) -> dict | None:
    """Try to parse JSON from model output, handling markdown fences and junk.

    Returns the parsed dict or None if parsing fails entirely.
    """
    # Strip markdown code fences if present
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the text
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


class BaseExtractor(ABC):
    """Abstract base for all LLM bill extractors."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier, e.g. 'claude-3.5-sonnet'."""
        ...

    @abstractmethod
    def _call_api(self, image_path: str) -> tuple[str, TokenUsage]:
        """Send the image + prompt to the provider and return (raw_text, token_usage).

        Subclasses implement provider-specific API calls here.
        """
        ...

    def extract(self, image_path: str) -> ExtractionResult:
        """Run extraction on a bill image. Returns ExtractionResult always —
        parse failures are recorded, never silently dropped.
        """
        start = time.time()

        try:
            raw_text, token_usage = self._call_api(image_path)
        except Exception as e:
            # API call itself failed
            logger.error(f"[{self.model_name}] API call failed for {image_path}: {e}")
            return ExtractionResult(
                model_name=self.model_name,
                image_file=Path(image_path).name,
                extraction=BillExtraction(
                    vendor_name="EXTRACTION_FAILED",
                    raw_model_output=str(e),
                ),
                token_usage=TokenUsage(),
                latency_seconds=time.time() - start,
                parse_success=False,
                parse_error=f"API error: {e}",
            )

        elapsed = time.time() - start

        # Try to parse the model's JSON output into our schema
        parsed = parse_model_json(raw_text)

        if parsed is None:
            logger.warning(
                f"[{self.model_name}] Failed to parse JSON from response for {image_path}. "
                f"Raw output: {raw_text[:200]}..."
            )
            return ExtractionResult(
                model_name=self.model_name,
                image_file=Path(image_path).name,
                extraction=BillExtraction(
                    vendor_name="PARSE_FAILED",
                    raw_model_output=raw_text,
                ),
                token_usage=token_usage,
                latency_seconds=elapsed,
                parse_success=False,
                parse_error="Could not parse JSON from model output",
            )

        # Build BillExtraction from parsed dict, preserving raw output
        parsed["raw_model_output"] = raw_text
        try:
            extraction = BillExtraction(**parsed)
        except Exception as e:
            logger.warning(
                f"[{self.model_name}] Pydantic validation failed for {image_path}: {e}"
            )
            return ExtractionResult(
                model_name=self.model_name,
                image_file=Path(image_path).name,
                extraction=BillExtraction(
                    vendor_name=parsed.get("vendor_name", "VALIDATION_FAILED"),
                    raw_model_output=raw_text,
                ),
                token_usage=token_usage,
                latency_seconds=elapsed,
                parse_success=False,
                parse_error=f"Pydantic validation: {e}",
            )

        return ExtractionResult(
            model_name=self.model_name,
            image_file=Path(image_path).name,
            extraction=extraction,
            token_usage=token_usage,
            latency_seconds=elapsed,
            parse_success=True,
        )
