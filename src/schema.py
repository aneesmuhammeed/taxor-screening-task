"""Pydantic models for bill extraction results and token usage tracking."""

from pydantic import BaseModel, Field
from typing import Optional


class BillExtraction(BaseModel):
    """Structured data extracted from a bill/receipt image."""

    vendor_name: str = Field(description="Name of the vendor/shop")
    invoice_number: Optional[str] = Field(
        default=None, description="Invoice or bill number, null if not present"
    )
    date: Optional[str] = Field(
        default=None, description="Date in ISO format (YYYY-MM-DD) if parseable"
    )
    date_raw: Optional[str] = Field(
        default=None, description="Date exactly as it appears on the bill"
    )
    amount: Optional[float] = Field(
        default=None, description="Total amount on the bill"
    )
    currency: str = Field(
        default="INR", description="Currency code, defaults to INR"
    )
    gst_details: Optional[str] = Field(
        default=None,
        description="Free text: GSTIN number, tax amount, tax rate if visible",
    )
    raw_model_output: str = Field(
        default="",
        description="Full raw text the model returned before parsing, for debugging",
    )


class TokenUsage(BaseModel):
    """Token counts from a single API call, for cost computation."""

    input_tokens: int = 0
    output_tokens: int = 0


class ExtractionResult(BaseModel):
    """Wraps a BillExtraction with metadata about the extraction run."""

    model_name: str
    image_file: str
    extraction: BillExtraction
    token_usage: TokenUsage
    latency_seconds: float = Field(description="Wall-clock time for the API call")
    parse_success: bool = Field(
        default=True, description="False if model output couldn't be parsed into schema"
    )
    parse_error: Optional[str] = Field(
        default=None, description="Error message if parsing failed"
    )
