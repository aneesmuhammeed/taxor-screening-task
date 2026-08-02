"""Zoho Books API client — OAuth2 self-client flow + expense creation.

Uses refresh token to get access tokens. Requires these env vars:
- ZOHO_CLIENT_ID
- ZOHO_CLIENT_SECRET
- ZOHO_REFRESH_TOKEN
- ZOHO_ORG_ID
- ZOHO_DOMAIN (default: .com)
"""

import json
import logging
import os
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


class ZohoClient:
    """Handles Zoho Books OAuth2 and expense creation."""

    def __init__(self):
        self.client_id = os.environ.get("ZOHO_CLIENT_ID")
        self.client_secret = os.environ.get("ZOHO_CLIENT_SECRET")
        self.refresh_token = os.environ.get("ZOHO_REFRESH_TOKEN")
        self.org_id = os.environ.get("ZOHO_ORG_ID")
        self.domain = os.environ.get("ZOHO_DOMAIN", ".com")

        missing = []
        if not self.client_id:
            missing.append("ZOHO_CLIENT_ID")
        if not self.client_secret:
            missing.append("ZOHO_CLIENT_SECRET")
        if not self.refresh_token:
            missing.append("ZOHO_REFRESH_TOKEN")
        if not self.org_id:
            missing.append("ZOHO_ORG_ID")

        if missing:
            raise ValueError(
                f"Missing Zoho env vars: {', '.join(missing)}. "
                "See .env.example for required variables."
            )

        self.access_token: str | None = None
        self.base_url = f"https://www.zohoapis{self.domain}/books/v3"
        self.auth_url = f"https://accounts.zoho{self.domain}/oauth/v2/token"

    def _refresh_access_token(self):
        """Exchange refresh token for a new access token."""
        logger.info("Refreshing Zoho access token...")

        resp = requests.post(
            self.auth_url,
            params={
                "refresh_token": self.refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "refresh_token",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        if "access_token" not in data:
            raise RuntimeError(f"Zoho token refresh failed: {data}")

        self.access_token = data["access_token"]
        logger.info("Zoho access token refreshed successfully")

    def _headers(self) -> dict:
        """Auth headers for API requests."""
        if not self.access_token:
            self._refresh_access_token()
        return {
            "Authorization": f"Zoho-oauthtoken {self.access_token}",
            "Content-Type": "application/json",
        }

    def create_expense(self, extraction: dict) -> dict:
        """Create an expense in Zoho Books from an extraction dict.

        Maps BillExtraction fields to Zoho expense fields:
        - vendor_name -> description (or vendor lookup)
        - amount -> amount
        - date -> date
        - gst_details -> notes (custom_fields if available)

        Returns the Zoho API response dict.
        """
        # Build the expense payload
        expense_data = {
            "account_id": "",  # User needs to set this — first expense account
            "date": extraction.get("date") or "2025-01-01",
            "amount": extraction.get("amount") or 0,
            "description": f"Bill from {extraction.get('vendor_name', 'Unknown')}",
            "reference_number": extraction.get("invoice_number") or "",
            "notes": "",
        }

        # Add GST details to notes if present
        notes_parts = []
        if extraction.get("gst_details"):
            notes_parts.append(f"GST: {extraction['gst_details']}")
        if extraction.get("currency") and extraction["currency"] != "INR":
            notes_parts.append(f"Currency: {extraction['currency']}")
        expense_data["notes"] = "; ".join(notes_parts)

        logger.info(
            f"Creating Zoho expense: {expense_data['description']} "
            f"amount={expense_data['amount']}"
        )

        resp = requests.post(
            f"{self.base_url}/expenses",
            headers=self._headers(),
            params={"organization_id": self.org_id},
            data=json.dumps(expense_data),
        )

        # If 401, try refreshing token once
        if resp.status_code == 401:
            logger.warning("Got 401, refreshing token and retrying...")
            self._refresh_access_token()
            resp = requests.post(
                f"{self.base_url}/expenses",
                headers=self._headers(),
                params={"organization_id": self.org_id},
                data=json.dumps(expense_data),
            )

        resp.raise_for_status()
        result = resp.json()

        logger.info(f"Zoho expense created: {result.get('expense', {}).get('expense_id', 'unknown')}")
        return result


def push_extractions_to_zoho(
    extractions: list[dict],
    output_path: str = "results/zoho_expenses.json",
) -> list[dict]:
    """Push a list of extractions to Zoho Books and log results.

    Args:
        extractions: list of BillExtraction dicts to push
        output_path: where to write the response log

    Returns:
        list of Zoho API responses
    """
    try:
        client = ZohoClient()
    except ValueError as e:
        logger.error(f"Skipping Zoho push: {e}")
        return []

    results = []
    for i, ext in enumerate(extractions):
        try:
            result = client.create_expense(ext)
            results.append({
                "bill": ext.get("vendor_name", f"bill_{i}"),
                "status": "success",
                "response": result,
            })
        except Exception as e:
            logger.error(f"Failed to create expense for bill {i}: {e}")
            results.append({
                "bill": ext.get("vendor_name", f"bill_{i}"),
                "status": "error",
                "error": str(e),
            })

    # Write results log
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    logger.info(f"Zoho expense log written to {output_path}")

    return results
