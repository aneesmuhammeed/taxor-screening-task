"""Cost tracking: compute per-bill and extrapolated costs from actual token usage.

Pricing is pulled from each provider's published rates. The date and source URL
are noted so these can be updated when pricing changes.
"""

# --------------------------------------------------------------------------- #
# Pricing constants — UPDATE THESE when provider pricing changes
# --------------------------------------------------------------------------- #
# All prices are per 1 million tokens (USD)
# Last updated: 2025-01-15

PRICING = {
    "claude-sonnet-4-20250514": {
        "input_per_1m": 3.00,    # $3.00 per 1M input tokens
        "output_per_1m": 15.00,  # $15.00 per 1M output tokens
        "source": "https://www.anthropic.com/pricing",
        "date": "2025-01-15",
    },
    "gemini-3.6-flash": {
        "input_per_1m": 0.075,   # $0.075 per 1M input tokens
        "output_per_1m": 0.30,   # $0.30 per 1M output tokens
        "source": "https://ai.google.dev/pricing",
        "date": "2025-01-15",
    },
    "openai/gpt-4o": {
        "input_per_1m": 2.50,    # $2.50 per 1M input tokens
        "output_per_1m": 10.00,  # $10.00 per 1M output tokens
        "source": "https://openrouter.ai/models/openai/gpt-4o",
        "date": "2025-01-15",
    },
    "pixtral-12b-2409": {
        "input_per_1m": 0.15,    # $0.15 per 1M input tokens
        "output_per_1m": 0.15,   # $0.15 per 1M output tokens
        "source": "https://mistral.ai/technology/#pricing",
        "date": "2025-01-15",
    },
    "qwen/qwen-2-vl-7b-instruct:free": {
        "input_per_1m": 0.00,    # 100% Free via OpenRouter
        "output_per_1m": 0.00,   # 100% Free via OpenRouter
        "source": "https://openrouter.ai/models/qwen/qwen-2-vl-7b-instruct:free",
        "date": "2025-01-15",
    },
}


def compute_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Compute the USD cost for a single API call.

    Args:
        model_name: must match a key in PRICING
        input_tokens: number of input/prompt tokens
        output_tokens: number of output/completion tokens

    Returns:
        Cost in USD (float)

    Raises:
        KeyError: if model_name not found in PRICING
    """
    if model_name not in PRICING:
        raise KeyError(
            f"No pricing data for model '{model_name}'. "
            f"Known models: {list(PRICING.keys())}"
        )

    rates = PRICING[model_name]
    input_cost = (input_tokens / 1_000_000) * rates["input_per_1m"]
    output_cost = (output_tokens / 1_000_000) * rates["output_per_1m"]
    return input_cost + output_cost


def extrapolate_cost(per_bill_costs: list[float], n: int = 100) -> float:
    """Extrapolate total cost for n bills based on observed per-bill costs.

    Uses the mean of observed costs. Returns 0.0 if no costs provided.
    """
    if not per_bill_costs:
        return 0.0
    avg = sum(per_bill_costs) / len(per_bill_costs)
    return avg * n


def format_cost_table_row(
    model_name: str, per_bill_costs: list[float]
) -> dict:
    """Build a summary row for the cost results table.

    Returns dict with: model, avg_cost_per_bill, total_cost_100_bills, num_bills
    """
    if not per_bill_costs:
        return {
            "model": model_name,
            "avg_cost_per_bill_usd": 0.0,
            "total_cost_100_bills_usd": 0.0,
            "num_bills_evaluated": 0,
        }

    avg = sum(per_bill_costs) / len(per_bill_costs)
    return {
        "model": model_name,
        "avg_cost_per_bill_usd": round(avg, 6),
        "total_cost_100_bills_usd": round(avg * 100, 4),
        "num_bills_evaluated": len(per_bill_costs),
    }
