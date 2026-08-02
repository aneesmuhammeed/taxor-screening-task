"""Main evaluation orchestrator.

Loops over all bill images, runs all configured extractors, scores against
ground truth, tracks costs, and writes results files.

Usage:
    python src/run_eval.py              # Run evaluation only
    python src/run_eval.py --zoho       # Also push best-model results to Zoho Books
    python src/run_eval.py --bills-dir data/bills --gt data/ground_truth.json
"""

import argparse
import json
import logging
import time
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Add src to path so imports work when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from schema import ExtractionResult
from eval.scorer import score_extraction
from eval.cost_tracker import compute_cost, format_cost_table_row

logger = logging.getLogger(__name__)


def setup_logging(results_dir: str = "results"):
    """Configure logging after ensuring the results directory exists."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f"{results_dir}/eval.log", mode="w", encoding="utf-8"
            ),
        ],
    )


def load_extractors() -> list:
    """Try to initialize each extractor. Skip any whose API key is missing."""
    extractors = []

    try:
        from extractors.gemini_extractor import GeminiExtractor
        extractors.append(GeminiExtractor())
        logger.info("[OK] Gemini extractor loaded")
    except (ValueError, ImportError) as e:
        logger.warning(f"[SKIP] Gemini extractor skipped: {e}")

    try:
        from extractors.nemotron_extractor import NemotronExtractor
        extractors.append(NemotronExtractor())
        logger.info("[OK] Nemotron extractor loaded")
    except (ValueError, ImportError) as e:
        logger.warning(f"[SKIP] Nemotron extractor skipped: {e}")

    try:
        from extractors.minimax_extractor import MinimaxExtractor
        extractors.append(MinimaxExtractor())
        logger.info("[OK] Minimax extractor loaded")
    except (ValueError, ImportError) as e:
        logger.warning(f"[SKIP] Minimax extractor skipped: {e}")



    return extractors


def run_evaluation(bills_dir: Path, gt_path: Path, results_dir: Path):
    """Core evaluation loop: extract → score → track costs → write results."""

    # Load ground truth
    ground_truth_list = json.loads(gt_path.read_text())
    gt_by_file = {gt["image_file"]: gt for gt in ground_truth_list}
    logger.info(f"Loaded {len(gt_by_file)} ground truth entries")

    # Find bill images
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    bill_images = sorted(
        f for f in bills_dir.iterdir()
        if f.suffix.lower() in image_extensions
    )
    logger.info(f"Found {len(bill_images)} bill images in {bills_dir}")

    if not bill_images:
        logger.error("No bill images found! Run 'python src/generate_bills.py' first.")
        sys.exit(1)

    # Load extractors
    extractors = load_extractors()
    if not extractors:
        logger.error(
            "No extractors loaded! Set at least one API key in .env file. "
            "See .env.example for required variables."
        )
        sys.exit(1)

    logger.info(f"Running {len(extractors)} models × {len(bill_images)} bills")
    print(f"\n{'='*60}")
    print(f"  EVALUATION: {len(extractors)} models × {len(bill_images)} bills")
    print(f"{'='*60}\n")

    # Storage for results
    all_extractions: list[dict] = []  # raw extractions for auditability
    all_scores: list[dict] = []       # per-bill, per-model scores
    cost_data: dict[str, list[float]] = {}  # model -> list of per-bill costs

    for idx, img_path in enumerate(bill_images):
        img_name = img_path.name
        gt = gt_by_file.get(img_name)

        if gt is None:
            logger.warning(f"No ground truth for {img_name}, skipping scoring")

        for extractor in extractors:
            model = extractor.model_name
            # Avoid Gemini free tier rate limits (5 RPM)
            if idx > 0 and "gemini" in model.lower():
                time.sleep(12)

            print(f"  [{model}] Processing {img_name}...", end=" ", flush=True)

            # Run extraction
            result: ExtractionResult = extractor.extract(str(img_path))

            # Store raw extraction
            raw_entry = {
                "image_file": img_name,
                "model": model,
                "extraction": result.extraction.model_dump(),
                "token_usage": result.token_usage.model_dump(),
                "latency_seconds": result.latency_seconds,
                "parse_success": result.parse_success,
                "parse_error": result.parse_error,
            }
            all_extractions.append(raw_entry)

            # Compute cost
            cost = compute_cost(
                model, result.token_usage.input_tokens, result.token_usage.output_tokens
            )
            if model not in cost_data:
                cost_data[model] = []
            cost_data[model].append(cost)

            # Score against ground truth
            if gt is not None and result.parse_success:
                scores = score_extraction(result.extraction.model_dump(), gt)
                score_entry = {
                    "image_file": img_name,
                    "model": model,
                    "cost_usd": cost,
                    "latency_s": result.latency_seconds,
                    **{f"{field}_correct": s.get("correct") for field, s in scores.items() if field != "gst_details"},
                    **{f"{field}_score": s.get("score") for field, s in scores.items() if field != "gst_details"},
                    "gst_presence_correct": scores["gst_details"]["presence_correct"],
                    "gst_gstin_correct": scores["gst_details"]["gstin_correct"],
                }
                all_scores.append(score_entry)
                status = "[OK]" if result.parse_success else "[FAIL]"
            elif not result.parse_success:
                score_entry = {
                    "image_file": img_name,
                    "model": model,
                    "cost_usd": cost,
                    "latency_s": result.latency_seconds,
                    "parse_failed": True,
                }
                all_scores.append(score_entry)
                status = "[FAIL] (PARSE ERROR)"
            else:
                status = "? (no GT)"

            print(f"${cost:.6f} | {result.latency_seconds:.1f}s | {status}")

    # --- Write results files --- #
    results_dir.mkdir(parents=True, exist_ok=True)

    # Raw extractions (full auditability)
    raw_path = results_dir / "raw_extractions.json"
    raw_path.write_text(json.dumps(all_extractions, indent=2, default=str))
    logger.info(f"Raw extractions written to {raw_path}")

    # Build summary table
    if all_scores:
        df = pd.DataFrame(all_scores)
        _print_summary(df, cost_data, results_dir)
    else:
        logger.warning("No scores computed — check ground truth and API keys")

    return all_extractions, cost_data


def _print_summary(df: pd.DataFrame, cost_data: dict, results_dir: Path):
    """Compute and print per-model, per-field accuracy + cost summary."""

    scored_fields = [
        "vendor_name_correct",
        "invoice_number_correct",
        "date_correct",
        "amount_correct",
        "currency_correct",
        "gst_presence_correct",
        "gst_gstin_correct",
    ]

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}\n")

    summary_rows = []
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        # Exclude parse failures from accuracy calculation
        if "parse_failed" in model_df.columns:
            valid = model_df[model_df["parse_failed"] != True]
        else:
            valid = model_df

        row = {"model": model, "bills_evaluated": len(valid)}

        for field in scored_fields:
            if field in valid.columns:
                # Exclude None values (e.g., ambiguous dates) from denominator
                field_vals = valid[field].dropna()
                if len(field_vals) > 0:
                    accuracy = field_vals.mean() * 100
                    row[field] = f"{accuracy:.1f}%"
                    row[f"{field}_raw"] = accuracy
                else:
                    row[field] = "N/A"
                    row[f"{field}_raw"] = None
            else:
                row[field] = "N/A"
                row[f"{field}_raw"] = None

        # Parse failure rate
        total = len(model_df)
        if "parse_failed" in model_df.columns:
            failures = len(model_df[model_df["parse_failed"] == True])
        else:
            failures = 0
        row["parse_failures"] = f"{failures}/{total}"

        # Cost
        costs = cost_data.get(model, [])
        if costs:
            row["avg_cost_per_bill"] = f"${sum(costs)/len(costs):.6f}"
            row["cost_per_100_bills"] = f"${sum(costs)/len(costs)*100:.4f}"
        else:
            row["avg_cost_per_bill"] = "N/A"
            row["cost_per_100_bills"] = "N/A"

        summary_rows.append(row)

    # Print table
    display_fields = [
        "model", "bills_evaluated",
        "vendor_name_correct", "invoice_number_correct",
        "date_correct", "amount_correct", "currency_correct",
        "gst_presence_correct", "gst_gstin_correct",
        "parse_failures", "avg_cost_per_bill", "cost_per_100_bills",
    ]
    summary_df = pd.DataFrame(summary_rows)[display_fields]
    print(summary_df.to_string(index=False))

    # Write to CSV
    csv_path = results_dir / "results_table.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"\nResults table written to {csv_path}")

    # Also write detailed per-bill scores
    detail_path = results_dir / "detailed_scores.csv"
    df.to_csv(detail_path, index=False)
    logger.info(f"Detailed scores written to {detail_path}")

    print(f"\n  Results saved to: {results_dir}/")


def run_zoho_push(extractions: list[dict], model_name: str | None = None):
    """Push best-model extractions to Zoho Books."""
    from zoho.zoho_client import push_extractions_to_zoho

    # Filter to successful extractions from best model
    if model_name:
        filtered = [
            e["extraction"]
            for e in extractions
            if e["model"] == model_name and e["parse_success"]
        ]
    else:
        # Use the first model's results
        models = list(set(e["model"] for e in extractions))
        if models:
            filtered = [
                e["extraction"]
                for e in extractions
                if e["model"] == models[0] and e["parse_success"]
            ]
        else:
            filtered = []

    if not filtered:
        logger.warning("No valid extractions to push to Zoho")
        return

    # Push first 5
    to_push = filtered[:5]
    logger.info(f"Pushing {len(to_push)} extractions to Zoho Books...")
    push_extractions_to_zoho(to_push)


def main():
    parser = argparse.ArgumentParser(description="Run LLM bill extraction evaluation")
    parser.add_argument(
        "--bills-dir", default="data/bills", help="Directory containing bill images"
    )
    parser.add_argument(
        "--gt", default="data/ground_truth.json", help="Ground truth JSON file"
    )
    parser.add_argument(
        "--results-dir", default="results", help="Directory for output files"
    )
    parser.add_argument(
        "--zoho", action="store_true", help="Also push results to Zoho Books"
    )
    parser.add_argument(
        "--zoho-model", default=None, help="Model name to use for Zoho push (default: first model)"
    )
    args = parser.parse_args()

    # Load env vars
    load_dotenv()

    # Set up logging (creates results dir if needed)
    setup_logging(args.results_dir)

    bills_dir = Path(args.bills_dir)
    gt_path = Path(args.gt)

    if not bills_dir.exists():
        logger.error(f"Bills directory not found: {bills_dir}")
        logger.info("Run 'python src/generate_bills.py' to create synthetic bills")
        sys.exit(1)

    if not gt_path.exists():
        logger.error(f"Ground truth file not found: {gt_path}")
        logger.info("Run 'python src/generate_bills.py' to create ground truth")
        sys.exit(1)

    extractions, cost_data = run_evaluation(bills_dir, gt_path, Path(args.results_dir))

    if args.zoho:
        run_zoho_push(extractions, args.zoho_model)

    print("\nDone! Check results/ directory for output files.")


if __name__ == "__main__":
    main()
