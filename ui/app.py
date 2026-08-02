"""Streamlit UI: Upload a bill image, run all extractors, compare results side by side.

Usage:
    streamlit run ui/app.py
"""

import json
import sys
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()


st.set_page_config(page_title="Bill Extractor Comparison", layout="wide")

st.title("🧾 Bill Extraction — Model Comparison")
st.caption("Upload a handwritten Indian bill and compare extraction results across LLMs")


@st.cache_resource
def get_extractors():
    """Load available extractors (cached so we don't re-init on every rerun)."""
    extractors = []
    errors = []

    try:
        from extractors.gemini_extractor import GeminiExtractor
        extractors.append(GeminiExtractor())
    except (ValueError, ImportError) as e:
        errors.append(f"Gemini: {e}")

    try:
        from extractors.minimax_extractor import MinimaxExtractor
        extractors.append(MinimaxExtractor())
    except (ValueError, ImportError) as e:
        errors.append(f"Minimax: {e}")

    try:
        from extractors.mistral_extractor import MistralExtractor
        extractors.append(MistralExtractor())
    except (ValueError, ImportError) as e:
        errors.append(f"Mistral: {e}")

    try:
        from extractors.nemotron_extractor import NemotronExtractor
        extractors.append(NemotronExtractor())
    except (ValueError, ImportError) as e:
        errors.append(f"Nemotron: {e}")

    return extractors, errors


def load_ground_truth():
    """Load ground truth if available."""
    gt_path = Path("data/ground_truth.json")
    if gt_path.exists():
        data = json.loads(gt_path.read_text())
        return {entry["image_file"]: entry for entry in data}
    return {}


# --- Sidebar --- #
st.sidebar.header("⚙️ Configuration")
extractors, init_errors = get_extractors()

if extractors:
    st.sidebar.success(f"{len(extractors)} model(s) loaded")
    for ext in extractors:
        st.sidebar.write(f"  ✓ {ext.model_name}")
else:
    st.sidebar.error("No models loaded! Set API keys in .env")

if init_errors:
    with st.sidebar.expander("Skipped models"):
        for err in init_errors:
            st.write(f"✗ {err}")


# --- Main content --- #
uploaded_file = st.file_uploader(
    "Upload a bill image", type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_file and extractors:
    # Save uploaded file temporarily
    temp_path = Path("data/bills") / uploaded_file.name
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(uploaded_file.getvalue())

    col_img, col_results = st.columns([1, 2])

    with col_img:
        st.image(uploaded_file, caption="Uploaded Bill", use_container_width=True)

    with col_results:
        # Load ground truth for comparison
        gt_data = load_ground_truth()
        gt = gt_data.get(uploaded_file.name)

        if gt:
            st.info(f"Ground truth found for {uploaded_file.name}")

        # Run all extractors
        results = {}
        with st.spinner("Running extractors..."):
            for ext in extractors:
                result = ext.extract(str(temp_path))
                results[ext.model_name] = result

        # Build comparison table
        fields = [
            "vendor_name", "invoice_number", "date", "date_raw",
            "amount", "currency", "gst_details"
        ]

        # Import scorer for highlighting
        from eval.scorer import score_extraction

        st.subheader("📊 Extraction Results")

        # Header row
        cols = st.columns(1 + len(results) + (1 if gt else 0))
        cols[0].markdown("**Field**")
        col_idx = 1
        if gt:
            cols[col_idx].markdown("**Ground Truth**")
            col_idx += 1
        for model_name in results:
            cols[col_idx].markdown(f"**{model_name}**")
            col_idx += 1

        # Data rows with color highlighting
        for field in fields:
            cols = st.columns(1 + len(results) + (1 if gt else 0))
            cols[0].write(field)
            col_idx = 1

            if gt:
                gt_val = gt.get(field, "—")
                cols[col_idx].write(str(gt_val) if gt_val is not None else "null")
                col_idx += 1

            for model_name, result in results.items():
                ext_dict = result.extraction.model_dump()
                val = ext_dict.get(field)
                display = str(val) if val is not None else "null"

                if gt and field != "date_raw":
                    # Score this field
                    scores = score_extraction(ext_dict, gt)
                    if field == "gst_details":
                        correct = scores["gst_details"]["presence_correct"]
                    elif field in scores:
                        correct = scores[field].get("correct")
                    else:
                        correct = None

                    if correct is True:
                        cols[col_idx].markdown(f"🟢 {display}")
                    elif correct is False:
                        cols[col_idx].markdown(f"🔴 {display}")
                    else:
                        cols[col_idx].write(display)
                else:
                    cols[col_idx].write(display)
                col_idx += 1

        # Cost and latency summary
        st.subheader("💰 Cost & Performance")
        from eval.cost_tracker import compute_cost

        cost_cols = st.columns(len(results))
        for i, (model_name, result) in enumerate(results.items()):
            cost = compute_cost(
                model_name,
                result.token_usage.input_tokens,
                result.token_usage.output_tokens,
            )
            with cost_cols[i]:
                st.metric(model_name, f"${cost:.6f}", f"{result.latency_seconds:.1f}s")
                st.caption(
                    f"Tokens: {result.token_usage.input_tokens} in / "
                    f"{result.token_usage.output_tokens} out"
                )

        # Raw model outputs (expandable)
        st.subheader("🔍 Raw Model Outputs")
        for model_name, result in results.items():
            with st.expander(f"{model_name} — raw output"):
                st.code(result.extraction.raw_model_output, language="json")
                if not result.parse_success:
                    st.error(f"Parse error: {result.parse_error}")

elif not extractors:
    st.warning(
        "No models loaded. Set API keys in your `.env` file:\n"
        "- `GOOGLE_API_KEY`\n"
        "- `MINIMAX_API_KEY`\n"
        "- `MISTRAL_API_KEY`\n"
        "- `NVIDIA_API_KEY`"
    )
