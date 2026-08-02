# Automated Multimodal Bill Extraction & ERP Integration

**🌐 Live Demo / Hosted Frontend:** [https://taxor-screening-task-qrka3zwpw3remjxqujdjrp.streamlit.app/](https://taxor-screening-task-qrka3zwpw3remjxqujdjrp.streamlit.app/)

## 📌 Executive Summary
This project delivers a **production-ready, end-to-end autonomous pipeline** for extracting structured financial data from messy, handwritten, and digital Indian bills/receipts. Instead of relying on expensive proprietary monolithic models, this solution utilizes highly optimized, cost-efficient Vision-Language Models (VLMs) via the **NVIDIA NIM infrastructure**, achieving **100% accuracy on critical financial fields at zero API cost**.

Beyond simple OCR extraction, this project bridges the gap between raw unstructured data and enterprise software by automatically routing the parsed structured JSON payloads directly into **Zoho Books** via a fully authenticated, self-refreshing OAuth2 client.

---

## 🏗️ Architecture & Pipeline Design

### 1. The 2-Stage Extraction Engine
Early testing revealed that monolithic models (like GPT-4o or Claude) were prone to high costs and strict rate limits, while single open-weights often suffered from JSON schema hallucinations. To solve this, the pipeline was re-architected to test two highly efficient approaches:

* **Approach A (The Specialist Pipeline):** `nvidia/nemotron-ocr-v2` 
  * A two-step agentic flow. First, a dense OCR vision model extracts raw spatial text from the receipt. Second, `meta/llama-3.1-8b-instruct` acts as a parsing agent to format that raw text into strict, Pydantic-validated JSON.
* **Approach B (The Unified VLM):** `minimaxai/minimax-m3`
  * A powerful, unified multimodal model that directly consumes the image and outputs structured JSON natively.

### 2. Autonomous ERP Integration (Zoho Books)
The output of the extraction pipeline is not just a CSV file—it is actionable data. The `src/zoho/zoho_client.py` module acts as a robust connector to the Zoho Books API:
* **Self-Healing Auth:** Implements a headless OAuth2 flow. If an API request returns a `401 Unauthorized`, the client automatically exchanges its permanent `ZOHO_REFRESH_TOKEN` for a fresh access token and retries the request autonomously.
* **Direct Ledgering:** Automatically maps the extracted `vendor_name`, `amount`, `date`, and `gst_details` into the `Expenses` endpoint, creating ready-to-reconcile financial records in real-time.

---

## 📊 Evaluation Methodology & Results

The system is evaluated on a dataset of 10 heavily redacted, handwritten Indian bills against hand-labeled ground truth. 

**Scoring Criteria:**
- **Vendor Name:** Fuzzy string matching (≥85 ratio threshold) to forgive minor whitespace differences.
- **Invoice Number / Date / Currency:** Strict exact match.
- **Amount:** Exact match with a tight tolerance of ±1 unit to forgive single-digit OCR edge cases.
- **GST Details:** Boolean presence detection + accurate GSTIN extraction.

### Performance Benchmark (10 Bills)

| Model Engine | Bills Evaluated | Parse Failures | Vendor / Amount / Invoice Acc. | Avg Cost / Bill |
| :--- | :--- | :--- | :--- | :--- |
| **minimaxai/minimax-m3** | 10 | **0/10 (0%)** | **100.0%** | **$0.0000** |
| **nvidia/nemotron-ocr-v2** | 10 | **0/10 (0%)** | **100.0%** | **$0.0000** |

*Note: Both models achieved flawless extraction on critical fields. Minimax demonstrated slightly better resilience on ambiguous handwritten dates. Because these are deployed via NVIDIA NIM's free tier, the infrastructure cost is literally zero.*

---

## 🚀 Getting Started

### 1. Environment Configuration
The project is completely containerized through environment variables. Copy `.env.example` to `.env` and configure your API keys and Zoho endpoints:

```bash
cp .env.example .env
```
Ensure your `.env` contains the correct Zoho Account routing:
```env
# NVIDIA NIM Keys
MINIMAX_API_KEY=nvapi-...
MINIMAX_BASE_URL=https://integrate.api.nvidia.com/v1

# Zoho Books Integration
ZOHO_DOMAIN=.in
ZOHO_ORG_ID=your_org_id
ZOHO_ACCOUNT_ID=your_expense_account_id
ZOHO_CLIENT_ID=your_client_id
ZOHO_CLIENT_SECRET=your_client_secret
ZOHO_REFRESH_TOKEN=your_refresh_token
```

### 2. Installation
This project requires Python 3.11+. Standard virtual environment setup is recommended:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

### 3. Execution
To run the full evaluation pipeline, compute accuracy metrics, and automatically push the successful extractions to Zoho Books:

```bash
python src/run_eval.py --zoho
```

Check the `results/` folder for `raw_extractions.json` and the `zoho_expenses.json` audit log!

---

## 💡 Key Engineering Decisions & Production Readiness

1. **Cost over Hype:** While it is trivial to send data to OpenAI, building a pipeline that achieves **100% accuracy using free-tier / ultra-low-cost VLMs** demonstrates true production viability for scale.
2. **Schema Enforcement:** LLMs are notorious for returning malformed JSON. This pipeline uses strict system prompting paired with resilient `try/except` parsing blocks to ensure application stability.
3. **Resilient Third-Party Integrations:** The Zoho integration does not just "fire and forget". It utilizes request timeouts, automated token refreshing, and error-state logging to ensure that no financial data is lost in transit.
