# ─────────────────────────────────────────────────────────────────────────────
# CV Assistant Platform – README
# ─────────────────────────────────────────────────────────────────────────────

## Overview

A production-grade **RAG + Information Extraction** system for HR candidate analysis.
Processes Banking and Information-Technology resumes (text-based and scanned PDFs), extracts structured entities, enables natural-language Q&A, and identifies cross-domain "outlier" candidates trainable for IT roles.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | `llama-3.3-70b-versatile` |
| Orchestration | LangChain |
| Vector Database | Pinecone |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace) |
| PDF Processing | PyMuPDF4LLM + pdf2image + easyocr |
| Entity Validation | Pydantic v2 |
| Evaluation | Cosine Similarity |
| UI | Streamlit |

---

## Project Structure

```
cv_rag_system/
├── config.yaml        ← ALL parameters
├── config.py          ← typed dataclass hierarchy + Settings root class
├── pdf_loader.py      ← PyMuPDF4LLM + OCRLoader (Fallback)
├── ingest.py          ← PDF ingestion pipeline → Pinecone
├── extractor.py       ← LangChain + LLAMA entity extraction (CV → JSON)
├── rag_chain.py       ← RAG retrieval & generation chain + Outlier logic
├── eval.py            ← Evaluation
├── app.py             ← Streamlit UI
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. System Dependencies (Ubuntu/Debian)

```bash
sudo apt update && sudo apt install -y tesseract-ocr poppler-utils
```

On macOS:
```bash
brew install tesseract poppler
```

### 2. Python Environment

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Fill in your API keys:

```
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
KAGGLE_API_TOKEN=your_kaggle_api_token_here
```

Get keys from:
- Groq: https://console.groq.com/keys
- Pinecone: https://app.pinecone.io
- Kaggle: https://www.kaggle.com/settings/api
---

## Usage

### Step 1: Ingest Documents

```bash
python ingest.py --data_dir ./resumes --index_name cv-rag-index
```

This will:
- Download dataset from Kaggle
- Extract text from PDFs (auto-detects text vs. scanned)
- Chunk and embed documents
- Upsert into a single Pinecone index with domain metadata

### Step 3: Run the App

```bash
streamlit run app.py
```

### Step 4: Run Evaluation (Optional)

```bash
python eval.py --output_path ./eval_results.json
```

---

## Architecture & Design Decisions

### PDF Processing Strategy

```
PDF Input
    ↓
PyMuPDF4LLM (Markdown)
    ↓
< 100 chars? → OCR fallback (pdf2image + easyocr)
    ↓
Raw text with domain metadata
```

### Single Pinecone Index + Metadata Filtering

Rather than separate indices per domain, a single index with `category` metadata
enables:
- Cross-domain queries (ALL)
- Domain-specific queries (`{"category": {"$eq": "BANKING"}}`)
- The Outlier query (retrieve BANKING, analyse for IT transferability)

This is more cost-effective and simpler to maintain.

### Entity Extraction via Structured Output

The extraction chain uses `JsonOutputParser(pydantic_object=CVEntity)` which:
1. Injects the Pydantic JSON schema into the system prompt
2. Asks LLAMA to return only valid JSON
3. Validates and coerces the output into the typed model
4. Fails gracefully with descriptive errors

### Outlier Logic

The outlier query works in two stages:
1. **Retrieve**: Pull top-K Banking chunks using a broad "analytical skills" query
2. **Generate**: Feed a specialised system prompt asking LLAMA to rank by
   transferable IT skills (SQL, maths, data, PM, systems thinking)

### RAG Evaluation Methodology

**Quantitative (Cosine Similarity):**
Calculates the Cosine Similarity score (ranging from `0.0` to `1.0`) between the RAG-generated answer and the predefined `ground_truth`.

**Qualitative (Manual):**
- Per-question human review of answer quality, source attribution, and completeness
- Stored in `eval_results.json` for audit

## Submission
- **GitHub Repo**: [CV-RAG](https://github.com/phngan05/CV-RAG)
- **Demo**: [your-demo-link]
