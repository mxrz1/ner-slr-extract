# SLR-Extract

An end-to-end pipeline for **Systematic Literature Review (SLR) extraction** from academic PDF papers. The system uses Mistral's OCR API to extract text and figures, stores them in an OpenAI Vector Store, and then uses GPT models to answer structured feature extraction questions for each paper.

---

## Pipeline Architecture

```
PDFs
 │
 ▼
[1] pdf_extraction_mistral_api.py   — OCR: extract text & figures via Mistral API
 │
 ▼
[2] populate_vector_store.py        — Upload chunked text to OpenAI Vector Store
 │
 ▼
[3] slr_extract_blocks.py           — Query vector store & extract SLR features via GPT
 │
 ▼
[4] export_to_excel.py              — Export results to Excel for analysis
```

---

## Prerequisites

- Python 3.12+
- A [Mistral AI](https://console.mistral.ai/) account with API access (for OCR)
- An [OpenAI](https://platform.openai.com/) account with API access (for GPT extraction and Vector Store)

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env and fill in your MISTRAL_API_KEY and OPENAI_API_KEY
```

---

## Input Preparation

The pipeline supports two modes for discovering PDF files:

### Zotero Mode (recommended)
If you manage references in Zotero, export your library as a BibTeX file (`.bib`) into your input folder. The extractor will read the `.bib` file and locate each PDF by its `file` field in the BibTeX entry. This mode produces nicely named output directories (e.g., `2024_Paper_Title_AuthorName`).

Set `use_zotero_mode = True` in `pdf_extraction_mistral_api.py` (see Configuration below).

### Direct Folder Scan Mode
Place all PDF files in a single input folder. The extractor will scan for all `.pdf` files directly. Output directories will be named after the PDF filename.

Set `use_zotero_mode = False` in `pdf_extraction_mistral_api.py`.

---

## Configuration

### Step 1 — PDF Extraction (`pdf-extraction-mistral-api/pdf_extraction_mistral_api.py`)

Edit the `main()` function (around line 499):

```python
def main() -> None:
    ...
    input_folder = "my_papers"           # Path to your folder with PDFs (and .bib file for Zotero mode)
    output_folder = "my_papers_extracted" # Path where extracted data will be saved
    use_zotero_mode = True               # True = Zotero/BibTeX mode, False = direct folder scan
```

### Step 2 — Vector Store Population (`populate_vector_store.py`)

Edit the `INPUT_ROOT` constant (line 22):

```python
INPUT_ROOT = "pdf-extraction-mistral-api/my_papers_extracted"  # Must match output_folder from Step 1
```

### Step 3 — SLR Feature Extraction (`slr_extract_blocks.py`)

Edit the constants (lines 45–50):

```python
PAPERS_ROOT           = Path("pdf-extraction-mistral-api/my_papers_extracted")  # Must match output_folder from Step 1
PROMPT_DIR            = Path("prompts")        # Directory with JSON prompt definitions
MODEL                 = "gpt-5-nano"           # Default GPT model for extraction
DEFAULT_REASONING_EFFORT = "medium"            # Default reasoning effort: "low", "medium", or "high"
```

### Step 4 — Excel Export (`export_to_excel.py`)

No code changes needed — the output folder is passed via CLI argument (see Running the Pipeline below).

---

## Running the Pipeline

### Step 1: Extract PDFs

```bash
cd pdf-extraction-mistral-api
python pdf_extraction_mistral_api.py
```

This creates one subdirectory per paper inside your configured `output_folder`, each containing:
- `content.md` — full OCR-extracted text in Markdown
- `img-*.jpeg` — extracted figures
- `metadata.json` — bibliographic metadata (Zotero mode only)
- `mistral_response.json` — raw Mistral API response

### Step 2: Upload to Vector Store

```bash
python populate_vector_store.py
```

This chunks each paper's `content.md` and uploads the chunks to an OpenAI Vector Store. A `chunk_mapping.json` is saved per paper so re-runs skip already-uploaded papers. The `vector_store_registry.json` in the root tracks all active vector stores.

### Step 3: Extract SLR Features

```bash
python slr_extract_blocks.py
```

For each paper, this queries the vector store with the questions defined in `prompts/*.json` and saves the GPT answers to `slr_features.json` inside each paper's directory.

### Step 4: Export to Excel

```bash
# Export all papers to individual Excel files
python export_to_excel.py --folder pdf-extraction-mistral-api/my_papers_extracted/

# Merge all papers into a single Excel (features as rows, papers as columns)
python export_to_excel.py --merge --folder pdf-extraction-mistral-api/my_papers_extracted/ --output merged_results.xlsx

# Transpose the merged output (papers as rows, features as columns)
python export_to_excel.py --merge --transpose --folder pdf-extraction-mistral-api/my_papers_extracted/ --output transposed_results.xlsx

# Filter to specific features using a YAML file
python export_to_excel.py --merge --feature-filter example_feature_filter.yaml --folder pdf-extraction-mistral-api/my_papers_extracted/
```

---

## Customizing Prompts

The research questions are defined as JSON files in `prompts/`. Each file covers a thematic block of questions (e.g., `A_Biblio.json`, `B_Methods.json`).

Example structure:

```json
{
  "block": "A_Biblio",
  "features": [
    {
      "name": "Publication Year",
      "question": "In which year was this paper published?",
      "guideline": "Return only the 4-digit year."
    },
    {
      "name": "Research Methodology",
      "question": "What research methodology does this paper use?",
      "guideline": "Describe the approach in 1-3 sentences.",
      "model": "gpt-5",
      "reasoning_effort": "high"
    }
  ]
}
```

**Fields:**
- `name` — Feature identifier (used as column header in Excel)
- `question` — The question posed to GPT about the paper
- `guideline` — Additional instructions to guide the GPT answer
- `model` *(optional)* — Override default model for this question (`"gpt-5"` or `"gpt-5-nano"`)
- `reasoning_effort` *(optional)* — Override default effort (`"low"`, `"medium"`, or `"high"`)

---

## Selective Re-extraction

To re-extract specific features for specific papers without running the full pipeline:

```bash
# List all available features
python selective_extract.py --list-features

# Re-extract one feature for one paper (dry run)
python selective_extract.py --single \
  --folder pdf-extraction-mistral-api/my_papers_extracted/ \
  --doc-id author_title_2024 \
  --feature "Publication Year" \
  --dry-run

# Re-extract one feature (actual, shows result in console)
python selective_extract.py --single \
  --folder pdf-extraction-mistral-api/my_papers_extracted/ \
  --doc-id author_title_2024 \
  --feature "Publication Year"

# Bulk re-extraction from a YAML task file
python selective_extract.py --yaml tasks.yaml \
  --folder pdf-extraction-mistral-api/my_papers_extracted/
```

---

## Output Structure

After running the full pipeline, each paper's directory looks like:

```
my_papers_extracted/
└── 2024_Paper_Title_Author/
    ├── content.md              # Full OCR text (Markdown)
    ├── metadata.json           # Bibliographic metadata
    ├── mistral_response.json   # Raw Mistral OCR response
    ├── img-0.jpeg              # Extracted figures
    ├── img-1.jpeg
    ├── chunk_mapping.json      # Vector store chunk index
    └── slr_features.json       # Extracted SLR features (GPT answers)
```

The `slr_features.json` contains one entry per feature with the GPT answer, reasoning, and source chunk references.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `MISTRAL_API_KEY` | Yes | Mistral API key for OCR (Step 1) |
| `OPENAI_API_KEY` | Yes | OpenAI API key for Vector Store and GPT extraction (Steps 2–3) |
| `VECTOR_STORE_ID` | No | Pre-existing OpenAI Vector Store ID to reuse (see note below). |

**Note on `VECTOR_STORE_ID` vs `vector_store_registry.json`:**

For most users, leave `VECTOR_STORE_ID` empty. When you run `populate_vector_store.py`, it automatically creates a new vector store and records it in `vector_store_registry.json`. This registry file is the primary source of truth for `slr_extract_blocks.py` — it supports multiple stores (a new one is created automatically when the 9,500-file capacity limit is reached).

`VECTOR_STORE_ID` is only needed if you have a **pre-existing** vector store you want to reuse. In that case, `populate_vector_store.py` will seed the registry with it on first run, and `slr_extract_blocks.py` will fall back to it if the registry is empty. The `vector_store_registry.json` file is auto-generated and should not be committed to version control.


