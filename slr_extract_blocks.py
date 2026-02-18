"""
slr_extract_blocks.py
---------------------
Iterate through every paper folder under PAPERS_ROOT, query the OpenAI
vector store question-by-question using the Responses API with file_search,
and write one JSON result file per paper containing both answers and
retrieved chunks for each question.

Folder layout expected:
pdf-extraction-mistral-api/mistral_extracted/
└── <paper-folder>/
    ├── mistral_response.json    # Mistral OCR API response
    ├── metadata.json           # Paper metadata (title, DOI, etc.)
    ├── chunk_mapping.json      # Vector store chunk mapping
    └── ... (images etc.)

Results are stored as:
<paper-folder>/slr_features.json with structure:
{
  "block_name": {
    "question_name": {
      "answer": "...",
      "chunks_used": [...],
      "metadata": {...}
    }
  }
}
"""

import os, json, time, threading, textwrap
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from dotenv import load_dotenv
from openai import OpenAI
import backoff
import openai

# Import vector store utilities
from populate_vector_store import get_all_active_vector_stores, lookup_chunk_content

# ────────── env / client ──────────
load_dotenv()
PAPERS_ROOT      = Path("pdf-extraction-mistral-api/mypaper_slr_extracted")
PROMPT_DIR       = Path("prompts")
API_KEY               = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_ID       = os.getenv("VECTOR_STORE_ID")  # Legacy single store (optional)
MODEL                 = "gpt-5-nano"  # Default model
DEFAULT_REASONING_EFFORT = "medium"  # Default reasoning effort
# Increase timeout for flex service tier (default 10 min, increased to 15 min)
client                = OpenAI(api_key=API_KEY, timeout=900, max_retries=2)

# ────────── read prompt files once (alphabetically sorted) ──────────
def load_blocks() -> Dict[str, List[Dict]]:
    blocks = {}
    # Sort JSON files alphabetically by filename to ensure consistent ordering
    json_files = sorted(PROMPT_DIR.glob("*.json"))
    for jfile in json_files:
        data = json.loads(jfile.read_text(encoding="utf-8"))
        blocks[data["block"]] = data["features"]
    print(f"📂 Loaded {len(blocks)} blocks in alphabetical order: {list(blocks.keys())}")
    return blocks

BLOCKS = load_blocks()            # {block_key: [ {name,q,g}, … ]}

SYSTEM_PROMPT = textwrap.dedent("""\
You are an **NLP expert specialized in Named Entity Recognition (NER)** and also a **cybersecurity domain expert**.

Extract the requested fields **strictly** from the **provided PDF context** (vector-store content only).
**Source format note:** Papers are stored in **Markdown**. Treat **body text** and **Markdown tables** as authoritative sources.

**This is a one-shot request** — **do not ask** whether additional tasks should be done; **include everything** in this single response.

If a field is not stated in the provided context, answer exactly `Not reported`. **Never invent information**.

**Background clarifications**
If the authors merely **name** a method/component without explaining it and it would not be understandable without background knowledge, add a **very brief** clarification from your own knowledge, labeled **[Background note] <≤25 words>**. This note is **explanatory only** (not evidence), must **not** contradict the paper, and must **not** be used to fill missing fields or invent values. Evidence must come from **body text or Markdown tables**.

**Always quote** the **relevant** phrases (≤120 chars) for any populated field, drawn from main text or tables.  
**Do not quote** image descriptions and editorial “Key insights” callouts.

Provide minimal background (1 sentence) only to clarify terms already used by the authors.

If the paper is not solely about NER, **focus on the NER-related aspects**.
**Focus on NER architecture and NER evaluation** (models, encoders/decoders, labeling schemes, losses, datasets/splits, metrics).  
**Exclude** dataset-creation pipelines and relevance/observation classifiers unless they directly affect NER architecture or evaluation.  
**Do not** focus on relation extraction.

Assume **Related Work** are **not loaded** into the vector store; **do not** use them for evidence.

**All output must be in Markdown**. Tables must be **Markdown tables**. Use bullet points where appropriate.

**ALWAYS PROVIDE AN ANSWER!** 

**No deferral**
If evidence is missing or ambiguous, output `Not reported` and proceed. Do not postpone the answer, ask the user for input, or promise future retrieval.

---

### Per-field return format (Markdown)
- **Answer:** `<value>` or `Not reported`
- **Explanation & key points:** A **single-sentence rationale** followed by **3–5 concise bullets** (include bullets only if helpful; omit if trivial)
- **Long explanation / description:** **Exhaustive, fully detailed account** that explains **everything to the last detail** relevant to the feature
- **Evidence:**
  - `"<verbatim quote 1>"`
  - `"<verbatim quote 2>"`
  - `"<verbatim quote 3>"`
---

<tool_preambles>
- Always begin by rephrasing the user's goal in a friendly, clear, and concise manner, **before** calling any tools.
- Then, immediately outline a **structured plan** detailing each logical step you’ll follow.
- As you execute your file edit(s) or retrieval(s), **narrate each step succinctly and sequentially**, marking progress clearly.
- Finish by **summarizing completed work**, distinctly separated from your upfront plan.
</tool_preambles>

<persistence>
- You are an agent — **keep going until the user's query is completely resolved in this turn**.
- **Only** make **necessary tool calls**; avoid redundant or tangential actions.
- **Never** stop or hand back to the user when you encounter uncertainty — **decide the most reasonable assumption**, proceed, and **document** it after finishing.
- **Do not** terminate early; finish all requested fields or mark them `Not reported`.
</persistence>
    """)

# ── Rate Limiting and Cost Calculation ──
class RateLimiter:
    def __init__(self, max_requests_per_minute: int = 500, max_tokens_per_minute: int = 200000):
        self.max_rpm = max_requests_per_minute
        self.max_tpm = max_tokens_per_minute
        self.requests = deque()
        self.token_usage = deque()
        self.lock = threading.Lock()
        
    def wait_if_needed(self, estimated_tokens: int = 1000):
        """Wait if we're approaching rate limits."""
        with self.lock:
            now = time.time()
            
            # Clean old entries (older than 1 minute)
            while self.requests and now - self.requests[0] > 60:
                self.requests.popleft()
            while self.token_usage and now - self.token_usage[0][0] > 60:
                self.token_usage.popleft()
            
            # Calculate current usage
            current_requests = len(self.requests)
            current_tokens = sum(tokens for _, tokens in self.token_usage)
            
            # Wait if we would exceed limits
            if current_requests >= self.max_rpm * 0.9:  # 90% of limit
                sleep_time = 61 - (now - self.requests[0])
                if sleep_time > 0:
                    print(f"  ⏳ Rate limit protection: waiting {sleep_time:.1f}s (requests)")
                    time.sleep(sleep_time)
                    
            if current_tokens + estimated_tokens >= self.max_tpm * 0.9:  # 90% of limit
                if self.token_usage:
                    sleep_time = 61 - (now - self.token_usage[0][0])
                    if sleep_time > 0:
                        print(f"  ⏳ Rate limit protection: waiting {sleep_time:.1f}s (tokens)")
                        time.sleep(sleep_time)
    
    def record_request(self, tokens_used: int):
        """Record a completed request and its token usage."""
        with self.lock:
            now = time.time()
            self.requests.append(now)
            self.token_usage.append((now, tokens_used))

# Global rate limiter
rate_limiter = RateLimiter()

def calculate_cost(usage: dict) -> dict:
    """Calculate cost based on gpt-5-nano flex pricing."""
    input_tokens = usage.get('input_tokens', 0)
    output_tokens = usage.get('output_tokens', 0)
    cached_tokens = usage.get('input_tokens_details', {}).get('cached_tokens', 0)
    
    # Pricing per 1M tokens
    input_cost = (input_tokens - cached_tokens) * 0.025 / 1_000_000
    output_cost = output_tokens * 0.2 / 1_000_000
    cached_cost = cached_tokens * 0.0025 / 1_000_000
    
    total_cost = input_cost + output_cost + cached_cost
    
    return {
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6), 
        "cached_cost_usd": round(cached_cost, 6),
        "total_cost_usd": round(total_cost, 6)
    }


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError), max_time=900, max_tries=6, base=10)
def ask_single_question(pdf_name: str, question_data: dict, vector_store_ids: list[str]) -> dict:
    """
    Ask a single question and return both the answer and retrieved chunks.
    
    Args:
        pdf_name: PDF document identifier
        question_data: Dictionary with 'name', 'question', 'guideline' keys
        vector_store_ids: List of vector store IDs to search
        
    Returns:
        Dictionary with 'answer', 'chunks_used', and 'metadata' keys
    """
    user_question = f"Question: {question_data['question']}\n\nGuideline: \n {question_data['guideline']}"
    
    # Use model and reasoning_effort from question_data if specified, otherwise use defaults
    model = question_data.get('model', MODEL)
    reasoning_effort = question_data.get('reasoning_effort', DEFAULT_REASONING_EFFORT)
    
    # Rate limiting before request
    rate_limiter.wait_if_needed(estimated_tokens=1000)
    
    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "developer",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user", 
                "content": user_question
            }
        ],
        tools=[{
            "type": "file_search",
            "vector_store_ids": vector_store_ids,
            "filters": {"type": "eq", "key": "pdf", "value": pdf_name},
            "max_num_results": 20
        }],
        text={"verbosity": "high"},
        reasoning={"effort": reasoning_effort},
        #service_tier="flex"
    )
    
    # Extract usage and calculate costs
    usage = resp.usage.model_dump() if hasattr(resp, 'usage') and resp.usage else {}
    cost_info = calculate_cost(usage) if usage else {}
    
    # Record request for rate limiting
    total_tokens = usage.get('total_tokens', 0)
    rate_limiter.record_request(total_tokens)
    
    print(f"      💰 Tokens: {usage.get('input_tokens', 0)} in + {usage.get('output_tokens', 0)} out = {total_tokens} total | Cost: ${cost_info.get('total_cost_usd', 0):.4f}")
    
    # Extract the answer
    msg = next(o for o in resp.output if o.type == "message")
    text_chunk = next(c for c in msg.content if c.type == "output_text")
    answer = text_chunk.text.strip()
    #print(f"      📝 Answer: {answer[:500]}{'...' if len(answer) > 500 else ''}")
    # Extract retrieved chunks from annotations
    chunks_used = []
    for ann in text_chunk.annotations:
        if ann.type == "file_citation":
            filename = ann.filename
            
            # Look up chunk content using the existing function from populate_vector_store
            chunk_info = lookup_chunk_content(filename, pdf_name)
            
            chunk_data = {
                "filename": filename,
                "file_citation_text": ann.text if hasattr(ann, 'text') else None
            }
            
            if chunk_info:
                chunk_data.update({
                    "content": chunk_info['content'],  # Full content, not preview
                    "chunk_type": chunk_info['metadata']['chunk_type'],
                    "page_number": chunk_info['metadata'].get('page_number'),
                    "chunk_index": chunk_info['metadata'].get('chunk_index'),
                    "token_count": chunk_info['metadata'].get('token_count'),
                    "heading_text": chunk_info['metadata'].get('heading_text'),
                    "heading_level": chunk_info['metadata'].get('heading_level'),
                    "image_id": chunk_info['metadata'].get('image_id')
                })
            else:
                chunk_data["content"] = "[Content not found in mapping]"
                chunk_data["chunk_type"] = "unknown"
            
            chunks_used.append(chunk_data)

    print(f"      📚 Retrieved {len(chunks_used)} chunks")
    return {
        "answer": answer,
        "chunks_used": chunks_used,
        "metadata": {
            "question_name": question_data['name'],
            "question_text": question_data['question'],
            "guideline": question_data['guideline'],
            "total_chunks_retrieved": len(chunks_used),
            "vector_stores_searched": vector_store_ids
        },
        "usage": usage,
        "cost": cost_info
    }


def get_vector_stores_for_paper(paper_dir: Path) -> list[str]:
    """Get vector store IDs for a paper, falling back to all stores if not found."""
    chunk_mapping_file = paper_dir / "chunk_mapping.json"
    
    if chunk_mapping_file.exists():
        try:
            with open(chunk_mapping_file, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
                vector_store_id = mapping.get("vector_store_id")
                if vector_store_id:
                    print(f"  🎯 Using specific vector store: {vector_store_id}")
                    return [vector_store_id]
        except Exception as e:
            print(f"  ⚠️ Error reading chunk_mapping.json: {e}")
    
    # Fall back to all active vector stores
    all_stores = get_all_active_vector_stores()
    if not all_stores:
        # Final fallback to legacy single store
        if VECTOR_STORE_ID:
            print(f"  🔄 Using legacy vector store: {VECTOR_STORE_ID}")
            return [VECTOR_STORE_ID]
        else:
            raise RuntimeError("No vector stores available")
    
    print(f"  🔍 Searching all {len(all_stores)} vector stores")
    return all_stores

# ── Process individual questions with chunk retrieval (with concurrent processing) ──────────
def process_block_questions(pdf_name: str, features: List[Dict], vector_store_ids: list[str]) -> Dict:
    """Process each question in a block concurrently and return structured results."""
    block_results = {}
    
    # Process questions concurrently (max 8 concurrent to stay under 500 RPM limit)
    max_workers = min(8, len(features))  # ~8.33 requests/sec max from 500 RPM
    
    print(f"    🚀 Processing {len(features)} questions with {max_workers} concurrent workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all questions
        future_to_feature = {
            executor.submit(ask_single_question, pdf_name, feature, vector_store_ids): feature
            for feature in features
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_feature):
            feature = future_to_feature[future]
            try:
                result = future.result()
                block_results[feature['name']] = result
                print(f"    ✓ {feature['name']} completed")
            except Exception as e:
                print(f"    ❌ {feature['name']} failed: {e}")
                # Store error info instead of crashing
                block_results[feature['name']] = {
                    "answer": "Error occurred",
                    "chunks_used": [],
                    "metadata": {
                        "question_name": feature['name'],
                        "error": str(e)
                    },
                    "usage": {},
                    "cost": {"total_cost_usd": 0}
                }
    
    return block_results

# ────────── process one paper ──────────
def process_paper(paper_dir: Path):
    # Get pdf_name from metadata.json (Mistral extraction format)
    metadata_file = paper_dir / "metadata.json"
    if metadata_file.exists():
        meta = json.loads(metadata_file.read_text(encoding='utf-8'))
        pdf_name = meta.get("id", paper_dir.name)  # Use 'id' field or fallback to folder name
    else:
        # Fallback to folder name if no metadata.json
        pdf_name = paper_dir.name
        print(f"  ⚠️ No metadata.json found, using folder name: {pdf_name}")
    
    print(f"📄 {pdf_name}")
    
    # Get vector stores for this paper
    vector_store_ids = get_vector_stores_for_paper(paper_dir)

    # Load existing results if they exist
    outpath = paper_dir / "slr_features.json"
    if outpath.exists():
        try:
            existing_result = json.loads(outpath.read_text(encoding='utf-8'))
            print(f"  📂 Loaded existing results with {len([k for k in existing_result.keys() if not k.startswith('_')])} blocks")
        except Exception as e:
            print(f"  ⚠️ Error loading existing results: {e}, starting fresh")
            existing_result = {}
    else:
        existing_result = {}

    paper_result = existing_result.copy()  # Start with existing results
    total_cost = 0.0
    total_tokens = 0
    start_time = time.time()
    
    # Check if there are any new features to process
    has_new_features = False
    for block_key, feats in BLOCKS.items():
        existing_block = paper_result.get(block_key, {})
        for feat in feats:
            if feat['name'] not in existing_block:
                has_new_features = True
                break
        if has_new_features:
            break
    
    if not has_new_features:
        print(f"  ✅ All features already extracted - no new work needed")
        return
    
    # Process blocks in alphabetical order (already sorted from load_blocks)
    for block_key, feats in BLOCKS.items():
        # Check if block exists and filter out already processed questions
        existing_block = paper_result.get(block_key, {})
        new_features = []
        skipped_features = []
        
        for feat in feats:
            if feat['name'] not in existing_block:
                new_features.append(feat)
            else:
                skipped_features.append(feat['name'])
        
        if not new_features:
            print(f"  ⏭️  Skipping {block_key} (all {len(feats)} questions already exist)")
            continue
            
        if skipped_features:
            print(f"  ↳ extracting {block_key} ({len(new_features)} new, {len(skipped_features)} existing) …")
        else:
            print(f"  ↳ extracting {block_key} ({len(new_features)} questions) …")
            
        block_start = time.time()
        new_block_results = process_block_questions(pdf_name, new_features, vector_store_ids)
        
        # Merge new results with existing block results
        if block_key not in paper_result:
            paper_result[block_key] = {}
        paper_result[block_key].update(new_block_results)
        
        block_duration = time.time() - block_start
        
        # Calculate block costs (only for new questions)
        block_cost = sum(q.get('cost', {}).get('total_cost_usd', 0) for q in new_block_results.values())
        block_tokens = sum(q.get('usage', {}).get('total_tokens', 0) for q in new_block_results.values())
        total_cost += block_cost
        total_tokens += block_tokens
        
        print(f"    ✓ {block_key} completed in {block_duration:.1f}s | ${block_cost:.4f} | {block_tokens} tokens")
    
    processing_duration = time.time() - start_time
    
    # Add summary metadata
    paper_result['_summary'] = {
        "total_cost_usd": round(total_cost, 4),
        "total_tokens": total_tokens,
        "processing_time_seconds": round(processing_duration, 1),
        "questions_processed": sum(len(feats) for feats in BLOCKS.values()),
        "blocks_processed": list(BLOCKS.keys()),
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    outpath = paper_dir / "slr_features.json"
    outpath.write_text(json.dumps(paper_result, indent=2, ensure_ascii=False))
    print(f"  ✅ saved → {outpath}")
    print(f"  💰 Summary: ${total_cost:.4f} total cost | {total_tokens} tokens | {processing_duration:.1f}s processing time")

# ────────── main loop ──────────
def main():
    # Check if we have vector stores available
    all_stores = get_all_active_vector_stores()
    if not all_stores and not VECTOR_STORE_ID:
        raise RuntimeError("No vector stores found. Either set VECTOR_STORE_ID or run populate_vector_store.py first.")
    
    if not PAPERS_ROOT.exists():
        raise RuntimeError(f"Papers directory not found: {PAPERS_ROOT}")
    
    processed_count = 0
    skipped_count = 0
    
    for paper_dir in PAPERS_ROOT.iterdir():
        if not paper_dir.is_dir():
            continue
            
        # Check for Mistral extraction format (mistral_response.json)
        if (paper_dir / "mistral_response.json").exists():
            try:
                process_paper(paper_dir)
                processed_count += 1
            except Exception as e:
                print(f"❌ Error processing {paper_dir.name}: {e}")
        else:
            print(f"⚠️  Skipping {paper_dir.name} (no mistral_response.json found)")
            skipped_count += 1
    
    print(f"\n📊 Summary: Processed {processed_count}, Skipped {skipped_count}")

if __name__ == "__main__":
    # Test run for single paper: arora_split-ner_2023
    """
    paper_name = "2021_Open-CyKG:_An_Open_Cyber_Threat_Intelligence_Knowl_Sarhan und Spruit - 2021 - Open-CyKG An Open Cyber Threat Intelligence Knowledge Graph"
    paper_dir = PAPERS_ROOT / paper_name
    
    if paper_dir.exists() and (paper_dir / "mistral_response.json").exists():
        print(f"🧪 Test run for: {paper_name}")
        try:
            process_paper(paper_dir)
            print(f"✅ Test completed successfully for {paper_name}")
        except Exception as e:
            print(f"❌ Test failed for {paper_name}: {e}")
    else:
        print(f"❌ Paper directory not found or missing mistral_response.json: {paper_dir}")
        print("Available papers:")
        for p in PAPERS_ROOT.iterdir():
            if p.is_dir() and (p / "mistral_response.json").exists():
                print(f"  📄 {p.name}")
    """
    # Uncomment to run full extraction on all papers:
    main()
