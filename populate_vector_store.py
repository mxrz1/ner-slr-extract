import os
import json
import time
import io
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI, OpenAIError

# Import chunking functionality from mistral_vector_store
import sys
sys.path.append('pdf-extraction-mistral-api')
from mistral_vector_store import DocumentLoader, ContentChunker, ProcessedDocument, remove_references_section

# Import tiktoken for consistent tokenization
import tiktoken

# ─────────── Configuration ───────────
load_dotenv()
INPUT_ROOT = "pdf-extraction-mistral-api/mypaper_slr_extracted"
MAX_PARALLEL = 50
MAX_FILES_PER_VECTOR_STORE = 9500  # Leave some buffer below 10k limit
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXISTING_VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")
VECTOR_STORE_REGISTRY_FILE = "vector_store_registry.json"
# ─────────────────────────────────────

client = OpenAI(api_key=OPENAI_API_KEY)


def load_vector_store_registry() -> dict:
    """Load the vector store registry from file."""
    registry_path = Path(VECTOR_STORE_REGISTRY_FILE)
    if registry_path.exists():
        try:
            with open(registry_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading vector store registry: {e}")
    
    # Initialize empty registry
    registry = {
        "vector_stores": [],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add existing vector store if specified
    if EXISTING_VECTOR_STORE_ID:
        print(f"📋 Adding existing vector store to registry: {EXISTING_VECTOR_STORE_ID}")
        try:
            vs_info = client.vector_stores.retrieve(EXISTING_VECTOR_STORE_ID)
            registry["vector_stores"].append({
                "id": EXISTING_VECTOR_STORE_ID,
                "name": vs_info.name or "mistral_extracted_papers",
                "file_count": vs_info.file_counts.total if vs_info.file_counts else 0,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "active": True
            })
        except Exception as e:
            print(f"⚠️ Could not retrieve existing vector store info: {e}")
            registry["vector_stores"].append({
                "id": EXISTING_VECTOR_STORE_ID,
                "name": "mistral_extracted_papers",
                "file_count": 0,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "active": True
            })
    
    return registry


def save_vector_store_registry(registry: dict) -> None:
    """Save the vector store registry to file."""
    registry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(VECTOR_STORE_REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def get_vector_store_with_capacity(required_slots: int = 1) -> str:
    """Get a vector store with available capacity or create a new one."""
    registry = load_vector_store_registry()
    
    # Find vector store with available capacity
    for vs_info in registry["vector_stores"]:
        if vs_info["active"] and vs_info["file_count"] + required_slots <= MAX_FILES_PER_VECTOR_STORE:
            print(f"📦 Using vector store {vs_info['id']} ({vs_info['file_count']}/{MAX_FILES_PER_VECTOR_STORE} files)")
            return vs_info["id"]
    
    # Create new vector store if none available
    store_number = len(registry["vector_stores"]) + 1
    store_name = f"mistral_extracted_papers_v2_{store_number}"
    
    print(f"🆕 Creating new vector store: {store_name}")
    vs = client.vector_stores.create(
        name=store_name,
        metadata={
            "purpose": "mistral_ocr_extracted_papers",
            "created_by": "populate_vector_store.py",
            "store_number": str(store_number)
        }
    )
    
    # Add to registry
    new_store = {
        "id": vs.id,
        "name": store_name,
        "file_count": 0,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "active": True
    }
    registry["vector_stores"].append(new_store)
    save_vector_store_registry(registry)
    
    print(f"✅ Created vector store: {vs.id}")
    return vs.id


def update_vector_store_file_count(vector_store_id: str, added_files: int) -> None:
    """Update the file count for a vector store in the registry."""
    registry = load_vector_store_registry()
    
    for vs_info in registry["vector_stores"]:
        if vs_info["id"] == vector_store_id:
            vs_info["file_count"] += added_files
            break
    else:
        print(f"⚠️ Vector store {vector_store_id} not found in registry")
        return
    
    save_vector_store_registry(registry)
    print(f"📊 Updated vector store {vector_store_id}: +{added_files} files (total: {vs_info['file_count']})")


def get_all_active_vector_stores() -> list[str]:
    """Get list of all active vector store IDs."""
    registry = load_vector_store_registry()
    return [vs["id"] for vs in registry["vector_stores"] if vs["active"]]



def upload_text_chunk(text: str, filename: str, attributes: Dict[str, Any], vector_store_id: str):
    """Upload a single text chunk to OpenAI vector store with markdown MIME type."""
    try:
        bio = io.BytesIO(text.encode("utf-8"))
        bio.name = filename.replace('.txt', '.md')  # Use .md extension for markdown
        # Create file with markdown content type
        file_obj = client.files.create(file=bio, purpose="assistants")
        file_id = file_obj.id
        
        client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_id,
            attributes=attributes
        )
        return True
    except Exception as e:
        print(f"❌ Failed to upload {filename}: {e}")
        return False


def upload_paper_to_vector_store(paper_dir: Path, vector_store_id: str):
    """Upload a single paper from Mistral extraction format to OpenAI vector store using advanced chunking.
    
    Args:
        paper_dir: Path to paper directory containing mistral_response.json and metadata.json
        vector_store_id: OpenAI vector store ID
        
    Returns:
        Document ID of the uploaded paper
    """
    # Initialize document loader and content chunker
    document_loader = DocumentLoader(str(paper_dir.parent))
    content_chunker = ContentChunker(chunk_size=512, chunk_overlap=64)
    
    # Load the document using the sophisticated loader
    doc = document_loader._load_single_document(paper_dir)
    if not doc:
        print(f"❌ Failed to load document from {paper_dir.name}")
        return None
    
    print(f"\n📄 {doc.paper_title} ({doc.document_id})")
    
    # Chunk the document using the advanced chunking mechanism
    try:
        chunked_doc = content_chunker.chunk_document(doc)
        print(f"📊 Generated {len(chunked_doc.chunks)} chunks")
        
        # Count chunk types
        text_chunks = sum(1 for chunk in chunked_doc.chunks if chunk.metadata['chunk_type'] == 'text')
        image_chunks = sum(1 for chunk in chunked_doc.chunks if chunk.metadata['chunk_type'] == 'image')
        print(f"   - {text_chunks} text chunks, {image_chunks} image chunks")
        
    except Exception as e:
        print(f"❌ Error chunking document: {e}")
        return None
    
    # Create chunk mapping for later retrieval
    chunk_mapping = {
        "document_id": doc.document_id,
        "paper_title": doc.paper_title,
        "vector_store_id": vector_store_id,
        "total_chunks": len(chunked_doc.chunks),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunks": {}
    }
    
    # Prepare chunk data for parallel upload
    upload_tasks = []
    
    for chunk in chunked_doc.chunks:
        metadata = chunk.metadata
        chunk_index = metadata['chunk_index']
        
        if metadata['chunk_type'] == 'text':
            fname = f"{doc.document_id}_text_chunk_{metadata['chunk_index']}.md"
            attrs = {
                "kind": "text_chunk",
                "pdf": doc.document_id,
                "page": metadata.get('page_number', 0),
                "paper_title": doc.paper_title,
                "chunk_type": metadata['chunk_type'],
                "chunk_index": metadata['chunk_index'],
                "token_count": metadata.get('token_count', 0),
                "heading_level": metadata.get('heading_level'),
                "heading_text": metadata.get('heading_text')
            }
            
            # Add to chunk mapping
            chunk_mapping["chunks"][str(chunk_index)] = {
                "filename": fname,
                "content": chunk.page_content,
                "metadata": metadata,
                "openai_attributes": attrs
            }
            
            # Prepare upload task
            upload_tasks.append((chunk.page_content, fname, attrs))
            
        else:  # image chunk
            fname = f"{doc.document_id}_image_{metadata.get('image_id', f'chunk_{metadata['chunk_index']}')}.md"
            attrs = {
                "kind": "image_chunk", 
                "pdf": doc.document_id,
                "page": metadata.get('page_number', 0),
                "paper_title": doc.paper_title,
                "chunk_type": metadata['chunk_type'],
                "chunk_index": metadata['chunk_index'],
                "token_count": metadata.get('token_count', 0),
                "image_id": metadata.get('image_id')
            }
            
            # Add to chunk mapping
            chunk_mapping["chunks"][str(chunk_index)] = {
                "filename": fname,
                "content": chunk.page_content,
                "metadata": metadata,
                "openai_attributes": attrs
            }
            
            # Prepare upload task
            upload_tasks.append((chunk.page_content, fname, attrs))
    
    # Update total chunk count
    chunk_mapping["total_chunks"] = len(upload_tasks)
    
    # Upload chunks in parallel
    print(f"🚀 Uploading {len(upload_tasks)} chunks in parallel (max {MAX_PARALLEL} concurrent)...")
    successful_uploads = 0
    failed_uploads = 0
    
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        # Submit all upload tasks
        future_to_filename = {
            executor.submit(upload_text_chunk, text, fname, attrs, vector_store_id): fname
            for text, fname, attrs in upload_tasks
        }
        
        # Process completed uploads with progress bar
        with tqdm(total=len(upload_tasks), desc="Uploading chunks", leave=False) as pbar:
            for future in as_completed(future_to_filename):
                filename = future_to_filename[future]
                try:
                    success = future.result()
                    if success:
                        successful_uploads += 1
                    else:
                        failed_uploads += 1
                except Exception as e:
                    print(f"❌ Upload failed for {filename}: {e}")
                    failed_uploads += 1
                pbar.update(1)
    
    print(f"✅ Upload complete: {successful_uploads} successful, {failed_uploads} failed")
    
    # Update vector store file count
    if successful_uploads > 0:
        update_vector_store_file_count(vector_store_id, successful_uploads)
    
    # Save chunk mapping to file
    mapping_file = paper_dir / "chunk_mapping.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(chunk_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved chunk mapping to {mapping_file}")
    
    return doc.document_id


def load_chunk_mapping(paper_directory: str) -> dict | None:
    """Load chunk mapping for a specific paper.
    
    Args:
        paper_directory: Path to paper directory containing chunk_mapping.json
        
    Returns:
        Dictionary with chunk mapping or None if not found
    """
    mapping_file = Path(paper_directory) / "chunk_mapping.json"
    if not mapping_file.exists():
        print(f"❌ No chunk mapping found at {mapping_file}")
        return None
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading chunk mapping: {e}")
        return None


def find_paper_directory(document_id: str, base_dir: str = INPUT_ROOT) -> Path | None:
    """Find paper directory by document ID.
    
    Args:
        document_id: Document ID to search for
        base_dir: Base directory to search in
        
    Returns:
        Path to paper directory or None if not found
    """
    base_path = Path(base_dir)
    for item in base_path.iterdir():
        if item.is_dir():
            metadata_file = item / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        if metadata.get('id') == document_id:
                            return item
                except:
                    continue
    return None


def lookup_chunk_content(filename: str, document_id: str) -> dict | None:
    """Look up chunk content by filename.
    
    Args:
        filename: Filename used for OpenAI upload
        document_id: Document ID to search in
        
    Returns:
        Dictionary with chunk info or None if not found
    """
    paper_dir = find_paper_directory(document_id)
    if not paper_dir:
        return None
    
    mapping = load_chunk_mapping(str(paper_dir))
    if not mapping:
        return None
    
    # Search through chunks to find matching filename
    for chunk_idx, chunk_info in mapping['chunks'].items():
        if chunk_info['filename'] == filename:
            return chunk_info
    
    return None


def delete_single_file(file_id: str, filename: str) -> bool:
    """Delete a single file from OpenAI storage."""
    try:
        client.files.delete(file_id)
        return True
    except Exception as e:
        print(f"❌ Failed to delete {filename}: {e}")
        return False


def delete_all_openai_files(confirm: bool = False) -> None:
    """Delete all files from OpenAI file storage using parallel requests.
    
    Args:
        confirm: If True, actually delete files. If False, just list them.
    """
    print("🔍 Listing all OpenAI files...")
    
    files = client.files.list()
    file_list = list(files.data)
    
    if not file_list:
        print("✅ No files found in OpenAI storage")
        return
    
    print(f"📁 Found {len(file_list)} files:")
    for i, file in enumerate(file_list, 1):
        print(f"  {i}. {file.filename} (ID: {file.id}) - {file.purpose}")
    
    if not confirm:
        print("\n⚠️  This was a dry run. To actually delete files, call delete_all_openai_files(confirm=True)")
        return
    
    print(f"\n🗑️  Deleting {len(file_list)} files in parallel (max 50 concurrent)...")
    
    deleted_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        # Submit all delete tasks
        future_to_file = {
            executor.submit(delete_single_file, file.id, file.filename): file
            for file in file_list
        }
        
        # Process completed deletions with progress bar
        with tqdm(total=len(file_list), desc="Deleting files") as pbar:
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    success = future.result()
                    if success:
                        deleted_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"❌ Delete failed for {file.filename}: {e}")
                    failed_count += 1
                pbar.update(1)
    
    print(f"\n✅ Deletion complete:")
    print(f"   - Deleted: {deleted_count}")
    print(f"   - Failed: {failed_count}")
    print(f"   - Total: {len(file_list)}")


def run_test_query(pdf_name: str, vector_store_id: str, query: str = "Explain the architecture of the paper"):
    print(f"\n🔍 Test Query for: {pdf_name}")
    response = client.responses.create(
        model="gpt-5-nano",
        input=query,
        tools=[{
            "type": "file_search",
            "vector_store_ids": [vector_store_id],
            "filters": {
                "type": "eq",
                "key": "pdf",
                "value": pdf_name
            }
        }],
        extra_body={"include": ["output[*].file_search_call.search_results"]},
        service_tier="flex"
    )

    # Print retrieved chunks based on annotations in the final message
    final_message = next((o for o in response.output if o.type == "message"), None)
    if final_message is None:
        print("❌ No assistant message found in the response.")
        return

    content_block = next((c for c in final_message.content if c.type == "output_text"), None)
    if content_block is None:
        print("❌ No output_text found in assistant message.")
        return

    print("📚 Retrieved Context Chunks:")
    print("-" * 80)
    
    chunk_count = 0
    for ann in content_block.annotations:
        if ann.type != "file_citation":
            continue

        filename = ann.filename
        chunk_count += 1
        
        # Look up the actual chunk content
        chunk_info = lookup_chunk_content(filename, pdf_name)
        
        if chunk_info:
            is_image = chunk_info['metadata']['chunk_type'] == 'image'
            
            if is_image:
                source_type = "🖼️ Image"
            else:
                source_type = "📄 Text"
                
            metadata = chunk_info['metadata']
            
            print(f"\n{chunk_count}. {source_type} → {filename}")
            print(f"   📍 Page {metadata.get('page_number', '?')}, Chunk {metadata.get('chunk_index', '?')}, Tokens: {metadata.get('token_count', '?')}")
            
            if metadata.get('heading_text'):
                heading_level = metadata.get('heading_level', 1)
                print(f"   📋 Section: {'#' * heading_level} {metadata.get('heading_text')}")
            
            if is_image and metadata.get('image_id'):
                print(f"   🖼️  Image ID: {metadata.get('image_id')}")
                
            print(f"   📄 Content:")
            content_lines = chunk_info['content'].split('\n')
            for i, line in enumerate(content_lines[:10]):  # Show first 10 lines
                print(f"      {line}")
            if len(content_lines) > 10:
                print(f"      ... ({len(content_lines) - 10} more lines)")
                
        else:
            print(f"\n{chunk_count}. ❓ Unknown → {filename}")
            print(f"   ⚠️  Content not found in mapping")
    
    print("-" * 80)

    # Print the actual answer
    print("\n📝 Final Answer:\n")
    print(content_block.text.strip())



def is_paper_already_uploaded(paper_dir: Path) -> bool:
    """Check if a paper has already been uploaded by looking for chunk_mapping.json."""
    return (paper_dir / "chunk_mapping.json").exists()


def get_vector_store_for_paper(document_id: str) -> str | None:
    """Get the vector store ID for a specific paper."""
    paper_dir = find_paper_directory(document_id)
    if not paper_dir:
        return None
    
    mapping = load_chunk_mapping(str(paper_dir))
    if not mapping:
        return None
    
    return mapping.get("vector_store_id")


def run_test_query_smart(pdf_name: str, query: str = "Explain the architecture of the paper"):
    """Run test query using the correct vector store for the paper or all stores if not found."""
    # Try to find the specific vector store for this paper
    vector_store_id = get_vector_store_for_paper(pdf_name)
    
    if vector_store_id:
        print(f"🎯 Found paper in vector store: {vector_store_id}")
        vector_store_ids = [vector_store_id]
    else:
        print("🔍 Paper not found, searching all vector stores")
        vector_store_ids = get_all_active_vector_stores()
        if not vector_store_ids:
            print("❌ No active vector stores found")
            return
    
    print(f"\n🔍 Test Query for: {pdf_name}")
    response = client.responses.create(
        model="gpt-5-nano",
        input=query,
        tools=[{
            "type": "file_search",
            "vector_store_ids": vector_store_ids,
            "filters": {
                "type": "eq",
                "key": "pdf",
                "value": pdf_name
            }
        }],
        extra_body={"include": ["output[*].file_search_call.search_results"]},
        service_tier="flex"
    )

    # Print retrieved chunks based on annotations in the final message
    final_message = next((o for o in response.output if o.type == "message"), None)
    if final_message is None:
        print("❌ No assistant message found in the response.")
        return

    content_block = next((c for c in final_message.content if c.type == "output_text"), None)
    if content_block is None:
        print("❌ No output_text found in assistant message.")
        return

    print("📚 Retrieved Context Chunks:")
    print("-" * 80)
    
    chunk_count = 0
    for ann in content_block.annotations:
        if ann.type != "file_citation":
            continue

        filename = ann.filename
        chunk_count += 1
        
        # Look up the actual chunk content
        chunk_info = lookup_chunk_content(filename, pdf_name)
        
        if chunk_info:
            is_image = chunk_info['metadata']['chunk_type'] == 'image'
            
            if is_image:
                source_type = "🖼️ Image"
            else:
                source_type = "📄 Text"
                
            metadata = chunk_info['metadata']
            
            print(f"\n{chunk_count}. {source_type} → {filename}")
            print(f"   📍 Page {metadata.get('page_number', '?')}, Chunk {metadata.get('chunk_index', '?')}, Tokens: {metadata.get('token_count', '?')}")
            
            if metadata.get('heading_text'):
                heading_level = metadata.get('heading_level', 1)
                print(f"   📋 Section: {'#' * heading_level} {metadata.get('heading_text')}")
            
            if is_image and metadata.get('image_id'):
                print(f"   🖼️  Image ID: {metadata.get('image_id')}")
                
            print(f"   📄 Content:")
            content_lines = chunk_info['content'].split('\n')
            for i, line in enumerate(content_lines[:10]):  # Show first 10 lines
                print(f"      {line}")
            if len(content_lines) > 10:
                print(f"      ... ({len(content_lines) - 10} more lines)")
                
        else:
            print(f"\n{chunk_count}. ❓ Unknown → {filename}")
            print(f"   ⚠️  Content not found in mapping")
    
    print("-" * 80)

    # Print the actual answer
    print("\n📝 Final Answer:\n")
    print(content_block.text.strip())


def update_existing_chunk_mappings_with_vector_store():
    """Update existing chunk_mapping.json files to include vector_store_id from registry."""
    if not EXISTING_VECTOR_STORE_ID:
        print("⚠️ No existing vector store ID provided, skipping update of existing mappings")
        return
    
    updated_count = 0
    input_path = Path(INPUT_ROOT)
    
    for item in input_path.iterdir():
        if not item.is_dir():
            continue
            
        mapping_file = item / "chunk_mapping.json"
        if not mapping_file.exists():
            continue
            
        try:
            # Load existing mapping
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            
            # Check if vector_store_id is missing
            if "vector_store_id" not in mapping:
                mapping["vector_store_id"] = EXISTING_VECTOR_STORE_ID
                mapping["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Save updated mapping
                with open(mapping_file, 'w', encoding='utf-8') as f:
                    json.dump(mapping, f, indent=2, ensure_ascii=False)
                
                updated_count += 1
                print(f"✅ Updated {item.name}/chunk_mapping.json with vector_store_id")
                
        except Exception as e:
            print(f"⚠️ Error updating {mapping_file}: {e}")
    
    if updated_count > 0:
        print(f"📋 Updated {updated_count} existing chunk mapping files with vector_store_id")
    else:
        print("📋 No existing chunk mappings needed vector_store_id updates")


def main():
    input_path = Path(INPUT_ROOT)
    if not input_path.exists():
        print(f"❌ Input directory not found: {INPUT_ROOT}")
        return

    # Update existing chunk mappings with vector store ID
    print("🔄 Checking existing chunk mappings for vector_store_id...")
    update_existing_chunk_mappings_with_vector_store()

    # Find all paper directories (each should contain mistral_response.json)
    paper_dirs = []
    for item in input_path.iterdir():
        if item.is_dir() and (item / "mistral_response.json").exists():
            paper_dirs.append(item)
    
    if not paper_dirs:
        print("❌ No paper directories with mistral_response.json found.")
        return
    
    paper_dirs = sorted(paper_dirs)
    print(f"Found {len(paper_dirs)} paper directories")

    # Separate already uploaded from new papers
    already_uploaded = []
    papers_to_upload = []
    
    for paper_dir in paper_dirs:
        if is_paper_already_uploaded(paper_dir):
            already_uploaded.append(paper_dir.name)
        else:
            papers_to_upload.append(paper_dir)
    
    if already_uploaded:
        print(f"📋 Found {len(already_uploaded)} already uploaded papers (skipping):")
        for name in already_uploaded[:5]:  # Show first 5
            print(f"   ✅ {name}")
        if len(already_uploaded) > 5:
            print(f"   ... and {len(already_uploaded) - 5} more")
    
    if not papers_to_upload:
        print("✅ All papers already uploaded!")
    else:
        print(f"\n🚀 Uploading {len(papers_to_upload)} new papers...")
    
    uploaded_docs = []
    for paper_dir in papers_to_upload:
        # Get document info first to estimate chunk count
        document_loader = DocumentLoader(str(paper_dir.parent))
        try:
            doc = document_loader._load_single_document(paper_dir)
            if doc:
                content_chunker = ContentChunker(chunk_size=512, chunk_overlap=64)
                chunked_doc = content_chunker.chunk_document(doc)
                estimated_chunks = len(chunked_doc.chunks)
                
                # Get vector store with capacity for this paper
                vector_store_id = get_vector_store_with_capacity(estimated_chunks)
                
                # Upload the paper
                document_id = upload_paper_to_vector_store(paper_dir, vector_store_id)
                if document_id:
                    uploaded_docs.append(document_id)
                    print(f"✅ Uploaded: {document_id}")
                    
        except Exception as e:
            print(f"❌ Error processing {paper_dir.name}: {e}")

    # Run test queries
    if uploaded_docs:
        print(f"\n🧪 Running test query on first uploaded document: {uploaded_docs[0]}")
        run_test_query_smart(uploaded_docs[0], "Describe the architecture or main methodology presented in this paper")
    
    # Run test query on the Split-NER paper (will work if already uploaded or newly uploaded)
    print(f"\n🧪 Running test query on Split-NER paper...")
    run_test_query_smart("arora_split-ner_2023", "What is Split-NER and how does it work?")


if __name__ == "__main__":
    # Uncomment the line below to delete all OpenAI files first (dry run by default)
    # delete_all_openai_files()
    #delete_all_openai_files(confirm=True)  # Actually delete files
    
    main()
