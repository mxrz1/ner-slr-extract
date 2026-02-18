#!/usr/bin/env python3
"""
PDF extraction using Mistral OCR API.

This module processes PDF files using Mistral's OCR API to extract text content
and images with base64 encoding from academic papers.
"""

import json
import base64
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import os
import time
import re
from uuid import uuid4
from tqdm import tqdm
from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model
from pydantic import BaseModel, Field
import bibtexparser


class FigureAnnotation(BaseModel):
    """Pydantic model for figure bbox annotation with description and insights."""
    
    description: str = Field(
        ...,
        description="""You are a domain-savvy AI assistant who explains scientific figures to graduate-level data-science readers in clear, accessible language (≈101–200-level).

Your job is to produce ONE well-structured paragraph (≈120–200 words) that:

1. **Identifies** what kind of figure it is (e.g. "bar chart", "Transformer architecture diagram") *without* labeling it "figure" or discussing slide/page formatting.

2. **Describes** every major component or axis:
   – For plots: explain axes, legends, key trends, notable data points.
   – For architectures: walk through components in processing order and how data flows.
   – For diagrams/flows: describe entities and their interactions step-by-step.

3. **Interprets** the significance in one sentence ("This suggests…", "These results confirm…").
Avoid irrelevant layout notes (coordinates, pixel positions, page numbers)."""
    )
    
    key_insights: str = Field(
        ...,
        description="Summarize the main scientific finding, conclusion, or message that this figure conveys in 1-2 concise sentences. Focus on what the figure proves, demonstrates, or reveals about the research. Be factual and avoid speculation."
    )


def get_pdf_files(folder_path: str, use_zotero: bool = True) -> list[dict]:
    """Get PDF files either from Zotero export or direct folder scanning.
    
    Args:
        folder_path: Path to folder containing PDF files (and optionally bibtex file).
        use_zotero: If True, parse bibtex file for metadata. If False, scan folder directly.
        
    Returns:
        List of dictionaries with PDF information.
        
    Raises:
        FileNotFoundError: If the folder doesn't exist or bibtex file missing (when use_zotero=True).
    """
    input_path = Path(folder_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {folder_path}")
    
    pdf_entries = []
    
    if use_zotero:
        # Zotero mode: Parse bibtex file and extract PDF metadata
        bib_files = list(input_path.glob("*.bib"))
        if not bib_files:
            raise FileNotFoundError(f"No .bib file found in {folder_path}")
        
        bib_file = bib_files[0]  # Use first .bib file found
        print(f"Found bibtex file: {bib_file}")
        
        # Parse bibtex file
        with open(bib_file, 'r', encoding='utf-8') as f:
            bib_database = bibtexparser.load(f)
        
        print(f"Processing {len(bib_database.entries)} entries from {bib_file}")
        for entry in bib_database.entries:
            # Check if entry has file field
            if 'file' not in entry:
                entry_id = entry.get('ID', 'unknown')
                print(f"Warning: Entry '{entry_id}' has no file field and will not be processed")
                continue
                
            # Extract file path from file field: split by semicolon first, then find PDF file
            file_field = entry['file']
            file_entries = file_field.split(';')
            
            # Find the first file entry that ends with "application/pdf"
            path = None
            for file_entry in file_entries:
                # Split by colon and check if last part is "application/pdf"
                parts = file_entry.split(':')
                if len(parts) >= 3 and parts[-1] == "application/pdf":
                    path = parts[-2]  # Path is second to last part
                    break
            
            if path is None:
                entry_id = entry.get('ID', 'unknown')
                print(f"Warning: Entry '{entry_id}' has no PDF file path and will not be processed")
                continue
            
            # Only include PDF files
            if path.lower().endswith('.pdf'):
                # Create full path
                full_path = input_path / path
                
                # Extract metadata from bibtex entry
                pdf_entry = {
                    'path': path,
                    'full_path': full_path,
                    'title': entry.get('title', ''),
                    'doi': entry.get('doi', ''),
                    'year': entry.get('year', ''),
                    'id': entry.get('ID', ''),  # Add bibtex ID for identification
                    'has_metadata': True
                }
                
                # Only add if PDF file actually exists
                if full_path.exists():
                    pdf_entries.append(pdf_entry)
                else:
                    entry_id = entry.get('ID', 'unknown')
                    print(f"Warning: PDF file not found for entry '{entry_id}': {full_path}")
            else:
                entry_id = entry.get('ID', 'unknown')
                print(f"Warning: Entry '{entry_id}' has non-PDF file path: {path}")
        
        print(f"Found {len(pdf_entries)} PDF files with metadata in {folder_path}")
    
    else:
        # Direct mode: Scan folder for PDF files
        pdf_files = list(input_path.glob("*.pdf"))
        print(f"Scanning {folder_path} for PDF files...")
        
        for pdf_file in pdf_files:
            if pdf_file.is_file():
                pdf_entry = {
                    'full_path': pdf_file,
                    'filename': pdf_file.stem,  # PDF filename without extension
                    'id': str(uuid4()),  # Generate UUID for non-bibtex files
                    'has_metadata': False
                }
                pdf_entries.append(pdf_entry)
        
        print(f"Found {len(pdf_entries)} PDF files in {folder_path}")
    
    return pdf_entries


def process_single_pdf(pdf_path: Path, client: Mistral) -> Optional[dict]:
    """Process a single PDF file using Mistral OCR API with bbox annotations.
    
    Args:
        pdf_path: Path to the PDF file.
        client: Mistral client instance.
        
    Returns:
        OCR response dictionary or None if error.
    """
    try:
        # Encode PDF to base64
        with open(pdf_path, "rb") as pdf_file:
            base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
        
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}"
            },
            bbox_annotation_format=response_format_from_pydantic_model(FigureAnnotation),
            include_image_base64=True
        )
        
        return ocr_response
        
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")
        return None


def check_extraction_exists(pdf_entry: dict, base_output_dir: str) -> bool:
    """Check if extraction already exists for this PDF.
    
    Args:
        pdf_entry: PDF entry dictionary (with or without metadata).
        base_output_dir: Base output directory.
        
    Returns:
        True if mistral_response.json already exists, False otherwise.
    """
    pdf_path = pdf_entry['full_path']
    
    # Create folder name based on available data (same logic as save_results)
    if pdf_entry.get('has_metadata', False):
        # Zotero mode: year_title_id
        year = pdf_entry.get('year', 'unknown')
        title = pdf_entry.get('title', '').replace(' ', '_').replace('/', '_').replace('\\', '_').replace("{", "").replace("}", "")[:50]
        pdf_id = pdf_path.stem
        folder_name = f"{year}_{title}_{pdf_id}"
    else:
        # Direct mode: just use PDF filename
        folder_name = pdf_path.stem
    
    # Check if mistral_response.json exists
    output_dir = Path(base_output_dir) / folder_name
    response_file = output_dir / "mistral_response.json"
    print(response_file)
    return response_file.exists()


def read_multiline_caption(lines: list[str], start_idx: int, initial_caption: str) -> str:
    """Read multi-line caption continuation until empty line.
    
    Args:
        lines: All markdown lines.
        start_idx: Index to start reading from.
        initial_caption: The first line of the caption.
        
    Returns:
        Complete multi-line caption text.
    """
    caption_parts = [initial_caption]
    
    # Pattern to detect other figures/images that would stop caption reading
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    figure_pattern = r'^\s*\*{0,2}((Figure|Fig\.)\s+\d+[\.\:]?)\*{0,2}\s*(.*)'
    
    i = start_idx + 1
    while i < len(lines):
        line = lines[i].strip()
        if not line:  # Empty line - stop reading
            break
        # Stop if we encounter another figure or image
        if re.search(image_pattern, line) or re.match(figure_pattern, line, re.IGNORECASE):
            break
        caption_parts.append(line)
        i += 1
    
    return " ".join(caption_parts)


def extract_captions_from_markdown(markdown_content: str) -> dict[str, str]:
    """Extract figure captions from markdown content.
    
    Args:
        markdown_content: Merged markdown content from OCR response.
        
    Returns:
        Dictionary mapping image IDs to their captions.
    """
    caption_mapping = {}
    lines = markdown_content.split('\n')
    
    # Patterns
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    figure_pattern = r'^\s*\*{0,2}((Figure|Fig\.)\s+\d+[\.\:]?)\*{0,2}\s*(.*)'
    subfig_pattern = r'^\s*\(([a-z])\)\s*(.+)'
    
    for i, line in enumerate(lines):
        # Step 1: Find image
        image_match = re.search(image_pattern, line)
        if image_match:
            image_file = image_match.group(2)  # Extract filename
            subfigures_temp = []  # Temporary list to collect subfigures
            
            # Step 2: Search up to 5 lines for captions
            main_caption = ""
            for j in range(1, min(6, len(lines) - i)):
                next_line = lines[i + j].strip()
                if not next_line:  # Skip empty lines
                    continue
                
                # Check for subfigure caption ONLY if we haven't hit another image yet
                subfig_match = re.match(subfig_pattern, next_line)
                if subfig_match and not main_caption:  # Only collect subfigures before finding main caption
                    # Found subfigure - use single line only (no multiline)
                    subfig_label = subfig_match.group(1)
                    subfig_text = subfig_match.group(2).strip()
                    subfig_caption = f"({subfig_label}) {subfig_text}"
                    subfigures_temp.append(subfig_caption)
                    continue
                
                # Stop collecting subfigures if we encounter another image
                if re.search(image_pattern, next_line):
                    # Don't break - continue searching for Figure caption, but stop collecting subfigures
                    pass
                
                # Check for Figure caption
                figure_match = re.match(figure_pattern, next_line, re.IGNORECASE)
                if figure_match:
                    # Found Figure caption - combine figure label and text
                    figure_label = figure_match.group(1).strip()  # "Figure 1." or "Fig. 1."
                    figure_text = figure_match.group(3).strip()   # Caption text after label
                    main_caption = f"{figure_label} {figure_text}".strip()
                    break  # Found Figure caption, we're done
            
            # Combine what we found for this specific image
            if main_caption and subfigures_temp:
                # This image has subfigures - combine main caption with its own subfigure only
                caption_mapping[image_file] = f"{main_caption} | {subfigures_temp[0]}"  # Only first subfigure belongs to this image
            elif main_caption:
                # Only main caption found
                caption_mapping[image_file] = main_caption
            elif subfigures_temp:
                # Only subfigures found (no main caption)
                caption_mapping[image_file] = subfigures_temp[0]  # Only first subfigure belongs to this image
    
    return caption_mapping


def save_results(ocr_response: dict, pdf_entry: dict, base_output_dir: str) -> None:
    """Save OCR results with merged markdown, extracted images, and metadata.
    
    Args:
        ocr_response: Mistral OCR response.
        pdf_entry: PDF entry dictionary (with or without metadata).
        base_output_dir: Base output directory.
    """
    # Convert response to dict if needed
    response_data = ocr_response.model_dump() if hasattr(ocr_response, 'model_dump') else dict(ocr_response)
    
    pdf_path = pdf_entry['full_path']
    
    # Create folder name based on available data
    if pdf_entry.get('has_metadata', False):
        # Zotero mode: year_title_id
        year = pdf_entry.get('year', 'unknown')
        title = pdf_entry.get('title', '').replace(' ', '_').replace('/', '_').replace('\\', '_').replace("{", "").replace("}", "")[:50]
        pdf_id = pdf_path.stem
        folder_name = f"{year}_{title}_{pdf_id}"
    else:
        # Direct mode: just use PDF filename
        folder_name = pdf_path.stem
    
    # Create output directory
    output_dir = Path(base_output_dir) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 3. Extract and save images, merge markdown
        merged_markdown = ""
        
        if 'pages' in response_data:
            for page in response_data['pages']:
                # Add page markdown to merged content
                if 'markdown' in page:
                    merged_markdown += page['markdown'] + "\n\n"
        
        # 4. Extract captions from markdown
        caption_mapping = extract_captions_from_markdown(merged_markdown)
        
        # Parse embedded JSON strings in image annotations and add captions
        if 'pages' in response_data:
            for page in response_data['pages']:
                if 'images' in page:
                    for image in page['images']:
                        if 'image_annotation' in image and isinstance(image['image_annotation'], str):
                            try:
                                # Parse the JSON string to convert it to actual JSON
                                parsed_annotation = json.loads(image['image_annotation'])
                                image['image_annotation'] = parsed_annotation
                            except json.JSONDecodeError:
                                # If parsing fails, leave it as string
                                pass
                        
                        # Add caption to image annotation if we found one
                        if 'id' in image:
                            image_id = image['id']
                            if image_id in caption_mapping:
                                # Ensure image_annotation exists as dict
                                if 'image_annotation' not in image:
                                    image['image_annotation'] = {}
                                elif isinstance(image['image_annotation'], str):
                                    # Try to parse it one more time
                                    try:
                                        image['image_annotation'] = json.loads(image['image_annotation'])
                                    except json.JSONDecodeError:
                                        image['image_annotation'] = {}
                                
                                # Add caption to image annotation
                                image['image_annotation']['caption'] = caption_mapping[image_id]
        
        # 1. Save processed JSON response
        json_file = output_dir / "mistral_response.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)
        
        # 2. Save metadata
        metadata_file = output_dir / "metadata.json"
        # Convert Path objects to strings for JSON serialization
        metadata_copy = pdf_entry.copy()
        metadata_copy['full_path'] = str(metadata_copy['full_path'])
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_copy, f, indent=2, ensure_ascii=False)
        
        # 5. Extract and save images from pages
        if 'pages' in response_data:
            for page in response_data['pages']:
                # Extract and save images from this page
                if 'images' in page:
                    for image in page['images']:
                        if 'id' in image and 'image_base64' in image:
                            image_id = image['id']
                            
                            # Remove data URL prefix (e.g., "data:image/jpeg;base64,")
                            base64_string = image['image_base64']
                            if base64_string.startswith('data:'):
                                base64_string = base64_string.split(',', 1)[1]
                            
                            image_data = base64.b64decode(base64_string)
                            
                            # Save image with original ID as filename
                            image_file = output_dir / image_id
                            with open(image_file, 'wb') as f:
                                f.write(image_data)
        
        # 6. Save merged markdown file
        markdown_file = output_dir / "content.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(merged_markdown)
            
        print(f"Saved results to: {output_dir}")
        
    except Exception as e:
        print(f"Error saving results for {pdf_path.name}: {e}")


def process_all_pdfs(pdf_entries: list[dict], client: Mistral, output_folder: str, input_folder: str) -> None:
    """Process all PDF files in the list.
    
    Args:
        pdf_entries: List of PDF entry dictionaries with metadata.
        client: Mistral client instance.
        output_folder: Output directory for extracted data.
        input_folder: Input folder for relative path calculation.
    """
    if not pdf_entries:
        print("No PDF files to process")
        return
    
    processed = 0
    failed = 0
    skipped = 0
    
    for pdf_entry in tqdm(pdf_entries, desc="Processing PDFs"):
        pdf_path = pdf_entry['full_path']
        
        # Check if extraction already exists
        if check_extraction_exists(pdf_entry, output_folder):
            print(f"Skipping {pdf_path.name} - extraction already exists")
            skipped += 1
            continue

        start_time = time.time()

        # Process PDF with Mistral
        ocr_response = process_single_pdf(pdf_path, client)

        if ocr_response:
            # Save results
            save_results(ocr_response, pdf_entry, output_folder)
            processed += 1
        else:
            failed += 1

        # Smart rate limiting - only sleep remaining time to ensure 1 second intervals
        elapsed = time.time() - start_time
        if elapsed < 1.1:
            time.sleep(1.0 - elapsed)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed}")
    print(f"Skipped (already extracted): {skipped}")
    print(f"Failed: {failed}")


def main() -> None:
    """Main function to run the PDF extraction pipeline."""
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set")
    
    # Create Mistral client
    client = Mistral(api_key=api_key)
    
    # Configuration
    input_folder = "mypaper_slr"  # Folder containing PDF files
    output_folder = "mypaper_slr_extracted"  # Folder to save extracted data
    use_zotero_mode = True  # Set to False to scan folder directly for PDFs
    
    # Get all PDF files
    pdf_files = get_pdf_files(input_folder, use_zotero=use_zotero_mode)

    # Process all PDFs
    process_all_pdfs(pdf_files, client, output_folder, input_folder)

    # test pdf
    #print(process_single_pdf("../papers_test/Arora und Park - 2023 - Split-NER Named Entity Recognition via Two Question-Answering-based Classifications.pdf", client))

if __name__ == "__main__":
    main()
