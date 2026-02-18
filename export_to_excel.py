"""
export_to_excel.py
------------------
Export SLR extracted data from JSON file to Excel format.
Each chunk gets its own column: block name, feature name, answer, chunk_1, chunk_2, etc.
"""

import json
import pandas as pd
from pathlib import Path
import argparse
import os
import yaml


def export_single_paper_to_excel(json_file_path: str) -> str | None:
    """
    Export SLR features from a single JSON file to Excel format.
    Each chunk gets its own column (chunk_1, chunk_2, etc.)
    
    Args:
        json_file_path: Path to the slr_features.json file
        
    Returns:
        Path to the created Excel file
    """
    
    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Generate output path in same folder
    json_path = Path(json_file_path)
    output_excel_path = str(json_path.parent / f"{json_path.stem}.xlsx")
    
    # First pass: determine maximum number of chunks across all questions
    max_chunks = 0
    for block_name, block_data in data.items():
        if block_name == '_summary':
            continue
        for feature_name, feature_data in block_data.items():
            chunks_used = feature_data.get('chunks_used', [])
            max_chunks = max(max_chunks, len(chunks_used))
    
    # Prepare Excel data
    excel_rows = []
    
    # Process each block
    for block_name, block_data in data.items():
        if block_name == '_summary':
            continue
            
        # Process each feature/question in the block
        for feature_name, feature_data in block_data.items():
            answer = feature_data.get('answer', '')
            chunks_used = feature_data.get('chunks_used', [])
            
            # Create row with basic info
            row = {
                'block_name': block_name,
                'feature_name': feature_name,
                'answer': answer
            }
            
            # Add chunk columns
            for i in range(max_chunks):
                chunk_col_name = f'chunk_{i+1}'
                if i < len(chunks_used):
                    chunk_content = chunks_used[i].get('content', '')
                    row[chunk_col_name] = chunk_content
                else:
                    row[chunk_col_name] = ''  # Empty for missing chunks
            
            excel_rows.append(row)
    
    # Write to Excel
    if excel_rows:
        # Create DataFrame
        df = pd.DataFrame(excel_rows)
        
        # Reorder columns to have basic info first
        column_order = ['block_name', 'feature_name', 'answer']
        chunk_columns = [f'chunk_{i+1}' for i in range(max_chunks)]
        column_order.extend(chunk_columns)
        df = df[column_order]
        
        # Write to Excel
        df.to_excel(output_excel_path, index=False, engine='openpyxl')
        
        print(f"✅ Exported {len(excel_rows)} rows to {output_excel_path}")
        print(f"📊 Max chunks per question: {max_chunks}")
        return output_excel_path
    else:
        print("❌ No data found to export")
        return None


def load_feature_filter(yaml_file: str) -> set[str] | None:
    """
    Load feature filter from YAML file.
    
    Args:
        yaml_file: Path to YAML file containing list of feature IDs to include
        
    Returns:
        Set of feature IDs to include, or None if all features should be included
    """
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if isinstance(data, list):
            return set(data)
        elif isinstance(data, dict) and 'features' in data:
            return set(data['features'])
        else:
            print(f"❌ Invalid YAML format. Expected list of features or dict with 'features' key")
            return None
    except Exception as e:
        print(f"❌ Error loading feature filter: {e}")
        return None


def export_merged_papers_to_excel(papers_folder: str, slr_filename: str = "slr_features.json", output_filename: str = "merged_papers.xlsx", transpose: bool = False, feature_filter_yaml: str = None) -> str | None:
    """
    Export SLR features from all papers merged into a single Excel file.
    Features are rows, papers are columns (unless transposed).
    
    Args:
        papers_folder: Path to folder containing paper subfolders
        slr_filename: Name of the SLR features JSON file (default: slr_features.json)
        output_filename: Name of the output Excel file (default: merged_papers.xlsx)
        transpose: If True, papers are rows and features are columns
        feature_filter_yaml: Path to YAML file with features to include
        
    Returns:
        Path to the created Excel file or None if failed
    """
    papers_path = Path(papers_folder)
    
    if not papers_path.exists():
        print(f"❌ Papers folder not found: {papers_folder}")
        return None
    
    if not papers_path.is_dir():
        print(f"❌ Path is not a directory: {papers_folder}")
        return None
    
    print(f"🔍 Merging papers from: {papers_folder}")
    
    # Load feature filter if provided
    allowed_features = None
    if feature_filter_yaml:
        allowed_features = load_feature_filter(feature_filter_yaml)
        if allowed_features is None:
            return None
        print(f"📋 Filtering to {len(allowed_features)} features from YAML")
    
    # Collect all paper data
    all_papers_data = {}
    all_features = set()  # Track all unique features across papers
    
    # First pass: collect all data and identify all features
    for paper_dir in papers_path.iterdir():
        if not paper_dir.is_dir():
            continue
            
        slr_json_file = paper_dir / slr_filename
        
        if not slr_json_file.exists():
            print(f"⏭️  Skipping {paper_dir.name} (no {slr_filename} found)")
            continue
        
        try:
            with open(slr_json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            paper_name = paper_dir.name
            all_papers_data[paper_name] = data
            
            # Collect all feature identifiers (block_name.feature_name)
            for block_name, block_data in data.items():
                if block_name == '_summary':
                    continue
                for feature_name in block_data.keys():
                    feature_id = f"{block_name}.{feature_name}"
                    # Apply feature filter if provided
                    if allowed_features is None or feature_id in allowed_features:
                        all_features.add(feature_id)
            
            print(f"📄 Loaded {paper_dir.name}")
            
        except Exception as e:
            print(f"❌ Error loading {paper_dir.name}: {e}")
            continue
    
    if not all_papers_data:
        print("❌ No valid papers found")
        return None
    
    # Sort features: use YAML order if filter provided, otherwise alphabetical
    if allowed_features and feature_filter_yaml:
        # Preserve order from YAML file
        with open(feature_filter_yaml, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        yaml_features = yaml_data if isinstance(yaml_data, list) else yaml_data.get('features', [])
        # Use YAML order for features that exist, then add any extras alphabetically
        sorted_features = [f for f in yaml_features if f in all_features]
        remaining_features = sorted(all_features - set(sorted_features))
        sorted_features.extend(remaining_features)
    else:
        sorted_features = sorted(all_features)
    
    sorted_papers = sorted(all_papers_data.keys())
    
    print(f"📊 Found {len(sorted_features)} unique features across {len(sorted_papers)} papers")
    
    # Build the merged data structure
    if transpose:
        # Transposed mode: papers as rows, features as columns
        excel_rows = []
        
        for paper_name in sorted_papers:
            paper_data = all_papers_data[paper_name]
            
            row = {'paper_name': paper_name}
            
            # Add data for each feature
            for feature_id in sorted_features:
                block_name, feature_name = feature_id.split('.', 1)
                
                # Get the answer for this feature from this paper
                if block_name in paper_data and feature_name in paper_data[block_name]:
                    answer = paper_data[block_name][feature_name].get('answer', '')
                else:
                    answer = ''  # Feature not present in this paper
                
                row[feature_id] = answer
            
            excel_rows.append(row)
    else:
        # Normal mode: features as rows, papers as columns
        excel_rows = []
        
        for feature_id in sorted_features:
            block_name, feature_name = feature_id.split('.', 1)
            
            row = {
                'block_name': block_name,
                'feature_name': feature_name,
                'feature_id': feature_id
            }
            
            # Add data for each paper
            for paper_name in sorted_papers:
                paper_data = all_papers_data[paper_name]
                
                # Get the answer for this feature from this paper
                if block_name in paper_data and feature_name in paper_data[block_name]:
                    answer = paper_data[block_name][feature_name].get('answer', '')
                else:
                    answer = ''  # Feature not present in this paper
                
                row[paper_name] = answer
            
            excel_rows.append(row)
    
    # Create DataFrame and save
    if excel_rows:
        df = pd.DataFrame(excel_rows)
        
        # Reorder columns based on mode
        if transpose:
            column_order = ['paper_name'] + sorted_features
        else:
            column_order = ['block_name', 'feature_name', 'feature_id'] + sorted_papers
        df = df[column_order]
        
        # Generate output path
        output_path = papers_path / output_filename
        
        # Write to Excel
        df.to_excel(output_path, index=False, engine='openpyxl')
        
        print(f"✅ Exported merged data to {output_path}")
        if transpose:
            print(f"📊 {len(excel_rows)} papers × {len(sorted_features)} features (transposed)")
        else:
            print(f"📊 {len(excel_rows)} features × {len(sorted_papers)} papers")
        return str(output_path)
    else:
        print("❌ No data to export")
        return None


def export_all_papers_to_excel(papers_folder: str, slr_filename: str = "slr_features.json") -> None:
    """
    Export SLR features from all papers in a folder to Excel format.
    
    Args:
        papers_folder: Path to folder containing paper subfolders
        slr_filename: Name of the SLR features JSON file (default: slr_features.json)
    """
    papers_path = Path(papers_folder)
    
    if not papers_path.exists():
        print(f"❌ Papers folder not found: {papers_folder}")
        return
    
    if not papers_path.is_dir():
        print(f"❌ Path is not a directory: {papers_folder}")
        return
    
    print(f"🔍 Searching for papers in: {papers_folder}")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Process each subfolder
    for paper_dir in papers_path.iterdir():
        if not paper_dir.is_dir():
            continue
            
        slr_json_file = paper_dir / slr_filename
        
        if not slr_json_file.exists():
            print(f"⏭️  Skipping {paper_dir.name} (no {slr_filename} found)")
            skipped_count += 1
            continue
        
        # Check if Excel file already exists
        excel_file = paper_dir / f"{slr_json_file.stem}.xlsx"
        if excel_file.exists():
            print(f"📄 {paper_dir.name} (Excel already exists)")
            skipped_count += 1
            continue
        
        print(f"📄 Processing {paper_dir.name}...")
        
        try:
            result = export_single_paper_to_excel(str(slr_json_file))
            if result:
                processed_count += 1
                print(f"  ✅ Created Excel file")
            else:
                error_count += 1
                print(f"  ❌ Failed to create Excel file")
        except Exception as e:
            error_count += 1
            print(f"  ❌ Error processing {paper_dir.name}: {e}")
    
    print(f"\n📊 Summary:")
    print(f"  • Processed: {processed_count}")
    print(f"  • Skipped: {skipped_count}")
    print(f"  • Errors: {error_count}")
    print(f"  • Total folders: {processed_count + skipped_count + error_count}")


def main():
    """Command line interface for the Excel export functionality."""
    parser = argparse.ArgumentParser(description="Export SLR features to Excel format")
    parser.add_argument("--folder", "-f", 
                        help="Folder containing paper subfolders (batch mode)")
    parser.add_argument("--file", "-F", 
                        help="Single JSON file to convert (single mode)")
    parser.add_argument("--merge", "-m", action="store_true",
                        help="Merge all papers into single Excel (features as rows, papers as columns)")
    parser.add_argument("--transpose", "-t", action="store_true",
                        help="Transpose merged output (papers as rows, features as columns)")
    parser.add_argument("--feature-filter", "-ff", 
                        help="YAML file containing list of features to include in output")
    parser.add_argument("--output", "-o", default="merged_papers.xlsx",
                        help="Output filename for merged Excel (default: merged_papers.xlsx)")
    parser.add_argument("--slr-filename", "-s", default="slr_features.json",
                        help="Name of SLR JSON file in each paper folder (default: slr_features.json)")
    
    args = parser.parse_args()
    
    if args.file:
        # Single file mode
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            return
        
        print(f"📄 Processing single file: {args.file}")
        result = export_single_paper_to_excel(args.file)
        if result:
            print(f"🎉 Successfully created: {result}")
        else:
            print("❌ Failed to create Excel file")
            
    elif args.folder:
        if args.merge:
            # Merge mode
            print(f"📊 Merging papers from: {args.folder}")
            result = export_merged_papers_to_excel(args.folder, args.slr_filename, args.output, args.transpose, args.feature_filter)
            if result:
                print(f"🎉 Successfully created merged Excel: {result}")
            else:
                print("❌ Failed to create merged Excel file")
        else:
            # Batch mode
            export_all_papers_to_excel(args.folder, args.slr_filename)
        
    else:
        # Default behavior - use the mistral_extracted folder
        default_folder = "pdf-extraction-mistral-api/mistral_extracted"
        if os.path.exists(default_folder):
            if args.merge:
                print(f"📊 No folder specified, merging from default: {default_folder}")
                result = export_merged_papers_to_excel(default_folder, args.slr_filename, args.output, args.transpose, args.feature_filter)
                if result:
                    print(f"🎉 Successfully created merged Excel: {result}")
                else:
                    print("❌ Failed to create merged Excel file")
            else:
                print(f"📁 No folder specified, using default: {default_folder}")
                export_all_papers_to_excel(default_folder, args.slr_filename)
        else:
            print("❌ Please specify either --folder or --file")
            parser.print_help()


if __name__ == "__main__":
    main()
