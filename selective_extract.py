"""
selective_extract.py
-------------------
Selectively re-extract specific SLR features for specific documents.
Supports single feature mode and bulk YAML mode with auto-discovery
of feature locations across prompt blocks. Results are printed to console.

Usage:
  # Single feature mode (re-extract and print to console)
  python selective_extract.py --single --folder mistral_extracted/ --doc-id paladini_you_2024 --feature "Innovative Approach" [--dry-run]
  
  # Bulk YAML mode  
  python selective_extract.py --yaml tasks.yaml --folder mistral_extracted/ [--dry-run]

YAML format:
  tasks:
    - feature: "Innovative Approach"
      doc_id: "paladini_you_2024"
    - feature: "Dataset size"
      doc_id: "chen_ner_2023"
"""

import os
import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Import reusable functions from slr_extract_blocks
from slr_extract_blocks import (
    load_blocks, ask_single_question, get_vector_stores_for_paper,
    SYSTEM_PROMPT, MODEL, DEFAULT_REASONING_EFFORT
)

class FeatureResolver:
    """Auto-discovery system for finding features across prompt blocks."""
    
    def __init__(self, blocks: Dict[str, List[Dict]]):
        self.blocks = blocks
        self._build_feature_index()
    
    def _build_feature_index(self):
        """Build index: feature_name -> (block_name, feature_data)"""
        self.feature_index: Dict[str, List[Tuple[str, Dict]]] = {}
        
        for block_name, features in self.blocks.items():
            for feature_data in features:
                feature_name = feature_data['name']
                if feature_name not in self.feature_index:
                    self.feature_index[feature_name] = []
                self.feature_index[feature_name].append((block_name, feature_data))
    
    def resolve_feature(self, feature_name: str) -> Tuple[str, Dict]:
        """Resolve feature name to (block_name, feature_data). Raises exception if not found or ambiguous."""
        if feature_name not in self.feature_index:
            available_features = list(self.feature_index.keys())
            raise ValueError(f"Feature '{feature_name}' not found. Available features: {available_features}")
        
        matches = self.feature_index[feature_name]
        if len(matches) > 1:
            block_list = [block_name for block_name, _ in matches]
            raise ValueError(f"Feature '{feature_name}' is ambiguous, found in multiple blocks: {block_list}. Please use a more specific name.")
        
        return matches[0]
    
    def list_all_features(self) -> List[str]:
        """Return sorted list of all available features."""
        return sorted(self.feature_index.keys())


class DocumentFinder:
    """Find documents by ID in specified folder."""
    
    def __init__(self, folder_path: Path):
        self.folder_path = folder_path
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    def find_document(self, doc_id: str) -> Path:
        """Find document folder by ID. Raises exception if not found."""
        # Search through all subdirectories for metadata.json with matching id
        for paper_dir in self.folder_path.iterdir():
            if not paper_dir.is_dir():
                continue
                
            metadata_file = paper_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
                    if metadata.get('id') == doc_id:
                        return paper_dir
                except (json.JSONDecodeError, KeyError):
                    continue
        
        raise FileNotFoundError(f"Document with ID '{doc_id}' not found in {self.folder_path}")


class SelectiveExtractor:
    """Main extractor for selective re-processing."""
    
    def __init__(self, folder_path: Path, dry_run: bool = False):
        load_dotenv()
        self.folder_path = folder_path
        self.dry_run = dry_run
        
        # Initialize components
        self.blocks = load_blocks()
        self.feature_resolver = FeatureResolver(self.blocks)
        self.doc_finder = DocumentFinder(folder_path)
        
        print(f"📁 Searching in folder: {folder_path}")
        print(f"🔍 Auto-discovery enabled for {len(self.feature_resolver.list_all_features())} features")
        if dry_run:
            print("🧪 DRY RUN MODE - No API calls will be made")
    
    def extract_single(self, doc_id: str, feature_name: str) -> bool:
        """Extract single feature for single document. Returns True if successful."""
        try:
            # Resolve feature
            block_name, feature_data = self.feature_resolver.resolve_feature(feature_name)
            print(f"🎯 Resolved '{feature_name}' → {block_name}")
            
            # Find document
            paper_dir = self.doc_finder.find_document(doc_id)
            print(f"📄 Found document: {paper_dir.name}")
            
            # Load existing slr_features.json
            slr_file = paper_dir / "slr_features.json"
            if not slr_file.exists():
                raise FileNotFoundError(f"No slr_features.json found in {paper_dir}")
            
            existing_data = json.loads(slr_file.read_text(encoding='utf-8'))
            
            if self.dry_run:
                print(f"🧪 DRY RUN: Would re-extract '{feature_name}' for '{doc_id}'")
                print(f"   Block: {block_name}")
                print(f"   Question: {feature_data['question'][:100]}...")
                return True
            
            # Get vector stores and extract
            vector_store_ids = get_vector_stores_for_paper(paper_dir)
            pdf_name = self._get_pdf_name(paper_dir)
            
            print(f"🔄 Re-extracting '{feature_name}'...")
            result = ask_single_question(pdf_name, feature_data, vector_store_ids)
            
            # Print result to console
            print(f"\n🔍 Result for '{feature_name}' in '{doc_id}':")
            print("=" * 60)
            
            if isinstance(result, dict):
                # Handle structured response format
                if 'answer' in result:
                    print(f"Answer: {result['answer']}")
                if 'reasoning' in result:
                    print(f"\nReasoning: {result['reasoning']}")
                if 'chunks' in result:
                    print(f"\nSource chunks: {len(result['chunks'])} chunks used")
            else:
                # Handle simple string format
                print(result)
            
            print("=" * 60)
            
            # Update existing data
            if block_name not in existing_data:
                existing_data[block_name] = {}
            existing_data[block_name][feature_name] = result
            
            # Save updated file
            slr_file.write_text(json.dumps(existing_data, indent=2, ensure_ascii=False), encoding='utf-8')
            print(f"✅ Updated {slr_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to extract '{feature_name}' for '{doc_id}': {e}")
            return False
    

    def extract_bulk(self, yaml_file: Path) -> Dict[str, bool]:
        """Extract multiple features from YAML file. Returns success status per task."""
        if not yaml_file.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_file}")
        
        with open(yaml_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'tasks' not in config:
            raise ValueError("YAML file must contain 'tasks' key with list of tasks")
        
        tasks = config['tasks']
        print(f"📋 Processing {len(tasks)} tasks from {yaml_file}")
        
        results = {}
        for i, task in enumerate(tasks, 1):
            if 'feature' not in task or 'doc_id' not in task:
                print(f"❌ Task {i}: Missing 'feature' or 'doc_id' keys")
                results[f"task_{i}"] = False
                continue
            
            print(f"\n📝 Task {i}/{len(tasks)}: {task['feature']} for {task['doc_id']}")
            success = self.extract_single(task['doc_id'], task['feature'])
            results[f"task_{i}"] = success
        
        # Summary
        successful = sum(results.values())
        print(f"\n📊 Summary: {successful}/{len(tasks)} tasks completed successfully")
        
        return results
    
    def _get_pdf_name(self, paper_dir: Path) -> str:
        """Get PDF name for vector store filtering."""
        metadata_file = paper_dir / "metadata.json"
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
            return metadata.get("id", paper_dir.name)
        return paper_dir.name


def main():
    parser = argparse.ArgumentParser(description="Selective SLR feature re-extraction")
    parser.add_argument("--folder", help="Folder containing extracted documents")
    parser.add_argument("--dry-run", action="store_true", help="Test mode - no API calls")
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    
    # Single feature mode
    mode_group.add_argument("--single", action="store_true", help="Single feature mode")
    parser.add_argument("--doc-id", help="Document ID (required for single mode)")
    parser.add_argument("--feature", help="Feature name (required for single mode)")
    
    # Bulk mode
    mode_group.add_argument("--yaml", help="YAML file with bulk tasks")
    
    
    # List mode
    mode_group.add_argument("--list-features", action="store_true", help="List all available features")
    
    args = parser.parse_args()
    
    # List features mode (doesn't need folder)
    if args.list_features:
        blocks = load_blocks()
        resolver = FeatureResolver(blocks)
        features = resolver.list_all_features()
        print(f"📝 Available features ({len(features)}):")
        for feature in features:
            print(f"  • {feature}")
        return
    
    # Validate folder for other modes
    if not args.folder:
        parser.error("--folder is required for single and yaml modes")
    
    folder_path = Path(args.folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Validate single mode arguments
    if args.single:
        if not args.doc_id or not args.feature:
            parser.error("--single mode requires both --doc-id and --feature")
    
    # Initialize extractor
    extractor = SelectiveExtractor(folder_path, dry_run=args.dry_run)
    
    try:
        if args.single:
            # Single feature mode
            success = extractor.extract_single(args.doc_id, args.feature)
            exit(0 if success else 1)
            
        elif args.yaml:
            # Bulk YAML mode
            yaml_file = Path(args.yaml)
            results = extractor.extract_bulk(yaml_file)
            failed_tasks = [task for task, success in results.items() if not success]
            exit(0 if not failed_tasks else 1)
            
    except Exception as e:
        print(f"💥 Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()