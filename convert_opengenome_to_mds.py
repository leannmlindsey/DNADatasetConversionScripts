#!/usr/bin/env python3
"""
OpenGenome to MDS Format Conversion Script

This script downloads the OpenGenome dataset from HuggingFace and converts it
to MDS format for use with ModernBERT training pipeline.

Author: Generated for DNADatasetConversionScripts
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from datasets import load_dataset
from streaming import MDSWriter
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories(base_path: str) -> Dict[str, str]:
    """Create directory structure for OpenGenome data"""
    base_dir = Path(base_path)
    
    directories = {
        'stage1': {
            'train': base_dir / 'opengenome_mds' / 'stage1' / 'train',
            'validation': base_dir / 'opengenome_mds' / 'stage1' / 'validation', 
            'test': base_dir / 'opengenome_mds' / 'stage1' / 'test'
        },
        'stage2': {
            'train': base_dir / 'opengenome_mds' / 'stage2' / 'train',
            'validation': base_dir / 'opengenome_mds' / 'stage2' / 'validation',
            'test': base_dir / 'opengenome_mds' / 'stage2' / 'test'
        },
        'raw': base_dir / 'opengenome_raw'
    }
    
    # Create all directories
    for stage_dirs in directories.values():
        if isinstance(stage_dirs, dict):
            for split_dir in stage_dirs.values():
                split_dir.mkdir(parents=True, exist_ok=True)
        else:
            stage_dirs.mkdir(parents=True, exist_ok=True)
    
    return directories

def format_genome_text(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Format genomic data for BERT training.
    
    The OpenGenome dataset contains taxonomic classification and genomic sequences.
    This function formats them into a single text field suitable for language modeling.
    """
    # Extract the two columns from OpenGenome dataset
    # Adjust field names based on actual dataset structure
    taxonomy = example.get('taxonomy', example.get('classification', ''))
    sequence = example.get('text', example.get('sequence', ''))
    
    # Format: [TAXONOMY] classification [SEQUENCE] dna_sequence
    formatted_text = f"[TAXONOMY] {taxonomy} [SEQUENCE] {sequence}"
    
    return {"text": formatted_text}

def download_and_inspect_dataset(stage: str, split: str) -> Any:
    """Download and inspect OpenGenome dataset"""
    logger.info(f"Downloading OpenGenome {stage} {split}...")
    
    try:
        dataset = load_dataset("LongSafari/open-genome", stage, split=split)
        
        # Inspect the first example to understand structure
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"Sample from {stage} {split}:")
            logger.info(f"Keys: {list(sample.keys())}")
            for key, value in sample.items():
                if isinstance(value, str):
                    preview = value[:100] + "..." if len(value) > 100 else value
                    logger.info(f"  {key}: {preview} (length: {len(value)})")
                else:
                    logger.info(f"  {key}: {value}")
        
        return dataset
    
    except Exception as e:
        logger.error(f"Error downloading {stage} {split}: {e}")
        return None

def convert_to_mds(dataset: Any, output_dir: Path, stage: str, split: str, 
                   shard_size: int = 100000) -> None:
    """Convert dataset to MDS format with sharding"""
    
    logger.info(f"Converting {stage} {split} to MDS format...")
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Output directory: {output_dir}")
    
    # Define the schema for MDS
    columns = {
        'text': 'str'
    }
    
    try:
        with MDSWriter(
            out=str(output_dir),
            columns=columns,
            size_limit=shard_size
        ) as writer:
            
            for i, example in enumerate(dataset):
                # Format the example
                formatted_example = format_genome_text(example)
                
                # Write to MDS
                writer.write(formatted_example)
                
                if (i + 1) % 10000 == 0:
                    logger.info(f"Processed {i + 1}/{len(dataset)} examples...")
        
        logger.info(f"Successfully converted {stage} {split} to MDS format")
        
        # Log shard information
        shard_files = list(output_dir.glob("*.mds"))
        logger.info(f"Created {len(shard_files)} shards")
        
    except Exception as e:
        logger.error(f"Error converting {stage} {split} to MDS: {e}")
        raise

def save_dataset_info(directories: Dict, base_path: str) -> None:
    """Save information about the converted datasets"""
    
    info = {
        "dataset": "OpenGenome",
        "source": "LongSafari/open-genome",
        "format": "MDS",
        "stages": {
            "stage1": {
                "context_length": "8k",
                "splits": ["train", "validation", "test"]
            },
            "stage2": {
                "context_length": "131k", 
                "splits": ["train", "validation", "test"]
            }
        },
        "directory_structure": {
            "stage1": {
                "train": str(directories['stage1']['train']),
                "validation": str(directories['stage1']['validation']),
                "test": str(directories['stage1']['test'])
            },
            "stage2": {
                "train": str(directories['stage2']['train']),
                "validation": str(directories['stage2']['validation']),
                "test": str(directories['stage2']['test'])
            }
        },
        "text_format": "[TAXONOMY] classification [SEQUENCE] dna_sequence"
    }
    
    info_path = Path(base_path) / 'opengenome_mds' / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Dataset info saved to {info_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert OpenGenome to MDS format')
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/DATA',
        help='Base path for data storage'
    )
    parser.add_argument(
        '--stages', 
        nargs='+', 
        choices=['stage1', 'stage2'], 
        default=['stage1', 'stage2'],
        help='Which stages to convert'
    )
    parser.add_argument(
        '--splits', 
        nargs='+', 
        choices=['train', 'validation', 'test'], 
        default=['train', 'validation', 'test'],
        help='Which splits to convert'
    )
    parser.add_argument(
        '--shard_size', 
        type=int, 
        default=100000,
        help='Number of examples per shard'
    )
    parser.add_argument(
        '--dry_run', 
        action='store_true',
        help='Only download and inspect data, do not convert'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting OpenGenome to MDS conversion...")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Stages: {args.stages}")
    logger.info(f"Splits: {args.splits}")
    
    # Setup directories
    directories = setup_directories(args.data_path)
    
    # Process each stage and split
    for stage in args.stages:
        for split in args.splits:
            logger.info(f"\n--- Processing {stage} {split} ---")
            
            # Download dataset
            dataset = download_and_inspect_dataset(stage, split)
            
            if dataset is None:
                logger.warning(f"Skipping {stage} {split} due to download error")
                continue
            
            if args.dry_run:
                logger.info(f"Dry run: would convert {len(dataset)} examples")
                continue
            
            # Convert to MDS
            output_dir = directories[stage][split]
            convert_to_mds(dataset, output_dir, stage, split, args.shard_size)
    
    if not args.dry_run:
        # Save dataset information
        save_dataset_info(directories, args.data_path)
        
        logger.info("\n=== Conversion Complete ===")
        logger.info(f"Converted datasets available at: {args.data_path}/opengenome_mds/")
        logger.info("Directory structure:")
        for stage in args.stages:
            logger.info(f"  {stage}/")
            for split in args.splits:
                shard_files = list(directories[stage][split].glob("*.mds"))
                logger.info(f"    {split}/ ({len(shard_files)} shards)")

if __name__ == "__main__":
    main()
