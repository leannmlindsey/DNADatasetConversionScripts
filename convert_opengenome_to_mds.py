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

def segment_sequence(sequence: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """
    Split a sequence into chunks of specified size with optional overlap.
    
    Args:
        sequence: The DNA sequence to segment
        chunk_size: The size of each chunk
        overlap: Number of overlapping characters between chunks (default: 0)
    
    Returns:
        List of sequence chunks
    """
    if len(sequence) <= chunk_size:
        return [sequence]
    
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(sequence) - overlap, step):
        chunk = sequence[i:i + chunk_size]
        if len(chunk) >= chunk_size // 2:  # Only include chunks that are at least half the target size
            chunks.append(chunk)
    
    return chunks

def format_genome_text(example: Dict[str, Any], chunk_size: int = None) -> List[Dict[str, str]]:
    """
    Format genomic data for BERT training, optionally segmenting into chunks.
    
    The OpenGenome dataset contains taxonomic classification and genomic sequences.
    This function formats them into text fields suitable for language modeling.
    
    Args:
        example: Dictionary containing the dataset example
        chunk_size: If provided, segment sequences into this size
    
    Returns:
        List of dictionaries with formatted text
    """
    # Extract the two columns from OpenGenome dataset
    # Adjust field names based on actual dataset structure
    taxonomy = example.get('taxonomy', example.get('classification', ''))
    sequence = example.get('text', example.get('sequence', ''))
    
    if chunk_size is None:
        # No segmentation, return as single item
        formatted_text = f"[TAXONOMY] {taxonomy} [SEQUENCE] {sequence}"
        return [{"text": formatted_text}]
    else:
        # Segment the sequence
        sequence_chunks = segment_sequence(sequence, chunk_size)
        formatted_examples = []
        
        for i, chunk in enumerate(sequence_chunks):
            # Include chunk index in the formatting for tracking
            formatted_text = f"[TAXONOMY] {taxonomy} [CHUNK] {i+1}/{len(sequence_chunks)} [SEQUENCE] {chunk}"
            formatted_examples.append({"text": formatted_text})
        
        return formatted_examples

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
                   shard_size: int = 100000, chunk_size: int = None,
                   shards_per_dir: int = 1000) -> None:
    """Convert dataset to MDS format with sharding and optional sequence segmentation
    
    Args:
        dataset: The dataset to convert
        output_dir: Base output directory
        stage: Stage name (stage1 or stage2)
        split: Split name (train, validation, test)
        shard_size: Number of examples per shard
        chunk_size: Size for sequence segmentation
        shards_per_dir: Maximum number of shards per subdirectory
    """
    
    logger.info(f"Converting {stage} {split} to MDS format...")
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Maximum shards per subdirectory: {shards_per_dir}")
    if chunk_size:
        logger.info(f"Segmenting sequences into chunks of {chunk_size}")
    
    # Define the schema for MDS
    columns = {
        'text': 'str'
    }
    
    # Process in batches to create multiple subdirectories
    total_chunks = 0
    shard_count = 0
    current_subdir_index = 0
    
    # Calculate approximate number of chunks
    approx_chunks_per_example = 1 if chunk_size is None else 10  # Estimate
    total_expected_chunks = len(dataset) * approx_chunks_per_example
    chunks_per_subdir = shards_per_dir * shard_size
    
    batch_size = max(1, chunks_per_subdir // approx_chunks_per_example)
    
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        
        # Create subdirectory for this batch
        subdir = output_dir / f"shard_group_{current_subdir_index:04d}"
        subdir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing batch {batch_start}-{batch_end} in {subdir}")
        
        try:
            with MDSWriter(
                out=str(subdir),
                columns=columns,
                size_limit=shard_size
            ) as writer:
                
                for i in range(batch_start, batch_end):
                    example = dataset[i]
                    # Format the example with optional segmentation
                    formatted_examples = format_genome_text(example, chunk_size)
                    
                    # Write each segment to MDS
                    for formatted_example in formatted_examples:
                        writer.write(formatted_example)
                        total_chunks += 1
                    
                    if (i + 1) % 10000 == 0:
                        logger.info(f"Processed {i + 1}/{len(dataset)} examples ({total_chunks} chunks)...")
            
            # Count shards in this subdirectory
            shard_files = list(subdir.glob("*.mds"))
            shard_count += len(shard_files)
            logger.info(f"Created {len(shard_files)} shards in {subdir.name}")
            
            current_subdir_index += 1
            
        except Exception as e:
            logger.error(f"Error converting batch {batch_start}-{batch_end} to MDS: {e}")
            raise
    
    logger.info(f"Successfully converted {stage} {split} to MDS format")
    logger.info(f"Total chunks written: {total_chunks}")
    logger.info(f"Total shards created: {shard_count} across {current_subdir_index} subdirectories")
    
    # Log shard information across all subdirectories
    total_shard_files = list(output_dir.glob("*/*.mds"))
    logger.info(f"Total MDS files: {len(total_shard_files)}")

def save_dataset_info(directories: Dict, base_path: str) -> None:
    """Save information about the converted datasets"""
    
    info = {
        "dataset": "OpenGenome",
        "source": "LongSafari/open-genome",
        "format": "MDS",
        "stages": {
            "stage1": {
                "original_context_length": "8k",
                "segmented_context_length": "1024",
                "splits": ["train", "validation", "test"]
            },
            "stage2": {
                "original_context_length": "131k",
                "segmented_context_length": "8192", 
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
        "text_format": "[TAXONOMY] classification [CHUNK] n/total [SEQUENCE] dna_sequence",
        "segmentation_note": "Sequences are segmented into chunks to fit ModernBERT context windows"
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
        '--shards_per_dir', 
        type=int, 
        default=1000,
        help='Maximum number of shards per subdirectory (default: 1000)'
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
            
            # Convert to MDS with appropriate chunk size
            output_dir = directories[stage][split]
            # Stage 1: segment 8k to 1024, Stage 2: segment 131k to 8192
            chunk_size = 1024 if stage == 'stage1' else 8192
            convert_to_mds(dataset, output_dir, stage, split, args.shard_size, chunk_size, args.shards_per_dir)
    
    if not args.dry_run:
        # Save dataset information
        save_dataset_info(directories, args.data_path)
        
        logger.info("\n=== Conversion Complete ===")
        logger.info(f"Converted datasets available at: {args.data_path}/opengenome_mds/")
        logger.info("Directory structure:")
        for stage in args.stages:
            logger.info(f"  {stage}/")
            for split in args.splits:
                # Count shards across all subdirectories
                shard_files = list(directories[stage][split].glob("**/*.mds"))
                subdirs = list(directories[stage][split].glob("shard_group_*"))
                logger.info(f"    {split}/ ({len(shard_files)} shards across {len(subdirs)} subdirectories)")

if __name__ == "__main__":
    main()
