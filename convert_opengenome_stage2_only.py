#!/usr/bin/env python3
"""
OpenGenome Stage 2 Only Conversion Script

This script processes only Stage 2 of the OpenGenome dataset from HuggingFace
and converts it to MDS format. This is useful when Stage 1 has already been
processed or when you need to resume after a quota error.

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

def setup_stage2_directories(base_path: str) -> Dict[str, Path]:
    """Create directory structure for Stage 2 OpenGenome data only"""
    base_dir = Path(base_path)
    
    directories = {
        'train': base_dir / 'opengenome_mds' / 'stage2' / 'train',
        'validation': base_dir / 'opengenome_mds' / 'stage2' / 'validation',
        'test': base_dir / 'opengenome_mds' / 'stage2' / 'test'
    }
    
    # Create all directories
    for split_dir in directories.values():
        split_dir.mkdir(parents=True, exist_ok=True)
    
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

def format_genome_text(example: Dict[str, Any], chunk_size: int = 8192) -> List[Dict[str, str]]:
    """
    Format genomic data for BERT training, segmenting into chunks.
    
    The OpenGenome dataset contains taxonomic classification and genomic sequences.
    This function formats them into text fields suitable for language modeling.
    
    Args:
        example: Dictionary containing the dataset example
        chunk_size: Size to segment sequences into (default: 8192 for stage 2)
    
    Returns:
        List of dictionaries with formatted text
    """
    # Extract the two columns from OpenGenome dataset
    taxonomy = example.get('taxonomy', example.get('classification', ''))
    sequence = example.get('text', example.get('sequence', ''))
    
    # Segment the sequence
    sequence_chunks = segment_sequence(sequence, chunk_size)
    formatted_examples = []
    
    for i, chunk in enumerate(sequence_chunks):
        # Include chunk index in the formatting for tracking
        formatted_text = f"[TAXONOMY] {taxonomy} [CHUNK] {i+1}/{len(sequence_chunks)} [SEQUENCE] {chunk}"
        formatted_examples.append({"text": formatted_text})
    
    return formatted_examples

def download_and_inspect_stage2(split: str) -> Any:
    """Download and inspect OpenGenome stage2 dataset"""
    logger.info(f"Downloading OpenGenome stage2 {split}...")
    
    try:
        dataset = load_dataset("LongSafari/open-genome", "stage2", split=split)
        
        # Inspect the first example to understand structure
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"Sample from stage2 {split}:")
            logger.info(f"Keys: {list(sample.keys())}")
            for key, value in sample.items():
                if isinstance(value, str):
                    preview = value[:100] + "..." if len(value) > 100 else value
                    logger.info(f"  {key}: {preview} (length: {len(value)})")
                else:
                    logger.info(f"  {key}: {value}")
        
        return dataset
    
    except Exception as e:
        logger.error(f"Error downloading stage2 {split}: {e}")
        return None

def convert_to_mds(dataset: Any, output_dir: Path, split: str, 
                   shard_size: int = 100000, chunk_size: int = 8192,
                   resume_from: int = 0) -> None:
    """
    Convert dataset to MDS format with sharding and sequence segmentation.
    
    Args:
        dataset: The dataset to convert
        output_dir: Output directory for MDS files
        split: Dataset split name
        shard_size: Number of examples per shard
        chunk_size: Size of sequence chunks
        resume_from: Example index to resume from (for interrupted conversions)
    """
    
    logger.info(f"Converting stage2 {split} to MDS format...")
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Segmenting sequences into chunks of {chunk_size}")
    if resume_from > 0:
        logger.info(f"Resuming from example {resume_from}")
    
    # Define the schema for MDS
    columns = {
        'text': 'str'
    }
    
    try:
        # Check if we're resuming
        existing_shards = list(output_dir.glob("*.mds"))
        append_mode = len(existing_shards) > 0 and resume_from > 0
        
        with MDSWriter(
            out=str(output_dir),
            columns=columns,
            size_limit=shard_size
        ) as writer:
            
            total_chunks = 0
            processed_count = 0
            
            # Start from resume_from index
            for i in range(resume_from, len(dataset)):
                example = dataset[i]
                
                try:
                    # Format the example with segmentation
                    formatted_examples = format_genome_text(example, chunk_size)
                    
                    # Write each segment to MDS
                    for formatted_example in formatted_examples:
                        writer.write(formatted_example)
                        total_chunks += 1
                    
                    processed_count += 1
                    
                    if processed_count % 10000 == 0:
                        logger.info(f"Processed {processed_count} examples from index {resume_from} "
                                  f"({i + 1}/{len(dataset)} total, {total_chunks} chunks)...")
                        
                except Exception as e:
                    logger.error(f"Error processing example {i}: {e}")
                    logger.info("Saving progress information for resume...")
                    
                    # Save resume information
                    resume_info = {
                        "split": split,
                        "last_processed_index": i - 1,
                        "next_index": i,
                        "total_chunks_written": total_chunks,
                        "total_examples": len(dataset)
                    }
                    
                    resume_path = output_dir / f"resume_info_{split}.json"
                    with open(resume_path, 'w') as f:
                        json.dump(resume_info, f, indent=2)
                    
                    logger.info(f"Resume information saved to {resume_path}")
                    raise
        
        logger.info(f"Successfully converted stage2 {split} to MDS format")
        logger.info(f"Total chunks written: {total_chunks}")
        logger.info(f"Processed examples: {processed_count}")
        
        # Log shard information
        shard_files = list(output_dir.glob("*.mds"))
        logger.info(f"Total shards in directory: {len(shard_files)}")
        
        # Clean up resume file if exists
        resume_path = output_dir / f"resume_info_{split}.json"
        if resume_path.exists():
            resume_path.unlink()
            logger.info("Removed resume information file (conversion completed)")
        
    except Exception as e:
        logger.error(f"Error converting stage2 {split} to MDS: {e}")
        raise

def check_resume_info(output_dir: Path, split: str) -> int:
    """Check if there's resume information from a previous interrupted run"""
    resume_path = output_dir / f"resume_info_{split}.json"
    
    if resume_path.exists():
        with open(resume_path, 'r') as f:
            resume_info = json.load(f)
        
        logger.info(f"Found resume information for {split}:")
        logger.info(f"  Last processed index: {resume_info['last_processed_index']}")
        logger.info(f"  Next index to process: {resume_info['next_index']}")
        logger.info(f"  Chunks written so far: {resume_info['total_chunks_written']}")
        
        return resume_info['next_index']
    
    return 0

def update_dataset_info(base_path: str) -> None:
    """Update the dataset info file to reflect stage2 completion"""
    
    info_path = Path(base_path) / 'opengenome_mds' / 'dataset_info.json'
    
    # Check if info file exists
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        # Update stage2 status
        info['stage2_conversion_status'] = 'completed'
        
    else:
        # Create new info for stage2 only
        info = {
            "dataset": "OpenGenome",
            "source": "LongSafari/open-genome",
            "format": "MDS",
            "stage2_only": True,
            "stage2": {
                "original_context_length": "131k",
                "segmented_context_length": "8192",
                "splits": ["train", "validation", "test"],
                "conversion_status": "completed"
            },
            "text_format": "[TAXONOMY] classification [CHUNK] n/total [SEQUENCE] dna_sequence",
            "segmentation_note": "Sequences are segmented into chunks to fit ModernBERT context windows"
        }
    
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Dataset info updated at {info_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert OpenGenome Stage 2 to MDS format')
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/DATA',
        help='Base path for data storage'
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
        '--chunk_size', 
        type=int, 
        default=8192,
        help='Size of sequence chunks (default: 8192 for stage 2)'
    )
    parser.add_argument(
        '--resume', 
        action='store_true',
        help='Resume from previous interrupted conversion'
    )
    parser.add_argument(
        '--dry_run', 
        action='store_true',
        help='Only download and inspect data, do not convert'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting OpenGenome Stage 2 to MDS conversion...")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Chunk size: {args.chunk_size}")
    
    # Setup directories for stage 2 only
    directories = setup_stage2_directories(args.data_path)
    
    # Process each split for stage 2
    for split in args.splits:
        logger.info(f"\n--- Processing stage2 {split} ---")
        
        # Check for resume information
        resume_from = 0
        if args.resume:
            resume_from = check_resume_info(directories[split], split)
        
        # Download dataset
        dataset = download_and_inspect_stage2(split)
        
        if dataset is None:
            logger.warning(f"Skipping stage2 {split} due to download error")
            continue
        
        if args.dry_run:
            logger.info(f"Dry run: would convert {len(dataset)} examples")
            continue
        
        # Convert to MDS
        convert_to_mds(
            dataset, 
            directories[split], 
            split, 
            args.shard_size, 
            args.chunk_size,
            resume_from
        )
    
    if not args.dry_run:
        # Update dataset information
        update_dataset_info(args.data_path)
        
        logger.info("\n=== Stage 2 Conversion Complete ===")
        logger.info(f"Converted datasets available at: {args.data_path}/opengenome_mds/stage2/")
        logger.info("Directory structure:")
        for split in args.splits:
            shard_files = list(directories[split].glob("*.mds"))
            logger.info(f"  {split}/ ({len(shard_files)} shards)")
            
            # Check for any remaining resume files
            resume_path = directories[split] / f"resume_info_{split}.json"
            if resume_path.exists():
                logger.warning(f"  Warning: Resume file still exists for {split}")

if __name__ == "__main__":
    main()