#!/usr/bin/env python3
"""
OpenGenome2 to MDS Format Conversion Script

This script downloads the OpenGenome2 dataset from HuggingFace and converts it
to MDS format for use with ModernBERT training pipeline.

Note: OpenGenome2 has a different structure than OpenGenome - it contains
raw DNA sequences without taxonomic classification.

Author: Generated for DNADatasetConversionScripts
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from datasets import load_dataset, Dataset
from streaming import MDSWriter
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories(base_path: str) -> Dict[str, str]:
    """Create directory structure for OpenGenome2 data"""
    base_dir = Path(base_path)
    
    directories = {
        'main': {
            'train': base_dir / 'opengenome2_mds' / 'train',
            'validation': base_dir / 'opengenome2_mds' / 'validation', 
            'test': base_dir / 'opengenome2_mds' / 'test'
        },
        'raw': base_dir / 'opengenome2_raw'
    }
    
    # Create all directories
    for split_dirs in directories.values():
        if isinstance(split_dirs, dict):
            for split_dir in split_dirs.values():
                split_dir.mkdir(parents=True, exist_ok=True)
        else:
            split_dirs.mkdir(parents=True, exist_ok=True)
    
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

def format_genome2_text(example: Dict[str, Any], chunk_size: int = None) -> List[Dict[str, str]]:
    """
    Format OpenGenome2 data for BERT training, optionally segmenting into chunks.
    
    OpenGenome2 contains raw DNA sequences without taxonomic information.
    This function handles the inconsistent schema (text vs record fields).
    
    Args:
        example: Dictionary containing the dataset example
        chunk_size: If provided, segment sequences into this size
    
    Returns:
        List of dictionaries with formatted text
    """
    # Handle inconsistent schema in OpenGenome2
    text_content = ""
    
    if 'text' in example and example['text']:
        text_content = example['text']
    elif 'record' in example and example['record']:
        text_content = example['record']
    else:
        # Try to find any string field
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 50:  # Assume DNA sequences are long
                text_content = value
                break
    
    if not text_content:
        logger.warning(f"No valid text content found in example: {list(example.keys())}")
        text_content = ""
    
    if chunk_size is None:
        # No segmentation, return as single item
        formatted_text = f"[DNA_SEQUENCE] {text_content}"
        return [{"text": formatted_text}]
    else:
        # Segment the sequence
        sequence_chunks = segment_sequence(text_content, chunk_size)
        formatted_examples = []
        
        for i, chunk in enumerate(sequence_chunks):
            # Include chunk index in the formatting for tracking
            formatted_text = f"[CHUNK] {i+1}/{len(sequence_chunks)} [DNA_SEQUENCE] {chunk}"
            formatted_examples.append({"text": formatted_text})
        
        return formatted_examples

def load_opengenome2_safely(config_name: Optional[str] = None, split: Optional[str] = None) -> Optional[Dataset]:
    """
    Safely load OpenGenome2 dataset handling schema inconsistencies
    """
    logger.info(f"Attempting to load OpenGenome2...")
    
    try:
        # Try to load with default configuration
        if config_name and split:
            dataset = load_dataset("arcinstitute/opengenome2", config_name, split=split)
        elif split:
            dataset = load_dataset("arcinstitute/opengenome2", split=split)
        else:
            dataset = load_dataset("arcinstitute/opengenome2")
        
        logger.info(f"Successfully loaded dataset")
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load with default method: {e}")
        
        # Try alternative loading methods
        try:
            logger.info("Trying to load specific configurations...")
            # List available configurations
            from datasets import get_dataset_config_names
            configs = get_dataset_config_names("arcinstitute/opengenome2")
            logger.info(f"Available configurations: {configs}")
            
            if configs:
                # Try loading the first available config
                dataset = load_dataset("arcinstitute/opengenome2", configs[0])
                logger.info(f"Successfully loaded with config: {configs[0]}")
                return dataset
                
        except Exception as e2:
            logger.error(f"Alternative loading also failed: {e2}")
            
        # Try loading as streaming dataset
        try:
            logger.info("Trying streaming dataset...")
            dataset = load_dataset("arcinstitute/opengenome2", streaming=True)
            # Convert streaming to regular dataset (for small samples)
            dataset = dataset.take(1000)  # Take sample for testing
            logger.info("Successfully loaded as streaming dataset (sample)")
            return dataset
            
        except Exception as e3:
            logger.error(f"Streaming loading failed: {e3}")
    
    return None

def inspect_dataset_structure(dataset: Dataset, name: str = "dataset") -> None:
    """Inspect the structure of the dataset to understand its format"""
    logger.info(f"\n--- Inspecting {name} ---")
    
    if hasattr(dataset, '__len__'):
        logger.info(f"Dataset size: {len(dataset)}")
    else:
        logger.info("Dataset size: Unknown (streaming)")
    
    # Look at first few examples
    try:
        if hasattr(dataset, '__iter__'):
            examples = list(dataset.take(3) if hasattr(dataset, 'take') else dataset[:3])
        else:
            examples = [dataset[i] for i in range(min(3, len(dataset)))]
            
        for i, example in enumerate(examples):
            logger.info(f"\nExample {i}:")
            logger.info(f"Keys: {list(example.keys())}")
            
            for key, value in example.items():
                if isinstance(value, str):
                    preview = value[:100] + "..." if len(value) > 100 else value
                    logger.info(f"  {key}: {preview} (length: {len(value)})")
                else:
                    logger.info(f"  {key}: {value}")
                    
    except Exception as e:
        logger.error(f"Error inspecting dataset: {e}")

def convert_to_mds(dataset: Dataset, output_dir: Path, split: str, 
                   shard_size: int = 100000, chunk_size: int = None) -> None:
    """Convert dataset to MDS format with sharding and optional sequence segmentation"""
    
    logger.info(f"Converting {split} to MDS format...")
    logger.info(f"Output directory: {output_dir}")
    if chunk_size:
        logger.info(f"Segmenting sequences into chunks of {chunk_size}")
    
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
            
            processed_count = 0
            total_chunks = 0
            error_count = 0
            
            # Handle both regular and streaming datasets
            if hasattr(dataset, '__iter__'):
                iterator = dataset
            else:
                iterator = (dataset[i] for i in range(len(dataset)))
            
            for i, example in enumerate(iterator):
                try:
                    # Format the example with optional segmentation
                    formatted_examples = format_genome2_text(example, chunk_size)
                    
                    # Write each segment to MDS
                    for formatted_example in formatted_examples:
                        # Skip empty examples
                        if not formatted_example['text'].strip():
                            error_count += 1
                            continue
                        
                        writer.write(formatted_example)
                        total_chunks += 1
                    
                    processed_count += 1
                    
                    if processed_count % 10000 == 0:
                        logger.info(f"Processed {processed_count} examples ({total_chunks} chunks)...")
                        
                except Exception as e:
                    logger.warning(f"Error processing example {i}: {e}")
                    error_count += 1
                    if error_count > 100:  # Stop if too many errors
                        logger.error("Too many errors, stopping conversion")
                        break
        
        logger.info(f"Successfully converted {split} to MDS format")
        logger.info(f"Processed: {processed_count} examples into {total_chunks} chunks, Errors: {error_count}")
        
        # Log shard information
        shard_files = list(output_dir.glob("*.mds"))
        logger.info(f"Created {len(shard_files)} shards")
        
    except Exception as e:
        logger.error(f"Error converting {split} to MDS: {e}")
        raise

def create_synthetic_splits(dataset: Dataset, train_ratio: float = 0.8, 
                          val_ratio: float = 0.1) -> Dict[str, Dataset]:
    """
    Create train/validation/test splits from a single dataset
    since OpenGenome2 might not have predefined splits
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Create splits
    splits = {
        'train': dataset.select(range(train_size)),
        'validation': dataset.select(range(train_size, train_size + val_size)),
        'test': dataset.select(range(train_size + val_size, total_size))
    }
    
    logger.info(f"Created splits - Train: {len(splits['train'])}, "
               f"Val: {len(splits['validation'])}, Test: {len(splits['test'])}")
    
    return splits

def save_dataset_info(directories: Dict, base_path: str, chunk_size: int = None) -> None:
    """Save information about the converted datasets"""
    
    info = {
        "dataset": "OpenGenome2",
        "source": "arcinstitute/opengenome2",
        "format": "MDS",
        "note": "Raw DNA sequences without taxonomic classification",
        "differences_from_opengenome1": [
            "No taxonomic classification data",
            "Raw DNA sequences only", 
            "Inconsistent schema (text vs record fields)",
            "Single configuration (no stage1/stage2)"
        ],
        "segmentation": {
            "enabled": chunk_size is not None,
            "chunk_size": chunk_size if chunk_size else "N/A",
            "note": "Sequences segmented to fit ModernBERT context windows" if chunk_size else "No segmentation applied"
        },
        "directory_structure": {
            "train": str(directories['main']['train']),
            "validation": str(directories['main']['validation']),
            "test": str(directories['main']['test'])
        },
        "text_format": "[CHUNK] n/total [DNA_SEQUENCE] raw_dna_sequence" if chunk_size else "[DNA_SEQUENCE] raw_dna_sequence"
    }
    
    info_path = Path(base_path) / 'opengenome2_mds' / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Dataset info saved to {info_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert OpenGenome2 to MDS format')
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/DATA',
        help='Base path for data storage'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='Dataset configuration to load'
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
    parser.add_argument(
        '--sample_size', 
        type=int, 
        default=None,
        help='Use only a sample of the dataset for testing'
    )
    parser.add_argument(
        '--chunk_size', 
        type=int, 
        default=8192,
        help='Size of sequence chunks (default: 8192 for ModernBERT)'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting OpenGenome2 to MDS conversion...")
    logger.info(f"Data path: {args.data_path}")
    
    # Setup directories
    directories = setup_directories(args.data_path)
    
    # Load dataset
    logger.info("Loading OpenGenome2 dataset...")
    dataset = load_opengenome2_safely()
    
    if dataset is None:
        logger.error("Failed to load OpenGenome2 dataset")
        return 1
    
    # Inspect dataset structure
    inspect_dataset_structure(dataset, "OpenGenome2")
    
    if args.dry_run:
        logger.info("Dry run complete - dataset structure inspected")
        return 0
    
    # Handle sampling if requested
    if args.sample_size and hasattr(dataset, 'select'):
        logger.info(f"Using sample of {args.sample_size} examples")
        dataset = dataset.select(range(min(args.sample_size, len(dataset))))
    
    # Create splits if dataset doesn't have them
    if not hasattr(dataset, 'train'):
        logger.info("Creating train/validation/test splits...")
        splits = create_synthetic_splits(dataset)
    else:
        splits = {
            'train': dataset['train'] if 'train' in dataset else dataset,
            'validation': dataset['validation'] if 'validation' in dataset else None,
            'test': dataset['test'] if 'test' in dataset else None
        }
    
    # Convert each split to MDS
    for split_name, split_dataset in splits.items():
        if split_dataset is None:
            logger.warning(f"Skipping {split_name} - no data available")
            continue
            
        logger.info(f"\n--- Converting {split_name} split ---")
        output_dir = directories['main'][split_name]
        convert_to_mds(split_dataset, output_dir, split_name, args.shard_size, args.chunk_size)
    
    # Save dataset information
    save_dataset_info(directories, args.data_path, args.chunk_size)
    
    logger.info("\n=== Conversion Complete ===")
    logger.info(f"Converted datasets available at: {args.data_path}/opengenome2_mds/")
    logger.info("Directory structure:")
    for split_name in ['train', 'validation', 'test']:
        shard_files = list(directories['main'][split_name].glob("*.mds"))
        logger.info(f"  {split_name}/ ({len(shard_files)} shards)")
    
    return 0

if __name__ == "__main__":
    exit(main())
