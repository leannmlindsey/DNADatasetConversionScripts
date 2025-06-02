# DNA Dataset Conversion Scripts

This repository contains scripts for converting DNA/genomic datasets to formats compatible with modern language models, particularly ModernBERT. It includes conversion scripts for two different genomic datasets:

1. **OpenGenome** (LongSafari/open-genome) - Genomic sequences with taxonomic classification
2. **OpenGenome2** (arcinstitute/opengenome2) - Raw DNA sequences without taxonomic data

## Dataset Overview

### OpenGenome (LongSafari/open-genome)
- **Source**: https://huggingface.co/datasets/LongSafari/open-genome
- **Features**: Contains DNA sequences WITH taxonomic classification
- **Structure**: Organized into stage1 (8k context) and stage2 (131k context)
- **Segmentation**: 
  - Stage1: Original 8k sequences → segmented to 1024 tokens
  - Stage2: Original 131k sequences → segmented to 8192 tokens

### OpenGenome2 (arcinstitute/opengenome2)
- **Source**: https://huggingface.co/datasets/arcinstitute/opengenome2
- **Features**: Contains raw DNA sequences WITHOUT taxonomic data
- **Structure**: Single dataset without phase distinctions
- **Segmentation**: All sequences → segmented to 8192 tokens (configurable)

## OpenGenome to MDS Conversion

The script `convert_opengenome_to_mds.py` converts the OpenGenome dataset from HuggingFace to MDS (MosaicML Streaming Dataset) format for efficient training with ModernBERT.

### Features

- Downloads OpenGenome dataset from HuggingFace (`LongSafari/open-genome`)
- Converts both Stage 1 (8k context) and Stage 2 (131k context) data
- **Automatic segmentation for ModernBERT compatibility**:
  - Stage 1: Segments 8k sequences into 1024-token chunks
  - Stage 2: Segments 131k sequences into 8192-token chunks
- Maintains original train/validation/test splits
- Creates sharded MDS files for efficient streaming
- Formats genomic data with taxonomy classification and DNA sequences

### Installation on Biowulf

```bash
# Clone the repository
git clone https://github.com/leannmlindsey/DNADatasetConversionScripts.git
cd DNADatasetConversionScripts

# Set up conda environment (Biowulf-compatible)
chmod +x setup_environment.sh
./setup_environment.sh

# Activate the environment
conda activate opengenome_conversion
```

Alternatively, you can set up the environment manually:

```bash
# Create conda environment 
conda create -n opengenome_conversion python=3.10
conda activate opengenome_conversion

# Install dependencies (in this specific order for Biowulf compatibility)
pip install torch transformers
pip install numpy pandas
pip install tqdm
pip install datasets
pip install mosaicml-streaming
pip install huggingface_hub
```

### Usage

#### Basic conversion (all stages and splits):
```bash
python convert_opengenome_to_mds.py
```

#### Convert specific stages:
```bash
python convert_opengenome_to_mds.py --stages stage1
```

#### Convert specific splits:
```bash
python convert_opengenome_to_mds.py --splits train validation
```

#### Custom data path:
```bash
python convert_opengenome_to_mds.py --data_path /your/data/path
```

#### Dry run (inspect data without converting):
```bash
python convert_opengenome_to_mds.py --dry_run
```

#### Full options:
```bash
python convert_opengenome_to_mds.py \
    --data_path /data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/DATA \
    --stages stage1 stage2 \
    --splits train validation test \
    --shard_size 100000
```

### Output Structure

The script creates the following directory structure:

```
/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/DATA/
└── opengenome_mds/
    ├── dataset_info.json
    ├── stage1/
    │   ├── train/
    │   │   ├── index.json
    │   │   ├── shard.00000.mds
    │   │   ├── shard.00001.mds
    │   │   └── ...
    │   ├── validation/
    │   │   ├── index.json
    │   │   └── shard.00000.mds
    │   └── test/
    │       ├── index.json
    │       └── shard.00000.mds
    └── stage2/
        ├── train/
        ├── validation/
        └── test/
```

### Data Format

The genomic data is formatted as:
```
# Without segmentation:
[TAXONOMY] r__Duplodnaviria;k__Heunggongvirae;p__Uroviricota;c__Caudoviricetes;;;; [SEQUENCE] ACAAAAAGCCACCGAAACCCCTCGGAAAATAAGGGAGAATCAATGAAAAAATCCTGGCGAATAAAAAACACTCAAG...

# With segmentation:
[TAXONOMY] r__Duplodnaviria;k__Heunggongvirae;p__Uroviricota;c__Caudoviricetes;;;; [CHUNK] 1/8 [SEQUENCE] ACAAAAAGCCACCGAAACCCCTCGGAAAATAAGGGAGAATCAATGAAAAAATCCTGGCGAATAAAAAACACTCAAG...
```

### Integration with ModernBERT

The generated MDS files can be used directly with ModernBERT's training pipeline:

```yaml
# In your ModernBERT config YAML
train_loader:
  name: text
  dataset:
    local: '/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/DATA/opengenome_mds/stage1/train'
    streaming: false  # or true for StreamingTextDataset
    split: train
    shuffle: true
    max_seq_len: 1024  # 1024 for stage1, 8192 for stage2 (after segmentation)
```

## OpenGenome2 to MDS Conversion

The script `convert_opengenome2_to_mds.py` converts the OpenGenome2 dataset from HuggingFace to MDS format. This dataset contains raw DNA sequences without taxonomic classification.

### Features

- Downloads OpenGenome2 dataset from HuggingFace (`arcinstitute/opengenome2`)
- Handles inconsistent schema (text vs record fields)
- **Configurable segmentation**: Default 8192 tokens (can be customized)
- Creates synthetic train/validation/test splits if not present
- Robust error handling for malformed data
- Creates sharded MDS files for efficient streaming

### Usage

#### Basic conversion with default 8192 token chunks:
```bash
python convert_opengenome2_to_mds.py
```

#### Custom chunk size:
```bash
# For 1024 token chunks
python convert_opengenome2_to_mds.py --chunk_size 1024

# For 16384 token chunks
python convert_opengenome2_to_mds.py --chunk_size 16384
```

#### Test with a sample:
```bash
python convert_opengenome2_to_mds.py --sample_size 1000 --dry_run
```

#### Full options:
```bash
python convert_opengenome2_to_mds.py \
    --data_path /data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/DATA \
    --chunk_size 8192 \
    --shard_size 100000 \
    --sample_size 10000
```

### Data Format

OpenGenome2 data is formatted as:
```
# Without segmentation:
[DNA_SEQUENCE] ATCGATCGATCGATCGATCGATCG...

# With segmentation:
[CHUNK] 1/5 [DNA_SEQUENCE] ATCGATCGATCGATCGATCGATCG...
```

### Output Structure

```
/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/DATA/
└── opengenome2_mds/
    ├── dataset_info.json
    ├── train/
    │   ├── index.json
    │   ├── shard.00000.mds
    │   └── ...
    ├── validation/
    └── test/
```

## Key Differences Between Datasets

| Feature | OpenGenome | OpenGenome2 |
|---------|------------|-------------|
| Taxonomic Data | Yes | No |
| Dataset Organization | stage1/stage2 | Single dataset |
| Default Segmentation | stage1: 1024, stage2: 8192 | 8192 (configurable) |
| Data Format | [TAXONOMY] ... [SEQUENCE] ... | [DNA_SEQUENCE] ... |
| HuggingFace Source | LongSafari/open-genome | arcinstitute/opengenome2 |

## Requirements

- Python 3.8+
- datasets
- streaming (MosaicML)
- huggingface_hub
- transformers
- torch

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
