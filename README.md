# DNA Dataset Conversion Scripts

This repository contains scripts for converting DNA/genomic datasets to formats compatible with modern language models, particularly ModernBERT.

## OpenGenome to MDS Conversion

The main script `convert_opengenome_to_mds.py` converts the OpenGenome dataset from HuggingFace to MDS (MosaicML Streaming Dataset) format for efficient training with ModernBERT.

### Features

- Downloads OpenGenome dataset from HuggingFace (`LongSafari/open-genome`)
- Converts both Stage 1 (8k context) and Stage 2 (131k context) data
- Maintains original train/validation/test splits
- Creates sharded MDS files for efficient streaming
- Formats genomic data with taxonomy classification and DNA sequences

### Installation

```bash
# Clone the repository
git clone https://github.com/leannmlindsey/DNADatasetConversionScripts.git
cd DNADatasetConversionScripts

# Install dependencies
pip install -r requirements.txt
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
[TAXONOMY] r__Duplodnaviria;k__Heunggongvirae;p__Uroviricota;c__Caudoviricetes;;;; [SEQUENCE] ACAAAAAGCCACCGAAACCCCTCGGAAAATAAGGGAGAATCAATGAAAAAATCCTGGCGAATAAAAAACACTCAAG...
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
    max_seq_len: 8192  # 8k for stage1, 131072 for stage2
```

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
