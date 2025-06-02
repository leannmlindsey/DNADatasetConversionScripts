#!/bin/bash

# Biowulf Environment Setup for OpenGenome Conversion
# This script sets up a conda environment compatible with Biowulf

set -e  # Exit on any error

echo "Setting up conda environment for OpenGenome conversion on Biowulf..."

# Create conda environment 
echo "Creating conda environment 'opengenome_conversion'..."
conda create -n opengenome_conversion python=3.10 -y

# Activate environment
echo "Activating environment..."
source activate opengenome_conversion

# Install core dependencies in order (following your working pattern)
echo "Installing core dependencies..."
pip install torch transformers
pip install numpy pandas
pip install tqdm

# Install dataset and streaming libraries
echo "Installing dataset conversion libraries..."
pip install datasets
pip install mosaicml-streaming
pip install huggingface_hub

# Optional: Install additional utilities that might be helpful
echo "Installing additional utilities..."
pip install scikit-learn  # For any data analysis needs
pip install matplotlib seaborn  # For visualization if needed
pip install scipy  # For advanced statistical calculations
pip install plotly
echo ""
echo "Environment setup complete!"
echo ""
echo "To activate this environment in the future, run:"
echo "conda activate opengenome_conversion"
echo ""
echo "To test the installation, run:"
echo "python -c \"import datasets; import streaming; print('Environment ready!')\""
