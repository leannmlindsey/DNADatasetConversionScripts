#!/usr/bin/env python3
"""
OpenGenome Taxonomic Distribution Analysis

This script analyzes and visualizes the taxonomic distribution in the OpenGenome
dataset training, validation, and test splits. It parses the taxonomic 
classification strings and creates comprehensive visualizations.

Note: This is for OpenGenome (with taxonomy), not OpenGenome2 (raw sequences only).

Author: Generated for DNADatasetConversionScripts
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
from datasets import load_dataset
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class TaxonomicAnalyzer:
    """Analyzer for OpenGenome taxonomic data"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Taxonomic levels mapping
        self.tax_levels = {
            'r__': 'Realm',
            'k__': 'Kingdom', 
            'd__': 'Domain',
            'p__': 'Phylum',
            'c__': 'Class',
            'o__': 'Order',
            'f__': 'Family',
            'g__': 'Genus',
            's__': 'Species'
        }
        
        # Store analysis results
        self.results = {}
        
    def parse_taxonomy_string(self, taxonomy_str: str) -> Dict[str, str]:
        """
        Parse taxonomy string into hierarchical levels
        
        Example input: "r__Duplodnaviria;k__Heunggongvirae;p__Uroviricota;c__Caudoviricetes;;;;"
        """
        taxonomy = {}
        
        if not taxonomy_str or taxonomy_str.strip() == '':
            return taxonomy
            
        # Split by semicolon and clean
        parts = [part.strip() for part in taxonomy_str.split(';') if part.strip()]
        
        for part in parts:
            if '__' in part:
                prefix = part[:3]  # e.g., 'r__', 'k__', etc.
                value = part[3:].strip()  # Remove prefix
                
                if prefix in self.tax_levels and value:
                    level_name = self.tax_levels[prefix]
                    taxonomy[level_name] = value
                    
        return taxonomy
    
    def analyze_split(self, dataset, split_name: str, stage: str = None, 
                     sample_size: int = None) -> Dict[str, Any]:
        """Analyze taxonomic distribution in a dataset split"""
        
        split_id = f"{stage}_{split_name}" if stage else split_name
        logger.info(f"Analyzing {split_id}...")
        
        # Sample dataset if requested
        if sample_size and len(dataset) > sample_size:
            logger.info(f"Sampling {sample_size} examples from {len(dataset)}")
            indices = np.random.choice(len(dataset), sample_size, replace=False)
            dataset = dataset.select(indices)
        
        # Extract taxonomic data
        taxonomies = []
        sequence_lengths = []
        
        logger.info(f"Processing {len(dataset)} examples...")
        
        for i, example in enumerate(dataset):
            # Get taxonomy - adjust field name based on actual dataset structure
            taxonomy_str = ""
            for possible_field in ['taxonomy', 'classification', 'text']:
                if possible_field in example:
                    if possible_field == 'text':
                        # If it's from MDS format, extract taxonomy from formatted text
                        text = example[possible_field]
                        if '[TAXONOMY]' in text and '[SEQUENCE]' in text:
                            taxonomy_str = text.split('[TAXONOMY]')[1].split('[SEQUENCE]')[0].strip()
                        else:
                            taxonomy_str = text
                    else:
                        taxonomy_str = example[possible_field]
                    break
            
            # Parse taxonomy
            parsed_tax = self.parse_taxonomy_string(taxonomy_str)
            taxonomies.append(parsed_tax)
            
            # Get sequence length if available
            sequence = ""
            for possible_field in ['sequence', 'text']:
                if possible_field in example:
                    if possible_field == 'text' and '[SEQUENCE]' in example[possible_field]:
                        sequence = example[possible_field].split('[SEQUENCE]')[1].strip()
                    else:
                        sequence = example[possible_field]
                    break
            
            sequence_lengths.append(len(sequence))
            
            if (i + 1) % 10000 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} examples")
        
        # Analyze distributions
        analysis = {
            'split_name': split_id,
            'total_examples': len(dataset),
            'taxonomies': taxonomies,
            'sequence_lengths': sequence_lengths,
            'level_distributions': {},
            'level_diversity': {},
            'sequence_stats': {
                'mean_length': np.mean(sequence_lengths),
                'median_length': np.median(sequence_lengths),
                'std_length': np.std(sequence_lengths),
                'min_length': np.min(sequence_lengths),
                'max_length': np.max(sequence_lengths)
            }
        }
        
        # Analyze each taxonomic level
        for level_name in self.tax_levels.values():
            level_values = [tax.get(level_name, 'Unknown') for tax in taxonomies]
            level_counts = Counter(level_values)
            
            analysis['level_distributions'][level_name] = dict(level_counts)
            analysis['level_diversity'][level_name] = {
                'unique_count': len([v for v in level_counts.keys() if v != 'Unknown']),
                'total_count': len(level_counts),
                'most_common': level_counts.most_common(10)
            }
        
        logger.info(f"Completed analysis for {split_id}")
        return analysis
    
    def create_taxonomic_visualizations(self, analyses: List[Dict[str, Any]]) -> None:
        """Create comprehensive taxonomic visualizations"""
        
        logger.info("Creating taxonomic visualizations...")
        
        # 1. Overall taxonomic diversity comparison
        self._plot_taxonomic_diversity(analyses)
        
        # 2. Distribution at each taxonomic level
        for level in self.tax_levels.values():
            self._plot_level_distribution(analyses, level)
        
        # 3. Sequence length distributions
        self._plot_sequence_length_distribution(analyses)
        
        # 4. Taxonomic composition heatmap
        self._plot_taxonomic_heatmap(analyses)
        
        # 5. Diversity metrics comparison
        self._plot_diversity_metrics(analyses)
        
        logger.info(f"Visualizations saved to {self.output_dir}")
    
    def _plot_taxonomic_diversity(self, analyses: List[Dict[str, Any]]) -> None:
        """Plot taxonomic diversity across splits"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Taxonomic Diversity Across Dataset Splits', fontsize=16, fontweight='bold')
        
        # Prepare data
        splits = [analysis['split_name'] for analysis in analyses]
        levels = list(self.tax_levels.values())
        
        # Unique counts matrix
        unique_counts = np.zeros((len(splits), len(levels)))
        total_counts = np.zeros((len(splits), len(levels)))
        
        for i, analysis in enumerate(analyses):
            for j, level in enumerate(levels):
                if level in analysis['level_diversity']:
                    unique_counts[i, j] = analysis['level_diversity'][level]['unique_count']
                    total_counts[i, j] = analysis['level_diversity'][level]['total_count']
        
        # Plot 1: Unique taxa count
        ax1 = axes[0, 0]
        x = np.arange(len(levels))
        width = 0.25
        for i, split in enumerate(splits):
            ax1.bar(x + i * width, unique_counts[i], width, label=split, alpha=0.8)
        ax1.set_xlabel('Taxonomic Level')
        ax1.set_ylabel('Number of Unique Taxa')
        ax1.set_title('Unique Taxa Count by Level')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(levels, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Total examples per split
        ax2 = axes[0, 1]
        total_examples = [analysis['total_examples'] for analysis in analyses]
        bars = ax2.bar(splits, total_examples, alpha=0.8, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax2.set_ylabel('Number of Examples')
        ax2.set_title('Dataset Split Sizes')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, total_examples):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_examples)*0.01,
                    f'{value:,}', ha='center', va='bottom')
        
        # Plot 3: Diversity ratio (unique/total)
        ax3 = axes[1, 0]
        diversity_ratios = unique_counts / np.maximum(total_counts, 1)  # Avoid division by zero
        
        for i, split in enumerate(splits):
            ax3.plot(levels, diversity_ratios[i], marker='o', linewidth=2, label=split)
        ax3.set_xlabel('Taxonomic Level')
        ax3.set_ylabel('Diversity Ratio (Unique/Total)')
        ax3.set_title('Taxonomic Diversity Ratio')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Missing data percentage
        ax4 = axes[1, 1]
        missing_percentages = np.zeros((len(splits), len(levels)))
        
        for i, analysis in enumerate(analyses):
            for j, level in enumerate(levels):
                if level in analysis['level_distributions']:
                    total = sum(analysis['level_distributions'][level].values())
                    unknown = analysis['level_distributions'][level].get('Unknown', 0)
                    missing_percentages[i, j] = (unknown / total) * 100 if total > 0 else 0
        
        x = np.arange(len(levels))
        for i, split in enumerate(splits):
            ax4.bar(x + i * width, missing_percentages[i], width, label=split, alpha=0.8)
        ax4.set_xlabel('Taxonomic Level')
        ax4.set_ylabel('Missing Data (%)')
        ax4.set_title('Missing Taxonomic Information')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(levels, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'taxonomic_diversity_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_level_distribution(self, analyses: List[Dict[str, Any]], level: str) -> None:
        """Plot distribution for a specific taxonomic level"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{level} Distribution Across Dataset Splits', fontsize=16, fontweight='bold')
        
        # Collect all unique values for this level
        all_values = set()
        for analysis in analyses:
            if level in analysis['level_distributions']:
                all_values.update(analysis['level_distributions'][level].keys())
        
        # Remove 'Unknown' and get top values
        all_values.discard('Unknown')
        
        # Get top N most common across all splits
        value_totals = defaultdict(int)
        for analysis in analyses:
            if level in analysis['level_distributions']:
                for value, count in analysis['level_distributions'][level].items():
                    if value != 'Unknown':
                        value_totals[value] += count
        
        top_values = [item[0] for item in sorted(value_totals.items(), 
                                               key=lambda x: x[1], reverse=True)[:15]]
        
        # Plot 1: Top taxa comparison
        ax1 = axes[0, 0]
        x = np.arange(len(top_values))
        width = 0.25
        
        for i, analysis in enumerate(analyses):
            counts = []
            for value in top_values:
                count = analysis['level_distributions'].get(level, {}).get(value, 0)
                counts.append(count)
            ax1.bar(x + i * width, counts, width, label=analysis['split_name'], alpha=0.8)
        
        ax1.set_xlabel(level)
        ax1.set_ylabel('Count')
        ax1.set_title(f'Top 15 {level} Taxa')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(top_values, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Pie chart for first split (usually train)
        ax2 = axes[0, 1]
        if analyses:
            first_analysis = analyses[0]
            if level in first_analysis['level_distributions']:
                level_data = first_analysis['level_distributions'][level].copy()
                level_data.pop('Unknown', None)  # Remove Unknown
                
                # Get top 10 for pie chart
                top_10 = dict(sorted(level_data.items(), key=lambda x: x[1], reverse=True)[:10])
                others_count = sum(level_data.values()) - sum(top_10.values())
                
                if others_count > 0:
                    top_10['Others'] = others_count
                
                if top_10:
                    wedges, texts, autotexts = ax2.pie(top_10.values(), labels=top_10.keys(), 
                                                      autopct='%1.1f%%', startangle=90)
                    ax2.set_title(f'{level} Distribution ({first_analysis["split_name"]})')
                    
                    # Improve text readability
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
        
        # Plot 3: Cumulative distribution
        ax3 = axes[1, 0]
        for analysis in analyses:
            if level in analysis['level_distributions']:
                level_data = analysis['level_distributions'][level].copy()
                level_data.pop('Unknown', None)
                
                if level_data:
                    counts = sorted(level_data.values(), reverse=True)
                    cumsum = np.cumsum(counts) / sum(counts) * 100
                    ax3.plot(range(1, len(cumsum) + 1), cumsum, 
                            marker='o', label=analysis['split_name'], linewidth=2)
        
        ax3.set_xlabel('Taxa Rank')
        ax3.set_ylabel('Cumulative Percentage')
        ax3.set_title(f'{level} Cumulative Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(left=1)
        
        # Plot 4: Shannon diversity index comparison
        ax4 = axes[1, 1]
        shannon_indices = []
        split_names = []
        
        for analysis in analyses:
            if level in analysis['level_distributions']:
                level_data = analysis['level_distributions'][level].copy()
                level_data.pop('Unknown', None)
                
                if level_data:
                    total = sum(level_data.values())
                    shannon = -sum((count/total) * np.log(count/total) 
                                 for count in level_data.values() if count > 0)
                    shannon_indices.append(shannon)
                    split_names.append(analysis['split_name'])
        
        if shannon_indices:
            bars = ax4.bar(split_names, shannon_indices, alpha=0.8, color=['skyblue', 'lightgreen', 'lightcoral'])
            ax4.set_ylabel('Shannon Diversity Index')
            ax4.set_title(f'{level} Shannon Diversity')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, shannon_indices):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(shannon_indices)*0.01,
                        f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        safe_level = level.replace('/', '_').replace(' ', '_')
        plt.savefig(self.output_dir / f'{safe_level.lower()}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sequence_length_distribution(self, analyses: List[Dict[str, Any]]) -> None:
        """Plot sequence length distributions"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sequence Length Distributions', fontsize=16, fontweight='bold')
        
        # Plot 1: Histograms
        ax1 = axes[0, 0]
        for analysis in analyses:
            lengths = analysis['sequence_lengths']
            if lengths:
                ax1.hist(lengths, bins=50, alpha=0.6, label=analysis['split_name'], density=True)
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Density')
        ax1.set_title('Sequence Length Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plots
        ax2 = axes[0, 1]
        length_data = []
        labels = []
        for analysis in analyses:
            if analysis['sequence_lengths']:
                length_data.append(analysis['sequence_lengths'])
                labels.append(analysis['split_name'])
        
        if length_data:
            ax2.boxplot(length_data, labels=labels)
            ax2.set_ylabel('Sequence Length')
            ax2.set_title('Sequence Length Box Plots')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Statistics comparison
        ax3 = axes[1, 0]
        stats_names = ['Mean', 'Median', 'Std']
        x = np.arange(len(stats_names))
        width = 0.25
        
        for i, analysis in enumerate(analyses):
            stats = analysis['sequence_stats']
            values = [stats['mean_length'], stats['median_length'], stats['std_length']]
            ax3.bar(x + i * width, values, width, label=analysis['split_name'], alpha=0.8)
        
        ax3.set_xlabel('Statistic')
        ax3.set_ylabel('Sequence Length')
        ax3.set_title('Sequence Length Statistics')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(stats_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Length percentiles
        ax4 = axes[1, 1]
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        
        for analysis in analyses:
            lengths = analysis['sequence_lengths']
            if lengths:
                perc_values = np.percentile(lengths, percentiles)
                ax4.plot(percentiles, perc_values, marker='o', label=analysis['split_name'], linewidth=2)
        
        ax4.set_xlabel('Percentile')
        ax4.set_ylabel('Sequence Length')
        ax4.set_title('Sequence Length Percentiles')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sequence_length_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_taxonomic_heatmap(self, analyses: List[Dict[str, Any]]) -> None:
        """Create taxonomic composition heatmap"""
        
        # Prepare data for heatmap
        levels = list(self.tax_levels.values())
        splits = [analysis['split_name'] for analysis in analyses]
        
        # Create matrix of unique counts
        data_matrix = np.zeros((len(splits), len(levels)))
        
        for i, analysis in enumerate(analyses):
            for j, level in enumerate(levels):
                if level in analysis['level_diversity']:
                    data_matrix[i, j] = analysis['level_diversity'][level]['unique_count']
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(data_matrix, 
                   xticklabels=levels, 
                   yticklabels=splits,
                   annot=True, 
                   fmt='g',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Number of Unique Taxa'})
        
        plt.title('Taxonomic Diversity Heatmap Across Splits', fontsize=14, fontweight='bold')
        plt.xlabel('Taxonomic Level')
        plt.ylabel('Dataset Split')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'taxonomic_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_diversity_metrics(self, analyses: List[Dict[str, Any]]) -> None:
        """Plot various diversity metrics"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Taxonomic Diversity Metrics', fontsize=16, fontweight='bold')
        
        levels = list(self.tax_levels.values())
        splits = [analysis['split_name'] for analysis in analyses]
        
        # Calculate diversity metrics for each level and split
        shannon_matrix = np.zeros((len(splits), len(levels)))
        simpson_matrix = np.zeros((len(splits), len(levels)))
        richness_matrix = np.zeros((len(splits), len(levels)))
        evenness_matrix = np.zeros((len(splits), len(levels)))
        
        for i, analysis in enumerate(analyses):
            for j, level in enumerate(levels):
                if level in analysis['level_distributions']:
                    level_data = analysis['level_distributions'][level].copy()
                    level_data.pop('Unknown', None)
                    
                    if level_data and sum(level_data.values()) > 0:
                        total = sum(level_data.values())
                        proportions = [count/total for count in level_data.values()]
                        
                        # Shannon diversity
                        shannon = -sum(p * np.log(p) for p in proportions if p > 0)
                        shannon_matrix[i, j] = shannon
                        
                        # Simpson diversity
                        simpson = 1 - sum(p**2 for p in proportions)
                        simpson_matrix[i, j] = simpson
                        
                        # Richness (number of taxa)
                        richness = len(level_data)
                        richness_matrix[i, j] = richness
                        
                        # Evenness (Shannon / log(richness))
                        if richness > 1:
                            evenness = shannon / np.log(richness)
                            evenness_matrix[i, j] = evenness
        
        # Plot Shannon diversity
        ax1 = axes[0, 0]
        x = np.arange(len(levels))
        width = 0.25
        for i, split in enumerate(splits):
            ax1.bar(x + i * width, shannon_matrix[i], width, label=split, alpha=0.8)
        ax1.set_xlabel('Taxonomic Level')
        ax1.set_ylabel('Shannon Diversity Index')
        ax1.set_title('Shannon Diversity')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(levels, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot Simpson diversity
        ax2 = axes[0, 1]
        for i, split in enumerate(splits):
            ax2.bar(x + i * width, simpson_matrix[i], width, label=split, alpha=0.8)
        ax2.set_xlabel('Taxonomic Level')
        ax2.set_ylabel('Simpson Diversity Index')
        ax2.set_title('Simpson Diversity')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(levels, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot Richness
        ax3 = axes[1, 0]
        for i, split in enumerate(splits):
            ax3.bar(x + i * width, richness_matrix[i], width, label=split, alpha=0.8)
        ax3.set_xlabel('Taxonomic Level')
        ax3.set_ylabel('Number of Taxa (Richness)')
        ax3.set_title('Taxonomic Richness')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(levels, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot Evenness
        ax4 = axes[1, 1]
        for i, split in enumerate(splits):
            ax4.bar(x + i * width, evenness_matrix[i], width, label=split, alpha=0.8)
        ax4.set_xlabel('Taxonomic Level')
        ax4.set_ylabel('Evenness Index')
        ax4.set_title('Taxonomic Evenness')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(levels, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'diversity_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_analysis_results(self, analyses: List[Dict[str, Any]]) -> None:
        """Save detailed analysis results to JSON"""
        
        # Prepare serializable results
        serializable_results = []
        
        for analysis in analyses:
            result = {
                'split_name': analysis['split_name'],
                'total_examples': analysis['total_examples'],
                'sequence_stats': analysis['sequence_stats'],
                'level_diversity': analysis['level_diversity'],
                'top_taxa_per_level': {}
            }
            
            # Add top taxa for each level
            for level, distribution in analysis['level_distributions'].items():
                # Remove 'Unknown' and get top 20
                clean_dist = {k: v for k, v in distribution.items() if k != 'Unknown'}
                top_taxa = sorted(clean_dist.items(), key=lambda x: x[1], reverse=True)[:20]
                result['top_taxa_per_level'][level] = top_taxa
            
            serializable_results.append(result)
        
        # Save to JSON
        output_file = self.output_dir / 'taxonomic_analysis_results.json'
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Analysis results saved to {output_file}")
        
        # Also save summary statistics
        summary = {
            'dataset': 'OpenGenome',
            'analysis_date': pd.Timestamp.now().isoformat(),
            'splits_analyzed': [r['split_name'] for r in serializable_results],
            'total_examples': sum(r['total_examples'] for r in serializable_results),
            'taxonomic_levels': list(self.tax_levels.values())
        }
        
        summary_file = self.output_dir / 'analysis_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Analysis summary saved to {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze OpenGenome taxonomic distribution')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/DATA/taxonomic_analysis',
        help='Directory to save analysis results and plots'
    )
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['stage1', 'stage2'],
        default=['stage1', 'stage2'],
        help='Which stages to analyze'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        choices=['train', 'validation', 'test'],
        default=['train', 'validation', 'test'],
        help='Which splits to analyze'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=None,
        help='Sample size for analysis (use for testing with large datasets)'
    )
    parser.add_argument(
        '--use_mds',
        action='store_true',
        help='Use MDS format data instead of original HuggingFace format'
    )
    parser.add_argument(
        '--mds_path',
        type=str,
        default='/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/DATA/opengenome_mds',
        help='Path to MDS formatted data'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting OpenGenome taxonomic analysis...")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Stages: {args.stages}")
    logger.info(f"Splits: {args.splits}")
    
    # Initialize analyzer
    analyzer = TaxonomicAnalyzer(args.output_dir)
    analyses = []
    
    # Analyze each stage and split
    for stage in args.stages:
        for split in args.splits:
            logger.info(f"\n--- Loading {stage} {split} ---")
            
            try:
                if args.use_mds:
                    # Load from MDS format
                    from streaming import StreamingDataset
                    mds_path = Path(args.mds_path) / stage / split
                    if mds_path.exists():
                        dataset = StreamingDataset(local=str(mds_path))
                        logger.info(f"Loaded MDS dataset from {mds_path}")
                    else:
                        logger.warning(f"MDS path not found: {mds_path}")
                        continue
                else:
                    # Load from HuggingFace
                    dataset = load_dataset("LongSafari/open-genome", stage, split=split)
                    logger.info(f"Loaded HuggingFace dataset: {stage} {split}")
                
                # Analyze the split
                analysis = analyzer.analyze_split(dataset, split, stage, args.sample_size)
                analyses.append(analysis)
                
            except Exception as e:
                logger.error(f"Error analyzing {stage} {split}: {e}")
                continue
    
    if not analyses:
        logger.error("No successful analyses completed")
        return 1
    
    # Create visualizations
    try:
        analyzer.create_taxonomic_visualizations(analyses)
        analyzer.save_analysis_results(analyses)
        
        logger.info("\n=== Analysis Complete ===")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info(f"Analyzed {len(analyses)} dataset splits")
        logger.info("Generated visualizations:")
        logger.info("  - taxonomic_diversity_overview.png")
        logger.info("  - [level]_distribution.png (for each taxonomic level)")
        logger.info("  - sequence_length_analysis.png")
        logger.info("  - taxonomic_heatmap.png")
        logger.info("  - diversity_metrics.png")
        logger.info("Saved data files:")
        logger.info("  - taxonomic_analysis_results.json")
        logger.info("  - analysis_summary.json")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
