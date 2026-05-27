"""
Exploratory Data Analysis (EDA) for NMR Spectra
=================================================
Generates visualizations and summary statistics for the NMR dataset.

Usage:
    python eda.py --data ../parsed/synthetic_spectra.csv --labels ../parsed/synthetic_labels.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import os
import argparse


def plot_spectra_overlay(spectra_df, labels_df, ppm_range=(0.0, 10.0), output_dir='../parsed/'):
    """
    Plot all spectra overlaid, colored by compound class.
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Extract ppm axis from column names
    ppm_values = np.array([float(c.replace('ppm_', '').replace('bin_', '')) 
                           for c in spectra_df.columns])
    
    colors = {'Plant_Extract_A': '#e74c3c', 'Plant_Extract_B': '#2ecc71', 'Plant_Extract_C': '#3498db'}
    
    for compound in labels_df['compound'].unique():
        mask = labels_df['compound'] == compound
        subset = spectra_df[mask]
        color = colors.get(compound, '#333333')
        
        for _, row in subset.iterrows():
            ax.plot(ppm_values, row.values, color=color, alpha=0.3, linewidth=0.5)
        
        # Plot mean spectrum with solid line
        mean_spectrum = subset.mean(axis=0)
        ax.plot(ppm_values, mean_spectrum.values, color=color, linewidth=2, label=f"{compound} (mean)")
    
    ax.set_xlabel('Chemical Shift (ppm)', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title('NMR Spectra Overlay by Compound Class', fontsize=14)
    ax.legend()
    ax.invert_xaxis()  # NMR convention: high ppm on left
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectra_overlay.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: spectra_overlay.png")


def plot_mean_difference(spectra_df, labels_df, output_dir='../parsed/'):
    """
    Plot the mean spectrum for each class and the difference between classes.
    """
    ppm_values = np.array([float(c.replace('ppm_', '').replace('bin_', '')) 
                           for c in spectra_df.columns])
    
    classes = labels_df['compound'].unique()
    n_classes = len(classes)
    
    fig, axes = plt.subplots(n_classes + 1, 1, figsize=(16, 4 * (n_classes + 1)), sharex=True)
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    means = {}
    for i, compound in enumerate(classes):
        mask = labels_df['compound'] == compound
        mean_spec = spectra_df[mask].mean(axis=0)
        means[compound] = mean_spec
        
        axes[i].plot(ppm_values, mean_spec.values, color=colors[i], linewidth=1.5)
        axes[i].set_ylabel('Intensity')
        axes[i].set_title(f'Mean Spectrum: {compound}')
        axes[i].grid(True, alpha=0.3)
    
    # Plot pairwise differences
    if n_classes >= 2:
        diff = means[classes[0]] - means[classes[1]]
        axes[-1].plot(ppm_values, diff.values, color='purple', linewidth=1)
        axes[-1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[-1].set_title(f'Difference: {classes[0]} − {classes[1]}')
        axes[-1].set_ylabel('Δ Intensity')
        axes[-1].fill_between(ppm_values, diff.values, 0, 
                               where=diff.values > 0, alpha=0.3, color='red', label='A > B')
        axes[-1].fill_between(ppm_values, diff.values, 0, 
                               where=diff.values < 0, alpha=0.3, color='green', label='B > A')
        axes[-1].legend()
    
    axes[-1].set_xlabel('Chemical Shift (ppm)')
    axes[-1].invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_difference.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: mean_difference.png")


def plot_feature_variance(spectra_df, output_dir='../parsed/'):
    """
    Plot the variance of each feature across samples to identify high-information regions.
    """
    ppm_values = np.array([float(c.replace('ppm_', '').replace('bin_', '')) 
                           for c in spectra_df.columns])
    
    variances = spectra_df.var(axis=0)
    
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(ppm_values, variances.values, color='#e67e22', linewidth=0.8)
    ax.fill_between(ppm_values, variances.values, alpha=0.3, color='#e67e22')
    ax.set_xlabel('Chemical Shift (ppm)', fontsize=12)
    ax.set_ylabel('Variance', fontsize=12)
    ax.set_title('Feature Variance Across Samples (High variance = potentially informative)', fontsize=14)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_variance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: feature_variance.png")


def generate_summary_stats(spectra_df, labels_df, output_dir='../parsed/'):
    """
    Generate and save summary statistics.
    """
    stats = {
        'Total samples': len(spectra_df),
        'Total features': spectra_df.shape[1],
        'Classes': labels_df['compound'].nunique(),
        'Class distribution': labels_df['compound'].value_counts().to_dict(),
        'Mean intensity (global)': float(spectra_df.values.mean()),
        'Std intensity (global)': float(spectra_df.values.std()),
        'Min intensity': float(spectra_df.values.min()),
        'Max intensity': float(spectra_df.values.max()),
        'Sparsity (% near-zero)': float((spectra_df.values < 0.01).mean() * 100),
    }
    
    report = "# EDA Summary Statistics\n\n"
    for key, value in stats.items():
        report += f"- **{key}:** {value}\n"
    
    with open(os.path.join(output_dir, 'eda_summary.md'), 'w') as f:
        f.write(report)
    
    print(f"  Saved: eda_summary.md")
    return stats


def main():
    parser = argparse.ArgumentParser(description='EDA for NMR spectra')
    parser.add_argument('--data', type=str, default='../parsed/synthetic_spectra.csv')
    parser.add_argument('--labels', type=str, default='../parsed/synthetic_labels.csv')
    parser.add_argument('--output', type=str, default='../parsed/')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("Loading data...")
    spectra = pd.read_csv(args.data)
    labels = pd.read_csv(args.labels)
    
    print(f"Spectra shape: {spectra.shape}")
    print(f"Labels: {labels['compound'].value_counts().to_dict()}")
    
    print("\nGenerating plots...")
    plot_spectra_overlay(spectra, labels, output_dir=args.output)
    plot_mean_difference(spectra, labels, output_dir=args.output)
    plot_feature_variance(spectra, output_dir=args.output)
    
    print("\nGenerating summary statistics...")
    stats = generate_summary_stats(spectra, labels, output_dir=args.output)
    
    print("\nEDA complete!")


if __name__ == '__main__':
    main()
