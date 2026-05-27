"""
Synthetic NMR Data Generator
=============================
Generates realistic synthetic 1H NMR spectra for 3 compound classes
(Plant Extract A, B, C) to enable POC development before real data arrives.

Usage:
    python generate_synthetic_data.py --n_samples 30 --output_dir ../parsed/

Output:
    - synthetic_spectra.csv  (N x 20001: ppm axis + intensities)
    - synthetic_labels.csv   (N x 1: compound class labels)
    - synthetic_spectra_binned.csv  (N x 500: after binning)
"""

import numpy as np
import pandas as pd
import os
import argparse


def generate_nmr_peak(ppm_axis, center, width, height, noise_std=0.01):
    """Generate a single Lorentzian NMR peak."""
    peak = height * (width**2) / ((ppm_axis - center)**2 + width**2)
    return peak


def generate_compound_spectrum(ppm_axis, peak_params, noise_level=0.02, shift_std=0.005):
    """
    Generate a spectrum for one compound with realistic peak shifting.
    
    Args:
        ppm_axis: array of ppm positions
        peak_params: list of (center, width, height) tuples
        noise_level: baseline noise standard deviation
        shift_std: random peak shift standard deviation (simulates pH/temp drift)
    """
    spectrum = np.zeros_like(ppm_axis)
    
    for center, width, height in peak_params:
        # Add small random shift to simulate chemical shift drift
        shifted_center = center + np.random.normal(0, shift_std)
        # Add small random variation to height
        varied_height = height * (1 + np.random.normal(0, 0.1))
        spectrum += generate_nmr_peak(ppm_axis, shifted_center, width, max(0, varied_height))
    
    # Add baseline noise
    spectrum += np.random.normal(0, noise_level, len(ppm_axis))
    # Ensure non-negative
    spectrum = np.maximum(spectrum, 0)
    
    return spectrum


def define_compound_profiles():
    """
    Define spectral signatures for 3 plant extract classes.
    Each compound has characteristic peaks at specific ppm positions.
    """
    # Plant Extract A: Rich in sugars (glucose-like) and amino acids
    extract_a = [
        (1.33, 0.015, 0.8),   # Lactate doublet
        (1.35, 0.015, 0.75),
        (2.08, 0.02, 0.4),    # Acetate
        (3.24, 0.015, 0.9),   # Glucose region
        (3.41, 0.015, 0.85),
        (3.49, 0.015, 0.7),
        (3.72, 0.02, 0.6),
        (3.84, 0.015, 0.65),
        (5.23, 0.015, 0.5),   # α-glucose anomeric
    ]
    
    # Plant Extract B: Rich in aromatic compounds (flavonoid-like)
    extract_b = [
        (1.18, 0.02, 0.3),    # Lipid region
        (2.35, 0.02, 0.5),    # Organic acids
        (3.55, 0.02, 0.4),    # Some sugar overlap
        (6.25, 0.02, 0.7),    # Aromatic region
        (6.85, 0.015, 0.9),   # Flavonoid peaks
        (7.12, 0.015, 0.85),
        (7.45, 0.02, 0.6),
        (7.88, 0.015, 0.4),
        (8.15, 0.02, 0.3),
    ]
    
    # Plant Extract C: Mixed profile (terpene-like + some sugar)
    extract_c = [
        (0.88, 0.02, 0.7),    # Terpene methyl groups
        (0.95, 0.015, 0.65),
        (1.25, 0.02, 0.5),    # Lipid chain
        (1.62, 0.02, 0.8),    # Terpene signature
        (2.02, 0.015, 0.45),
        (3.30, 0.02, 0.35),   # Slight sugar overlap
        (4.52, 0.02, 0.3),
        (5.10, 0.015, 0.55),  # Olefinic terpene
        (5.35, 0.02, 0.4),
    ]
    
    return {
        'Plant_Extract_A': extract_a,
        'Plant_Extract_B': extract_b,
        'Plant_Extract_C': extract_c,
    }


def generate_dataset(n_samples_per_class=10, n_points=20001, ppm_range=(0.0, 10.0)):
    """
    Generate full synthetic dataset.
    
    Args:
        n_samples_per_class: number of samples per compound class
        n_points: number of ppm positions (features)
        ppm_range: (min_ppm, max_ppm)
    
    Returns:
        spectra_df: DataFrame (N x n_points+1) with ppm as column names
        labels_df: DataFrame (N x 1) with compound labels
    """
    ppm_axis = np.linspace(ppm_range[0], ppm_range[1], n_points)
    compounds = define_compound_profiles()
    
    all_spectra = []
    all_labels = []
    
    for compound_name, peak_params in compounds.items():
        for i in range(n_samples_per_class):
            spectrum = generate_compound_spectrum(
                ppm_axis, peak_params,
                noise_level=0.015 + np.random.uniform(0, 0.01),  # Variable noise
                shift_std=0.005 + np.random.uniform(0, 0.003),   # Variable drift
            )
            all_spectra.append(spectrum)
            all_labels.append(compound_name)
    
    # Create DataFrames
    col_names = [f"ppm_{p:.4f}" for p in ppm_axis]
    spectra_df = pd.DataFrame(all_spectra, columns=col_names)
    labels_df = pd.DataFrame({'compound': all_labels})
    
    # Store ppm axis as metadata row
    ppm_df = pd.DataFrame([ppm_axis], columns=col_names, index=['ppm_axis'])
    
    return spectra_df, labels_df, ppm_df


def spectral_binning(spectra_df, bin_width=0.02, ppm_range=(0.0, 10.0)):
    """
    Perform spectral binning (bucketing) to reduce dimensionality.
    
    Args:
        spectra_df: DataFrame (N x P) of raw spectra
        bin_width: width of each bin in ppm
        ppm_range: (min, max) ppm range
    
    Returns:
        binned_df: DataFrame (N x n_bins) with bin AUC values
    """
    n_bins = int((ppm_range[1] - ppm_range[0]) / bin_width)
    ppm_values = np.array([float(c.replace('ppm_', '')) for c in spectra_df.columns])
    
    binned_data = []
    bin_labels = []
    
    for i in range(n_bins):
        bin_start = ppm_range[0] + i * bin_width
        bin_end = bin_start + bin_width
        bin_center = (bin_start + bin_end) / 2
        
        mask = (ppm_values >= bin_start) & (ppm_values < bin_end)
        if mask.any():
            bin_values = spectra_df.iloc[:, mask].sum(axis=1)
            binned_data.append(bin_values)
            bin_labels.append(f"bin_{bin_center:.3f}")
    
    binned_df = pd.DataFrame(dict(zip(bin_labels, binned_data)))
    return binned_df


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic NMR data')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Samples per class (default: 10, total = 3x this)')
    parser.add_argument('--output_dir', type=str, default='../parsed/',
                        help='Output directory for CSV files')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating synthetic NMR data: {args.n_samples} samples per class...")
    spectra_df, labels_df, ppm_df = generate_dataset(n_samples_per_class=args.n_samples)
    
    print(f"  Spectra shape: {spectra_df.shape}")
    print(f"  Labels shape: {labels_df.shape}")
    print(f"  Classes: {labels_df['compound'].value_counts().to_dict()}")
    
    # Save raw spectra
    spectra_df.to_csv(os.path.join(args.output_dir, 'synthetic_spectra.csv'), index=False)
    labels_df.to_csv(os.path.join(args.output_dir, 'synthetic_labels.csv'), index=False)
    ppm_df.to_csv(os.path.join(args.output_dir, 'ppm_axis.csv'))
    
    # Perform binning
    print("Performing spectral binning (0.02 ppm bins)...")
    binned_df = spectral_binning(spectra_df)
    binned_df.to_csv(os.path.join(args.output_dir, 'synthetic_spectra_binned.csv'), index=False)
    print(f"  Binned shape: {binned_df.shape}")
    
    print(f"\nFiles saved to {args.output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
