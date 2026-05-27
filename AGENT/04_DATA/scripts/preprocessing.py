"""
NMR Preprocessing Pipeline
============================
Implements spectral binning and PQN normalization for NMR data.

Usage:
    from preprocessing import preprocess_pipeline
    X_processed, feature_names = preprocess_pipeline(raw_spectra_df)
"""

import numpy as np
import pandas as pd
from scipy import stats


def spectral_binning(spectra_df, bin_width=0.02, ppm_range=(0.0, 10.0)):
    """
    Perform spectral binning (bucketing) to reduce dimensionality.
    Collapses 20,000+ features into ~500 bins by calculating the
    area under the curve (AUC) for each ppm interval.
    
    Args:
        spectra_df: DataFrame (N x P) where columns encode ppm positions
        bin_width: width of each bin in ppm (default: 0.02)
        ppm_range: tuple (min_ppm, max_ppm)
    
    Returns:
        binned_df: DataFrame (N x n_bins) with bin AUC values
    """
    n_bins = int((ppm_range[1] - ppm_range[0]) / bin_width)
    
    # Extract ppm values from column names
    ppm_values = np.array([float(c.replace('ppm_', '')) for c in spectra_df.columns])
    
    binned_data = []
    bin_labels = []
    
    for i in range(n_bins):
        bin_start = ppm_range[0] + i * bin_width
        bin_end = bin_start + bin_width
        bin_center = (bin_start + bin_end) / 2
        
        mask = (ppm_values >= bin_start) & (ppm_values < bin_end)
        if mask.any():
            # AUC approximation: sum of intensities in bin
            bin_values = spectra_df.iloc[:, mask].sum(axis=1)
            binned_data.append(bin_values)
            bin_labels.append(f"bin_{bin_center:.3f}")
    
    binned_df = pd.DataFrame(dict(zip(bin_labels, binned_data)))
    return binned_df


def pqn_normalization(spectra_df):
    """
    Probabilistic Quotient Normalization (PQN).
    
    Steps:
    1. Calculate the reference spectrum (median across all samples)
    2. For each sample, divide by the reference to get quotients
    3. Calculate the median quotient for each sample
    4. Divide each sample by its median quotient
    
    This corrects for varying dilution effects across biological replicates.
    
    Args:
        spectra_df: DataFrame (N x P) of spectral data
    
    Returns:
        normalized_df: DataFrame (N x P) after PQN normalization
    """
    data = spectra_df.values.copy()
    
    # Step 1: Calculate reference spectrum (median of all samples)
    reference = np.median(data, axis=0)
    
    # Avoid division by zero
    reference[reference == 0] = np.finfo(float).eps
    
    # Step 2-3: For each sample, calculate quotients and find median
    normalized = np.zeros_like(data)
    for i in range(data.shape[0]):
        quotients = data[i, :] / reference
        # Remove zeros and infinities for median calculation
        valid_quotients = quotients[(quotients > 0) & np.isfinite(quotients)]
        if len(valid_quotients) > 0:
            median_quotient = np.median(valid_quotients)
            normalized[i, :] = data[i, :] / median_quotient
        else:
            normalized[i, :] = data[i, :]
    
    return pd.DataFrame(normalized, columns=spectra_df.columns, index=spectra_df.index)


def remove_water_region(spectra_df, water_range=(4.5, 5.0)):
    """
    Remove the water/solvent signal region from the spectrum.
    
    Args:
        spectra_df: DataFrame with ppm-labeled columns
        water_range: (min_ppm, max_ppm) of the water region to exclude
    
    Returns:
        filtered_df: DataFrame with water region columns removed
    """
    ppm_values = np.array([float(c.replace('ppm_', '').replace('bin_', '')) 
                           for c in spectra_df.columns])
    mask = ~((ppm_values >= water_range[0]) & (ppm_values <= water_range[1]))
    return spectra_df.iloc[:, mask]


def baseline_correction(spectra_df, method='median'):
    """
    Simple baseline correction by subtracting the baseline estimate.
    
    Args:
        spectra_df: DataFrame (N x P)
        method: 'median' or 'min'
    
    Returns:
        corrected_df: DataFrame (N x P) after baseline subtraction
    """
    data = spectra_df.values.copy()
    
    if method == 'median':
        # Use the 5th percentile as baseline estimate
        baseline = np.percentile(data, 5, axis=1, keepdims=True)
    elif method == 'min':
        baseline = np.min(data, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    corrected = data - baseline
    corrected = np.maximum(corrected, 0)  # Ensure non-negative
    
    return pd.DataFrame(corrected, columns=spectra_df.columns, index=spectra_df.index)


def preprocess_pipeline(raw_spectra_df, bin_width=0.02, remove_water=True):
    """
    Full preprocessing pipeline: binning → water removal → PQN → baseline correction.
    
    Args:
        raw_spectra_df: DataFrame (N x 20000+) of raw spectral data
        bin_width: ppm bin width for spectral binning
        remove_water: whether to remove the water region
    
    Returns:
        processed_df: DataFrame (N x ~500) fully preprocessed
        feature_names: list of bin center labels
    """
    print("Step 1: Spectral binning...")
    binned = spectral_binning(raw_spectra_df, bin_width=bin_width)
    print(f"  Reduced {raw_spectra_df.shape[1]} → {binned.shape[1]} features")
    
    if remove_water:
        print("Step 2: Removing water region (4.5-5.0 ppm)...")
        binned = remove_water_region(binned)
        print(f"  Remaining features: {binned.shape[1]}")
    
    print("Step 3: PQN normalization...")
    normalized = pqn_normalization(binned)
    
    print("Step 4: Baseline correction...")
    corrected = baseline_correction(normalized)
    
    print(f"Final shape: {corrected.shape}")
    return corrected, list(corrected.columns)


if __name__ == '__main__':
    # Quick test with synthetic data
    print("Loading synthetic data for testing...")
    spectra = pd.read_csv('../parsed/synthetic_spectra.csv')
    labels = pd.read_csv('../parsed/synthetic_labels.csv')
    
    processed, features = preprocess_pipeline(spectra)
    processed.to_csv('../parsed/synthetic_preprocessed.csv', index=False)
    print(f"\nPreprocessed data saved. Shape: {processed.shape}")
    print(f"Label distribution:\n{labels['compound'].value_counts()}")
