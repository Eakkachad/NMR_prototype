# Parsed Data Directory
Generated/parsed data files go here.

## Expected Files (after running scripts):
- `synthetic_spectra.csv` — synthetic raw spectra (N × 20001)
- `synthetic_labels.csv` — compound class labels
- `synthetic_spectra_binned.csv` — after spectral binning
- `synthetic_preprocessed.csv` — after full preprocessing
- `ppm_axis.csv` — ppm axis metadata
- `spectra_overlay.png` — EDA plot
- `mean_difference.png` — EDA plot
- `feature_variance.png` — EDA plot
- `eda_summary.md` — EDA statistics

## How to Generate:
```bash
cd AGENT/04_DATA/scripts/
python generate_synthetic_data.py --n_samples 10 --output_dir ../parsed/
python eda.py --data ../parsed/synthetic_spectra.csv --labels ../parsed/synthetic_labels.csv --output ../parsed/
```
