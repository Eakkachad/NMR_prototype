# Data Dictionary

## Matrix Structure

| Dimension | Description | Size |
|-----------|-------------|------|
| Rows (N) | Individual biological samples | Unknown (TBD when dataset arrives) |
| Columns (P) | Signal intensity at each ppm position | 20,000+ |

## Column Meaning

Each column represents a specific chemical shift position measured in **parts per million (ppm)**. The ppm axis typically ranges from 0 to ~10 ppm for ¹H NMR.

## Expected Value Ranges

| Property | Range | Notes |
|----------|-------|-------|
| ppm axis | 0.0 - 10.0 ppm (typical ¹H) | Higher ppm = more deshielded protons |
| Signal intensity | 0.0 - variable | Arbitrary units, depends on concentration and instrument |
| Baseline noise | ~0 ± small noise | Should be near zero in non-compound regions |

## Compound Targets (Preliminary)

Based on preplane.md, the classification targets are:

| Class | Description |
|-------|-------------|
| Plant Extract A | Specific compound profile TBD |
| Plant Extract B | Specific compound profile TBD |
| Plant Extract C | Specific compound profile TBD |

## Key ppm Regions for Common Metabolites

| ppm Range | Typical Compounds |
|-----------|-------------------|
| 0.8-1.5 | Lipids, branched-chain amino acids (valine, leucine, isoleucine) |
| 1.3-1.4 | Lactate (doublet) |
| 2.0-2.5 | Acetyl groups, glutamate, glutamine |
| 3.0-4.0 | Amino acids, sugars, choline |
| 3.2-3.9 | Glucose (multiple peaks) |
| 4.6-5.0 | Water (residual, usually suppressed) |
| 5.0-5.5 | Anomeric protons (sugars) |
| 6.5-8.5 | Aromatic compounds, histidine, phenylalanine, tyrosine |

## Processing Pipeline Outputs

| Stage | Output | Shape |
|-------|--------|-------|
| Raw extraction | `raw_matrix.csv` | (N, ~20000) |
| After binning | `binned_matrix.csv` | (N, ~500) |
| After PQN | `normalized_matrix.csv` | (N, ~500) |
| After feature selection | `selected_features.csv` | (N, ~50-100) |
| Labels | `labels.csv` | (N, 1) |
