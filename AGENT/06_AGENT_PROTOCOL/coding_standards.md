# Coding Standards

## Language & Environment
- **Python 3.10+**
- **Package manager:** pip (or conda if needed for specific packages)
- **Compute target:** RTX 4060 (local) or Google Colab

## Style Guide
- Follow PEP 8
- Use type hints where practical
- Maximum line length: 100 characters
- Use f-strings for string formatting

## File Organization
```
04_DATA/
├── scripts/           # All Python scripts
│   ├── generate_synthetic_data.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── pdf_parser.py
│   ├── baseline_model.py    # PLS-DA + SVM + RF
│   ├── advanced_model.py    # Neural ODE + EBM (if attempted)
│   └── dashboard.py         # Streamlit/Gradio app
├── parsed/            # Generated/parsed data files
│   ├── synthetic_spectra.csv
│   ├── synthetic_labels.csv
│   └── ...
└── raw/               # Original unmodified files
    └── (copy of PDF)
```

## Naming Conventions
| Element | Convention | Example |
|---------|-----------|---------|
| Files | snake_case.py | `baseline_model.py` |
| Functions | snake_case | `train_svm_model()` |
| Classes | PascalCase | `NMRPipeline` |
| Constants | UPPER_SNAKE | `MAX_FEATURES = 20000` |
| Variables | snake_case | `binned_spectra` |

## Documentation
- Every script starts with a module docstring explaining purpose and usage
- Every function has a docstring with Args, Returns, and optionally Raises
- Non-obvious logic gets inline comments

## Dependencies
Maintain a `requirements.txt` at `04_DATA/scripts/requirements.txt`:
```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
plotly>=5.0
pdfplumber>=0.10
streamlit>=1.30
torch>=2.0
torchdiffeq>=0.2
```

## Testing
- For the POC, informal testing is acceptable (assert statements, print checks)
- Every script should be runnable standalone with `python script.py`
- Include `if __name__ == '__main__':` blocks for testing
- Log results to files, not just stdout

## Error Handling
- Use try/except for external dependencies (PDF parsing, file I/O)
- Fail gracefully with clear error messages
- Never silently swallow exceptions
