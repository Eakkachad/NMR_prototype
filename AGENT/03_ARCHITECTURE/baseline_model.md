# Baseline Model — Core Layer Specification

> Priority: 🔴 CRITICAL — This is the guaranteed demo. Build this first.

## Overview

The baseline model uses classical, well-proven ML techniques that work reliably in high-dimensional, low-sample settings. This guarantees a working prototype for the Feasibility criterion (30 pts).

## Pipeline

```
Binned Data (N × ~500) → PLS-DA (feature ranking) → Top Features → SVM/RF → Prediction
```

## Step 1: Feature Selection via PLS-DA

### What
Partial Least Squares Discriminant Analysis (PLS-DA) is a supervised dimensionality reduction technique specifically designed for high-dimensional chemical data.

### How
1. Fit PLS-DA model on binned/normalized spectra with compound labels
2. Calculate Variable Importance in Projection (VIP) scores for each bin
3. Select bins with VIP > 1.0 (standard threshold)
4. These represent the ppm regions most discriminative for compound classification

### Implementation
```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer
import numpy as np

def pls_da_vip(X, y, n_components=2):
    """
    PLS-DA with VIP score calculation.
    X: (N, P) binned spectra
    y: (N,) compound labels
    Returns: VIP scores (P,)
    """
    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, Y)
    
    # VIP calculation
    t = pls.x_scores_       # (N, n_components)
    w = pls.x_weights_      # (P, n_components)
    q = pls.y_loadings_     # (Q, n_components)
    
    p, h = w.shape
    vips = np.zeros(p)
    
    s = np.diag(t.T @ t @ q.T @ q).reshape(1, -1)
    total_s = np.sum(s)
    
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p * (s @ weight.reshape(-1, 1)) / total_s)
    
    return vips
```

## Step 2: Classification

### SVM with RBF Kernel
- **Why:** Excellent in high-dimensional, small-sample settings. RBF kernel handles nonlinear decision boundaries.
- **Hyperparameters:** C and gamma via GridSearchCV

### Random Forest
- **Why:** Provides built-in feature importance, resistant to overfitting with proper tuning
- **Hyperparameters:** n_estimators, max_depth via GridSearchCV

### Implementation
```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def train_and_evaluate(X_selected, y, cv_folds=5):
    """
    Train SVM and RF, report cross-validated performance.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    models = {
        'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_selected, y, cv=cv, scoring='f1_weighted')
        results[name] = {
            'mean_f1': scores.mean(),
            'std_f1': scores.std(),
            'scores': scores,
        }
    
    return results
```

## Expected Outputs

1. **Classification report:** Accuracy, Precision, Recall, F1 per compound class
2. **Confusion matrix:** Visual heatmap
3. **VIP plot:** Bar chart of top features with ppm positions
4. **Spectral overlay:** Input spectra colored by predicted class

## Success Criteria

| Metric | Target (synthetic data) | Target (real data) |
|--------|-------------------------|---------------------|
| Accuracy | >80% | >70% |
| F1 (weighted) | >0.80 | >0.70 |
| Cross-val stability | std < 0.10 | std < 0.15 |

## Dependencies

```
scikit-learn >= 1.3
numpy >= 1.24
pandas >= 2.0
matplotlib >= 3.7
seaborn >= 0.12
```
