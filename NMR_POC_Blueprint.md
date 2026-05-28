นี่คือไฟล์คู่มือและพิมพ์เขียวเชิงวิศวกรรม (Detailed Technical Blueprint & Skeleton Code) ที่ออกแบบมาเป็นภาษาอังกฤษอย่างละเอียด เพื่อให้คุณสามารถคัดลอกไปสร้างไฟล์ `.md` หรือป้อนเข้าสู่ Context Window ของ AI Agent ได้ทันที ตัวเอกสารจะบังคับกรอบการทำงาน กำหนดไลบรารีมาตรฐานสากล และมีโครงสร้างโค้ดสเกเลตัน (Skeleton Code) ที่ชัดเจน ซึ่งจะช่วยประหยัด Token และป้องกันไม่ให้ Agent เขียนโค้ดหลุดสโคปไปไกลครับ

---

```markdown
# Engineering Specification Blueprint: Working POC for Hybrid Physics-Aware NMR AI Pipeline

## 1. System Execution Context & Tech Stack
To minimize token consumption, prevent architectural hallucination, and ensure a stable TRL 4/5 implementation, the Agent MUST strictly utilize the following standardized ecosystem libraries rather than writing core signal-processing mathematical functions from scratch.

### Core Software Dependencies
* **Signal Parsing & I/O:** `nmrglue` (Standard for Bruker/Varian file parsing, baseline correction, and flattening).
* **Mathematical Baseline Alignment:** `pyicoshift` (Python 3 port of the interval correlation shifting algorithm).
* **Synthetic Signal Generation:** `nmrsim` (Physics-based multi-spin quantum mechanical simulation tool).
* **Time-Series / Feature Backbone:** `ibm-granite/granite-timeseries-patchtst-fm-r1` (Hugging Face open-weight foundation network for 1D sequence processing) or native PyTorch 1D-ResNet blocks.
* **Structural Shape Alignment:** `scipy.signal` (for peak detection) and `fastdtw` (for constrained Dynamic Time Warping calculations).

---

## 2. Directory Structure & File Inventory
The Agent must maintain and develop within the following file directory mapping:


```

AGENT/
└── 04_DATA/
└── scripts/
├── run_poc.py                  # Master batch inference & evaluation loop
├── app.py                      # Clinical Light-Theme Streamlit Dashboard
├── generate_synthetic_nmr.py   # Sim2Real Stochastic Generator
├── models_core.py              # Encoder, Autoencoder-Alignment, & Patch EBM
└── database_matcher.py         # Local HMDB Dictionary Mapping & DTW Scorer

```

---

## 3. Comprehensive Code Skeletons & Implementation Logic

### Module 1: Physics-Driven Stochastic Simulation (Sim2Real Pipeline)
* **File Target:** `AGENT/04_DATA/scripts/generate_synthetic_nmr.py`
* **Objective:** Synthesize clean baseline spectra representing clinical biomarkers, then apply randomized physical perturbations to create training pairs `(noisy_drifted_spectrum, clean_aligned_spectrum)`.

```python
import numpy as np
import scipy.signal as signal

class NMRDataSimulator:
    def __init__(self, num_points=4000, ppm_start=0.5, ppm_end=9.0):
        self.num_points = num_points
        self.ppm_axis = np.linspace(ppm_start, ppm_end, num_points)
        
    def generate_lorentzian_peak(self, x0, intensity, lw):
        """Calculates a strict physical Lorentzian peak shape."""
        return intensity * (lw**2 / ((self.ppm_axis - x0)**2 + lw**2))

    def create_clean_biomarker_profile(self, biomarker_dict):
        """
        Accepts a dictionary of known chemical shifts and generates a clean spectrum.
        Example: {'Alanine': [(1.48, 1.0, 0.02), (1.49, 1.0, 0.02)]}
        """
        clean_spectrum = np.zeros(self.num_points)
        for molecule, peaks in biomarker_dict.items():
            for (x0, intensity, lw) in peaks:
                clean_spectrum += self.generate_lorentzian_peak(x0, intensity, lw)
        return clean_spectrum

    def inject_stochastic_disturbances(self, clean_spectrum, max_drift=0.05, noise_level=0.02):
        """
        Stochastically applies physical degradation: Chemical Shift Drift and Ghost Peaks.
        """
        noisy_spectrum = np.copy(clean_spectrum)
        
        # 1. Simulate Chemical Shift Drift (Constant shifting for local regions)
        drift_delta = np.random.uniform(-max_drift, max_drift)
        shift_idx = int((drift_delta / (self.ppm_axis[1] - self.ppm_axis[0])))
        noisy_spectrum = np.roll(noisy_spectrum, shift_idx)
        
        # 2. Inject Quantum-Rule-Violating Ghost Peaks (Anomalies for Stage 3 EBM)
        if np.random.rand() > 0.5:
            ghost_pos = np.random.uniform(3.0, 5.5) # Inject into carbohydrate zone
            ghost_peak = self.generate_lorentzian_peak(ghost_pos, intensity=0.4, lw=0.01)
            noisy_spectrum += ghost_peak
            
        # 3. Add Baseline Gaussian Noise
        noise = np.random.normal(0, noise_level, self.num_points)
        noisy_spectrum += noise
        
        return noisy_spectrum

```

### Module 2: Sequence Encoder & Alignment Network

* **File Target:** `AGENT/04_DATA/scripts/models_core.py`
* **Objective:** Compress raw inputs to an aligned representation and unpack back to 1D space.
* **Critical Architectural Rule:** Do NOT use raw U-Net style skip connections from the initial encoder layers to the final decoder layers, as this bypasses the alignment trajectory and leaks positional physical errors.

```python
import torch
import torch.nn as nn

class SequenceAwareEncoder(nn.Module):
    def __init__(self, input_dim=4000, latent_dim=512):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv1d(16, 64, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(128)
        )
        self.fc_latent = nn.Linear(64 * 128, latent_dim)

    def forward(self, x):
        # x shape: [Batch_Size, 1, 4000]
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        latent = self.fc_latent(features)
        return latent

class SpectrumDecoder(nn.Module):
    def __init__(self, latent_dim=512, output_dim=4000):
        super().__init__()
        self.fc_expand = nn.Linear(latent_dim, 64 * 128)
        self.reconstruct = nn.Sequential(
            nn.ConvTranspose1d(64, 16, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(16, 1, kernel_size=15, stride=2, padding=7, output_padding=1)
        )
        # Linear adjustment layer to match exact output resolution
        self.out_layer = nn.Linear(4096, output_dim)

    def forward(self, latent):
        x = self.fc_expand(latent)
        x = x.view(x.size(0), 64, 128)
        x = self.reconstruct(x)
        x = self.out_layer(x)
        return x # Output shape: [Batch_Size, 1, 4000]

```

### Module 3: Localized Patch Energy-Based Model (Stage 3 Physics Verifier)

* **File Target:** Incorporated inside `AGENT/04_DATA/scripts/models_core.py`
* **Objective:** Calculate structural physical validity scores across segmented chemical domains.

```python
class LocalizedPatchEBM(nn.Module):
    def __init__(self, num_points=4000):
        super().__init__()
        # Small networks optimized to calculate local structural energy criteria
        self.patch_net = nn.Sequential(
            nn.Linear(1333, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, aligned_spectrum):
        """
        Slices reconstructed 4000-point vector into 3 distinct operational zones:
        Aliphatic (0.5-3.0 ppm), Carbohydrate (3.0-5.5 ppm), and Aromatic (5.5-9.0 ppm).
        """
        # Assumes uniform ppm slicing for structural simplification
        patch_size = aligned_spectrum.size(2) // 3
        patch1 = aligned_spectrum[:, :, 0:patch_size].squeeze(1)
        patch2 = aligned_spectrum[:, :, patch_size:2*patch_size].squeeze(1)
        patch3 = aligned_spectrum[:, :, 2*patch_size:3*patch_size].squeeze(1)
        
        energy1 = self.patch_net(patch1)
        energy2 = self.patch_net(patch2)
        energy3 = self.patch_net(patch3)
        
        # Mathematical Formulation: Global Energy = Sum of weighted localized energy patches
        global_energy = 0.4 * energy1 + 0.4 * energy2 + 0.2 * energy3
        return global_energy

```

### Module 4: Local Database Matching & Hybrid Scorer (Stage 4)

* **File Target:** `AGENT/04_DATA/scripts/database_matcher.py`
* **Objective:** Compute constrained DTW and peak bipartite matching scores against a local library.

```python
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class LocalDatabaseMatcher:
    def __init__(self):
        # Local mock library mimicking Human Metabolome Database (HMDB)
        self.hmdb_local_dict = {
            "L-Lactate": np.sin(np.linspace(0, 10, 4000)) * 0.5,  # Replace with true profile vector
            "Glucose": np.cos(np.linspace(0, 10, 4000)) * 0.8,
            "Acetate": np.zeros(4000)
        }

    def compute_hybrid_score(self, aligned_spec, ebm_score):
        best_match = None
        highest_confidence = -1.0
        
        for metabolite, ref_profile in self.hmdb_local_dict.items():
            # Calculate Constrained Dynamic Time Warping distance
            distance, _ = fastdtw(aligned_spec, ref_profile, radius=10, dist=euclidean)
            dtw_similarity = 1.0 / (1.0 + distance)
            
            # Simulated peak alignment assignment score
            peak_assignment_score = 0.85 
            
            # Formula: Match Confidence = 0.45 * assignment + 0.35 * dtw + 0.20 * ebm
            ebm_normalized_score = torch.sigmoid(torch.tensor(-ebm_score)).item()
            
            match_confidence = (0.45 * peak_assignment_score) + \
                               (0.35 * dtw_similarity) + \
                               (0.20 * ebm_normalized_score)
                               
            if match_confidence > highest_confidence:
                highest_confidence = match_confidence
                best_match = metabolite
                
        return {
            "candidate_name": best_match,
            "match_confidence": round(highest_confidence, 4),
            "ebm_physics_score": float(ebm_score.detach().cpu().numpy()[0][0])
        }

```

---

## 4. Master Orchestration Interface (Working POC Run Loop)

* **File Target:** `AGENT/04_DATA/scripts/run_poc.py`
* **Objective:** Execute full inference end-to-end and output structured Data Contract payloads.

```python
import torch
import json
from generate_synthetic_nmr import NMRDataSimulator
from models_core import SequenceAwareEncoder, SpectrumDecoder, LocalizedPatchEBM
from database_matcher import LocalDatabaseMatcher

def execute_pipeline():
    # 1. Initialize Pipeline components
    simulator = NMRDataSimulator()
    encoder = SequenceAwareEncoder()
    decoder = SpectrumDecoder()
    ebm = LocalizedPatchEBM()
    matcher = LocalDatabaseMatcher()
    
    # 2. Simulate incoming live clinical sample data
    mock_profile = {"L-Lactate": [(1.33, 1.0, 0.01), (4.11, 0.3, 0.01)]}
    clean_signal = simulator.create_clean_biomarker_profile(mock_profile)
    noisy_input = simulator.inject_stochastic_disturbances(clean_signal)
    
    # Convert numpy data to torch tensor matrix contract
    input_tensor = torch.tensor(noisy_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # 3. Process through Neural Architecture
    latent_space = encoder(input_tensor)
    reconstructed_ppm = decoder(latent_space)
    energy_score = ebm(reconstructed_ppm)
    
    # 4. Perform Local HMDB Match Matching
    aligned_array = reconstructed_ppm.squeeze(0).squeeze(0).detach().numpy()
    diagnostic_results = matcher.compute_hybrid_score(aligned_array, energy_score)
    
    # 5. Formulate final TRL 4/5 Structured Electronic Medical Record JSON Payload
    emr_report = {
        "sample_metadata": {
            "sample_id": "NMR-SAMPLE-2026-X99",
            "clinical_status": "Processed_Success",
            "dimension_points": len(aligned_array)
        },
        "diagnostic_metrics": {
            "detected_biomarker": diagnostic_results["candidate_name"],
            "match_confidence": diagnostic_results["match_confidence"],
            "ebm_physics_meter": diagnostic_results["ebm_physics_score"]
        }
    }
    
    # Output to disk
    with open("clinical_report.json", "w") as f:
        json.dump(emr_report, f, indent=4)
        
    print("Execution Success. Structured JSON EMR Report generated.")

if __name__ == "__main__":
    execute_pipeline()

```

---

## 5. Verification Checklist for Agent Compliance

Before final deployment verification, the Agent must validate the following conditions:

1. Running `python AGENT/04_DATA/scripts/run_poc.py` outputs a clean `clinical_report.json` with zero array dimension mismatches.
2. The Streamlit dashboard (`app.py`) updates its visualization canvas based on the actual 4,000 downsampled data dimensions without visual lag.
3. Every anomaly marker or flag triggered by clinical input saves trace configurations to `anomalous_peaks_feedback.json` for active learning telemetry.

```
