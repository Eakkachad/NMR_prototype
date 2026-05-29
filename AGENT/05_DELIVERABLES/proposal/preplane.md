# Project Development Plan: Automated AI Pipeline for NMR Spectroscopy

## 1. Dataset Context and Characteristics

* 
**Track and Domain**: The project belongs to the Phenome track, specifically focusing on Metabolomics and Compound Profiling.


* 
**Data Source**: The dataset originates from the National Phenome Institute.


* 
**Data Dimension**: The input data consists of raw two-dimensional (2D) matrix formats containing more than 20,000 features or columns per individual sample.


* 
**Sample Profiles**: Initial sample data includes processed nuclear magnetic resonance (NMR) spectra from plant extracts (Plant Extract A, B, and C) across multiple subjects.


* 
**Current Annotation State**: The baseline peaks within the provided data are manually annotated using existing commercial software that requires processing each spectral image individually.



---

## 2. Problem Statement and Technical Challenges

* 
**Lack of Universal Standard**: There is currently no optimized, end-to-end standard solution for feature extraction, feature selection, and automated feature annotation for high-dimensional NMR signals.


* 
**Scalability Bottleneck**: The existing workflow relies heavily on manual intervention and per-image processing, making it impossible to scale for high-throughput screening or massive datasets.


* 
**Signal Overlapping**: NMR spectra present highly complex and dense signal overlaps where different chemical compounds share nearly identical chemical shifts, leading to significant classification difficulty.


* 
**Algorithmic Errors in Statistical Tools**: Current automated optimization tools introduce severe technical artifacts:


* 
**Ghost Peaks**: The generation of false-positive signal peaks that do not correspond to any actual physical or chemical compound in the sample.


* 
**Miss-identification**: Computational mislabeling of chemical compounds in overcrowded spectral areas due to rigid mathematical peak-fitting constraints.





---

## 3. Benchmarking Current Tools and Limitations

| Tool Name | Core Algorithm | Key Advantages | Critical Bottlenecks |
| --- | --- | --- | --- |
| **Chenomx** | Human-guided template matching 

 | Extremely high accuracy in heavily overlapped spectral regions because human experts can differentiate mixed signals visually.

 | Highly unscalable, slow (takes hours or days per sample), introduces human operator bias, and requires expensive commercial licensing.

 |
| **Bayesil** | Sequential Monte Carlo (SMC) probabilistic graphical models 

 | Fully automated, web-based, fast execution (2 to 7 minutes per sample), and eliminates human bias completely.

 | Rigid and inflexible; only functions with specific biological fluids (e.g., serum, plasma) and strict instrument frequencies.

 |
| **BATMAN** | Markov Chain Monte Carlo (MCMC) Bayesian deconvolution 

 | Rigorous statistical foundation that can model minor variations in chemical shift drifts.

 | Extremely slow and computationally heavy; fails to scale when applied to massive datasets with 20,000+ features.

 |
| **ASICS** | Penalized Linear Regression (LASSO/Global Optimization) 

 | Highly scalable and exceptionally fast, making it ideal for processing massive data matrices within high-throughput pipelines.

 | Prone to generating ghost peaks and misidentifying compounds when encountering noise or out-of-library signals.

 |

---

## 4. Proposed Hybrid AI Architecture

To address the limitations of current software, the proposed pipeline integrates deep learning, continuous-time differential equations, and energy constraints into a unified three-stage framework:

### Stage 1: Dimensionality Reduction and Feature Extraction

* This layer utilizes deep learning architectures such as Transformers or Convolutional Neural Networks (CNNs) to rapidly project the raw 20,000+ dimensional spectral data into a compact latent space representation.


* By compressing the global features first, it bypasses the heavy computational overhead that slows down traditional Bayesian methods.



### Stage 2: Continuous Alignment and Drift Correction

* This layer applies a Latent Neural ODE (Ordinary Differential Equation) framework to model signal shifts.


* Instead of processing discrete steps, the dynamics of chemical shift drifts (caused by minor environmental variations like pH or temperature) are learned as continuous trajectories within the low-dimensional latent space according to the continuous-depth equation:

$$\frac{dh(t)}{dt} = f(h(t), t, \theta)$$


* This enables precise alignment without demanding immense computational power or manual warping.



### Stage 3: Physics-Constrained Deconvolution and Identification

* This layer deploys an Energy-Based Model (EBM) as a generative verification gate to suppress false positives.


* The model's loss function incorporates rigid physical and chemical constraints, such as theoretical spin-spin coupling constants and fixed peak-height ratios inherent to specific molecular structures.


* If the network attempts to insert an mathematically convenient but physically impossible signal, the EBM assigns it a high energy score and rejects it, systematically eliminating ghost peaks.



### External Data Integration for Unknown Compound Discovery

* Unannotated peak clusters that do not match the local library are isolated using spatial correlation clustering algorithms like SPA-STOCSY.


* The pipeline automatically extracts these unidentified structural motifs and cross-references them via automated API queries to trusted international databases like HMDB and PubChem to discover novel biomarkers or unknown compounds.



---

## 5. Key Literature and Open-Source References

### Theme 1: Transformers and Attention Mechanisms for NMR

* 
**NMRformer** (Zhou et al., 2025): Treats 1D NMR spectra as sequences of peaks, using a transformer encoder with self-attention and peak-height ratios to achieve over 88% peak assignment accuracy.


* 
**NMRTrans** (Yang et al., 2026): Models spectra as unordered peak sets using a Set Transformer, aligning better with the physical nature of chemical shifts.


* 
**FlavorFormer** (Liao et al., 2025): A hybrid CNN-Transformer model designed to extract both local spectral features and global dependencies in mixtures.


* 
**MolDeTr** (Schmid et al., 2026): A detection transformer that unifies peak picking, multiplet identification, and proton-count extraction in a single pass.



### Theme 2: Neural ODEs and Continuous Transformations

* 
**Latent Space Energy-Based Neural ODEs** (2024): Combines energy priors with latent ODE state evolution, offering a blueprint for continuous modeling of spectral variations.


* 
**Latent ODEs for Irregularly-Sampled Time Series** (Rubanova et al., 2019): Provides the foundational PyTorch implementation (`YuliaRubanova/latent_ode`) for continuous-time sequence modeling.


* 
**SPIN-ODE** (2025): Demonstrates a physics-constrained Neural ODE strategy for chemical networks, enforcing physical boundary rules during integration.



### Theme 3: Energy-Based Models and Physics-Informed ML

* 
**DEEP Picker & Voigt Fitter1D** (Li et al., 2021, 2023): Deep neural networks trained on synthetic spin-dynamics simulations to perform fully automated peak picking and analytical deconvolution in overlapped regions.


* 
**EB-gMCR** (2024): An energy-based generative deep learning model for signal unmixing that enforces parsimony, automatically discovering the correct number of components and removing spurious ghost signals.


* 
**MIST-CF** (2023): Uses an energy-based function built over a Spectrum Transformer backbone to rank chemical formulas, inherently suppressing false positives.



### Primary Accessible Code Repositories

* 
**NMRQNet**: Quantitative deep learning for plasma metabolites (`github.com/LiuzLab/NMRQNet`).


* 
**SPA-STOCSY**: Automated tool for clustering and identifying non-annotated metabolites (`github.com/LiuzLab/SPA-STOCSY`).


* 
**DEEP Picker**: Advanced peak picking and deconvolution code base (`github.com/lidawei1975/deep`).
