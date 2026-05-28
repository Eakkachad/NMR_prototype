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
