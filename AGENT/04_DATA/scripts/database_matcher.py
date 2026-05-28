from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import torch

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
            distance, _ = fastdtw(aligned_spec, ref_profile, radius=10, dist=lambda a, b: abs(a - b))
            dtw_similarity = 1.0 / (1.0 + distance)
            
            # Simulated peak alignment assignment score
            peak_assignment_score = 0.85 
            
            # Formula: Match Confidence = 0.45 * assignment + 0.35 * dtw + 0.20 * ebm
            ebm_normalized_score = torch.sigmoid(-ebm_score.detach().clone()).item()
            
            match_confidence = (0.45 * peak_assignment_score) + \
                               (0.35 * dtw_similarity) + \
                               (0.20 * ebm_normalized_score)
                               
            if match_confidence > highest_confidence:
                highest_confidence = match_confidence
                best_match = metabolite
                
        try:
            ebm_physics_score = float(ebm_score.detach().cpu().numpy()[0][0])
        except Exception:
            ebm_physics_score = float(ebm_score.detach().cpu().numpy().flatten()[0])

        return {
            "candidate_name": best_match,
            "match_confidence": round(highest_confidence, 4),
            "ebm_physics_score": ebm_physics_score
        }
