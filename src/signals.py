# Update src/signals.py to this:

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_composite_signal(z_mom, z_vol, z_qual, z_esg=None, weights={'mom': 0.40, 'vol': 0.30, 'qual': 0.30, 'esg': 0.0}):
    """
    Combines factors into a single Alpha score. 
    Supports an optional ESG tilt.
    """
    logging.info(f"Building composite signal with weights: {weights}")
    
    # Align core dataframes
    z_mom, z_vol = z_mom.align(z_vol)
    z_mom, z_qual = z_mom.align(z_qual)
    
    # Base 3-factor score
    score = (weights['mom'] * z_mom) + \
            (weights['vol'] * z_vol) + \
            (weights['qual'] * z_qual)
            
    # Add ESG tilt if provided
    if z_esg is not None and weights.get('esg', 0) > 0:
        z_mom, z_esg = z_mom.align(z_esg)
        score += (weights['esg'] * z_esg)
        
    return score