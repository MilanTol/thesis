"""
Here we construct a DMO power spectrum and plot the relative contributions
of the smooth/clump auto/cross correlations.
"""

import numpy as np

from clumpy_halo_model import Config
from clumpy_halo_model.models import *

from pathlib import Path

root = Path(__file__).resolve().parents[2]
data_dir = root/"data/clump_massbins_contributions" 

# load config file
cfg = Config.from_yaml(root/"experiments/clump_massbins_contributions/config.yaml")

M_MINS = [
    13,
    12,
    11, 
    10, 
    9,
    8,
]


ratios: dict[str, float] = {}

k = cfg.k_grid()
POWER = with_baryons(cfg)
TOT = POWER.components(k)['P_tot']


P_1h_bin_previous = np.zeros_like(k)

# for each massbin compute the power spectrum and save to data
for logm_min in M_MINS:
    m_min = 10**logm_min
    cfg_bin = cfg.replace(m_min=m_min)
    
    # Construct a DMO power spectrum:
    POWER_bin = with_baryons(cfg_bin)

    # extract values
    P_1h_bin = POWER_bin.components(k)['P_1h'] - POWER_bin.components(k)['P_1h_ss']
    
    # compute difference:
    P_1h_bin_clumps = P_1h_bin - P_1h_bin_previous 

    # compute relative contributions of each term
    ratios[f'{m_min}'] = P_1h_bin_clumps / TOT
    
    # reset P_1h_bin_previous
    P_1h_bin_previous = P_1h_bin

# safe data
np.savez(data_dir / "ratios.npz", k=k, **ratios)








