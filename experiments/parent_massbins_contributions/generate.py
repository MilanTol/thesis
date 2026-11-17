"""
Here we construct a DMO power spectrum and plot the relative contributions
of the smooth/clump auto/cross correlations.
"""

import numpy as np

from clumpy_halo_model import Config
from clumpy_halo_model.models import *

from pathlib import Path
root = Path(__file__).resolve().parents[2]

# load config file
cfg = Config.from_yaml(root/"experiments/parent_massbins_contributions/config.yaml")

MASSBINS = [
    # [8, 9],
    # [9, 10],
    [10, 11],
    [11, 12],
    [12, 13],
    [13, 14],
    [14, 15],
]


ratios: dict[str, float] = {}

k = cfg.k_grid()
POWER = with_baryons(cfg)
TOT = POWER.components(k)['P_tot']


# for each massbin compute the power spectrum and save to data
for massbin in MASSBINS:
    
    M_min, M_max = 10**massbin[0], 10**massbin[1]
    cfg_bin = cfg.replace(M_min=M_min, M_max=M_max)
    
    print(np.log10(cfg_bin.M_min), np.log10(cfg_bin.M_max))


    # Construct a DMO power spectrum:
    POWER_bin = with_baryons(cfg_bin)

    # extract values
    P_1h_bin = POWER_bin.components(k)['P_1h']

    # compute relative contributions of each term
    ratios[f'{massbin}'] = P_1h_bin / TOT


# safe data
data_dir = root/"data/parent_massbins_contributions" 
np.savez(data_dir / "ratios.npz", k=k, **ratios)








