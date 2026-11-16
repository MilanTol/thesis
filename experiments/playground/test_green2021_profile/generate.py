"""
Here we construct a DMO power spectrum and plot the relative contributions
of the smooth/clump auto/cross correlations.
"""

import numpy as np

from clumpy_halo_model import Config
from clumpy_halo_model.models import *

from pathlib import Path
root = Path(__file__).resolve().parents[3]

# load config file
cfg = Config.from_yaml(root/"experiments/playground/test_green2021_profile/config.yaml")


# Construct a DMO power spectrum:
DMO_POWER = dark_matter_only(cfg)


# extract values
k = cfg.k_grid()
values = DMO_POWER.components(k)


# compute relative contributions of each term
ratios: dict[str, float] = {}
ratios['ss'] = values['P_1h_ss'] / values['P_1h']
ratios['sc'] = values['P_1h_sc'] / values['P_1h']
ratios['self_c'] = values['P_1h_self_c'] / values['P_1h']
ratios['cc'] = values['P_1h_cc'] / values['P_1h']


# safe data
data_dir = root/"experiments/playground/test_green2021_profile/data" 
np.savez(data_dir / "ratios.npz", k=k, **ratios)








