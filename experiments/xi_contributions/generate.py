"""
Here we construct a DMO angular power spectrum and plot the relative contributions
of the smooth/clump auto/cross correlations.
"""

import numpy as np

from clumpy_halo_model import Config
from clumpy_halo_model import dark_matter_only
from clumpy_halo_model import AngularPower

from pathlib import Path
root = Path(__file__).resolve().parents[2]

# load config file
cfg = Config.from_yaml(root/"experiments/Pm_contributions/config.yaml")


# Construct a DMO power spectrum:
ANGULAR_POWER = AngularPower(cfg, dark_matter_only)


# extract values
ell = cfg.ell_grid()
values = ANGULAR_POWER.components(ell)


# compute relative contributions of each term
ratios: dict[str, float] = {}
ratios['ss'] = values['P_1h_ss'] / values['P_1h']
ratios['sc'] = values['P_1h_sc'] / values['P_1h']
ratios['self_c'] = values['P_1h_self_c'] / values['P_1h']
ratios['cc'] = values['P_1h_cc'] / values['P_1h']


# safe data
data_dir = root/"data/Pm_contributions" 
np.savez(data_dir / "ratios.npz", k=k, **ratios)
