"""
Here we construct 2 power spectra (with baryons).
One for WDM and one for CDM. We then compute the ratio of the power spectrum. 
"""

import numpy as np

from clumpy_halo_model import Config
from clumpy_halo_model.models import *

from pathlib import Path
root = Path(__file__).resolve().parents[2]

# load config file
cfg_WDM = Config.from_yaml(root/"experiments/Pm_WDM-CDM/config.yaml")
cfg_CDM = cfg_WDM.replace(alpha=0, beta=0)


# Construct a DMO power spectrum:
WDM_POWER = with_baryons(cfg_WDM)
CDM_POWER = with_baryons(cfg_CDM)


# extract values
k = cfg_WDM.k_grid() 
WDM_values = WDM_POWER.components(k) 
CDM_values = CDM_POWER.components(k)

# compute relative contributions of each term
ratios: dict[str, float] = {}
ratios['ss'] = WDM_values['P_1h_ss'] / WDM_values['P_1h_ss']
ratios['sc'] = WDM_values['P_1h_sc'] / CDM_values['P_1h_sc']
ratios['self_c'] = WDM_values['P_1h_self_c'] / CDM_values['P_1h_self_c']
ratios['cc'] = WDM_values['P_1h_cc'] / CDM_values['P_1h_cc']
ratios['tot'] = WDM_values['P_tot'] / CDM_values['P_tot']

# safe data
data_dir = root/"data/Pm_WDM-CDM" 
np.savez(data_dir / "ratios.npz", k=k, **ratios)