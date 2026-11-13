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


# Construct DMO power spectra:
WDM_DMO = dark_matter_only(cfg_WDM)
CDM_DMO = dark_matter_only(cfg_CDM)


# Construct baryon power spectra:
WDM_BAR = with_baryons(cfg_WDM)
CDM_BAR = with_baryons(cfg_CDM)


# extract values
k = cfg_WDM.k_grid() 
WDM_DMO_comp = WDM_DMO.components(k) 
CDM_DMO_comp = CDM_DMO.components(k)

WDM_BAR_comp = WDM_BAR.components(k) 
CDM_BAR_comp = CDM_BAR.components(k) 

# compute relative contributions of each term
ratios_dmo: dict[str, float] = {}
# ratios_dmo['ss'] = WDM_values['P_1h_ss'] / CDM_values['P_1h_ss']
# ratios_dmo['sc'] = WDM_values['P_1h_sc'] / CDM_values['P_1h_sc']
# ratios_dmo['self_c'] = WDM_values['P_1h_self_c'] / CDM_values['P_1h_self_c']
# ratios_dmo['cc'] = WDM_values['P_1h_cc'] / CDM_values['P_1h_cc']
ratios_dmo['tot'] = WDM_DMO_comp['P_tot'] / CDM_DMO_comp['P_tot']

ratios_bar: dict[str, float] = {}
# ratios_bar['ss'] = WDM_values['P_1h_ss'] / CDM_values['P_1h_ss']
# ratios_bar['sc'] = WDM_values['P_1h_sc'] / CDM_values['P_1h_sc']
# ratios_bar['self_c'] = WDM_values['P_1h_self_c'] / CDM_values['P_1h_self_c']
# ratios_bar['cc'] = WDM_values['P_1h_cc'] / CDM_values['P_1h_cc']
ratios_bar['tot'] = WDM_BAR_comp['P_tot'] / CDM_BAR_comp['P_tot']

# safe data
data_dir = root/"data/Pm_WDM-CDM" 
np.savez(data_dir / "ratios_dmo.npz", k=k, **ratios_dmo)
np.savez(data_dir / "ratios_bar.npz", k=k, **ratios_bar)