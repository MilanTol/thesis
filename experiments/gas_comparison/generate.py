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
cfg_dmo_wdm = Config.from_yaml(root/"experiments/gas_comparison/config_debackere.yaml")
cfg_deb_wdm = Config.from_yaml(root/"experiments/gas_comparison/config_debackere.yaml")
cfg_sie_wdm = Config.from_yaml(root/"experiments/gas_comparison/config_siegel.yaml")

cfg_dmo_cdm = cfg_dmo_wdm.replace(alpha=0, beta=0)
cfg_deb_cdm = cfg_deb_wdm.replace(alpha=0, beta=0)
cfg_sie_cdm = cfg_sie_wdm.replace(alpha=0, beta=0)


# Construct power spectra:
P_dmo_wdm = dark_matter_only(cfg_dmo_wdm)
P_deb_wdm = with_baryons(cfg_deb_wdm)
P_sie_wdm = with_baryons(cfg_sie_wdm)

P_dmo_cdm = dark_matter_only(cfg_dmo_cdm)
P_deb_cdm = with_baryons(cfg_deb_cdm)
P_sie_cdm = with_baryons(cfg_sie_cdm)


# extract values
k = cfg_deb_wdm.k_grid() 
P_dmo_wdm_comp = P_dmo_wdm.components(k)
P_deb_wdm_comp = P_deb_wdm.components(k) 
P_sie_wdm_comp = P_sie_wdm.components(k)

P_dmo_cdm_comp = P_dmo_cdm.components(k)
P_deb_cdm_comp = P_deb_cdm.components(k)
P_sie_cdm_comp = P_sie_cdm.components(k)


# compute relative contributions of each term
ratios: dict[str, float] = {}
ratios['dmo'] = P_dmo_wdm_comp['P_tot'] / P_dmo_cdm_comp['P_tot']
ratios['debackere'] = P_deb_wdm_comp['P_tot'] / P_deb_cdm_comp['P_tot']
ratios['siegel'] = P_sie_wdm_comp['P_tot'] / P_sie_cdm_comp['P_tot']


# safe data
data_dir = root/"data/gas_comparison" 
np.savez(data_dir / "ratios.npz", k=k, **ratios)
