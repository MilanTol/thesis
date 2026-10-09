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
cfg = Config.from_yaml(root/"experiments/baryonification/config.yaml")


# Construct a DMO power spectrum:
DMO_POWER = dark_matter_only(cfg)
BARYON_POWER = with_baryons(cfg)


# extract values
k = cfg.k_grid()
dmo_power = DMO_POWER.components(k)['P_1h']
baryon_power = BARYON_POWER.components(k)['P_1h']


# construct ratio of power in baryons vs dmo:
reaction = baryon_power / dmo_power


# safe data 
data_dir = root / "data/baryonification"
np.savez(data_dir / "our_reaction.npz", k=k, reaction=reaction) 

