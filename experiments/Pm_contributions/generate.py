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
cfg = Config.from_yaml(root/"experiments/Pm_contributions/config.yaml")

# Construct a DMO power spectrum:
DMO_POWER = dark_matter_only(cfg)

k = cfg.k_grid()
values = DMO_POWER.components(k)
print(values)

# safe data
data_dir = root/"data/Pm_contributions" 
np.savez(data_dir / "data.npz", k=k, **values)








