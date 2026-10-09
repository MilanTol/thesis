"""
Here we construct a DMO power spectrum and plot the relative contributions
of the smooth/clump auto/cross correlations.
"""

from genericpath import isfile
import os

from clumpy_halo_model.config.config import Config
from clumpy_halo_model.models import *

from pathlib import Path

# load Pm_contributions folder
base = Path(__file__).resolve().parents[0]
# load thesis folder
root = Path(__file__).resolve().parents[2]

# load a config object if existing:
if os.isfile(base + 'config.yaml'):
    cfg = Config()




# # Construct a DMO power spectrum:
# DMO_POWER = dark_matter_only(cfg)







