"""
Here we construct a DMO power spectrum and plot the relative contributions
of the smooth/clump auto/cross correlations.
"""

import os

from halo_model.config.config import Config
from halo_model.models import *


# instantiate a config object:
cfg = Config()


# Construct a DMO power spectrum:
DMO_POWER = dark_matter_only(cfg)







