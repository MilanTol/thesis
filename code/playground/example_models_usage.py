"""
Example on how to create MatterPower objects from a config file and
access the power spectrum values of various components.
"""


# fix pwd
import os
code_dir = '/home/milan/Desktop/thesis/code/'
os.chdir(code_dir)


# imports: ["Ingredients", "dark_matter_only", "with_baryons"]
from halo_model.config.config import Config
from halo_model.models import *


# instantiate a config object:
cfg = Config()


# Construct a DMO power spectrum:
DMO_POWER = dark_matter_only(cfg)
dmo_sample = DMO_POWER.components(k=1)


# construct a baryon power spectrum:
BARYON_POWER = with_baryons(cfg)
baryon_sample = BARYON_POWER.components(k=1)


print(dmo_sample['P_1h_ss'])
print(baryon_sample['P_1h_ss'])

