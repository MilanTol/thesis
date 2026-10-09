"""
Example on how to create MatterPower objects from a config file and
access the power spectrum values of various components.
"""


from clumpy_halo_model import Config
from clumpy_halo_model import dark_matter_only, with_baryons


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

