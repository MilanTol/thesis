"""
Recreate the Debackere et al. gas fractions (Figure 2) with ProfileGasDebackere:
f_gas_500c
"""

import matplotlib.pyplot as plt
import numpy as np

from clumpy_halo_model.config.config import Config
from clumpy_halo_model.halos.concentration.models.neto2007 import ConcentrationNeto2007
from clumpy_halo_model.halos.profile.models.gas_debackere2022 import ProfileGasDebackere2022
from clumpy_halo_model.halos.shmr.models.moster2013 import SHMRMoster2013

from clumpy_halo_model.algorithms.mass_converter import NFWMassConverter

from pathlib import Path
root = Path(__file__).resolve().parents[2]

# load config file
cfg = Config.from_yaml(root/"experiments/debackere_gas_fractions/config.yaml")

cosmo, z = cfg.cosmo, cfg.z
concentration = ConcentrationNeto2007(cfg)  
shmr = SHMRMoster2013(cfg, concentration)    


# instantiate gas profile object
gas_profile = ProfileGasDebackere2022(cfg, shmr, concentration)

# create M500c grid
M500 = np.geomspace(1e13, 3e15, 100)

# convert the M500 grid to m200c vals to compute the halo masses. 
# Note that this only has to be done to compute the stellar fraction of these halos
# to subsequentally check whether the cosmic baryon fraction is exceed to reduce the gas fraction
# for most haloes the " m200c halo mass" is not required to compute the gas fraction

# compute gas fractions:
gas_fractions = [gas_profile.f_gas_500(M500_t, 
                                       NFWMassConverter(cfg).mass_converter("500c", "200c", M500_t, concentration(cosmo, M500_t, z), z), 
                                       z) 
                 for M500_t in M500]



# -------------------------
# # Interpolate debackere figure 2 data:
# -------------------------

root = Path(__file__).resolve().parents[2]
figures_dir = root / "figures/debackere_gas_fractions" 
data_dir = root / "data/debackere_gas_fractions"

x, y = np.loadtxt(
    data_dir / "figure2_fit.csv",
    delimiter=",",
    unpack=True
)

idx = np.argsort(x)
x = x[idx]
y = y[idx]

from scipy.interpolate import interp1d
f_gas = interp1d(
    np.log(x), y,
    kind="linear",
    bounds_error=False,
    fill_value="extrapolate"
)

gas_fractions_debackere = f_gas(np.log(M500))



# Plotting

plt.figure(figsize=(8,6))
plt.plot(M500, gas_fractions, label='ours')
plt.plot(M500, gas_fractions_debackere, label='debackere')

plt.xscale("log")

plt.legend()

plt.ylim(0, 0.17)
 
plt.savefig(figures_dir/"gas_fractions.pdf")
