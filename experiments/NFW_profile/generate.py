"""
Here we construct an nfw profile class and inspect the real and fourier profiles.
"""

import numpy as np

from clumpy_halo_model import Config
from clumpy_halo_model.models import *

from clumpy_halo_model.halos.profile.models.nfw import ProfileNFW
from clumpy_halo_model.halos.concentration.models.neto2007 import ConcentrationNeto2007
from clumpy_halo_model.halos.r_vir.models.SO import R_virSO

from pathlib import Path
root = Path(__file__).resolve().parents[2]


# load config file
cfg = Config.from_yaml(root/"experiments/NFW_profile/config.yaml")

# construct r_vir, concentration, NFW_profile
r_vir = R_virSO(cfg=cfg)
c = ConcentrationNeto2007(cfg=cfg)
NFW = ProfileNFW(cfg=cfg, c=c, r_vir=r_vir)

# set some mass and redshift:
M = 1e12
z = 0

# compute real and fourier values over a grid:
r = np.geomspace(1e-3, 1e3, 100)
real = NFW.real(cfg.cosmo, r=r, M=M, z=z)

k = np.geomspace(1e-3, 1e3, 100)
fourier = NFW.fourier(cfg.cosmo, k=k, M=M, z=z)
fourier_analytic = NFW.fourier_analytic(cfg.cosmo, k=k, M=M, z=z)

# safe data 
data_dir = root / "data/NFW_profile"
np.savez(data_dir / "real.npz", r=r, real=real)
np.savez(data_dir / "fourier.npz", k=k, fourier=fourier, fourier_analytic=fourier_analytic) 

print(f"fourier profile value at k = {k[0]:.2f} is {fourier[0]:.2f}")
print(f"analytic fourier profile value at k = {k[0]:.2f} is {fourier_analytic[0]:.2f}")