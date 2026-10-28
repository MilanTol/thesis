"""
Compute the individual components of the profile to compare their real profiles.
NOTE: one cannot use profiles that do not have a "real" component implemented
e.g. one cannot use the pointmass profile.
"""

import numpy as np

from clumpy_halo_model.config.config import Config
from clumpy_halo_model.config.registry import COMPONENTS, MODELS

from clumpy_halo_model.halos.profile.profile import CompositeProfile
from clumpy_halo_model.halos.profile.weights import (
    CentralStarWeight, ClumpDarkMatterWeight, ClumpStarWeight, GasWeight,
    SmoothDarkMatterWeight, SmoothMassWeight, Unity)

from clumpy_halo_model.models import Ingredients

# load config file
from pathlib import Path
root = Path(__file__).resolve().parents[2]
cfg = Config.from_yaml(root/"experiments/halo_profile/config.yaml")

# load in DM halo ingredients:
parts = Ingredients.build(cfg)

# set concentration model:
concentration = parts.concentration

# load in baryon components from config selection:
shmr = MODELS[COMPONENTS['shmr']][cfg.models['shmr']][1](cfg=cfg, c=concentration)
shmr_centrals = MODELS[COMPONENTS['shmr_centrals']][cfg.models['shmr_centrals']][1](cfg=cfg, c=concentration)
stellar_profile = MODELS[COMPONENTS['star_profile']][cfg.models['star_profile']][1](cfg=cfg, c=concentration)
gas_profile = MODELS[COMPONENTS['gas_profile']][cfg.models['gas_profile']][1](cfg=cfg, shmr=shmr, c=concentration)

# construct dictionary for all profiles of smooth with weights and argweights
smooth = {
    "components": {
        "dm": parts.smooth_dm_profile,
        "stellar": stellar_profile,
        "gas": gas_profile,
    },   
    
    "weights": {
        "dm": SmoothDarkMatterWeight(parts.clump_mass_func),
        "stellar": CentralStarWeight(shmr_centrals),
        "gas": GasWeight(shmr),
    },
    
    "arg_weights": {
        "dm": SmoothDarkMatterWeight(parts.clump_mass_func),
        "stellar": Unity(),
        "gas": Unity(),
    }
}

# set mass and redshift of some halo:
M = 1e13
z = 0

# set reference density (critical density like in Debackere)
rho_c = cfg.cosmo.rho_c(z) * 1e9 # convert to Mpc

# compute contributions at a range of r values for smooth component:
r = np.geomspace(1e-2, 1e1, 100)
dm_real = (
    smooth["weights"]["dm"](cfg.cosmo, M, z)
    * smooth["components"]["dm"].real(cfg.cosmo, r, M*smooth["arg_weights"]["dm"](cfg.cosmo, M, z), z)
) * M / rho_c
stellar_real = (
    smooth["weights"]["stellar"](cfg.cosmo, M, z)
    * smooth["components"]["stellar"].real(cfg.cosmo, r, M*smooth["arg_weights"]["stellar"](cfg.cosmo, M, z), z)    
) * M / rho_c
gas_real = (
    smooth["weights"]["gas"](cfg.cosmo, M, z)
    * smooth["components"]["gas"].real(cfg.cosmo, r, M*smooth["arg_weights"]["gas"](cfg.cosmo, M, z), z)    
) * M / rho_c


# compute contributions at a range of k values for smooth component:
k = np.geomspace(1e-1, 1e4, 100)
dm_fourier = (
    smooth["weights"]["dm"](cfg.cosmo, M, z)
    * smooth["components"]["dm"].fourier(cfg.cosmo, k, M*smooth["arg_weights"]["dm"](cfg.cosmo, M, z), z)
) * M / rho_c
stellar_fourier = (
    smooth["weights"]["stellar"](cfg.cosmo, M, z)
    * smooth["components"]["stellar"].fourier(cfg.cosmo, k, M*smooth["arg_weights"]["stellar"](cfg.cosmo, M, z), z)    
) * M / rho_c
gas_fourier = (
    smooth["weights"]["gas"](cfg.cosmo, M, z)
    * smooth["components"]["gas"].fourier(cfg.cosmo, k, M*smooth["arg_weights"]["gas"](cfg.cosmo, M, z), z)    
) * M / rho_c


# safe data 
data_dir = root / "data/halo_profile"
np.savez(data_dir / "real.npz", r=r, dm=dm_real, stellar=stellar_real, gas=gas_real) 
np.savez(data_dir / "fourier.npz", k=k, dm=dm_fourier, stellar=stellar_fourier, gas=gas_fourier) 


