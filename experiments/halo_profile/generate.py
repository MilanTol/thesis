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
M = 1e14
z = 0

# compute contributions at a range of r values for smooth component:
r = np.geomspace(1e-3, 1e2)
dm_weight = smooth["weights"]["dm"](cfg.cosmo, M, z)
dm_argweight = smooth["arg_weights"]["dm"](cfg.cosmo, M, z)
dm = (
    smooth["weights"]["dm"](cfg.cosmo, M, z)
    * smooth["components"]["dm"].real(cfg.cosmo, r, smooth["arg_weights"]["dm"](cfg.cosmo, M, z), z)
)
stellar = (
    smooth["weights"]["stellar"](cfg.cosmo, M, z)
    * smooth["components"]["stellar"].real(cfg.cosmo, r, smooth["arg_weights"]["stellar"](cfg.cosmo, M, z), z)    
)
gas = (
    smooth["weights"]["gas"](cfg.cosmo, M, z)
    * smooth["components"]["gas"].real(cfg.cosmo, r, smooth["arg_weights"]["gas"](cfg.cosmo, M, z), z)    
)

# safe data 
data_dir = root / "data/halo_profile"
np.savez(data_dir / "contributions.npz", r=r, dm=dm, stellar=stellar, gas=gas) 