R"""
Here all the model choices are stored in a dictionary. This has to be done in a separate file
to prevent circular imports. 
"""

from .params import *

# profiles
from ..halos.profile.profile import Profile
from ..halos.profile.models.gas_cored_powerlaw import ProfileGasCoredPowerLaw
from ..halos.profile.models.gas_debackere2022 import ProfileGasDebackere2022
from ..halos.profile.models.pointmass import ProfilePointMass
from ..halos.profile.models.stellar_truncated_powerlaw import ProfileStellarTruncatedPowerLaw

# shmr
from ..halos.shmr.shmr import SHMR
from ..halos.shmr.models.moster2013 import SHMRMoster2013, SHMRMoster2013Centrals
from ..halos.shmr.models.niemiec2022 import SMHRNiemiec2022


# tells what kind of object belongs to a component
COMPONENTS = {
    "gas_profile": "profile",
    "star_profile": "profile",
    "shmr": "shmr",
    "shmr_centrals": "shmr",
}

# tells what models are available for each object
MODELS = {
    "profile": {
        "pointmass": (PointMass, ProfilePointMass),
        "gassiegel2025":    (GasSiegel2025, ProfileGasCoredPowerLaw),
        "gasdebackere2022": (GasDebackere2022, ProfileGasDebackere2022),
        "starsiegel2025": (StarSiegel2025, ProfileStellarTruncatedPowerLaw)
    },
    "shmr": {
        "moster2013": (Moster2013, SHMRMoster2013),
        "moster2013centrals": (Moster2013, SHMRMoster2013Centrals),
        "niemiec2022": (Niemiec2022, SMHRNiemiec2022),
    },
}
    