R"""
Here all the model choices are stored in a dictionary. This has to be done in a separate file
to prevent circular imports. 
"""

from .params import *
# gas profiles
from ..halos.profile.models.gas_cored_powerlaw import ProfileGasCoredPowerLaw
from ..halos.profile.models.gas_debackere import ProfileGasDebackere
# shmr
from ..halos.shmr.models.moster2013 import SHMRMoster2013
from ..halos.shmr.models.niemiec2022 import SMHRNiemiec2022


    
COMPONENTS = {
    "gas_profile": {
        "siegel2025":    (GasSiegel2025, ProfileGasCoredPowerLaw),
        "debackere2022": (GasDebackere2022, ProfileGasDebackere),
    },
    "shmr": {
        "moster2013": (ShmrMoster2013, SHMRMoster2013),
        "niemiec2022": (ShmrNiemiec2022, SMHRNiemiec2022),
    },
}
    