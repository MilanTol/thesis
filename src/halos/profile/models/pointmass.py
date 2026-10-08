from ..profile import Profile
from ...concentration.concentration import Concentration
from ...r_vir.r_vir import R_vir
from .nfw_helper_functions.Si_Ci_integrals import Si, Ci

import numpy as np


class ProfilePointMass(Profile):
    """
    Instantiates a normalized profile object.
    The profile follows delta_function to model the dark matter + stellar component.
    In fourier space this is simply a constant
    """   

    def _fourier(self, cosmo, k, M, z, **kwargs):
        return 1
