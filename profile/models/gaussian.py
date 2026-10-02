from ..profile import Profile
from ...shmr.shmr import SMHR
import numpy as np

class ProfileGaussian(Profile):
    def __init__(self, smhr:SMHR):
        self._smhr = smhr
        
    def _fourier(self, cosmo, k, M, z, **kwargs):
        M_star = M*self._smhr(cosmo, M, z)
        if np.log10(M_star) < 9:
            sig = 1.5
        else:
            sig = 2.5*(np.log10(M_star) - 9) + 1.5 #some scaling with stellar mass
        sig *= 1e-3 # set units to Mpc
        return np.exp(-0.5 * k**2 * sig**2)
    