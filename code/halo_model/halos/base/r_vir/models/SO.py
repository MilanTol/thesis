from ..r_vir import R_vir
from .....config.config import Config

import numpy as np

# spherical overdensity model for computing virial radius

class R_virSO(R_vir):

    def __init__(self, cfg:Config):
        """      
        :param delta: average overdensity of halo as a fraction of 
        background matter density at z=0 contained within virial radius.
        """
        self.delta = int(cfg.massdef[:-1])
        self.ref = cfg.massdef[-1]
        if (self.ref != "c") and (self.ref != "m"):
            raise Exception("reference density not valid for SO model")
    
    def R_vir(self, cosmo, M, z):
        if self.ref == "c":
            rho_bg = cosmo.rho_c(z=z)*1e9 #converting to Mpc
        elif self.ref == "m":
            rho_bg = cosmo.rho_m(z=z)*1e9 #converting to Mpc
        else:
            raise Exception("reference density not valid for SO model")
        return (3 * M / (4 * np.pi * self.delta * rho_bg))**(1/3)